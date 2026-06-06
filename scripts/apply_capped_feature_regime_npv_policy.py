#!/usr/bin/env python3
"""Apply a capped-economics feature-regime NPV policy for Deliverable A.

The useful regime clue is not a single calendar-day shock. The largest
train-to-future feature shift is `observed_revenue_trend_3mo`: validation/test
live in a weaker-revenue regime than most of the training history. Under the
current 60-day draw-capped economics, broad PD regime offsets do not improve
NPV much, but a direct value model trained on this future-like feature regime
does improve capped validation NPV.

The value model is an ensemble over fixed HGB seeds, aggregated by a trimmed
mean, because the capped NPV frontier is narrow and a single fitted tree model
is too seed-sensitive.

This script changes only A decisions. PD point estimates and intervals remain
from the active calibrated PD model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
ARCHIVE_DIR = REPORT_DIR / "archive"
SUBMISSION_A = SUBMISSION_DIR / "submission_A_decisions.csv"

REVENUE_TREND_CUTOFF = -0.20
PRIOR_DECLINED_SCORE_FLOOR = 0.03
ENSEMBLE_SEEDS = [1, 2, 3, 5, 7, 11, 17, 23, 29, 37, 41, 43, 53, 71, 101, 137, 2026, 2443]
TRIM_EACH_SIDE = 2


def make_model(numeric: list[str], categorical: list[str], *, seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_iter=260,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    loss="squared_error",
                    random_state=seed,
                ),
            ),
        ]
    )


def trimmed_mean_prediction(predictions: list[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(predictions)
    if matrix.shape[0] <= 2 * TRIM_EACH_SIDE:
        return matrix.mean(axis=0)
    return np.sort(matrix, axis=0)[TRIM_EACH_SIDE:-TRIM_EACH_SIDE].mean(axis=0)


def future_like_revenue_regime(frame: pd.DataFrame) -> np.ndarray:
    trend = pd.to_numeric(frame["observed_revenue_trend_3mo"], errors="coerce")
    return (trend.isna() | (trend < REVENUE_TREND_CUTOFF)).to_numpy()


def choose_threshold(
    score: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    realized_validation: np.ndarray,
    labeled_validation: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    n_val = len(validation)
    val_score = score[:n_val]
    test_score = score[n_val:]
    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0

    candidates = np.unique(
        np.r_[
            np.linspace(-0.08, 0.12, 81),
            np.quantile(val_score[labeled_validation], np.linspace(0.01, 0.99, 99)),
        ]
    )
    rows: list[dict[str, float | int]] = []
    for threshold in candidates:
        val_decision = (val_score > threshold) & (
            ~val_prior_declined | (val_score > PRIOR_DECLINED_SCORE_FLOOR)
        )
        test_decision = (test_score > threshold) & (
            ~test_prior_declined | (test_score > PRIOR_DECLINED_SCORE_FLOOR)
        )
        labeled_decision = val_decision[labeled_validation]
        if labeled_decision.sum() == 0:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "validation_labeled_realized_npv": float(
                    realized_validation[labeled_decision].sum()
                ),
                "validation_labeled_approved": int(labeled_decision.sum()),
                "validation_all_approval_rate": float(val_decision.mean()),
                "test_approval_rate": float(test_decision.mean()),
                "approved_total": int(val_decision.sum() + test_decision.sum()),
                "prior_declined_approved_total": int(
                    (val_decision & val_prior_declined).sum()
                    + (test_decision & test_prior_declined).sum()
                ),
            }
        )
    sweep = pd.DataFrame(rows).sort_values(
        ["validation_labeled_realized_npv", "approved_total"],
        ascending=[False, True],
    )
    return float(sweep.iloc[0]["threshold"]), sweep.reset_index(drop=True)


def summarize_policy(
    policy: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    decision: np.ndarray,
    labeled_validation: np.ndarray,
    realized_validation: np.ndarray,
    score: np.ndarray,
    threshold: float | None,
) -> dict[str, float | int | str | None]:
    n_val = len(validation)
    val_decision = decision[:n_val]
    test_decision = decision[n_val:]
    prior_declined = np.r_[
        validation["prior_decision"].to_numpy() == 0,
        test["prior_decision"].to_numpy() == 0,
    ]
    labeled_decision = val_decision[labeled_validation]
    return {
        "policy": policy,
        "threshold": threshold,
        "validation_labeled_realized_npv": float(realized_validation[labeled_decision].sum()),
        "validation_labeled_approved": int(labeled_decision.sum()),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(decision.sum()),
        "prior_declined_approved_total": int((decision & prior_declined).sum()),
        "mean_score_approved": float(score[decision].mean()) if decision.sum() else float("nan"),
    }


def run(*, promote: bool) -> dict[str, object]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission = pd.read_csv(SUBMISSION_A)

    n_val = len(validation)
    if len(submission) != n_val + len(test):
        raise ValueError(f"submission A row count mismatch: {len(submission)}")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _feature_cols, numeric, categorical = feature_columns(train_fe)
    feature_cols = numeric + categorical

    labeled_train = train["default_flag"].notna().to_numpy()
    regime_train = future_like_revenue_regime(train_fe)
    fit_mask = labeled_train & regime_train
    if fit_mask.sum() < 1_000:
        raise ValueError(f"too few future-like labeled training rows: {fit_mask.sum()}")

    y_margin = realized_npv(train.loc[fit_mask]) / np.maximum(
        train.loc[fit_mask, "requested_amount"].to_numpy(float),
        1.0,
    )
    val_predictions = []
    test_predictions = []
    for seed in ENSEMBLE_SEEDS:
        model = make_model(numeric, categorical, seed=seed)
        model.fit(train_fe.loc[fit_mask, feature_cols], y_margin)
        val_predictions.append(model.predict(validation_fe[feature_cols]))
        test_predictions.append(model.predict(test_fe[feature_cols]))

    val_score = trimmed_mean_prediction(val_predictions)
    test_score = trimmed_mean_prediction(test_predictions)
    score = np.r_[val_score, test_score]

    labeled_validation = validation["default_flag"].notna().to_numpy()
    realized_validation = realized_npv(validation.loc[labeled_validation])
    threshold, sweep = choose_threshold(
        score,
        validation,
        test,
        realized_validation,
        labeled_validation,
    )

    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0
    val_decision = (val_score > threshold) & (
        ~val_prior_declined | (val_score > PRIOR_DECLINED_SCORE_FLOOR)
    )
    test_decision = (test_score > threshold) & (
        ~test_prior_declined | (test_score > PRIOR_DECLINED_SCORE_FLOOR)
    )
    candidate_decision = np.r_[val_decision, test_decision]
    active_decision = submission["decision"].to_numpy(int).astype(bool)

    active_summary = summarize_policy(
        "active_submission_before_capped_feature_regime",
        validation,
        test,
        active_decision,
        labeled_validation,
        realized_validation,
        score,
        None,
    )
    candidate_summary = summarize_policy(
        "capped_feature_revenue_trend_future_like",
        validation,
        test,
        candidate_decision,
        labeled_validation,
        realized_validation,
        score,
        threshold,
    )

    sweep.to_csv(REPORT_DIR / "capped_feature_regime_npv_threshold_sweep.csv", index=False)

    updated = submission.copy()
    updated["decision"] = candidate_decision.astype(int)
    candidate_path = OUTPUT_DIR / "candidates" / "capped_feature_regime_npv" / "submission_A_decisions.csv"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(candidate_path, index=False)

    promoted = False
    if promote:
        if (
            candidate_summary["validation_labeled_realized_npv"]
            <= active_summary["validation_labeled_realized_npv"]
        ):
            raise RuntimeError("candidate did not improve capped validation NPV; refusing to promote")
        backup = ARCHIVE_DIR / "submission_A_decisions_before_capped_feature_regime_npv.csv"
        if not backup.exists():
            shutil.copy2(SUBMISSION_A, backup)
        updated.to_csv(SUBMISSION_A, index=False)
        promoted = True

    report: dict[str, object] = {
        "objective": "Capped-economics feature-regime direct NPV policy for Deliverable A.",
        "promoted": promoted,
        "regime_definition": {
            "feature": "observed_revenue_trend_3mo",
            "future_like_rule": f"missing or < {REVENUE_TREND_CUTOFF}",
            "labeled_train_rows": int(labeled_train.sum()),
            "future_like_labeled_train_rows": int(fit_mask.sum()),
            "validation_future_like_share": float(future_like_revenue_regime(validation_fe).mean()),
            "test_future_like_share": float(future_like_revenue_regime(test_fe).mean()),
        },
        "ensemble": {
            "seeds": ENSEMBLE_SEEDS,
            "aggregation": f"trimmed_mean_drop_{TRIM_EACH_SIDE}_each_side",
        },
        "prior_declined_score_floor": PRIOR_DECLINED_SCORE_FLOOR,
        "active_summary": active_summary,
        "candidate_summary": candidate_summary,
        "delta_validation_labeled_realized_npv": float(
            candidate_summary["validation_labeled_realized_npv"]
            - active_summary["validation_labeled_realized_npv"]
        ),
        "changed_decisions": int((candidate_decision != active_decision).sum()),
        "outputs": {
            "threshold_sweep": str(REPORT_DIR / "capped_feature_regime_npv_threshold_sweep.csv"),
            "candidate_submission_a": str(candidate_path),
        },
    }
    (REPORT_DIR / "capped_feature_regime_npv_policy_summary.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(promote=args.promote), indent=2))


if __name__ == "__main__":
    main()
