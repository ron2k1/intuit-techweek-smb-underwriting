#!/usr/bin/env python3
"""Purged temporal OOF validation for the active raw-valid-prior A policy.

The official train/validation/test split is fixed. This diagnostic improves the
internal model-selection audit by using expanding temporal folds, purging
businesses that appear in each validation fold, producing out-of-fold PD/NPV
margins, and comparing an OOF-selected threshold with the current active
official-validation-selected threshold.

It does not overwrite the active submission.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_compact_feature_reject_bakeoff import (  # noqa: E402
    PRIOR_DECLINED_MIN_MARGIN,
    choose_threshold,
    raw_valid_prior_columns,
)
from src.economics import expected_npv, realized_npv  # noqa: E402
from src.timing import fit_default_day_model, fit_recovery_model  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
CANDIDATE_DIR = OUTPUT_DIR / "candidates" / "purged_temporal_oof"
RANDOM_SEED = 2031


def prepare_raw_frames(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    feature_cols, _numeric, categorical = raw_valid_prior_columns(train)
    train_x = train[feature_cols].copy()
    validation_x = validation[feature_cols].copy()
    test_x = test[feature_cols].copy()
    for col in categorical:
        train_x[col] = train_x[col].astype("category")
        validation_x[col] = validation_x[col].astype("category")
        test_x[col] = test_x[col].astype("category")
    return feature_cols, categorical, train_x, validation_x, test_x


def temporal_model_cal_split(train: pd.DataFrame, idx: np.ndarray, fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    ordered = train.loc[idx].sort_values("application_timestamp").index.to_numpy()
    split_at = max(1, min(len(ordered) - 1, int(len(ordered) * fraction)))
    model_idx = ordered[:split_at]
    cal_idx = ordered[split_at:]
    cal_businesses = set(train.loc[cal_idx, "business_id"])
    model_idx = np.asarray([i for i in model_idx if train.loc[i, "business_id"] not in cal_businesses])
    return model_idx, cal_idx


def fit_pd_model(
    train_x: pd.DataFrame,
    train: pd.DataFrame,
    model_idx: np.ndarray,
    cal_idx: np.ndarray,
    categorical: list[str],
    *,
    seed: int,
):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.028,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        train_x.loc[model_idx],
        train.loc[model_idx, "default_flag"].astype(int).to_numpy(),
        categorical_feature=categorical,
    )
    raw_cal = model.predict_proba(train_x.loc[cal_idx])[:, 1]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, cal_y)
    return model, calibrator


def predict_pd(model: LGBMClassifier, calibrator: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    return np.clip(calibrator.predict(model.predict_proba(x)[:, 1]), 0.001, 0.999)


def policy_metrics(
    frame: pd.DataFrame,
    margin: np.ndarray,
    threshold: float,
    *,
    apply_prior_guard: bool,
) -> dict[str, float | int]:
    decision = margin > threshold
    if apply_prior_guard:
        decision = decision & (
            (frame["prior_decision"].to_numpy() == 1) | (margin > PRIOR_DECLINED_MIN_MARGIN)
        )
    realized = (
        frame["realized_npv"].to_numpy(float)
        if "realized_npv" in frame.columns
        else realized_npv(frame)
    )
    y = frame["default_flag"].astype(int).to_numpy()
    return {
        "threshold": float(threshold),
        "approved": int(decision.sum()),
        "approval_rate": float(decision.mean()),
        "realized_npv": float(realized[decision].sum()),
        "observed_default_rate_approved": float(y[decision].mean()) if decision.sum() else float("nan"),
        "mean_margin_approved": float(margin[decision].mean()) if decision.sum() else float("nan"),
    }


def build_oof(train: pd.DataFrame, train_x: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = train[train["default_flag"].notna()].sort_values("application_timestamp").copy()
    blocks = np.array_split(labeled.index.to_numpy(), 5)
    oof_rows = []
    fold_rows = []
    for fold_id in range(1, 5):
        val_idx = blocks[fold_id]
        train_pool_idx = np.concatenate(blocks[:fold_id])
        val_businesses = set(train.loc[val_idx, "business_id"])
        purged_train_idx = np.asarray(
            [i for i in train_pool_idx if train.loc[i, "business_id"] not in val_businesses]
        )
        if len(purged_train_idx) < 1000 or len(val_idx) < 1000:
            continue

        model_idx, cal_idx = temporal_model_cal_split(train, purged_train_idx)
        model, calibrator = fit_pd_model(
            train_x,
            train,
            model_idx,
            cal_idx,
            categorical,
            seed=RANDOM_SEED + fold_id,
        )
        val_pd = predict_pd(model, calibrator, train_x.loc[val_idx])

        # Day-level timing and recovery are trained on the same purged historical window. This
        # validates the whole expected-profit decision surface, not only PD rank.
        day_model = fit_default_day_model(
            train_x.loc[purged_train_idx],
            train.loc[purged_train_idx][train.loc[purged_train_idx, "default_flag"] == 1],
        )
        t_star = day_model.predict_day(train_x.loc[val_idx])
        recovery = fit_recovery_model(train_x, train.loc[purged_train_idx][train.loc[purged_train_idx, "default_flag"] == 1])
        recovery_rate = recovery.predict_rate(train_x.loc[val_idx])

        val_frame = train.loc[val_idx].copy()
        margin = expected_npv(
            val_frame["requested_amount"].to_numpy(float),
            val_pd,
            t_star,
            recovery_rate,
        ) / np.maximum(val_frame["requested_amount"].to_numpy(float), 1.0)

        y = val_frame["default_flag"].astype(int).to_numpy()
        fold_sweep_threshold, fold_sweep = choose_threshold(margin, realized_npv(val_frame))
        fold_rows.append(
            {
                "fold": fold_id,
                "train_rows_before_group_purge": int(len(train_pool_idx)),
                "train_rows_after_group_purge": int(len(purged_train_idx)),
                "pd_model_rows": int(len(model_idx)),
                "pd_calibration_rows": int(len(cal_idx)),
                "purged_train_rows": int(len(train_pool_idx) - len(purged_train_idx)),
                "purged_model_cal_rows": int(len(purged_train_idx) - len(model_idx) - len(cal_idx)),
                "validation_rows": int(len(val_idx)),
                "validation_start": str(pd.to_datetime(val_frame["application_timestamp"]).min()),
                "validation_end": str(pd.to_datetime(val_frame["application_timestamp"]).max()),
                "default_rate": float(y.mean()),
                "auroc": float(roc_auc_score(y, val_pd)),
                "log_loss": float(log_loss(y, val_pd, labels=[0, 1])),
                "brier": float(brier_score_loss(y, val_pd)),
                "mean_pd": float(val_pd.mean()),
                "best_fold_threshold": float(fold_sweep_threshold),
                "best_fold_realized_npv": float(fold_sweep.iloc[0]["realized_npv"]),
                "best_fold_approved": int(fold_sweep.iloc[0]["approved"]),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_id,
                    "row_index": val_idx,
                    "business_id": val_frame["business_id"].to_numpy(),
                    "application_timestamp": val_frame["application_timestamp"].to_numpy(),
                    "default_flag": y,
                    "requested_amount": val_frame["requested_amount"].to_numpy(float),
                    "predicted_pd": val_pd,
                    "expected_t_star": t_star,
                    "expected_recovery_rate": recovery_rate,
                    "expected_npv_margin": margin,
                    "realized_npv": realized_npv(val_frame),
                }
            )
        )

    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def official_threshold_comparison(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    active_threshold: float,
    oof_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission_a = pd.read_csv(OUTPUT_DIR / "submission" / "submission_A_decisions.csv")
    curves = np.load(OUTPUT_DIR / "deliverable_a_curves.npz")
    val_pd = submission_a.iloc[: len(validation)]["predicted_pd"].to_numpy(float)
    test_pd = submission_a.iloc[len(validation) :]["predicted_pd"].to_numpy(float)
    val_margin = expected_npv(
        validation["requested_amount"].to_numpy(float),
        val_pd,
        curves["validation_t_star"],
        curves["validation_recovery"],
    ) / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
    test_margin = expected_npv(
        test["requested_amount"].to_numpy(float),
        test_pd,
        curves["test_t_star"],
        curves["test_recovery"],
    ) / np.maximum(test["requested_amount"].to_numpy(float), 1.0)

    labeled = validation["default_flag"].notna().to_numpy()
    rows = []
    candidate_frames = []
    for name, threshold in [("active_threshold", active_threshold), ("oof_selected_threshold", oof_threshold)]:
        val_decision = (val_margin > threshold) & (
            (validation["prior_decision"].to_numpy() == 1) | (val_margin > PRIOR_DECLINED_MIN_MARGIN)
        )
        test_decision = (test_margin > threshold) & (
            (test["prior_decision"].to_numpy() == 1) | (test_margin > PRIOR_DECLINED_MIN_MARGIN)
        )
        all_decision = np.r_[val_decision, test_decision]
        all_prior_declined = pd.concat([validation["prior_decision"], test["prior_decision"]], ignore_index=True).to_numpy() == 0
        all_enpv = np.r_[
            expected_npv(validation["requested_amount"].to_numpy(float), val_pd, curves["validation_t_star"], curves["validation_recovery"]),
            expected_npv(test["requested_amount"].to_numpy(float), test_pd, curves["test_t_star"], curves["test_recovery"]),
        ]
        realized = realized_npv(validation.loc[labeled])
        rows.append(
            {
                "policy": name,
                "threshold": float(threshold),
                "validation_labeled_realized_npv": float(realized[val_decision[labeled]].sum()),
                "validation_labeled_approved": int(val_decision[labeled].sum()),
                "validation_labeled_default_rate_approved": float(
                    validation.loc[labeled, "default_flag"].to_numpy(float)[val_decision[labeled]].mean()
                ),
                "validation_all_approval_rate": float(val_decision.mean()),
                "test_approval_rate": float(test_decision.mean()),
                "approved_total": int(all_decision.sum()),
                "prior_declined_approved_total": int((all_decision & all_prior_declined).sum()),
                "headline_expected_npv": float(all_enpv[all_decision].sum()),
            }
        )
        candidate_frames.append(
            (
                name,
                pd.concat(
                    [
                        submission_a.iloc[: len(validation)].assign(decision=val_decision.astype(int)),
                        submission_a.iloc[len(validation) :].assign(decision=test_decision.astype(int)),
                    ],
                    ignore_index=True,
                ),
            )
        )
    return pd.DataFrame(rows), pd.DataFrame()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    _feature_cols, categorical, train_x, _validation_x, _test_x = prepare_raw_frames(train, validation, test)

    oof, folds = build_oof(train, train_x, categorical)
    oof_threshold, oof_sweep = choose_threshold(
        oof["expected_npv_margin"].to_numpy(float),
        oof["realized_npv"].to_numpy(float),
    )
    active_summary_path = REPORT_DIR / "lightgbm_no_prior_active_policy_summary.json"
    active_threshold = json.loads(active_summary_path.read_text()).get("threshold", -0.025)
    official_compare, _ = official_threshold_comparison(
        validation,
        test,
        active_threshold=float(active_threshold),
        oof_threshold=float(oof_threshold),
    )

    oof_active = policy_metrics(oof, oof["expected_npv_margin"].to_numpy(float), float(active_threshold), apply_prior_guard=False)
    oof_oof = policy_metrics(oof, oof["expected_npv_margin"].to_numpy(float), float(oof_threshold), apply_prior_guard=False)
    summary = {
        "objective": "Purged expanding-window temporal OOF validation for raw-valid-prior A policy.",
        "fold_count": int(len(folds)),
        "oof_rows": int(len(oof)),
        "active_threshold": float(active_threshold),
        "oof_selected_threshold": float(oof_threshold),
        "oof_active_threshold": oof_active,
        "oof_selected_threshold_metrics": oof_oof,
        "oof_pd_metrics": {
            "auroc": float(roc_auc_score(oof["default_flag"], oof["predicted_pd"])),
            "log_loss": float(log_loss(oof["default_flag"], oof["predicted_pd"], labels=[0, 1])),
            "brier": float(brier_score_loss(oof["default_flag"], oof["predicted_pd"])),
            "mean_pd": float(oof["predicted_pd"].mean()),
            "actual_default_rate": float(oof["default_flag"].mean()),
        },
        "official_validation_comparison": official_compare.to_dict("records"),
        "interpretation": [
            "Official train/validation/test split is unchanged.",
            "OOF folds are expanding-window temporal folds with business_id purge from validation fold.",
            "OOF threshold is used as a robustness diagnostic; current active can be kept if official-validation NPV gain outweighs OOF overfit risk.",
        ],
    }

    oof.to_csv(REPORT_DIR / "purged_temporal_oof_predictions.csv", index=False)
    folds.to_csv(REPORT_DIR / "purged_temporal_oof_fold_metrics.csv", index=False)
    oof_sweep.to_csv(REPORT_DIR / "purged_temporal_oof_threshold_sweep.csv", index=False)
    official_compare.to_csv(REPORT_DIR / "purged_temporal_oof_official_threshold_comparison.csv", index=False)
    (REPORT_DIR / "purged_temporal_oof_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
