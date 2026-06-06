#!/usr/bin/env python3
"""Build a reject-inference simple-augmentation candidate A policy.

This does not overwrite the active submission. It writes a candidate A file and
summary under outputs/candidates/reject_simple_gamma_*/ so we can compare before
promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DELIVERABLE_A_FEATURE_SET", "all_engineered")

from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table  # noqa: E402
from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


PRIOR_DECLINED_MIN_MARGIN = 0.03
MIN_PD_INTERVAL_HALF_WIDTH = 0.06
RANDOM_SEED = 2026


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def fit_model(
    fit_x: pd.DataFrame,
    fit_y: np.ndarray,
    cal_x: pd.DataFrame,
    cal_y: np.ndarray,
    categorical: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[LGBMClassifier, IsotonicRegression]:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(fit_x, fit_y, sample_weight=sample_weight, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, cal_y)
    return model, calibrator


def predict_pd(model: LGBMClassifier, calibrator: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    return np.clip(calibrator.predict(model.predict_proba(x)[:, 1]), 0.001, 0.999)


def choose_threshold(margin: np.ndarray, realized: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.r_[
            np.linspace(-0.05, 0.08, 53),
            np.quantile(margin, np.linspace(0.01, 0.99, 79)),
        ]
    )
    rows = []
    for threshold in candidates:
        decision = margin > threshold
        if decision.sum() == 0:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "approved": int(decision.sum()),
                "approval_rate": float(decision.mean()),
                "realized_npv": float(realized[decision].sum()),
                "mean_realized_npv_approved": float(realized[decision].mean()),
                "observed_default_rate_approved": float(np.mean(realized[decision] < 0)),
            }
        )
    sweep = pd.DataFrame(rows).sort_values("realized_npv", ascending=False).reset_index(drop=True)
    return float(sweep.iloc[0]["threshold"]), sweep


def build_candidate(gamma: float, reject_weight_scale: float) -> dict[str, object]:
    root = PROJECT_ROOT
    csv_dir = root / "data" / "csv-files"
    output_dir = root / "outputs"
    candidate_dir = output_dir / "candidates" / f"reject_simple_gamma_{str(gamma).replace('.', 'p')}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(csv_dir / "train.csv")
    validation = pd.read_csv(csv_dir / "validation.csv")
    test = pd.read_csv(csv_dir / "test.csv")
    curves = np.load(output_dir / "deliverable_a_curves.npz")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    feature_cols = numeric + categorical
    train_x = train_fe[feature_cols].copy()
    val_x = validation_fe[feature_cols].copy()
    test_x = test_fe[feature_cols].copy()

    for col in categorical:
        train_x[col] = train_x[col].astype("category")
        val_x[col] = val_x[col].astype("category")
        test_x[col] = test_x[col].astype("category")

    model_idx, cal_idx = time_split_labeled(train)
    reject_idx = train.index[train["default_flag"].isna()].to_numpy()
    fit_y = train.loc[model_idx, "default_flag"].astype(int).to_numpy()
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()

    base_model, base_cal = fit_model(
        train_x.loc[model_idx],
        fit_y,
        train_x.loc[cal_idx],
        cal_y,
        categorical,
        random_state=RANDOM_SEED,
    )
    reject_base_pd = predict_pd(base_model, base_cal, train_x.loc[reject_idx])
    reject_pd_stressed = expit(logit(np.clip(reject_base_pd, 0.001, 0.999)) + np.log(gamma))
    accept_bad_rate = float(train.loc[model_idx, "default_flag"].mean())
    target_reject_bad_rate = min(max(accept_bad_rate * gamma, accept_bad_rate), 0.85)
    cutoff = float(np.quantile(reject_pd_stressed, 1.0 - target_reject_bad_rate))
    pseudo_y = (reject_pd_stressed >= cutoff).astype(int)

    augmented_x = pd.concat([train_x.loc[model_idx], train_x.loc[reject_idx]], ignore_index=True)
    augmented_y = np.r_[fit_y, pseudo_y]
    weights = np.r_[np.ones(len(fit_y)), np.full(len(pseudo_y), reject_weight_scale)]

    model, calibrator = fit_model(
        augmented_x,
        augmented_y,
        train_x.loc[cal_idx],
        cal_y,
        categorical,
        sample_weight=weights,
        random_state=RANDOM_SEED + int(gamma * 100),
    )
    cal_pd = predict_pd(model, calibrator, train_x.loc[cal_idx])
    val_pd = predict_pd(model, calibrator, val_x)
    test_pd = predict_pd(model, calibrator, test_x)

    bin_table = pd_interval_bin_table(cal_pd, cal_y, n_bins=10)
    val_lower, val_upper = build_pd_intervals(val_pd, val_pd[:, None], bin_table)
    test_lower, test_upper = build_pd_intervals(test_pd, test_pd[:, None], bin_table)
    val_lower = np.minimum(val_lower, np.clip(val_pd - MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    val_upper = np.maximum(val_upper, np.clip(val_pd + MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    test_lower = np.minimum(test_lower, np.clip(test_pd - MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    test_upper = np.maximum(test_upper, np.clip(test_pd + MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))

    val_enpv = expected_npv(
        validation["requested_amount"].to_numpy(float),
        val_pd,
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    test_enpv = expected_npv(
        test["requested_amount"].to_numpy(float),
        test_pd,
        curves["test_t_star"],
        curves["test_recovery"],
    )
    val_margin = val_enpv / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
    test_margin = test_enpv / np.maximum(test["requested_amount"].to_numpy(float), 1.0)
    labeled_val = validation["default_flag"].notna().to_numpy()
    realized_val = realized_npv(validation.loc[labeled_val])
    threshold, sweep = choose_threshold(val_margin[labeled_val], realized_val)

    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0
    val_decision = (
        (val_margin > threshold)
        & (~val_prior_declined | (val_margin > PRIOR_DECLINED_MIN_MARGIN))
    ).astype(int)
    test_decision = (
        (test_margin > threshold)
        & (~test_prior_declined | (test_margin > PRIOR_DECLINED_MIN_MARGIN))
    ).astype(int)

    submission = pd.concat(
        [
            pd.DataFrame(
                {
                    "applicant_id": validation["applicant_id"],
                    "decision": val_decision,
                    "predicted_pd": val_pd,
                    "pd_lower_90": np.minimum(val_lower, val_pd),
                    "pd_upper_90": np.maximum(val_upper, val_pd),
                }
            ),
            pd.DataFrame(
                {
                    "applicant_id": test["applicant_id"],
                    "decision": test_decision,
                    "predicted_pd": test_pd,
                    "pd_lower_90": np.minimum(test_lower, test_pd),
                    "pd_upper_90": np.maximum(test_upper, test_pd),
                }
            ),
        ],
        ignore_index=True,
    )
    submission.to_csv(candidate_dir / "submission_A_decisions.csv", index=False)
    sweep.to_csv(candidate_dir / "threshold_sweep.csv", index=False)
    bin_table.to_csv(candidate_dir / "pd_interval_bins.csv", index=False)

    val_y = validation.loc[labeled_val, "default_flag"].astype(int).to_numpy()
    all_prior_decision = pd.concat([validation["prior_decision"], test["prior_decision"]], ignore_index=True).to_numpy()
    all_decision = np.r_[val_decision, test_decision].astype(bool)
    coverage = bin_level_coverage(val_pd[labeled_val], val_y, val_lower[labeled_val], val_upper[labeled_val])
    summary = {
        "policy": "reject_inference_simple_augmentation_candidate",
        "gamma": gamma,
        "reject_weight_scale": reject_weight_scale,
        "accept_bad_rate_model_split": accept_bad_rate,
        "target_reject_bad_rate": target_reject_bad_rate,
        "pseudo_reject_bad_rate": float(pseudo_y.mean()),
        "pseudo_reject_bad_cutoff": cutoff,
        "threshold": threshold,
        "prior_declined_min_margin": PRIOR_DECLINED_MIN_MARGIN,
        "validation_labeled_realized_npv": float(realized_val[val_decision[labeled_val] == 1].sum()),
        "validation_labeled_approved": int(val_decision[labeled_val].sum()),
        "validation_labeled_approval_rate": float(val_decision[labeled_val].mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized_val[val_decision[labeled_val] == 1] < 0)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "prior_declined_approved_total": int((all_decision & (all_prior_decision == 0)).sum()),
        "prior_declined_approval_rate": float((all_decision & (all_prior_decision == 0)).sum() / max((all_prior_decision == 0).sum(), 1)),
        "validation_pd_metrics": {
            "auroc": float(roc_auc_score(val_y, val_pd[labeled_val])),
            "log_loss": float(log_loss(val_y, val_pd[labeled_val], labels=[0, 1])),
            "brier": float(brier_score_loss(val_y, val_pd[labeled_val])),
            "mean_pd": float(np.mean(val_pd[labeled_val])),
            "actual_default_rate": float(np.mean(val_y)),
        },
        "validation_interval_coverage_90": coverage,
    }
    (candidate_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--reject-weight-scale", type=float, default=0.35)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.gamma, args.reject_weight_scale), indent=2))


if __name__ == "__main__":
    main()
