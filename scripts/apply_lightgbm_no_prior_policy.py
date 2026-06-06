#!/usr/bin/env python3
"""Promote the LightGBM/no-prior-score NPV policy to active submission A.

This applies the best candidate from `experiment_model_family_npv_bakeoff.py`:

calibrated LightGBM PDs, no prior-underwriter-score/proxy features,
expected NPV via the active timing/recovery curves, and a validation-selected
NPV margin threshold.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table  # noqa: E402
from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


DROP_PRIOR_SCORE_PROXIES = {
    "prior_underwriter_score",
    "prior_score_logit",
    "selection_support_index",
}
MIN_PD_INTERVAL_HALF_WIDTH = 0.06
PRIOR_DECLINED_MIN_MARGIN = 0.03


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


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


def main() -> None:
    root = Path(".")
    csv_dir = root / "data" / "csv-files"
    output_dir = root / "outputs"
    report_dir = output_dir / "reports"
    submission_dir = output_dir / "submission"
    archive_dir = report_dir / "archive"
    report_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(csv_dir / "train.csv")
    validation = pd.read_csv(csv_dir / "validation.csv")
    test = pd.read_csv(csv_dir / "test.csv")
    current_submission = pd.read_csv(submission_dir / "submission_A_decisions.csv")
    curves = np.load(output_dir / "deliverable_a_curves.npz")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    numeric = [c for c in numeric if c not in DROP_PRIOR_SCORE_PROXIES]
    categorical = [c for c in categorical if c not in DROP_PRIOR_SCORE_PROXIES]
    feature_cols = numeric + categorical

    model_idx, cal_idx = time_split_labeled(train)
    train_x = train_fe.loc[model_idx, feature_cols].copy()
    cal_x = train_fe.loc[cal_idx, feature_cols].copy()
    val_x = validation_fe[feature_cols].copy()
    test_x = test_fe[feature_cols].copy()
    train_y = train.loc[model_idx, "default_flag"].astype(int)
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()

    for col in categorical:
        train_x[col] = train_x[col].astype("category")
        cal_x[col] = cal_x[col].astype("category")
        val_x[col] = val_x[col].astype("category")
        test_x[col] = test_x[col].astype("category")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=1103,
        verbosity=-1,
    )
    model.fit(train_x, train_y, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, cal_y)

    cal_pd = np.clip(calibrator.predict(raw_cal), 0.001, 0.999)
    val_pd = np.clip(calibrator.predict(model.predict_proba(val_x)[:, 1]), 0.001, 0.999)
    test_pd = np.clip(calibrator.predict(model.predict_proba(test_x)[:, 1]), 0.001, 0.999)

    bin_table = pd_interval_bin_table(cal_pd, cal_y, n_bins=10)
    val_lower, val_upper = build_pd_intervals(val_pd, val_pd[:, None], bin_table)
    test_lower, test_upper = build_pd_intervals(test_pd, test_pd[:, None], bin_table)
    # The single LightGBM model has little ensemble dispersion; the calibration
    # bins alone were too narrow on labeled validation. Keep a conservative
    # half-width floor so Scal is not sacrificed for the small NPV lift.
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

    backup = archive_dir / "submission_A_decisions_before_lightgbm_no_prior.csv"
    if not backup.exists():
        current_submission.to_csv(backup, index=False)

    updated = pd.concat(
        [
            pd.DataFrame(
                {
                    "applicant_id": validation["applicant_id"],
                    "decision": val_decision,
                    "predicted_pd": val_pd,
                    "pd_lower_90": val_lower,
                    "pd_upper_90": val_upper,
                }
            ),
            pd.DataFrame(
                {
                    "applicant_id": test["applicant_id"],
                    "decision": test_decision,
                    "predicted_pd": test_pd,
                    "pd_lower_90": test_lower,
                    "pd_upper_90": test_upper,
                }
            ),
        ],
        ignore_index=True,
    )
    updated["pd_lower_90"] = np.minimum(updated["pd_lower_90"].clip(0, 1), updated["predicted_pd"])
    updated["pd_upper_90"] = np.maximum(updated["pd_upper_90"].clip(0, 1), updated["predicted_pd"])
    updated.to_csv(submission_dir / "submission_A_decisions.csv", index=False)

    val_y = validation.loc[labeled_val, "default_flag"].astype(int).to_numpy()
    coverage = bin_level_coverage(val_pd[labeled_val], val_y, val_lower[labeled_val], val_upper[labeled_val])
    summary = {
        "policy": "lightgbm_no_prior_score",
        "threshold": threshold,
        "prior_declined_min_margin": PRIOR_DECLINED_MIN_MARGIN,
        "validation_labeled_realized_npv": float(realized_val[val_decision[labeled_val] == 1].sum()),
        "validation_labeled_approved": int(val_decision[labeled_val].sum()),
        "validation_labeled_approval_rate": float(val_decision[labeled_val].mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized_val[val_decision[labeled_val] == 1] < 0)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(val_decision.sum() + test_decision.sum()),
        "prior_declined_approved_total": int(
            (
                np.r_[val_decision, test_decision].astype(bool)
                & (pd.concat([validation["prior_decision"], test["prior_decision"]], ignore_index=True).to_numpy() == 0)
            ).sum()
        ),
        "validation_pd_metrics": {
            "auroc": float(roc_auc_score(val_y, val_pd[labeled_val])),
            "log_loss": float(log_loss(val_y, val_pd[labeled_val], labels=[0, 1])),
            "brier": float(brier_score_loss(val_y, val_pd[labeled_val])),
            "mean_pd": float(np.mean(val_pd[labeled_val])),
            "actual_default_rate": float(np.mean(val_y)),
        },
        "validation_interval_coverage_90": coverage,
        "pd_interval_half_width_floor": MIN_PD_INTERVAL_HALF_WIDTH,
        "replaced_policy_note": "Promoted from model-family NPV bakeoff; prior-declined guardrail reduces unverified reject-region exposure.",
    }
    sweep.to_csv(report_dir / "lightgbm_no_prior_policy_threshold_sweep.csv", index=False)
    bin_table.to_csv(report_dir / "lightgbm_no_prior_pd_interval_bins.csv", index=False)
    (report_dir / "lightgbm_no_prior_active_policy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
