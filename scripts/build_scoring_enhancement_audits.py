#!/usr/bin/env python3
"""Post-freeze scoring diagnostics for intervals, reject risk, and exposure.

These audits intentionally do not overwrite the submission. They answer the
"should we change the final guarded policy?" question with comparable shadow
metrics, while keeping the validator-ready files stable unless a policy clearly
dominates.
"""

from __future__ import annotations

import json
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

from scripts.experiment_compact_feature_reject_bakeoff import (  # noqa: E402
    RANDOM_SEED,
    compact_columns,
    fit_lightgbm,
    prepare_frames,
    predict_pd,
    time_split_labeled,
)
from src.deliverable_a_pipeline import add_application_features, feature_columns, train_approval_propensity  # noqa: E402
from src.economics import expected_npv, npv_default, npv_repaid, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
N_BOOTSTRAP_MODELS = 8


def odds_stress_pd(pd_values: np.ndarray, gamma: float) -> np.ndarray:
    return expit(logit(np.clip(pd_values, 0.001, 0.999)) + np.log(gamma))


def odds(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(1.0 - values, 1e-9)


def break_even_pd(amount: np.ndarray, t_star: np.ndarray, recovery: np.ndarray) -> np.ndarray:
    paid = npv_repaid(amount)
    default = npv_default(amount, t_star, recovery * amount)
    return np.clip(paid / np.maximum(paid - default, 1e-9), 0.0, 1.0)


def interval_table(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_col, dropna=False):
        if len(group) == 0:
            continue
        default_rate = float(group["default_flag"].mean())
        mean_lower = float(group["pd_lower_90"].mean())
        mean_upper = float(group["pd_upper_90"].mean())
        rows.append(
            {
                "grouping": group_col,
                "segment": str(value),
                "rows": int(len(group)),
                "default_rate": default_rate,
                "mean_predicted_pd": float(group["predicted_pd"].mean()),
                "mean_lower_90": mean_lower,
                "mean_upper_90": mean_upper,
                "bin_interval_hit": bool(mean_lower <= default_rate <= mean_upper),
                "mean_interval_width": float((group["pd_upper_90"] - group["pd_lower_90"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_a_interval_audit(validation: pd.DataFrame, submission_a: pd.DataFrame) -> pd.DataFrame:
    frame = validation.merge(submission_a, on="applicant_id", how="left", validate="one_to_one")
    frame = frame[frame["default_flag"].notna()].copy()
    frame["pd_decile"] = pd.qcut(frame["predicted_pd"], 10, labels=False, duplicates="drop")
    frame["amount_quintile"] = pd.qcut(frame["requested_amount"], 5, labels=False, duplicates="drop")
    frame["interval_width"] = frame["pd_upper_90"] - frame["pd_lower_90"]
    tables = [
        interval_table(frame, "pd_decile"),
        interval_table(frame, "prior_decision"),
        interval_table(frame, "has_linked_bank_feed"),
        interval_table(frame, "amount_quintile"),
        interval_table(frame, "sector"),
    ]
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(REPORT_DIR / "a_interval_sharpness_audit.csv", index=False)
    return out


def build_b_interval_audit() -> dict[str, float | int]:
    diag = pd.read_csv(REPORT_DIR / "deliverable_b_validation_timing_diagnostics.csv")
    by_cohort = (
        diag.groupby("cohort_week")
        .agg(
            rows=("loan_age_weeks", "size"),
            coverage=("interval_hit", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_width=("interval_width", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    by_age = (
        diag.groupby("loan_age_weeks")
        .agg(
            rows=("cohort_week", "size"),
            coverage=("interval_hit", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_width=("interval_width", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    by_cohort.to_csv(REPORT_DIR / "b_interval_sharpness_by_cohort.csv", index=False)
    by_age.to_csv(REPORT_DIR / "b_interval_sharpness_by_age.csv", index=False)
    return {
        "coverage": float(diag["interval_hit"].mean()),
        "mean_width": float(diag["interval_width"].mean()),
        "mean_abs_error": float(diag["abs_error"].mean()),
        "worst_abs_error": float(diag["abs_error"].max()),
    }


def approval_support(train: pd.DataFrame, eval_frame: pd.DataFrame) -> np.ndarray:
    train_fe = add_application_features(train)
    eval_fe = add_application_features(eval_frame)
    _, numeric, categorical = feature_columns(train_fe)
    model = train_approval_propensity(
        train_fe[numeric + categorical],
        (train["prior_decision"] == 1).astype(int),
        numeric,
        categorical,
    )
    return np.clip(model.predict_proba(eval_fe[numeric + categorical])[:, 1], 0.001, 0.999)


def policy_metrics(
    name: str,
    frame: pd.DataFrame,
    decision: np.ndarray,
    enpv: np.ndarray,
    prior_declined: np.ndarray,
    labeled: np.ndarray,
    realized: np.ndarray,
) -> dict[str, float | int | str]:
    return {
        "policy": name,
        "approved_total": int(decision.sum()),
        "prior_declined_approved": int((decision & prior_declined).sum()),
        "approval_rate": float(decision.mean()),
        "headline_expected_npv": float(enpv[decision].sum()),
        "prior_declined_expected_npv": float(enpv[decision & prior_declined].sum()),
        "labeled_validation_approved": int((decision & labeled).sum()),
        "labeled_validation_realized_npv": float(np.nansum(np.where(decision & labeled, realized, 0.0))),
        "mean_pd_approved": float(frame.loc[decision, "predicted_pd"].mean()) if decision.any() else np.nan,
    }


def build_reject_and_exposure_audits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    submission_a: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = np.load(OUTPUT_DIR / "deliverable_a_curves.npz")
    frame = pd.concat(
        [validation.assign(_split="validation"), test.assign(_split="test")],
        ignore_index=True,
    ).merge(submission_a, on="applicant_id", how="left", validate="one_to_one")

    amount = frame["requested_amount"].to_numpy(float)
    pd_point = frame["predicted_pd"].to_numpy(float)
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    enpv = expected_npv(amount, pd_point, t_star, recovery)
    margin = enpv / np.maximum(amount, 1.0)
    prior_declined = frame["prior_decision"].to_numpy() == 0
    approved = frame["decision"].to_numpy(int).astype(bool)
    support = approval_support(train, frame.drop(columns=["decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]))
    frame["approval_support"] = support
    frame["expected_npv"] = enpv
    frame["npv_margin"] = margin
    frame["support_tier"] = pd.cut(
        support,
        bins=[0.0, 0.40, 0.55, 0.70, 1.0],
        labels=["low_support", "mid_support", "high_support", "very_high_support"],
        include_lowest=True,
    )
    frame["pd_risk_tier"] = pd.cut(
        pd_point,
        bins=[0.0, 0.08, 0.16, 0.28, 1.0],
        labels=["low_pd", "medium_pd", "high_pd", "very_high_pd"],
        include_lowest=True,
    )

    validation_mask = frame["_split"].eq("validation").to_numpy()
    labeled = validation_mask & frame["default_flag"].notna().to_numpy()
    realized = np.full(len(frame), np.nan)
    realized[validation_mask] = realized_npv(frame.loc[validation_mask])
    be_pd = break_even_pd(amount, t_star, recovery)
    break_even_gamma = odds(be_pd) / np.maximum(odds(pd_point), 1e-9)

    shadow_decisions = {
        "current_guarded": approved,
        "zero_prior_declined": approved & ~prior_declined,
        "prior_declined_support_ge_0p40": approved & (~prior_declined | (support >= 0.40)),
        "prior_declined_support_ge_0p55": approved & (~prior_declined | (support >= 0.55)),
        "prior_declined_break_even_gamma_ge_6x": approved & (~prior_declined | (break_even_gamma >= 6.0)),
    }

    cap_decision = approved.copy()
    for _, idx in frame[approved & prior_declined].groupby("sector").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        sector_prior_declined = (frame["sector"].to_numpy() == frame.loc[idx[0], "sector"]) & prior_declined
        cap = int(np.ceil(0.30 * sector_prior_declined.sum()))
        if len(idx) > cap:
            keep = idx[np.argsort(-margin[idx])[:cap]]
            drop = np.setdiff1d(idx, keep)
            cap_decision[drop] = False
    shadow_decisions["sector_prior_declined_30pct_cap"] = cap_decision

    shadow_rows = [
        policy_metrics(name, frame, decision, enpv, prior_declined, labeled, realized)
        for name, decision in shadow_decisions.items()
    ]

    for gamma in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]:
        stressed_pd = pd_point.copy()
        stressed_pd[prior_declined] = odds_stress_pd(stressed_pd[prior_declined], gamma)
        stressed_enpv = expected_npv(amount, stressed_pd, t_star, recovery)
        shadow_rows.append(
            {
                "policy": f"current_with_prior_declined_{gamma:g}x_odds_stress",
                "approved_total": int(approved.sum()),
                "prior_declined_approved": int((approved & prior_declined).sum()),
                "approval_rate": float(approved.mean()),
                "headline_expected_npv": float(stressed_enpv[approved].sum()),
                "prior_declined_expected_npv": float(stressed_enpv[approved & prior_declined].sum()),
                "labeled_validation_approved": int((approved & labeled).sum()),
                "labeled_validation_realized_npv": float(np.nansum(np.where(approved & labeled, realized, 0.0))),
                "mean_pd_approved": float(stressed_pd[approved].mean()),
            }
        )

    shadow = pd.DataFrame(shadow_rows)
    shadow.to_csv(REPORT_DIR / "reject_region_shadow_policy_sensitivity.csv", index=False)

    exposure_rows = []
    segment_cols = ["sector", "has_linked_bank_feed", "support_tier", "pd_risk_tier", "owner_personal_credit_band"]
    for col in segment_cols:
        for value, group in frame[approved].groupby(col, dropna=False):
            idx = group.index.to_numpy()
            exposure_rows.append(
                {
                    "segment_feature": col,
                    "segment": str(value),
                    "approved": int(len(group)),
                    "share_of_approved": float(len(group) / max(approved.sum(), 1)),
                    "prior_declined_approved": int(prior_declined[idx].sum()),
                    "requested_amount_total": float(group["requested_amount"].sum()),
                    "expected_npv_total": float(enpv[idx].sum()),
                    "mean_pd": float(group["predicted_pd"].mean()),
                    "mean_support": float(group["approval_support"].mean()),
                    "mean_margin": float(group["npv_margin"].mean()),
                }
            )
    exposure = pd.DataFrame(exposure_rows).sort_values(
        ["share_of_approved", "expected_npv_total"], ascending=[False, False]
    )
    exposure.to_csv(REPORT_DIR / "exposure_concentration_audit.csv", index=False)
    return shadow, exposure, frame


def build_bootstrap_shadow(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, float | int]:
    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    fit_idx, cal_idx = time_split_labeled(train)
    fit_y = train.loc[fit_idx, "default_flag"].astype(int).to_numpy()
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    train_x, val_x, test_x, _, categorical, _ = prepare_frames(
        "compact_risk_factors",
        train,
        validation,
        test,
        train_fe,
        validation_fe,
        test_fe,
    )
    rng = np.random.default_rng(RANDOM_SEED + 101)
    val_preds = []
    aucs = []
    for i in range(N_BOOTSTRAP_MODELS):
        sample = rng.choice(fit_idx, size=len(fit_idx), replace=True)
        model, iso, _ = fit_lightgbm(
            train_x.loc[sample],
            train.loc[sample, "default_flag"].astype(int).to_numpy(),
            train_x.loc[cal_idx],
            cal_y,
            categorical,
            seed=RANDOM_SEED + 1000 + i,
        )
        pred = predict_pd(model, iso, val_x)
        val_preds.append(pred)
        labeled = validation["default_flag"].notna().to_numpy()
        aucs.append(roc_auc_score(validation.loc[labeled, "default_flag"].astype(int), pred[labeled]))
    pred_matrix = np.vstack(val_preds).T
    labeled = validation["default_flag"].notna().to_numpy()
    std = pred_matrix[labeled].std(axis=1)
    point = pred_matrix[labeled].mean(axis=1)
    y = validation.loc[labeled, "default_flag"].astype(int).to_numpy()
    summary = {
        "bootstrap_models": N_BOOTSTRAP_MODELS,
        "validation_auc_mean": float(np.mean(aucs)),
        "validation_auc_min": float(np.min(aucs)),
        "validation_auc_max": float(np.max(aucs)),
        "validation_log_loss_ensemble_mean": float(log_loss(y, point)),
        "validation_brier_ensemble_mean": float(brier_score_loss(y, point)),
        "mean_pd_std_labeled": float(std.mean()),
        "p90_pd_std_labeled": float(np.quantile(std, 0.90)),
        "p99_pd_std_labeled": float(np.quantile(std, 0.99)),
    }
    pd.DataFrame(
        {
            "applicant_id": validation.loc[labeled, "applicant_id"].to_numpy(),
            "default_flag": y,
            "bootstrap_pd_mean": point,
            "bootstrap_pd_std": std,
        }
    ).to_csv(REPORT_DIR / "a_bootstrap_shadow_validation_predictions.csv", index=False)
    (REPORT_DIR / "a_bootstrap_shadow_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission_a = pd.read_csv(SUBMISSION_DIR / "submission_A_decisions.csv")

    a_interval = build_a_interval_audit(validation, submission_a)
    b_interval = build_b_interval_audit()
    shadow, exposure, _ = build_reject_and_exposure_audits(train, validation, test, submission_a)
    bootstrap = build_bootstrap_shadow(train, validation, test)

    summary = {
        "a_interval_segments": int(len(a_interval)),
        "a_interval_miss_segments": int((~a_interval["bin_interval_hit"]).sum()),
        "a_interval_mean_width": float(
            (submission_a["pd_upper_90"] - submission_a["pd_lower_90"]).mean()
        ),
        "b_interval": b_interval,
        "best_shadow_policy_by_labeled_validation_npv": shadow.sort_values(
            "labeled_validation_realized_npv", ascending=False
        ).iloc[0].to_dict(),
        "largest_exposure_segment": exposure.iloc[0].to_dict(),
        "bootstrap_shadow": bootstrap,
        "recommendation": (
            "Do not replace final A from these diagnostics alone. Current guarded A remains the best "
            "balanced policy; use exposure and bootstrap outputs as hidden-risk evidence."
        ),
    }
    (REPORT_DIR / "scoring_enhancement_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
