#!/usr/bin/env python3
"""Build a Deliverable A candidate adapted for the validation/test regime.

The organizer hint points to gradual regime shift rather than a one-day event.
This script uses covariate-shift/adversarial-validation weighting:

1. Train a domain classifier to distinguish historical train rows from
   validation/test rows using only applicant features.
2. Convert its future-regime odds into density-ratio sample weights.
3. Refit the compact risk-factor PD model with those weights, optionally adding
   reject-inference pseudo labels.
4. Tune the expected-NPV policy threshold on labeled validation while applying
   the same prior-declined stress guardrail used by the active submission.

It writes a candidate under outputs/candidates/regime_shift_adapted/ and does
not overwrite the active submission unless called with --promote.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    MIN_PD_INTERVAL_HALF_WIDTH,
    PRIOR_DECLINED_MIN_MARGIN,
    RANDOM_SEED,
    compact_columns,
    odds_stress_pd,
    prepare_frames,
    time_split_labeled,
)
from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table  # noqa: E402
from src.deliverable_a_pipeline import add_application_features  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
CANDIDATE_DIR = OUTPUT_DIR / "candidates" / "regime_shift_adapted"
SUBMISSION_A = SUBMISSION_DIR / "submission_A_decisions.csv"
CURVES_PATH = OUTPUT_DIR / "deliverable_a_curves.npz"

DOMAIN_WEIGHT_CLIP = (0.25, 6.0)
REJECT_WEIGHT_SCALE = 0.35
PRIOR_DECLINED_ODDS_STRESS_GAMMA = 3.0
PRIOR_DECLINED_STRESSED_MARGIN_FLOOR = 0.0


def fit_weighted_lightgbm(
    fit_x: pd.DataFrame,
    fit_y: np.ndarray,
    cal_x: pd.DataFrame,
    cal_y: np.ndarray,
    categorical: list[str],
    *,
    fit_weight: np.ndarray | None = None,
    cal_weight: np.ndarray | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[LGBMClassifier, IsotonicRegression, np.ndarray]:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=850,
        learning_rate=0.026,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.06,
        reg_lambda=0.40,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(fit_x, fit_y, sample_weight=fit_weight, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, cal_y, sample_weight=cal_weight)
    cal_pd = np.clip(calibrator.predict(raw_cal), 0.001, 0.999)
    return model, calibrator, cal_pd


def predict_pd(model: LGBMClassifier, calibrator: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    return np.clip(calibrator.predict(model.predict_proba(x)[:, 1]), 0.001, 0.999)


def fit_domain_weights(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate train-row importance weights for the validation/test regime."""
    # Direct calendar features trivially identify the official split. Exclude
    # them so the density ratio reflects covariate regime shift, not just time.
    domain_cols = [col for col in train_x.columns if not col.startswith("application_")]
    domain_categorical = [col for col in categorical if col in domain_cols]
    domain_train = train_x[domain_cols].copy()
    domain_future = pd.concat([val_x[domain_cols], test_x[domain_cols]], ignore_index=True).copy()
    domain_x = pd.concat([domain_train, domain_future], ignore_index=True)
    for col in domain_categorical:
        if col in domain_x.columns:
            domain_x[col] = domain_x[col].astype("category")

    y = np.r_[np.zeros(len(domain_train), dtype=int), np.ones(len(domain_future), dtype=int)]
    class_weight = np.where(
        y == 0,
        len(y) / (2.0 * max(len(domain_train), 1)),
        len(y) / (2.0 * max(len(domain_future), 1)),
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=0.50,
        random_state=RANDOM_SEED + 701,
        verbosity=-1,
    )
    model.fit(domain_x, y, sample_weight=class_weight, categorical_feature=domain_categorical)
    p_future_train = np.clip(model.predict_proba(domain_train)[:, 1], 0.001, 0.999)
    raw_ratio = p_future_train / np.maximum(1.0 - p_future_train, 1e-9)
    weights = np.clip(raw_ratio, *DOMAIN_WEIGHT_CLIP)
    weights = weights / np.mean(weights)
    meta = {
        "domain_rows_train": int(len(domain_train)),
        "domain_rows_future": int(len(domain_future)),
        "domain_feature_count": int(len(domain_cols)),
        "domain_excluded_calendar_features": [col for col in train_x.columns if col.startswith("application_")],
        "domain_in_sample_auc": float(roc_auc_score(y, model.predict_proba(domain_x)[:, 1])),
        "domain_weight_min": float(np.min(weights)),
        "domain_weight_p25": float(np.quantile(weights, 0.25)),
        "domain_weight_median": float(np.quantile(weights, 0.50)),
        "domain_weight_p75": float(np.quantile(weights, 0.75)),
        "domain_weight_max": float(np.max(weights)),
    }
    return weights, meta


def fit_regime_model(
    train: pd.DataFrame,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    domain_weights: np.ndarray,
    *,
    reject_gamma: float | None,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    model_idx, cal_idx = time_split_labeled(train)
    fit_x = train_x.loc[model_idx]
    fit_y = train.loc[model_idx, "default_flag"].astype(int).to_numpy()
    cal_x = train_x.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()

    fit_weight = domain_weights[model_idx]
    fit_weight = fit_weight / np.mean(fit_weight)
    cal_weight = domain_weights[cal_idx]
    cal_weight = cal_weight / np.mean(cal_weight)

    meta: dict[str, object] = {
        "model_rows": int(len(model_idx)),
        "calibration_rows": int(len(cal_idx)),
        "fit_weight_mean": float(np.mean(fit_weight)),
        "fit_weight_p90": float(np.quantile(fit_weight, 0.90)),
        "cal_weight_mean": float(np.mean(cal_weight)),
        "cal_weight_p90": float(np.quantile(cal_weight, 0.90)),
        "reject_gamma": reject_gamma,
    }

    if reject_gamma is None:
        model, calibrator, cal_pd = fit_weighted_lightgbm(
            fit_x,
            fit_y,
            cal_x,
            cal_y,
            categorical,
            fit_weight=fit_weight,
            cal_weight=cal_weight,
            seed=RANDOM_SEED + seed_offset,
        )
    else:
        base_model, base_calibrator, _ = fit_weighted_lightgbm(
            fit_x,
            fit_y,
            cal_x,
            cal_y,
            categorical,
            fit_weight=fit_weight,
            cal_weight=cal_weight,
            seed=RANDOM_SEED + seed_offset - 1,
        )
        reject_idx = train.index[train["default_flag"].isna()].to_numpy()
        reject_pd = predict_pd(base_model, base_calibrator, train_x.loc[reject_idx])
        reject_pd_stressed = odds_stress_pd(reject_pd, reject_gamma)
        accept_bad_rate = float(np.mean(fit_y))
        target_reject_bad_rate = min(max(accept_bad_rate * reject_gamma, accept_bad_rate), 0.85)
        cutoff = float(np.quantile(reject_pd_stressed, 1.0 - target_reject_bad_rate))
        pseudo_y = (reject_pd_stressed >= cutoff).astype(int)
        aug_x = pd.concat([fit_x, train_x.loc[reject_idx]], ignore_index=True)
        aug_y = np.r_[fit_y, pseudo_y]
        reject_weight = domain_weights[reject_idx]
        reject_weight = reject_weight / np.mean(reject_weight)
        aug_weight = np.r_[fit_weight, reject_weight * REJECT_WEIGHT_SCALE]
        model, calibrator, cal_pd = fit_weighted_lightgbm(
            aug_x,
            aug_y,
            cal_x,
            cal_y,
            categorical,
            fit_weight=aug_weight,
            cal_weight=cal_weight,
            seed=RANDOM_SEED + seed_offset + int(reject_gamma * 100),
        )
        meta.update(
            {
                "reject_rows": int(len(reject_idx)),
                "reject_weight_scale": REJECT_WEIGHT_SCALE,
                "accept_bad_rate": accept_bad_rate,
                "target_reject_bad_rate": target_reject_bad_rate,
                "pseudo_reject_bad_rate": float(np.mean(pseudo_y)),
                "pseudo_reject_bad_cutoff": cutoff,
            }
        )

    return cal_pd, predict_pd(model, calibrator, val_x), predict_pd(model, calibrator, test_x), meta


def policy_decision(
    margin: np.ndarray,
    threshold: float,
    prior_declined: np.ndarray,
    p_default: np.ndarray,
    amount: np.ndarray,
    t_star: np.ndarray,
    recovery: np.ndarray,
) -> np.ndarray:
    decision = (margin > threshold) & (
        ~prior_declined | (margin > PRIOR_DECLINED_MIN_MARGIN)
    )
    stressed_pd = p_default.copy()
    stressed_pd[prior_declined] = odds_stress_pd(
        stressed_pd[prior_declined],
        PRIOR_DECLINED_ODDS_STRESS_GAMMA,
    )
    stressed_npv = expected_npv(amount, stressed_pd, t_star, recovery)
    stressed_margin = stressed_npv / np.maximum(amount, 1.0)
    return decision & (~prior_declined | (stressed_margin > PRIOR_DECLINED_STRESSED_MARGIN_FLOOR))


def choose_policy_threshold(
    margin: np.ndarray,
    realized: np.ndarray,
    prior_declined: np.ndarray,
    p_default: np.ndarray,
    amount: np.ndarray,
    t_star: np.ndarray,
    recovery: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.r_[
            np.linspace(-0.06, 0.08, 57),
            np.quantile(margin, np.linspace(0.01, 0.99, 99)),
        ]
    )
    rows = []
    for threshold in candidates:
        decision = policy_decision(
            margin,
            float(threshold),
            prior_declined,
            p_default,
            amount,
            t_star,
            recovery,
        )
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


def candidate_summary(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cal_pd: np.ndarray,
    val_pd: np.ndarray,
    test_pd: np.ndarray,
    curves: np.lib.npyio.NpzFile,
    meta: dict[str, object],
    domain_meta: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model_idx, cal_idx = time_split_labeled(train)
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    labeled_val = validation["default_flag"].notna().to_numpy()
    val_y = validation.loc[labeled_val, "default_flag"].astype(int).to_numpy()
    realized_val = realized_npv(validation.loc[labeled_val])

    bin_table = pd_interval_bin_table(cal_pd, cal_y, n_bins=10)
    val_lower, val_upper = build_pd_intervals(val_pd, val_pd[:, None], bin_table)
    test_lower, test_upper = build_pd_intervals(test_pd, test_pd[:, None], bin_table)
    val_lower = np.minimum(val_lower, np.clip(val_pd - MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    val_upper = np.maximum(val_upper, np.clip(val_pd + MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    test_lower = np.minimum(test_lower, np.clip(test_pd - MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))
    test_upper = np.maximum(test_upper, np.clip(test_pd + MIN_PD_INTERVAL_HALF_WIDTH, 0.0, 1.0))

    val_amount = validation["requested_amount"].to_numpy(float)
    test_amount = test["requested_amount"].to_numpy(float)
    val_enpv = expected_npv(val_amount, val_pd, curves["validation_t_star"], curves["validation_recovery"])
    test_enpv = expected_npv(test_amount, test_pd, curves["test_t_star"], curves["test_recovery"])
    val_margin = val_enpv / np.maximum(val_amount, 1.0)
    test_margin = test_enpv / np.maximum(test_amount, 1.0)

    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0
    threshold, sweep = choose_policy_threshold(
        val_margin[labeled_val],
        realized_val,
        val_prior_declined[labeled_val],
        val_pd[labeled_val],
        val_amount[labeled_val],
        curves["validation_t_star"][labeled_val],
        curves["validation_recovery"][labeled_val],
    )
    val_decision = policy_decision(
        val_margin,
        threshold,
        val_prior_declined,
        val_pd,
        val_amount,
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    test_decision = policy_decision(
        test_margin,
        threshold,
        test_prior_declined,
        test_pd,
        test_amount,
        curves["test_t_star"],
        curves["test_recovery"],
    )

    all_decision = np.r_[val_decision, test_decision]
    all_prior_declined = np.r_[val_prior_declined, test_prior_declined]
    all_pd = np.r_[val_pd, test_pd]
    all_enpv = np.r_[val_enpv, test_enpv]
    all_amount = np.r_[val_amount, test_amount]
    prior_declined_approved = all_decision & all_prior_declined
    val_labeled_decision = val_decision[labeled_val]

    submission = pd.DataFrame(
        {
            "applicant_id": pd.concat(
                [validation["applicant_id"], test["applicant_id"]],
                ignore_index=True,
            ),
            "decision": all_decision.astype(int),
            "predicted_pd": all_pd,
            "pd_lower_90": np.r_[np.minimum(val_lower, val_pd), np.minimum(test_lower, test_pd)],
            "pd_upper_90": np.r_[np.maximum(val_upper, val_pd), np.maximum(test_upper, test_pd)],
        }
    )

    row = {
        "variant": name,
        "feature_set": "compact_risk_factors",
        "threshold": threshold,
        "validation_labeled_realized_npv": float(realized_val[val_labeled_decision].sum()),
        "validation_labeled_approved": int(val_labeled_decision.sum()),
        "validation_labeled_approval_rate": float(val_labeled_decision.mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized_val[val_labeled_decision] < 0)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "prior_declined_approved_total": int(prior_declined_approved.sum()),
        "prior_declined_approval_rate": float(prior_declined_approved.sum() / max(all_prior_declined.sum(), 1)),
        "headline_expected_npv": float(all_enpv[all_decision].sum()),
        "expected_roic_approved_amount": float(all_enpv[all_decision].sum() / all_amount[all_decision].sum()),
        "prior_declined_expected_npv": float(all_enpv[prior_declined_approved].sum()),
        "prior_declined_mean_pd": float(all_pd[prior_declined_approved].mean()) if prior_declined_approved.sum() else np.nan,
        "auroc": float(roc_auc_score(val_y, val_pd[labeled_val])),
        "log_loss": float(log_loss(val_y, val_pd[labeled_val], labels=[0, 1])),
        "brier": float(brier_score_loss(val_y, val_pd[labeled_val])),
        "mean_pd": float(np.mean(val_pd[labeled_val])),
        "actual_default_rate": float(np.mean(val_y)),
        "interval_bin_coverage": bin_level_coverage(
            val_pd[labeled_val],
            val_y,
            val_lower[labeled_val],
            val_upper[labeled_val],
        )["bin_coverage"],
        **domain_meta,
        **meta,
    }
    return row, submission, sweep


def active_summary(validation: pd.DataFrame, test: pd.DataFrame, curves: np.lib.npyio.NpzFile) -> dict[str, object]:
    sub = pd.read_csv(SUBMISSION_A)
    n_val = len(validation)
    val_sub = sub.iloc[:n_val].reset_index(drop=True)
    test_sub = sub.iloc[n_val:].reset_index(drop=True)
    labeled = validation["default_flag"].notna().to_numpy()
    val_y = validation.loc[labeled, "default_flag"].astype(int).to_numpy()
    val_pd = val_sub["predicted_pd"].to_numpy(float)
    test_pd = test_sub["predicted_pd"].to_numpy(float)
    val_decision = val_sub["decision"].to_numpy(int).astype(bool)
    test_decision = test_sub["decision"].to_numpy(int).astype(bool)
    realized = realized_npv(validation.loc[labeled])
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
    all_decision = np.r_[val_decision, test_decision]
    all_enpv = np.r_[val_enpv, test_enpv]
    all_amount = np.r_[
        validation["requested_amount"].to_numpy(float),
        test["requested_amount"].to_numpy(float),
    ]
    val_labeled_decision = val_decision[labeled]
    return {
        "variant": "active_submission",
        "threshold": None,
        "validation_labeled_realized_npv": float(realized[val_labeled_decision].sum()),
        "validation_labeled_approved": int(val_labeled_decision.sum()),
        "validation_labeled_approval_rate": float(val_labeled_decision.mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized[val_labeled_decision] < 0)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "headline_expected_npv": float(all_enpv[all_decision].sum()),
        "expected_roic_approved_amount": float(all_enpv[all_decision].sum() / all_amount[all_decision].sum()),
        "auroc": float(roc_auc_score(val_y, val_pd[labeled])),
        "log_loss": float(log_loss(val_y, val_pd[labeled], labels=[0, 1])),
        "brier": float(brier_score_loss(val_y, val_pd[labeled])),
        "mean_pd": float(np.mean(val_pd[labeled])),
        "actual_default_rate": float(np.mean(val_y)),
    }


def run(promote: bool) -> None:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    curves = np.load(CURVES_PATH)

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    train_x, val_x, test_x, _numeric, categorical, extra = prepare_frames(
        "compact_risk_factors",
        train,
        validation,
        test,
        train_fe,
        validation_fe,
        test_fe,
    )
    extra["feature_count"] = int(train_x.shape[1])
    domain_weights, domain_meta = fit_domain_weights(train_x, val_x, test_x, categorical)

    rows = [active_summary(validation, test, curves)]
    candidate_outputs: dict[str, tuple[dict[str, object], pd.DataFrame, pd.DataFrame]] = {}
    configs = [
        ("regime_weighted", None, 10),
        ("regime_weighted_reject_gamma_1p5", 1.5, 20),
        ("regime_weighted_reject_gamma_2", 2.0, 30),
    ]
    for name, reject_gamma, seed_offset in configs:
        print(f"Running {name}...")
        cal_pd, val_pd, test_pd, meta = fit_regime_model(
            train,
            train_x,
            val_x,
            test_x,
            categorical,
            domain_weights,
            reject_gamma=reject_gamma,
            seed_offset=seed_offset,
        )
        meta.update(extra)
        row, submission, sweep = candidate_summary(
            name,
            train,
            validation,
            test,
            cal_pd,
            val_pd,
            test_pd,
            curves,
            meta,
            domain_meta,
        )
        candidate_outputs[name] = (row, submission, sweep)
        rows.append(row)
        variant_dir = CANDIDATE_DIR / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        submission.to_csv(variant_dir / "submission_A_decisions.csv", index=False)
        sweep.to_csv(variant_dir / "threshold_sweep.csv", index=False)
        (variant_dir / "summary.json").write_text(json.dumps(row, indent=2))

    result = pd.DataFrame(rows).sort_values(
        ["validation_labeled_realized_npv", "auroc"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    result.to_csv(CANDIDATE_DIR / "summary.csv", index=False)
    result.to_json(CANDIDATE_DIR / "summary.json", orient="records", indent=2)
    result.to_csv(REPORT_DIR / "regime_shift_adapted_policy_comparison.csv", index=False)

    best = result.iloc[0].to_dict()
    promoted = False
    if promote and best["variant"] != "active_submission":
        chosen_name = str(best["variant"])
        archive = REPORT_DIR / "archive" / "submission_A_decisions_before_regime_shift_adaptation.csv"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(SUBMISSION_A, archive)
        candidate_outputs[chosen_name][1].to_csv(SUBMISSION_A, index=False)
        promoted = True

    report = {
        "objective": "Regime-shift adapted Deliverable A policy using adversarial-validation density-ratio weights.",
        "promoted": promoted,
        "best_variant": best["variant"],
        "active_variant_rank": int(result.index[result["variant"].eq("active_submission")][0]) + 1
        if result["variant"].eq("active_submission").any()
        else None,
        "domain_weighting": domain_meta,
        "comparison": result.to_dict(orient="records"),
    }
    (REPORT_DIR / "regime_shift_adapted_policy_summary.json").write_text(json.dumps(report, indent=2))
    print(result[
        [
            "variant",
            "validation_labeled_realized_npv",
            "validation_labeled_default_rate_approved",
            "validation_all_approval_rate",
            "test_approval_rate",
            "approved_total",
            "headline_expected_npv",
            "auroc",
            "log_loss",
            "brier",
        ]
    ].to_string(index=False))
    if promoted:
        print(f"Promoted {best['variant']} to {SUBMISSION_A}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="overwrite active A submission with the best candidate")
    args = parser.parse_args()
    run(promote=args.promote)


if __name__ == "__main__":
    main()
