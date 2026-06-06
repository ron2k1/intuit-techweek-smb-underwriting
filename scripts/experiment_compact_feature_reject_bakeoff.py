#!/usr/bin/env python3
"""Compact feature and reject-inference bakeoff for Deliverable A.

This script does not overwrite the active submission. It compares scoped A-model
feature sets using the same labeled-validation NPV objective and prior-declined
margin guardrail as the active policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table  # noqa: E402
from src.deliverable_a_pipeline import (  # noqa: E402
    CATEGORICAL_BASE,
    DROP_FOR_PD,
    PRIOR_POLICY_TOKENS,
    add_application_features,
    feature_columns,
)
from src.economics import expected_npv, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
CANDIDATE_DIR = OUTPUT_DIR / "candidates" / "compact_feature_bakeoff"
MIN_PD_INTERVAL_HALF_WIDTH = 0.08
PRIOR_DECLINED_MIN_MARGIN = 0.03
RANDOM_SEED = 2026

RAW_EXCLUDE = DROP_FOR_PD | {"prior_underwriter_score", "prior_approved_amount"}
COMPACT_ENGINEERED_FEATURES = [
    "application_month",
    "application_day_of_week",
    "application_weekofyear",
    "application_days_since_2024_01_01",
    "requested_to_stated_annual_revenue",
    "debt_to_stated_annual_revenue",
    "requested_to_observed_annual_revenue",
    "observed_to_stated_revenue_ratio",
    "cash_balance_to_requested_amount",
    "cash_balance_to_observed_monthly_revenue",
    "debt_to_requested_amount",
    "prior_default_rate",
    "has_external_decline_record",
    "has_inquiry_elsewhere_record",
    "bank_feed_missing_count",
    "cash_balance_negative",
    "has_prior_default",
    "has_prior_loan_history",
    "log_requested_amount",
    "log_stated_annual_revenue",
    "log_observed_monthly_revenue",
    "log_existing_debt",
    "signed_log_cash_balance_p10",
    "log_account_age_days",
    "repayment_burden_index",
    "credit_stress_index",
    "cash_stress_index",
    "maturity_index",
    "revenue_scale_index",
    "platform_engagement_index",
    "bank_feed_support_index",
    "utilization_x_delinquency",
    "utilization_x_repayment_burden",
    "cash_x_payroll_regularity",
    "volatility_x_repayment_burden",
    "bank_feed_x_utilization",
    "bank_feed_x_repayment_burden",
    "maturity_x_repayment_burden",
    "credit_x_cash_stress",
]


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def safe_feature_lists(columns: list[str]) -> tuple[list[str], list[str]]:
    feature_cols = [c for c in columns if not any(token in c for token in PRIOR_POLICY_TOKENS)]
    categorical = sorted(c for c in feature_cols if c in CATEGORICAL_BASE)
    numeric = sorted(c for c in feature_cols if c not in categorical)
    return numeric, categorical


def raw_valid_prior_columns(train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    cols = [
        c
        for c in train.columns
        if c not in RAW_EXCLUDE and not any(token in c for token in PRIOR_POLICY_TOKENS)
    ]
    numeric, categorical = safe_feature_lists(cols)
    return numeric + categorical, numeric, categorical


def full_engineered_columns(train_fe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    _, numeric, categorical = feature_columns(train_fe)
    return numeric + categorical, numeric, categorical


def compact_columns(train: pd.DataFrame, train_fe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    raw_cols, _, _ = raw_valid_prior_columns(train)
    compact = [c for c in COMPACT_ENGINEERED_FEATURES if c in train_fe.columns]
    cols = list(dict.fromkeys(raw_cols + compact))
    numeric, categorical = safe_feature_lists(cols)
    return numeric + categorical, numeric, categorical


def add_pca_components(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    numeric: list[str],
    *,
    n_components: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    pca_numeric = [c for c in numeric if c in train_x.columns]
    transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
        ]
    )
    train_pc = transformer.fit_transform(train_x[pca_numeric])
    val_pc = transformer.transform(val_x[pca_numeric])
    test_pc = transformer.transform(test_x[pca_numeric])
    pc_cols = [f"latent_pc_{i + 1}" for i in range(n_components)]
    for frame, arr in [(train_x, train_pc), (val_x, val_pc), (test_x, test_pc)]:
        for i, col in enumerate(pc_cols):
            frame[col] = arr[:, i]
    return train_x, val_x, test_x, pc_cols


def prepare_frames(
    variant: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_fe: pd.DataFrame,
    validation_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, object]]:
    if variant == "raw_valid_prior":
        feature_cols, numeric, categorical = raw_valid_prior_columns(train)
        source_train, source_val, source_test = train, validation, test
    elif variant == "compact_risk_factors":
        feature_cols, numeric, categorical = compact_columns(train, train_fe)
        source_train, source_val, source_test = train_fe, validation_fe, test_fe
    elif variant == "compact_risk_factors_pca8":
        feature_cols, numeric, categorical = compact_columns(train, train_fe)
        source_train, source_val, source_test = train_fe, validation_fe, test_fe
    elif variant == "full_engineered":
        feature_cols, numeric, categorical = full_engineered_columns(train_fe)
        source_train, source_val, source_test = train_fe, validation_fe, test_fe
    else:
        raise ValueError(f"Unknown variant: {variant}")

    train_x = source_train[feature_cols].copy()
    val_x = source_val[feature_cols].copy()
    test_x = source_test[feature_cols].copy()

    extra: dict[str, object] = {"base_feature_count": len(feature_cols)}
    if variant == "compact_risk_factors_pca8":
        train_x, val_x, test_x, pc_cols = add_pca_components(train_x, val_x, test_x, numeric)
        numeric = sorted(numeric + pc_cols)
        feature_cols = numeric + categorical
        extra["pca_components"] = pc_cols

    for col in categorical:
        train_x[col] = train_x[col].astype("category")
        val_x[col] = val_x[col].astype("category")
        test_x[col] = test_x[col].astype("category")
    return train_x, val_x, test_x, numeric, categorical, extra


def fit_lightgbm(
    fit_x: pd.DataFrame,
    fit_y: np.ndarray,
    cal_x: pd.DataFrame,
    cal_y: np.ndarray,
    categorical: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[LGBMClassifier, IsotonicRegression, np.ndarray]:
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
    model.fit(fit_x, fit_y, sample_weight=sample_weight, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, cal_y)
    cal_pd = np.clip(calibrator.predict(raw_cal), 0.001, 0.999)
    return model, calibrator, cal_pd


def predict_pd(model: LGBMClassifier, calibrator: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    return np.clip(calibrator.predict(model.predict_proba(x)[:, 1]), 0.001, 0.999)


def choose_threshold(margin: np.ndarray, realized: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.r_[
            np.linspace(-0.06, 0.08, 57),
            np.quantile(margin, np.linspace(0.01, 0.99, 99)),
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


def odds_stress_pd(pd_values: np.ndarray, gamma: float) -> np.ndarray:
    return expit(logit(np.clip(pd_values, 0.001, 0.999)) + np.log(gamma))


def build_model_predictions(
    train: pd.DataFrame,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    *,
    reject_gamma: float | None = None,
    reject_weight_scale: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    model_idx, cal_idx = time_split_labeled(train)
    fit_x = train_x.loc[model_idx]
    fit_y = train.loc[model_idx, "default_flag"].astype(int).to_numpy()
    cal_x = train_x.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    meta: dict[str, object] = {"model_rows": int(len(model_idx)), "calibration_rows": int(len(cal_idx))}

    if reject_gamma is None:
        model, calibrator, cal_pd = fit_lightgbm(fit_x, fit_y, cal_x, cal_y, categorical)
    else:
        base_model, base_calibrator, _ = fit_lightgbm(
            fit_x,
            fit_y,
            cal_x,
            cal_y,
            categorical,
            seed=RANDOM_SEED - 1,
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
        weights = np.r_[np.ones(len(fit_y)), np.full(len(pseudo_y), reject_weight_scale)]
        model, calibrator, cal_pd = fit_lightgbm(
            aug_x,
            aug_y,
            cal_x,
            cal_y,
            categorical,
            sample_weight=weights,
            seed=RANDOM_SEED + int(reject_gamma * 100),
        )
        meta.update(
            {
                "reject_gamma": reject_gamma,
                "reject_weight_scale": reject_weight_scale,
                "reject_rows": int(len(reject_idx)),
                "accept_bad_rate": accept_bad_rate,
                "target_reject_bad_rate": target_reject_bad_rate,
                "pseudo_reject_bad_rate": float(np.mean(pseudo_y)),
                "pseudo_reject_bad_cutoff": cutoff,
            }
        )

    val_pd = predict_pd(model, calibrator, val_x)
    test_pd = predict_pd(model, calibrator, test_x)
    return cal_pd, val_pd, test_pd, meta


def evaluate_variant(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    curves: np.lib.npyio.NpzFile,
    extra: dict[str, object],
    *,
    reject_gamma: float | None = None,
) -> dict[str, object]:
    cal_pd, val_pd, test_pd, meta = build_model_predictions(
        train,
        train_x,
        val_x,
        test_x,
        categorical,
        reject_gamma=reject_gamma,
    )
    labeled_val = validation["default_flag"].notna().to_numpy()
    val_y = validation.loc[labeled_val, "default_flag"].astype(int).to_numpy()
    realized_val = realized_npv(validation.loc[labeled_val])

    bin_table = pd_interval_bin_table(cal_pd, train.loc[time_split_labeled(train)[1], "default_flag"].astype(int).to_numpy(), n_bins=10)
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
    threshold, sweep = choose_threshold(val_margin[labeled_val], realized_val)

    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0
    val_decision = (
        (val_margin > threshold)
        & (~val_prior_declined | (val_margin > PRIOR_DECLINED_MIN_MARGIN))
    )
    test_decision = (
        (test_margin > threshold)
        & (~test_prior_declined | (test_margin > PRIOR_DECLINED_MIN_MARGIN))
    )
    all_decision = np.r_[val_decision, test_decision]
    all_prior_declined = pd.concat(
        [validation["prior_decision"], test["prior_decision"]], ignore_index=True
    ).to_numpy() == 0
    all_enpv = np.r_[val_enpv, test_enpv]
    all_pd = np.r_[val_pd, test_pd]

    prior_declined_approved = all_decision & all_prior_declined
    row = {
        "variant": name,
        "feature_count": int(train_x.shape[1]),
        "threshold": threshold,
        "validation_labeled_realized_npv": float(realized_val[val_decision[labeled_val]].sum()),
        "validation_labeled_approved": int(val_decision[labeled_val].sum()),
        "validation_labeled_approval_rate": float(val_decision[labeled_val].mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized_val[val_decision[labeled_val]] < 0)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "prior_declined_approved_total": int(prior_declined_approved.sum()),
        "prior_declined_approval_rate": float(prior_declined_approved.sum() / max(all_prior_declined.sum(), 1)),
        "headline_expected_npv": float(all_enpv[all_decision].sum()),
        "prior_declined_expected_npv": float(all_enpv[prior_declined_approved].sum()),
        "prior_declined_mean_pd": float(all_pd[prior_declined_approved].mean()) if prior_declined_approved.sum() else np.nan,
        "prior_declined_expected_npv_gamma_3x": float(
            expected_npv(
                np.r_[validation["requested_amount"].to_numpy(float), test["requested_amount"].to_numpy(float)][prior_declined_approved],
                odds_stress_pd(all_pd[prior_declined_approved], 3.0),
                np.r_[curves["validation_t_star"], curves["test_t_star"]][prior_declined_approved],
                np.r_[curves["validation_recovery"], curves["test_recovery"]][prior_declined_approved],
            ).sum()
        )
        if prior_declined_approved.sum()
        else 0.0,
        "auroc": float(roc_auc_score(val_y, val_pd[labeled_val])),
        "log_loss": float(log_loss(val_y, val_pd[labeled_val], labels=[0, 1])),
        "brier": float(brier_score_loss(val_y, val_pd[labeled_val])),
        "mean_pd": float(np.mean(val_pd[labeled_val])),
        "actual_default_rate": float(np.mean(val_y)),
        "interval_bin_coverage": bin_level_coverage(
            val_pd[labeled_val], val_y, val_lower[labeled_val], val_upper[labeled_val]
        )["bin_coverage"],
        **extra,
        **meta,
    }

    variant_dir = CANDIDATE_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "applicant_id": pd.concat([validation["applicant_id"], test["applicant_id"]], ignore_index=True),
            "decision": all_decision.astype(int),
            "predicted_pd": all_pd,
            "pd_lower_90": np.r_[np.minimum(val_lower, val_pd), np.minimum(test_lower, test_pd)],
            "pd_upper_90": np.r_[np.maximum(val_upper, val_pd), np.maximum(test_upper, test_pd)],
        }
    ).to_csv(variant_dir / "submission_A_decisions.csv", index=False)
    sweep.to_csv(variant_dir / "threshold_sweep.csv", index=False)
    bin_table.to_csv(variant_dir / "pd_interval_bins.csv", index=False)
    (variant_dir / "summary.json").write_text(json.dumps(row, indent=2))
    return row


def load_reference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active_path = REPORT_DIR / "lightgbm_no_prior_active_policy_summary.json"
    if active_path.exists():
        active = json.loads(active_path.read_text())
        gov_path = REPORT_DIR / "segment_governance_summary.json"
        gov = json.loads(gov_path.read_text()) if gov_path.exists() else {}
        rows.append(
            {
                "variant": "reference_active_full_engineered",
                "validation_labeled_realized_npv": active.get("validation_labeled_realized_npv"),
                "approved_total": active.get("approved_total"),
                "prior_declined_approved_total": active.get("prior_declined_approved_total"),
                "headline_expected_npv": gov.get("approved_expected_npv_total"),
                "auroc": active.get("validation_pd_metrics", {}).get("auroc"),
                "log_loss": active.get("validation_pd_metrics", {}).get("log_loss"),
                "brier": active.get("validation_pd_metrics", {}).get("brier"),
                "reference_only": True,
            }
        )
    reject_path = OUTPUT_DIR / "candidates" / "reject_simple_gamma_1p5" / "summary.json"
    if reject_path.exists():
        reject = json.loads(reject_path.read_text())
        rows.append(
            {
                "variant": "reference_reject_simple_gamma_1p5",
                "validation_labeled_realized_npv": reject.get("validation_labeled_realized_npv"),
                "approved_total": reject.get("approved_total"),
                "prior_declined_approved_total": reject.get("prior_declined_approved_total"),
                "auroc": reject.get("validation_pd_metrics", {}).get("auroc"),
                "log_loss": reject.get("validation_pd_metrics", {}).get("log_loss"),
                "brier": reject.get("validation_pd_metrics", {}).get("brier"),
                "reference_only": True,
            }
        )
    return rows


def main() -> None:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    curves = np.load(OUTPUT_DIR / "deliverable_a_curves.npz")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)

    rows = load_reference_rows()
    variants: list[tuple[str, str, float | None]] = [
        ("raw_valid_prior", "raw_valid_prior", None),
        ("compact_risk_factors", "compact_risk_factors", None),
        ("compact_risk_factors_pca8", "compact_risk_factors_pca8", None),
        ("compact_risk_factors_reject_gamma_1p5", "compact_risk_factors", 1.5),
        ("compact_risk_factors_pca8_reject_gamma_1p5", "compact_risk_factors_pca8", 1.5),
        ("full_engineered_refit_control", "full_engineered", None),
    ]

    for name, feature_variant, reject_gamma in variants:
        print(f"Running {name}...")
        train_x, val_x, test_x, _numeric, categorical, extra = prepare_frames(
            feature_variant,
            train,
            validation,
            test,
            train_fe,
            validation_fe,
            test_fe,
        )
        extra["feature_variant"] = feature_variant
        rows.append(
            evaluate_variant(
                name,
                train,
                validation,
                test,
                train_x,
                val_x,
                test_x,
                categorical,
                curves,
                extra,
                reject_gamma=reject_gamma,
            )
        )

    result = pd.DataFrame(rows)
    sort_cols = ["validation_labeled_realized_npv", "auroc"]
    result = result.sort_values(sort_cols, ascending=[False, False], na_position="last").reset_index(drop=True)
    result.to_csv(CANDIDATE_DIR / "summary.csv", index=False)
    result.to_json(CANDIDATE_DIR / "summary.json", orient="records", indent=2)
    result.to_csv(REPORT_DIR / "compact_feature_reject_bakeoff.csv", index=False)
    print(result[
        [
            "variant",
            "validation_labeled_realized_npv",
            "headline_expected_npv",
            "approved_total",
            "prior_declined_approved_total",
            "auroc",
            "log_loss",
            "brier",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
