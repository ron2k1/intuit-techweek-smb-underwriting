#!/usr/bin/env python3
"""Build segment-governance, hidden-risk, and split-validation diagnostics.

The report is intentionally policy-facing rather than model-leaderboard-only:
it shows how the current A submission behaves across credit, cashflow,
payroll, burden, sector, prior-decision, and intervention-query features.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DELIVERABLE_A_FEATURE_SET", "all_engineered")

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, npv_default, npv_repaid, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SUBMISSION_A = PROJECT_ROOT / "outputs" / "submission" / "submission_A_decisions.csv"
CURVES_PATH = PROJECT_ROOT / "outputs" / "deliverable_a_curves.npz"
INTERVENTION_QUERIES = PROJECT_ROOT / "data" / "intervention_queries.csv"

DROP_PRIOR_SCORE_PROXY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)
N_BOOTSTRAPS = 2000
RANDOM_SEED = 2026

CORE_SEGMENT_FEATURES = [
    "owner_personal_credit_band",
    "observed_cash_balance_p10",
    "payroll_regularity_score",
    "aggregate_credit_utilization",
    "invoice_payment_delinquency_rate",
    "requested_amount_to_observed_revenue",
    "sector",
    "prior_decision",
    "observed_overdraft_count_3mo",
    "existing_debt_obligations",
    "days_since_last_external_decline",
    "prior_loans_count",
    "prior_loans_default_count",
    "prior_loans_amount_total",
    "days_since_last_inquiry_elsewhere",
    "intended_use_of_funds",
    "has_linked_bank_feed",
    "employee_count_bucket",
    "geography_region",
    "recent_inquiries_count_6mo",
    "multi_lender_inquiry_count_30d",
    "account_age_days",
    "platform_active_months",
    "bookkeeping_recency_days",
    "stated_time_in_business",
    "vintage_years",
    "requested_amount",
    "stated_annual_revenue",
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "application_channel",
]

ENGINEERED_SEGMENT_FEATURES = [
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


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.lib.npyio.NpzFile]:
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission = pd.read_csv(SUBMISSION_A)
    curves = np.load(CURVES_PATH)
    return train, validation, test, submission, curves


def expected_npv_for_eval(eval_frame: pd.DataFrame, curves: np.lib.npyio.NpzFile) -> np.ndarray:
    n_val = int((eval_frame["_split"] == "validation").sum())
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    if len(t_star) != len(eval_frame):
        raise ValueError(f"curve length mismatch: {len(t_star)} vs {len(eval_frame)}")
    return expected_npv(
        eval_frame["requested_amount"].to_numpy(float),
        eval_frame["predicted_pd"].to_numpy(float),
        t_star,
        recovery,
    )


def break_even_pd(eval_frame: pd.DataFrame, curves: np.lib.npyio.NpzFile) -> np.ndarray:
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    amount = eval_frame["requested_amount"].to_numpy(float)
    paid = npv_repaid(amount)
    default = npv_default(amount, t_star, recovery * amount)
    denom = paid - default
    return np.clip(np.divide(paid, denom, out=np.full_like(paid, np.nan), where=denom > 0), 0.0, 1.0)


def fit_approval_support(train_fe: pd.DataFrame, eval_fe: pd.DataFrame) -> np.ndarray:
    _, numeric, categorical = feature_columns(train_fe)
    numeric = [c for c in numeric if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
    categorical = [c for c in categorical if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
    feature_cols = numeric + categorical
    train_x = train_fe[feature_cols].copy()
    eval_x = eval_fe[feature_cols].copy()
    for col in categorical:
        train_x[col] = train_x[col].astype("category")
        eval_x[col] = eval_x[col].astype("category")
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        min_samples_leaf=50,
        random_state=RANDOM_SEED,
    )
    # HGB in sklearn cannot consume categoricals directly, so factorize here for
    # a lightweight support score. This is diagnostics only, not the PD model.
    train_num = train_x.copy()
    eval_num = eval_x.copy()
    for col in categorical:
        cats = pd.Categorical(pd.concat([train_num[col], eval_num[col]], ignore_index=True))
        train_num[col] = pd.Categorical(train_num[col], categories=cats.categories).codes
        eval_num[col] = pd.Categorical(eval_num[col], categories=cats.categories).codes
    train_num = train_num.replace([np.inf, -np.inf], np.nan).fillna(train_num.median(numeric_only=True))
    eval_num = eval_num.replace([np.inf, -np.inf], np.nan).fillna(train_num.median(numeric_only=True))
    model.fit(train_num, (train_fe["prior_decision"] == 1).astype(int))
    return np.clip(model.predict_proba(eval_num)[:, 1], 0.001, 0.999)


def qcut_labels(series: pd.Series, q: int, prefix: str) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < 3:
        return series.map(lambda v: f"{prefix}:missing" if pd.isna(v) else f"{prefix}:{v}")
    try:
        bins = pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        return series.map(lambda v: f"{prefix}:missing" if pd.isna(v) else f"{prefix}:{v}")
    return bins.astype(str).where(series.notna(), f"{prefix}:missing")


def add_governance_segments(eval_frame: pd.DataFrame, train_fe: pd.DataFrame, eval_fe: pd.DataFrame) -> pd.DataFrame:
    out = eval_frame.copy()
    for col in CORE_SEGMENT_FEATURES + ENGINEERED_SEGMENT_FEATURES:
        if col in eval_fe.columns:
            out[col] = eval_fe[col].to_numpy()

    out["predicted_pd_quintile"] = qcut_labels(out["predicted_pd"], 5, "pd_q")
    out["approval_support_quintile"] = qcut_labels(out["approval_support"], 5, "support_q")
    out["npv_margin_quintile"] = qcut_labels(out["expected_npv_margin"], 5, "margin_q")

    p25, p75 = np.nanquantile(out["predicted_pd"], [0.25, 0.75])
    out["pd_risk_tier"] = np.select(
        [out["predicted_pd"] <= p25, out["predicted_pd"] <= p75],
        ["low", "medium"],
        default="high",
    )
    support25, support75 = np.nanquantile(out["approval_support"], [0.25, 0.75])
    out["support_tier"] = np.select(
        [out["approval_support"] <= support25, out["approval_support"] <= support75],
        ["low_support", "medium_support"],
        default="high_support",
    )

    if {"aggregate_credit_utilization", "invoice_payment_delinquency_rate"}.issubset(out.columns):
        util_q = pd.qcut(out["aggregate_credit_utilization"], 4, labels=["util_q1", "util_q2", "util_q3", "util_q4"], duplicates="drop")
        delin_q = pd.qcut(out["invoice_payment_delinquency_rate"], 4, labels=["delin_q1", "delin_q2", "delin_q3", "delin_q4"], duplicates="drop")
        out["utilization_delinquency_segment"] = util_q.astype(str) + "|" + delin_q.astype(str)
    if "requested_amount_to_observed_revenue" in out.columns:
        out["revenue_burden_quintile"] = qcut_labels(out["requested_amount_to_observed_revenue"], 5, "burden_q")
    if "observed_cash_balance_p10" in out.columns:
        out["cash_balance_quintile"] = qcut_labels(out["observed_cash_balance_p10"], 5, "cash_q")
    if "payroll_regularity_score" in out.columns:
        out["payroll_quintile"] = qcut_labels(out["payroll_regularity_score"], 5, "payroll_q")
    if "credit_stress_index" in out.columns:
        out["credit_stress_quintile"] = qcut_labels(out["credit_stress_index"], 5, "credit_stress_q")
    if "cash_stress_index" in out.columns:
        out["cash_stress_quintile"] = qcut_labels(out["cash_stress_index"], 5, "cash_stress_q")
    return out


def aggregate_segment(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    labeled = frame["default_flag"].notna()
    approved = frame["decision"] == 1
    prior_declined = frame["prior_decision"] == 0
    rows = []
    for value, group in frame.groupby(feature, dropna=False):
        g_labeled = group["default_flag"].notna()
        g_approved = group["decision"] == 1
        g_prior_declined = group["prior_decision"] == 0
        g_labeled_approved = g_labeled & g_approved
        rows.append(
            {
                "feature": feature,
                "segment": str(value),
                "rows": int(len(group)),
                "validation_rows": int((group["_split"] == "validation").sum()),
                "test_rows": int((group["_split"] == "test").sum()),
                "approved_count": int(g_approved.sum()),
                "approval_rate": float(g_approved.mean()),
                "prior_declined_rows": int(g_prior_declined.sum()),
                "prior_declined_approved": int((g_prior_declined & g_approved).sum()),
                "prior_declined_approval_rate": float((g_prior_declined & g_approved).sum() / max(g_prior_declined.sum(), 1)),
                "labeled_rows": int(g_labeled.sum()),
                "observed_default_rate": float(group.loc[g_labeled, "default_flag"].mean()) if g_labeled.sum() else np.nan,
                "approved_labeled_rows": int(g_labeled_approved.sum()),
                "approved_observed_default_rate": float(group.loc[g_labeled_approved, "default_flag"].mean()) if g_labeled_approved.sum() else np.nan,
                "mean_predicted_pd": float(group["predicted_pd"].mean()),
                "mean_pd_approved": float(group.loc[g_approved, "predicted_pd"].mean()) if g_approved.sum() else np.nan,
                "mean_pd_interval_width": float((group["pd_upper_90"] - group["pd_lower_90"]).mean()),
                "mean_approval_support": float(group["approval_support"].mean()),
                "expected_npv_total_approved": float(group.loc[g_approved, "expected_npv"].sum()),
                "expected_npv_margin_approved": float(group.loc[g_approved, "expected_npv_margin"].mean()) if g_approved.sum() else np.nan,
                "labeled_validation_realized_npv_approved": float(group.loc[g_labeled_approved, "realized_npv"].sum()) if g_labeled_approved.sum() else 0.0,
                "mean_break_even_pd_approved": float(group.loc[g_approved, "break_even_pd"].mean()) if g_approved.sum() else np.nan,
                "mean_pd_headroom_approved": float(group.loc[g_approved, "pd_headroom"].mean()) if g_approved.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["feature", "rows"], ascending=[True, False])


def build_segment_tables(frame: pd.DataFrame, intervention_features: set[str]) -> pd.DataFrame:
    segment_features = [
        "pd_risk_tier",
        "predicted_pd_quintile",
        "approval_support_quintile",
        "support_tier",
        "npv_margin_quintile",
        "owner_personal_credit_band",
        "cash_balance_quintile",
        "payroll_quintile",
        "utilization_delinquency_segment",
        "revenue_burden_quintile",
        "sector",
        "prior_decision",
        "has_linked_bank_feed",
        "employee_count_bucket",
        "geography_region",
        "intended_use_of_funds",
        "application_channel",
        "credit_stress_quintile",
        "cash_stress_quintile",
    ]
    segment_features.extend(sorted(intervention_features))
    segment_features.extend(ENGINEERED_SEGMENT_FEATURES)
    seen = []
    for col in segment_features:
        if col in frame.columns and col not in seen:
            seen.append(col)

    tables = []
    for col in seen:
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 12:
            temp = frame.copy()
            temp[f"{col}_quintile"] = qcut_labels(values, 5, f"{col}_q")
            tables.append(aggregate_segment(temp, f"{col}_quintile"))
        else:
            tables.append(aggregate_segment(frame, col))
    return pd.concat(tables, ignore_index=True)


def prior_declined_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    focus = frame[(frame["prior_decision"] == 0) & (frame["decision"] == 1)].copy()
    if focus.empty:
        return pd.DataFrame()
    features = [
        "pd_risk_tier",
        "support_tier",
        "cash_balance_quintile",
        "payroll_quintile",
        "revenue_burden_quintile",
        "credit_stress_quintile",
        "cash_stress_quintile",
        "sector",
        "owner_personal_credit_band",
        "has_linked_bank_feed",
    ]
    rows = []
    for feature in [f for f in features if f in focus.columns]:
        for value, group in focus.groupby(feature, dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "segment": str(value),
                    "prior_declined_approved": int(len(group)),
                    "share_of_prior_declined_approved": float(len(group) / len(focus)),
                    "mean_predicted_pd": float(group["predicted_pd"].mean()),
                    "predicted_defaults": float(group["predicted_pd"].sum()),
                    "mean_approval_support": float(group["approval_support"].mean()),
                    "expected_npv_total": float(group["expected_npv"].sum()),
                    "mean_expected_npv_margin": float(group["expected_npv_margin"].mean()),
                    "mean_break_even_pd": float(group["break_even_pd"].mean()),
                    "mean_pd_headroom": float(group["pd_headroom"].mean()),
                    "mean_break_even_multiplier": float(group["break_even_multiplier"].replace([np.inf, -np.inf], np.nan).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["expected_npv_total", "prior_declined_approved"], ascending=[True, False])


def stress_table(frame: pd.DataFrame, curves: np.lib.npyio.NpzFile) -> pd.DataFrame:
    prior = (frame["prior_decision"] == 0) & (frame["decision"] == 1)
    approved = frame["decision"] == 1
    rows = []
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    for multiplier in [1.0, 1.10, 1.25, 1.50, 2.0, 3.0]:
        pd_stressed = frame["predicted_pd"].to_numpy(float).copy()
        pd_stressed[prior.to_numpy()] = np.clip(pd_stressed[prior.to_numpy()] * multiplier, 0.0, 0.999)
        enpv = expected_npv(frame["requested_amount"].to_numpy(float), pd_stressed, t_star, recovery)
        rows.append(
            {
                "prior_declined_pd_multiplier": multiplier,
                "approved_expected_npv_total": float(enpv[approved.to_numpy()].sum()),
                "prior_declined_approved_expected_npv": float(enpv[prior.to_numpy()].sum()),
                "prior_declined_approved_mean_pd": float(pd_stressed[prior.to_numpy()].mean()) if prior.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_validation_npv(frame: pd.DataFrame) -> dict[str, float]:
    validation = frame[(frame["_split"] == "validation") & frame["default_flag"].notna()].copy()
    if validation.empty:
        return {}
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.where(validation["decision"].to_numpy(int) == 1, validation["realized_npv"].to_numpy(float), 0.0)
    boots = np.array([rng.choice(values, size=len(values), replace=True).sum() for _ in range(N_BOOTSTRAPS)])
    return {
        "labeled_validation_rows": int(len(values)),
        "point_estimate": float(values.sum()),
        "bootstrap_p05": float(np.quantile(boots, 0.05)),
        "bootstrap_p50": float(np.quantile(boots, 0.50)),
        "bootstrap_p95": float(np.quantile(boots, 0.95)),
        "bootstrap_std": float(boots.std(ddof=1)),
    }


def time_split_labeled(labeled: pd.DataFrame, fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * fraction)
    return ordered[:split_at], ordered[split_at:]


def fit_lightgbm_fold(train_fe: pd.DataFrame, train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> dict[str, float]:
    _, numeric, categorical = feature_columns(train_fe)
    numeric = [c for c in numeric if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
    categorical = [c for c in categorical if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
    feature_cols = numeric + categorical

    fit_idx, cal_idx = time_split_labeled(train_rows)
    fit_x = train_fe.loc[fit_idx, feature_cols].copy()
    cal_x = train_fe.loc[cal_idx, feature_cols].copy()
    val_x = train_fe.loc[val_rows.index, feature_cols].copy()
    fit_y = train_rows.loc[fit_idx, "default_flag"].astype(int)
    cal_y = train_rows.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    val_y = val_rows["default_flag"].astype(int).to_numpy()

    for col in categorical:
        fit_x[col] = fit_x[col].astype("category")
        cal_x[col] = cal_x[col].astype("category")
        val_x[col] = val_x[col].astype("category")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    model.fit(fit_x, fit_y, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(raw_cal, cal_y)
    pred = np.clip(iso.predict(model.predict_proba(val_x)[:, 1]), 0.001, 0.999)
    return {
        "rows": int(len(val_rows)),
        "default_rate": float(val_y.mean()),
        "auroc": float(roc_auc_score(val_y, pred)),
        "log_loss": float(log_loss(val_y, pred, labels=[0, 1])),
        "brier": float(brier_score_loss(val_y, pred)),
        "mean_pd": float(pred.mean()),
    }


def grouped_time_cv(train: pd.DataFrame, train_fe: pd.DataFrame) -> pd.DataFrame:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp")
    blocks = np.array_split(ordered.index.to_numpy(), 5)
    rows = []
    for fold_id in range(1, 5):
        val_idx = blocks[fold_id]
        train_idx = np.concatenate(blocks[:fold_id])
        val_businesses = set(train.loc[val_idx, "business_id"])
        purged_train_idx = np.array([i for i in train_idx if train.loc[i, "business_id"] not in val_businesses])
        if len(purged_train_idx) < 100 or len(val_idx) < 100:
            continue
        fold_train = train.loc[purged_train_idx].copy()
        fold_val = train.loc[val_idx].copy()
        metrics = fit_lightgbm_fold(train_fe, fold_train, fold_val)
        rows.append(
            {
                "fold": fold_id,
                "train_rows_before_group_purge": int(len(train_idx)),
                "train_rows_after_group_purge": int(len(purged_train_idx)),
                "purged_business_overlap_rows": int(len(train_idx) - len(purged_train_idx)),
                "validation_start": str(pd.to_datetime(fold_val["application_timestamp"]).min()),
                "validation_end": str(pd.to_datetime(fold_val["application_timestamp"]).max()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def split_leakage_audit(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    labeled = train[train["default_flag"].notna()].copy()
    model_idx, cal_idx = time_split_labeled(labeled)
    model_businesses = set(train.loc[model_idx, "business_id"])
    cal_businesses = set(train.loc[cal_idx, "business_id"])
    return {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "labeled_train_rows": int(len(labeled)),
        "model_split_rows": int(len(model_idx)),
        "calibration_split_rows": int(len(cal_idx)),
        "businesses_in_both_model_and_calibration": int(len(model_businesses & cal_businesses)),
        "business_overlap_train_validation": int(len(set(train["business_id"]) & set(validation["business_id"]))),
        "business_overlap_train_test": int(len(set(train["business_id"]) & set(test["business_id"]))),
        "business_overlap_validation_test": int(len(set(validation["business_id"]) & set(test["business_id"]))),
    }


def write_markdown(summary: dict[str, object], top_risks: pd.DataFrame) -> None:
    lines = [
        "# Segment Governance Report",
        "",
        "## Current Policy",
        f"- Feature set: `{summary['feature_set']}`",
        f"- Approved total: {summary['approved_total']:,}",
        f"- Prior-declined approved: {summary['prior_declined_approved_total']:,}",
        f"- Labeled validation NPV: ${summary['bootstrap_validation_npv']['point_estimate']:,.0f}",
        f"- Bootstrap 90% interval: ${summary['bootstrap_validation_npv']['bootstrap_p05']:,.0f} to ${summary['bootstrap_validation_npv']['bootstrap_p95']:,.0f}",
        "",
        "## Hidden Prior-Declined Risk",
        "- Local validation has no direct labels for prior-declined applicants, so this report treats that region as hidden-outcome risk.",
        "- `approval_support` estimates how close an applicant is to the historically prior-approved/labeled support region.",
        "- `break_even_pd` is the default probability that would make the applicant's expected NPV zero under the slide cash-flow formula.",
        "",
        "## Highest-Risk Prior-Declined Approved Segments",
    ]
    if top_risks.empty:
        lines.append("- None.")
    else:
        for row in top_risks.head(12).itertuples(index=False):
            lines.append(
                f"- `{row.feature}={row.segment}`: {row.prior_declined_approved:,} approvals, "
                f"mean PD {row.mean_predicted_pd:.3f}, mean support {row.mean_approval_support:.3f}, "
                f"expected NPV ${row.expected_npv_total:,.0f}, headroom {row.mean_pd_headroom:.3f}"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `segment_governance_by_factor.csv`",
            "- `prior_declined_hidden_risk_by_segment.csv`",
            "- `prior_declined_stress_test.csv`",
            "- `grouped_time_cv_diagnostics.csv`",
            "- `split_leakage_audit.json`",
            "- `intervention_feature_inventory.csv`",
        ]
    )
    (REPORT_DIR / "segment_governance_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, test, submission, curves = load_frames()
    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    eval_raw = pd.concat([validation.assign(_split="validation"), test.assign(_split="test")], ignore_index=True)
    eval_fe = pd.concat([validation_fe, test_fe], ignore_index=True)
    eval_frame = eval_raw.join(submission[["decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]])
    eval_frame["approval_support"] = fit_approval_support(train_fe, eval_fe)
    eval_frame["expected_npv"] = expected_npv_for_eval(eval_frame, curves)
    eval_frame["expected_npv_margin"] = eval_frame["expected_npv"] / np.maximum(eval_frame["requested_amount"].to_numpy(float), 1.0)
    eval_frame["break_even_pd"] = break_even_pd(eval_frame, curves)
    eval_frame["pd_headroom"] = eval_frame["break_even_pd"] - eval_frame["predicted_pd"]
    eval_frame["break_even_multiplier"] = eval_frame["break_even_pd"] / np.maximum(eval_frame["predicted_pd"], 1e-6)
    eval_frame["realized_npv"] = np.nan
    val_labeled = validation["default_flag"].notna().to_numpy()
    eval_frame.loc[: len(validation) - 1, "realized_npv"] = np.where(
        val_labeled,
        realized_npv(validation.fillna(np.nan)),
        np.nan,
    )
    eval_frame = add_governance_segments(eval_frame, train_fe, eval_fe)

    queries = pd.read_csv(INTERVENTION_QUERIES)
    intervention_features = set(map(str, queries["feature_name"].unique()))
    intervention_inventory = (
        queries.groupby("feature_name", dropna=False)
        .agg(query_count=("query_id", "size"), applicant_count=("applicant_id", "nunique"))
        .reset_index()
    )
    intervention_inventory["present_in_raw_data"] = intervention_inventory["feature_name"].isin(eval_raw.columns)
    intervention_inventory["present_in_engineered_frame"] = intervention_inventory["feature_name"].isin(eval_fe.columns)
    intervention_inventory["included_in_segment_report"] = intervention_inventory["feature_name"].isin(
        set(CORE_SEGMENT_FEATURES + ENGINEERED_SEGMENT_FEATURES)
    )

    segment_table = build_segment_tables(eval_frame, intervention_features)
    hidden_risk = prior_declined_risk_table(eval_frame)
    stress = stress_table(eval_frame, curves)
    cv = grouped_time_cv(train, train_fe)
    leakage = split_leakage_audit(train, validation, test)
    bootstrap = bootstrap_validation_npv(eval_frame)

    segment_table.to_csv(REPORT_DIR / "segment_governance_by_factor.csv", index=False)
    hidden_risk.to_csv(REPORT_DIR / "prior_declined_hidden_risk_by_segment.csv", index=False)
    stress.to_csv(REPORT_DIR / "prior_declined_stress_test.csv", index=False)
    cv.to_csv(REPORT_DIR / "grouped_time_cv_diagnostics.csv", index=False)
    intervention_inventory.to_csv(REPORT_DIR / "intervention_feature_inventory.csv", index=False)
    (REPORT_DIR / "split_leakage_audit.json").write_text(json.dumps(leakage, indent=2))

    approved = eval_frame["decision"] == 1
    prior_declined_approved = approved & (eval_frame["prior_decision"] == 0)
    summary = {
        "feature_set": os.getenv("DELIVERABLE_A_FEATURE_SET", "all_engineered"),
        "approved_total": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "prior_declined_approved_total": int(prior_declined_approved.sum()),
        "prior_declined_approval_rate": float(prior_declined_approved.sum() / max((eval_frame["prior_decision"] == 0).sum(), 1)),
        "approved_expected_npv_total": float(eval_frame.loc[approved, "expected_npv"].sum()),
        "prior_declined_approved_expected_npv": float(eval_frame.loc[prior_declined_approved, "expected_npv"].sum()),
        "prior_declined_approved_mean_pd": float(eval_frame.loc[prior_declined_approved, "predicted_pd"].mean()),
        "prior_declined_approved_mean_support": float(eval_frame.loc[prior_declined_approved, "approval_support"].mean()),
        "intervention_features_total": int(len(intervention_features)),
        "intervention_features_missing_from_raw_data": intervention_inventory.loc[
            ~intervention_inventory["present_in_raw_data"], "feature_name"
        ].astype(str).tolist(),
        "bootstrap_validation_npv": bootstrap,
        "split_leakage_audit": leakage,
        "grouped_time_cv_mean": cv[["auroc", "log_loss", "brier"]].mean(numeric_only=True).to_dict() if not cv.empty else {},
    }
    (REPORT_DIR / "segment_governance_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, hidden_risk)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
