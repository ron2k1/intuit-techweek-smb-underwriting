"""Deliverable C counterfactual predictions.

The challenge asks for do(feature=value) PDs, but the data are observational and
labels are selected by the prior underwriter. This module follows the project
DAG memo by using a causal-safe PD feature set, deterministic single-feature
interventions, duplicate-query caching, and wider intervals for historical,
proxy, policy-derived, or out-of-support interventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table
from src.deliverable_a_pipeline import (
    CATEGORICAL_BASE,
    CSV_DIR,
    DROP_FOR_PD,
    EXPERIMENTAL_CONFOUNDER_FEATURES,
    REPORT_DIR,
    SUBMISSION_DIR,
    add_application_features,
    ensemble_predictions,
    fit_calibrated_models,
    metric_summary,
    train_approval_propensity,
)


PRIOR_POLICY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)
POLICY_DERIVED_FEATURES = {
    "prior_underwriter_score",
    "prior_decision",
    "prior_approved_amount",
}
RISK_INCREASING_FEATURES = {
    "requested_amount",
    "aggregate_credit_utilization",
    "recent_inquiries_count_6mo",
    "existing_debt_obligations",
    "observed_overdraft_count_3mo",
    "invoice_payment_delinquency_rate",
    "multi_lender_inquiry_count_30d",
    "observed_revenue_volatility",
    "bookkeeping_recency_days",
    "prior_loans_default_count",
    "requested_amount_to_observed_revenue",
}
RISK_DECREASING_FEATURES = {
    "observed_cash_balance_p10",
    "payroll_regularity_score",
    "stated_annual_revenue",
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "stated_time_in_business",
    "vintage_years",
    "account_age_days",
    "platform_active_months",
    "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere",
}
MONOTONIC_NEUTRALIZE_FEATURES = RISK_INCREASING_FEATURES | RISK_DECREASING_FEATURES
CUSTOM_TREATMENTS = {
    # Requested amount is the cleanest product lever: it changes repayment burden.
    "requested_amount": (
        "amount_burden_intervention",
        1.0,
        0.010,
        "Direct loan-size intervention; engineered burden ratios are recomputed.",
    ),
    # Self-reported fields are application statements. They can move model score,
    # but they are weaker causal levers than observed business state.
    "stated_annual_revenue": (
        "self_report_proxy",
        0.30,
        0.040,
        "Reported revenue is confounded by true business scale and reporting behavior; shrink toward baseline.",
    ),
    "stated_time_in_business": (
        "self_report_proxy",
        0.35,
        0.035,
        "Reported age is partly historical identity/proxy rather than a manipulable operating lever.",
    ),
    # Observed bank-feed metrics are closer to business-state interventions, but
    # still depend on having a linked feed and recent measurement window.
    "observed_monthly_revenue_avg_3mo": (
        "observed_business_state",
        0.85,
        0.015,
        "Observed revenue is a business-state proxy; use most of the model delta and recompute revenue ratios.",
    ),
    "observed_revenue_trend_3mo": (
        "observed_business_state",
        0.85,
        0.015,
        "Revenue trend is measured business state; retain most model effect with modest measurement uncertainty.",
    ),
    "observed_revenue_volatility": (
        "observed_business_state",
        0.85,
        0.015,
        "Revenue volatility is measured business state; retain most model effect with modest measurement uncertainty.",
    ),
    "observed_cash_balance_p10": (
        "observed_business_state",
        0.85,
        0.015,
        "Cash balance is measured liquidity; retain most model effect with modest measurement uncertainty.",
    ),
    "observed_overdraft_count_3mo": (
        "observed_business_state",
        0.90,
        0.010,
        "Overdrafts are measured cash stress; retain most model effect.",
    ),
    "payroll_regularity_score": (
        "observed_business_state",
        0.85,
        0.015,
        "Payroll regularity is measured operating stability; retain most model effect.",
    ),
    # Bureau/current credit state is a plausible intervention target only through
    # balance-sheet behavior, so keep the sign/signal but avoid overclaiming.
    "aggregate_credit_utilization": (
        "credit_state_intervention",
        0.90,
        0.010,
        "Credit utilization is current credit stress; retain most model effect.",
    ),
    "existing_debt_obligations": (
        "credit_state_intervention",
        0.90,
        0.010,
        "Debt burden is current credit state; retain most model effect and recompute debt ratios.",
    ),
    "recent_inquiries_count_6mo": (
        "credit_context_intervention",
        0.75,
        0.020,
        "Recent inquiries are partly demand/urgency proxy; shrink modestly.",
    ),
    "owner_personal_credit_band": (
        "credit_state_intervention",
        0.85,
        0.015,
        "Credit band is a creditworthiness proxy; retain most effect but widen for coarse binning.",
    ),
    "multi_lender_inquiry_count_30d": (
        "application_context_proxy",
        0.65,
        0.025,
        "Multi-lender inquiry count is an urgency/context proxy; shrink toward baseline.",
    ),
    "application_channel": (
        "application_context_proxy",
        0.50,
        0.035,
        "Channel is confounded by applicant mix and acquisition context; do not treat as a pure risk lever.",
    ),
    "invoice_payment_delinquency_rate": (
        "platform_state_intervention",
        0.85,
        0.015,
        "Invoice delinquency is observed payment behavior; retain most effect.",
    ),
    "has_linked_bank_feed": (
        "measurement_process_intervention",
        0.15,
        0.070,
        "Bank-feed linkage changes observability and selection, not business health itself.",
    ),
}
DEFAULT_DIRECT_TREATMENT = (
    "direct_intervention",
    1.0,
    0.0,
    "Data dictionary marks this as intervenable; apply do-value with standard support checks.",
)
DEFAULT_HISTORICAL_TREATMENT = (
    "historical_or_proxy",
    0.35,
    0.035,
    "Non-intervenable historical/context/proxy field; shrink model delta and widen interval.",
)
DEFAULT_POLICY_TREATMENT = (
    "policy_artifact",
    0.0,
    0.08,
    "Prior-underwriter artifact/proxy; exclude from causal model and neutralize causal delta.",
)
DEFAULT_AMBIGUOUS_TREATMENT = (
    "ambiguous",
    0.50,
    0.025,
    "Ambiguous intervention semantics; partial shrinkage.",
)
SIGN_EFFECT_EPS = 0.0025


@dataclass(frozen=True)
class CounterfactualArtifacts:
    submission: pd.DataFrame
    query_diagnostics: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    treatment_plan: pd.DataFrame
    sign_violations: pd.DataFrame
    metrics: dict[str, Any]


def causal_safe_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return all engineered model columns with prior-policy proxies removed.

    This intentionally ignores DELIVERABLE_A_FEATURE_SET so C always sees the
    full risk/segment feature engineering layer. C then removes prior-policy
    artifacts, because they are predictive summaries but not causal drivers.
    """
    cols = [c for c in frame.columns if c not in DROP_FOR_PD]
    safe_cols = [
        col
        for col in cols
        if not any(token in col for token in PRIOR_POLICY_TOKENS)
    ]
    safe_categorical = sorted(c for c in safe_cols if c in CATEGORICAL_BASE)
    safe_numeric = [col for col in safe_cols if col not in safe_categorical]
    return safe_cols, safe_numeric, safe_categorical


def _dictionary_maps(dictionary: pd.DataFrame) -> tuple[dict[str, bool], dict[str, str]]:
    name_col = "field" if "field" in dictionary.columns else dictionary.columns[0]
    intervenable = {
        str(row[name_col]): bool(row["intervenable"])
        for _, row in dictionary.iterrows()
        if "intervenable" in dictionary.columns
    }
    group = {
        str(row[name_col]): str(row["group"])
        for _, row in dictionary.iterrows()
        if "group" in dictionary.columns
    }
    return intervenable, group


def classify_intervention(
    feature: str,
    *,
    intervenable: dict[str, bool],
    group: dict[str, str],
) -> tuple[str, float, float, str]:
    """Return class, delta shrink factor, and interval widening.

    Direct interventions keep the model-implied delta. Historical/proxy fields
    are answerable per the rules, but the effect is shrunk because changing a
    historical summary is not the same as a clean business intervention.
    """
    if feature in CUSTOM_TREATMENTS:
        return CUSTOM_TREATMENTS[feature]
    if feature in POLICY_DERIVED_FEATURES or any(token in feature for token in PRIOR_POLICY_TOKENS):
        return DEFAULT_POLICY_TREATMENT
    if intervenable.get(feature, False):
        return DEFAULT_DIRECT_TREATMENT
    feature_group = group.get(feature, "unknown")
    if feature_group in {
        "business_identity",
        "platform_engagement",
        "application_context",
        "bank_feed",
        "self_reported",
        "bureau_credit",
    }:
        return DEFAULT_HISTORICAL_TREATMENT
    return DEFAULT_AMBIGUOUS_TREATMENT


def coerce_intervention_value(series: pd.Series, raw_value: Any) -> Any:
    """Coerce CSV intervention values to the applicant column's native type."""
    if pd.isna(raw_value):
        return np.nan
    if pd.api.types.is_bool_dtype(series):
        return bool(int(float(raw_value)))
    if pd.api.types.is_integer_dtype(series):
        return int(round(float(raw_value)))
    if pd.api.types.is_float_dtype(series):
        return float(raw_value)
    return str(raw_value)


def intervention_support(
    train: pd.DataFrame,
    feature: str,
    raw_value: Any,
) -> dict[str, Any]:
    if feature not in train.columns:
        return {
            "feature_exists": False,
            "support_status": "missing_feature",
            "outside_min_max": True,
            "outside_p01_p99": True,
        }

    series = train[feature]
    value = coerce_intervention_value(series, raw_value)
    out: dict[str, Any] = {
        "feature_exists": True,
        "support_status": "in_support",
        "outside_min_max": False,
        "outside_p01_p99": False,
        "unseen_category": False,
    }
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        train_num = pd.to_numeric(series, errors="coerce")
        val = float(value)
        p01 = float(train_num.quantile(0.01))
        p99 = float(train_num.quantile(0.99))
        min_v = float(train_num.min())
        max_v = float(train_num.max())
        out.update(
            {
                "train_min": min_v,
                "train_p01": p01,
                "train_p99": p99,
                "train_max": max_v,
                "intervention_numeric": val,
                "outside_min_max": bool(val < min_v or val > max_v),
                "outside_p01_p99": bool(val < p01 or val > p99),
            }
        )
    else:
        seen = {str(v) for v in series.dropna().astype(str).unique()}
        val = str(value)
        out.update(
            {
                "intervention_category": val,
                "n_train_levels": len(seen),
                "unseen_category": val not in seen,
            }
        )

    if out["outside_min_max"] or out["unseen_category"]:
        out["support_status"] = "outside_train_support"
    elif out["outside_p01_p99"]:
        out["support_status"] = "tail_support"
    return out


def support_widening(support: dict[str, Any]) -> float:
    if support.get("support_status") == "missing_feature":
        return 0.10
    if support.get("outside_min_max") or support.get("unseen_category"):
        return 0.08
    if support.get("outside_p01_p99"):
        return 0.035
    return 0.0


def sign_expectation(feature: str) -> int:
    if feature in RISK_INCREASING_FEATURES:
        return 1
    if feature in RISK_DECREASING_FEATURES:
        return -1
    return 0


def intervention_direction(
    applicants: pd.DataFrame,
    queries: pd.DataFrame,
) -> np.ndarray:
    """Return sign(intervention_value - original_value) for numeric features."""
    by_applicant = applicants.set_index("applicant_id", drop=False)
    directions = []
    for query in queries.itertuples(index=False):
        feature = str(query.feature_name)
        if feature not in applicants.columns:
            directions.append(np.nan)
            continue
        series = applicants[feature]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            directions.append(np.nan)
            continue
        original = by_applicant.loc[query.applicant_id, feature]
        try:
            delta = float(query.intervention_value) - float(original)
        except (TypeError, ValueError):
            directions.append(np.nan)
            continue
        directions.append(float(np.sign(delta)))
    return np.asarray(directions, dtype=float)


def _make_counterfactual_rows(
    applicants: pd.DataFrame,
    queries: pd.DataFrame,
    train: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_applicant = applicants.set_index("applicant_id", drop=False)
    intervenable, group = _dictionary_maps(dictionary)
    rows = []
    diagnostics = []
    cache: dict[tuple[str, str, str], int] = {}

    for query in queries.itertuples(index=False):
        key = (str(query.applicant_id), str(query.feature_name), str(query.intervention_value))
        if key in cache:
            source_idx = cache[key]
            row = rows[source_idx].copy()
            row["query_id"] = query.query_id
            row["_duplicate_source_query_id"] = diagnostics[source_idx]["query_id"]
            rows.append(row)
        else:
            if query.applicant_id not in by_applicant.index:
                raise KeyError(f"query applicant_id not found: {query.applicant_id}")
            row = by_applicant.loc[query.applicant_id].copy()
            if query.feature_name not in row.index:
                raise KeyError(f"query feature_name not found: {query.feature_name}")
            value = coerce_intervention_value(applicants[query.feature_name], query.intervention_value)
            row[query.feature_name] = value
            row["query_id"] = query.query_id
            row["_duplicate_source_query_id"] = ""
            cache[key] = len(rows)
            rows.append(row)

        intervention_class, shrink, class_widening, treatment_reason = classify_intervention(
            str(query.feature_name),
            intervenable=intervenable,
            group=group,
        )
        support = intervention_support(train, str(query.feature_name), query.intervention_value)
        diagnostics.append(
            {
                "query_id": query.query_id,
                "applicant_id": query.applicant_id,
                "feature_name": query.feature_name,
                "intervention_value": query.intervention_value,
                "intervention_class": intervention_class,
                "delta_shrink_factor": shrink,
                "class_interval_widening": class_widening,
                "treatment_reason": treatment_reason,
                "support_interval_widening": support_widening(support),
                "is_duplicate_intervention": key in cache and cache[key] != len(rows) - 1,
                "duplicate_source_query_id": rows[-1]["_duplicate_source_query_id"],
                **support,
            }
        )

    raw_cf = pd.DataFrame(rows).drop(columns=["_duplicate_source_query_id"])
    diagnostics_df = pd.DataFrame(diagnostics)
    return raw_cf.reset_index(drop=True), diagnostics_df


def _feature_diagnostics(query_diag: pd.DataFrame) -> pd.DataFrame:
    agg = (
        query_diag.groupby("feature_name", dropna=False)
        .agg(
            count=("query_id", "size"),
            intervention_class=("intervention_class", lambda s: ",".join(sorted(set(map(str, s))))),
            outside_min_max_rate=("outside_min_max", "mean"),
            outside_p01_p99_rate=("outside_p01_p99", "mean"),
            unseen_category_rate=("unseen_category", "mean"),
            duplicate_count=("is_duplicate_intervention", "sum"),
            monotonic_guard_count=("monotonic_guard_applied", "sum"),
            raw_sign_violation_count=("raw_sign_violation", "sum"),
            mean_delta_raw=("delta_raw", "mean"),
            mean_delta_pre_guard=("delta_pre_guard", "mean"),
            mean_delta_final=("delta_final", "mean"),
            mean_base_pd=("base_pd", "mean"),
            mean_predicted_pd_cf=("predicted_pd_cf", "mean"),
            mean_interval_width=("interval_width", "mean"),
            sign_check_eligible=("sign_check_material", "sum"),
            sign_violation_count=("sign_violation", "sum"),
            sign_check_rate=("sign_check_pass", lambda s: float(np.nanmean(s)) if np.any(pd.notna(s)) else np.nan),
            treatment_reason=("treatment_reason", lambda s: " | ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    return agg.sort_values(["count", "feature_name"], ascending=[False, True])


def _treatment_plan(dictionary: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    intervenable, group = _dictionary_maps(dictionary)
    counts = queries["feature_name"].value_counts().to_dict()
    rows = []
    for feature in sorted(queries["feature_name"].unique()):
        treatment_class, shrink, widening, reason = classify_intervention(
            feature,
            intervenable=intervenable,
            group=group,
        )
        rows.append(
            {
                "feature_name": feature,
                "query_count": int(counts.get(feature, 0)),
                "dictionary_group": group.get(feature, "unknown"),
                "dictionary_intervenable": bool(intervenable.get(feature, False)),
                "causal_treatment_class": treatment_class,
                "delta_shrink_factor": shrink,
                "extra_interval_widening": widening,
                "expected_delta_sign_per_unit_increase": sign_expectation(feature),
                "predictive_allowed": feature not in POLICY_DERIVED_FEATURES
                and not any(token in feature for token in PRIOR_POLICY_TOKENS),
                "causal_claim_allowed": treatment_class in {
                    "amount_burden_intervention",
                    "observed_business_state",
                    "credit_state_intervention",
                    "credit_context_intervention",
                    "platform_state_intervention",
                    "direct_intervention",
                },
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["query_count", "feature_name"], ascending=[False, True])


def _sign_violation_report(query_diag: pd.DataFrame) -> pd.DataFrame:
    violations = query_diag[query_diag["sign_violation"]].copy()
    if violations.empty:
        return violations
    explanations = {
        "self_report_proxy": "Self-reported fields are confounded by true business quality and reporting behavior; a reported change is not the same as changing latent business health.",
        "observed_business_state": "Observed bank-feed features are correlated with missingness, seasonality, and business scale; model deltas can flip when conditioning on all other features.",
        "credit_state_intervention": "Credit state variables are correlated with unobserved owner/business quality; local perturbations can conflict with correlated controls held fixed.",
        "credit_context_intervention": "Inquiry counts are urgency/context proxies; holding all other applicant stress signals fixed can make the marginal model effect unstable.",
        "application_context_proxy": "Application context is confounded by applicant mix and timing; naive perturbation is not a pure causal effect.",
        "platform_state_intervention": "Platform behavior is partly a proxy for business process maturity; local deltas can be dominated by correlated engineered indices.",
        "historical_or_proxy": "Historical/proxy fields are not clean interventions; the pipeline shrinks their deltas and widens intervals rather than forcing a sign.",
    }
    violations["diagnostic_explanation"] = violations["intervention_class"].map(explanations).fillna(
        "Observed-data perturbation disagrees with expected monotonic direction; retained as uncertainty signal, not forcibly corrected."
    )
    keep = [
        "query_id",
        "applicant_id",
        "feature_name",
        "intervention_value",
        "intervention_class",
        "intervention_direction",
        "expected_delta_sign",
        "base_pd",
        "raw_counterfactual_pd",
        "predicted_pd_cf",
        "delta_raw",
        "delta_pre_guard",
        "delta_final",
        "raw_sign_violation",
        "monotonic_guard_applied",
        "pd_cf_lower_90",
        "pd_cf_upper_90",
        "support_status",
        "treatment_reason",
        "diagnostic_explanation",
    ]
    return violations[keep].sort_values(["feature_name", "query_id"])


def build_counterfactuals() -> CounterfactualArtifacts:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    queries = pd.read_csv(Path("data") / "intervention_queries.csv")
    dictionary = pd.read_csv(Path("data") / "data_dictionary.csv")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = causal_safe_feature_columns(train_fe)

    train_x = train_fe[numeric + categorical]
    validation_x = validation_fe[numeric + categorical]
    test_x = test_fe[numeric + categorical]

    prop_model = train_approval_propensity(
        train_x,
        (train["prior_decision"] == 1).astype(int),
        numeric,
        categorical,
    )
    prop_train = np.clip(prop_model.predict_proba(train_x)[:, 1], 0.001, 0.999)

    labeled = train[train["default_flag"].notna()].copy()
    models, _, cal_idx = fit_calibrated_models(train_x, labeled, numeric, categorical, prop_train)
    cal_x = train_x.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    cal_point, cal_matrix = ensemble_predictions(models, cal_x)

    validation_point, _ = ensemble_predictions(models, validation_x)
    val_labeled = validation["default_flag"].notna().to_numpy()
    validation_metrics = metric_summary(
        validation.loc[val_labeled, "default_flag"].astype(int).to_numpy(),
        validation_point[val_labeled],
    )

    pd_bins = pd_interval_bin_table(cal_point, cal_y.astype(float), n_bins=10)
    pd_bins.to_csv(REPORT_DIR / "deliverable_c_pd_interval_bins.csv", index=False)

    applicants = pd.concat([validation, test], ignore_index=True, sort=False)
    base_raw = applicants.set_index("applicant_id").loc[queries["applicant_id"]].reset_index()
    base_fe = add_application_features(base_raw)
    base_x = base_fe[numeric + categorical]
    base_point, base_matrix = ensemble_predictions(models, base_x)

    cf_raw, query_diag = _make_counterfactual_rows(applicants, queries, train, dictionary)
    cf_fe = add_application_features(cf_raw)
    cf_x = cf_fe[numeric + categorical]
    cf_point_raw, cf_matrix_raw = ensemble_predictions(models, cf_x)
    cf_prop = np.clip(prop_model.predict_proba(cf_x)[:, 1], 0.001, 0.999)

    shrink = query_diag["delta_shrink_factor"].to_numpy(float)
    expected_delta_sign = np.asarray([sign_expectation(f) for f in query_diag["feature_name"]], dtype=float)
    direction = intervention_direction(applicants, queries)
    expected_effect_sign = expected_delta_sign * direction
    delta_pre_guard = shrink * (cf_point_raw - base_point)
    guardable = query_diag["feature_name"].isin(MONOTONIC_NEUTRALIZE_FEATURES).to_numpy()
    monotonic_guard = (
        guardable
        & (expected_delta_sign != 0)
        & ~np.isnan(direction)
        & (direction != 0)
        & (np.abs(delta_pre_guard) >= SIGN_EFFECT_EPS)
        & (np.sign(delta_pre_guard) != expected_effect_sign)
    )
    delta_final = np.where(monotonic_guard, 0.0, delta_pre_guard)
    cf_point = np.clip(base_point + delta_final, 0.001, 0.999)
    cf_matrix = np.clip(base_matrix + shrink[:, None] * (cf_matrix_raw - base_matrix), 0.001, 0.999)
    cf_matrix[monotonic_guard] = base_matrix[monotonic_guard]

    lower, upper = build_pd_intervals(cf_point, cf_matrix, pd_bins)
    extra_width = (
        0.045 * (1.0 - cf_prop)
        + query_diag["class_interval_widening"].to_numpy(float)
        + query_diag["support_interval_widening"].to_numpy(float)
        + np.where(monotonic_guard, 0.020, 0.0)
    )
    lower = np.clip(np.minimum(lower, cf_point - extra_width), 0.0, 1.0)
    upper = np.clip(np.maximum(upper, cf_point + extra_width), 0.0, 1.0)
    lower = np.minimum(lower, cf_point)
    upper = np.maximum(upper, cf_point)

    query_diag["base_pd"] = base_point
    query_diag["raw_counterfactual_pd"] = cf_point_raw
    query_diag["predicted_pd_cf"] = cf_point
    query_diag["delta_raw"] = cf_point_raw - base_point
    query_diag["delta_pre_guard"] = delta_pre_guard
    query_diag["delta_final"] = cf_point - base_point
    query_diag["monotonic_guard_applied"] = monotonic_guard
    query_diag["prior_approval_propensity_cf"] = cf_prop
    query_diag["pd_cf_lower_90"] = lower
    query_diag["pd_cf_upper_90"] = upper
    query_diag["interval_width"] = upper - lower
    query_diag["expected_delta_sign"] = expected_delta_sign
    query_diag["intervention_direction"] = direction
    material_effect = np.abs(query_diag["delta_final"].to_numpy(float)) >= SIGN_EFFECT_EPS
    sign_eligible = (
        (expected_delta_sign != 0)
        & ~np.isnan(direction)
        & (direction != 0)
    )
    sign_material = sign_eligible & material_effect
    raw_material = sign_eligible & (np.abs(delta_pre_guard) >= SIGN_EFFECT_EPS)
    query_diag["raw_sign_violation"] = raw_material & (np.sign(delta_pre_guard) != expected_effect_sign)
    query_diag["sign_check_pass"] = np.where(
        ~sign_material,
        np.nan,
        np.sign(query_diag["delta_final"]) == expected_effect_sign,
    )
    query_diag["sign_check_material"] = sign_material
    query_diag["sign_violation"] = sign_material & (query_diag["sign_check_pass"] == False)  # noqa: E712

    submission = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "predicted_pd_cf": cf_point,
            "pd_cf_lower_90": lower,
            "pd_cf_upper_90": upper,
        }
    )
    submission["predicted_pd_cf"] = submission["predicted_pd_cf"].clip(0.0, 1.0)
    submission["pd_cf_lower_90"] = np.minimum(
        submission["pd_cf_lower_90"].clip(0.0, 1.0),
        submission["predicted_pd_cf"],
    )
    submission["pd_cf_upper_90"] = np.maximum(
        submission["pd_cf_upper_90"].clip(0.0, 1.0),
        submission["predicted_pd_cf"],
    )

    feature_diag = _feature_diagnostics(query_diag)
    treatment_plan = _treatment_plan(dictionary, queries)
    sign_violations = _sign_violation_report(query_diag)
    cal_lower, cal_upper = build_pd_intervals(cal_point, cal_matrix, pd_bins)
    metrics = {
        "rows": int(len(submission)),
        "unique_applicants": int(queries["applicant_id"].nunique()),
        "unique_features": int(queries["feature_name"].nunique()),
        "causal_safe_feature_count": int(len(numeric) + len(categorical)),
        "causal_safe_numeric_feature_count": int(len(numeric)),
        "causal_safe_categorical_feature_count": int(len(categorical)),
        "engineered_segment_features_used": sorted(
            str(col)
            for col in (set(numeric + categorical) & set(EXPERIMENTAL_CONFOUNDER_FEATURES))
        ),
        "duplicate_applicant_feature_value_rows": int(
            queries.duplicated(["applicant_id", "feature_name", "intervention_value"]).sum()
        ),
        "validation_labeled_only_causal_safe_model": validation_metrics,
        "calibration_holdout_interval_coverage_proxy": bin_level_coverage(
            cal_point,
            cal_y.astype(int),
            cal_lower,
            cal_upper,
            n_bins=10,
        ),
        "mean_predicted_pd_cf": float(submission["predicted_pd_cf"].mean()),
        "mean_interval_width": float((submission["pd_cf_upper_90"] - submission["pd_cf_lower_90"]).mean()),
        "outside_min_max_queries": int(query_diag["outside_min_max"].sum()),
        "outside_p01_p99_queries": int(query_diag["outside_p01_p99"].sum()),
        "unseen_category_queries": int(query_diag["unseen_category"].sum()),
        "intervention_class_counts": {
            str(k): int(v)
            for k, v in query_diag["intervention_class"].value_counts().sort_index().items()
        },
        "historical_or_proxy_queries": int((query_diag["intervention_class"] == "historical_or_proxy").sum()),
        "direct_intervention_queries": int(
            query_diag["intervention_class"].isin(
                [
                    "amount_burden_intervention",
                    "observed_business_state",
                    "credit_state_intervention",
                    "credit_context_intervention",
                    "platform_state_intervention",
                    "direct_intervention",
                ]
            ).sum()
        ),
        "sign_effect_epsilon": SIGN_EFFECT_EPS,
        "raw_sign_violation_queries": int(query_diag["raw_sign_violation"].sum()),
        "monotonic_guard_applied_queries": int(query_diag["monotonic_guard_applied"].sum()),
        "sign_check_eligible_queries": int(query_diag["sign_check_material"].sum()),
        "sign_violation_queries": int(query_diag["sign_violation"].sum()),
        "excluded_prior_policy_tokens": list(PRIOR_POLICY_TOKENS),
    }

    return CounterfactualArtifacts(
        submission=submission,
        query_diagnostics=query_diag,
        feature_diagnostics=feature_diag,
        treatment_plan=treatment_plan,
        sign_violations=sign_violations,
        metrics=metrics,
    )


def write_counterfactual_outputs(artifacts: CounterfactualArtifacts) -> None:
    artifacts.submission.to_csv(SUBMISSION_DIR / "submission_C_counterfactuals.csv", index=False)
    artifacts.query_diagnostics.to_csv(REPORT_DIR / "deliverable_c_query_diagnostics.csv", index=False)
    artifacts.feature_diagnostics.to_csv(REPORT_DIR / "deliverable_c_feature_diagnostics.csv", index=False)
    artifacts.treatment_plan.to_csv(REPORT_DIR / "deliverable_c_feature_treatment_plan.csv", index=False)
    artifacts.sign_violations.to_csv(REPORT_DIR / "deliverable_c_sign_violations.csv", index=False)
    (REPORT_DIR / "deliverable_c_summary.json").write_text(json.dumps(artifacts.metrics, indent=2))
