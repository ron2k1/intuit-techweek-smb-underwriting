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
    CSV_DIR,
    REPORT_DIR,
    SUBMISSION_DIR,
    add_application_features,
    ensemble_predictions,
    feature_columns,
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
    "aggregate_credit_utilization",
    "recent_inquiries_count_6mo",
    "existing_debt_obligations",
    "observed_overdraft_count_3mo",
    "invoice_payment_delinquency_rate",
    "multi_lender_inquiry_count_30d",
    "prior_loans_default_count",
    "requested_amount_to_observed_revenue",
}
RISK_DECREASING_FEATURES = {
    "observed_cash_balance_p10",
    "payroll_regularity_score",
    "stated_annual_revenue",
    "observed_monthly_revenue_avg_3mo",
    "stated_time_in_business",
    "vintage_years",
    "account_age_days",
    "platform_active_months",
}


@dataclass(frozen=True)
class CounterfactualArtifacts:
    submission: pd.DataFrame
    query_diagnostics: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    metrics: dict[str, Any]


def causal_safe_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return model columns with prior-underwriter outputs/proxies removed."""
    _, numeric, categorical = feature_columns(frame)
    cols = numeric + categorical
    safe_cols = [
        col
        for col in cols
        if not any(token in col for token in PRIOR_POLICY_TOKENS)
    ]
    safe_categorical = [col for col in categorical if col in safe_cols]
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
) -> tuple[str, float, float]:
    """Return class, delta shrink factor, and interval widening.

    Direct interventions keep the model-implied delta. Historical/proxy fields
    are answerable per the rules, but the effect is shrunk because changing a
    historical summary is not the same as a clean business intervention.
    """
    if feature in POLICY_DERIVED_FEATURES or any(token in feature for token in PRIOR_POLICY_TOKENS):
        return "policy_artifact", 0.0, 0.08
    if intervenable.get(feature, False):
        return "direct_intervention", 1.0, 0.0
    feature_group = group.get(feature, "unknown")
    if feature_group in {
        "business_identity",
        "platform_engagement",
        "application_context",
        "bank_feed",
        "self_reported",
    }:
        return "historical_or_proxy", 0.35, 0.035
    return "ambiguous", 0.50, 0.025


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

        intervention_class, shrink, class_widening = classify_intervention(
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
            mean_delta_raw=("delta_raw", "mean"),
            mean_delta_final=("delta_final", "mean"),
            mean_base_pd=("base_pd", "mean"),
            mean_predicted_pd_cf=("predicted_pd_cf", "mean"),
            mean_interval_width=("interval_width", "mean"),
            sign_check_rate=("sign_check_pass", lambda s: float(np.nanmean(s)) if np.any(pd.notna(s)) else np.nan),
        )
        .reset_index()
    )
    return agg.sort_values(["count", "feature_name"], ascending=[False, True])


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
    cf_point = np.clip(base_point + shrink * (cf_point_raw - base_point), 0.001, 0.999)
    cf_matrix = np.clip(base_matrix + shrink[:, None] * (cf_matrix_raw - base_matrix), 0.001, 0.999)

    lower, upper = build_pd_intervals(cf_point, cf_matrix, pd_bins)
    extra_width = (
        0.045 * (1.0 - cf_prop)
        + query_diag["class_interval_widening"].to_numpy(float)
        + query_diag["support_interval_widening"].to_numpy(float)
    )
    lower = np.clip(np.minimum(lower, cf_point - extra_width), 0.0, 1.0)
    upper = np.clip(np.maximum(upper, cf_point + extra_width), 0.0, 1.0)
    lower = np.minimum(lower, cf_point)
    upper = np.maximum(upper, cf_point)

    query_diag["base_pd"] = base_point
    query_diag["raw_counterfactual_pd"] = cf_point_raw
    query_diag["predicted_pd_cf"] = cf_point
    query_diag["delta_raw"] = cf_point_raw - base_point
    query_diag["delta_final"] = cf_point - base_point
    query_diag["prior_approval_propensity_cf"] = cf_prop
    query_diag["pd_cf_lower_90"] = lower
    query_diag["pd_cf_upper_90"] = upper
    query_diag["interval_width"] = upper - lower
    query_diag["expected_delta_sign"] = [sign_expectation(f) for f in query_diag["feature_name"]]
    query_diag["intervention_direction"] = intervention_direction(applicants, queries)
    expected_effect_sign = (
        query_diag["expected_delta_sign"].to_numpy(float)
        * query_diag["intervention_direction"].to_numpy(float)
    )
    query_diag["sign_check_pass"] = np.where(
        (query_diag["expected_delta_sign"].to_numpy(float) == 0)
        | np.isnan(query_diag["intervention_direction"].to_numpy(float))
        | (query_diag["intervention_direction"].to_numpy(float) == 0),
        np.nan,
        np.sign(query_diag["delta_final"]) == expected_effect_sign,
    )

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
    cal_lower, cal_upper = build_pd_intervals(cal_point, cal_matrix, pd_bins)
    metrics = {
        "rows": int(len(submission)),
        "unique_applicants": int(queries["applicant_id"].nunique()),
        "unique_features": int(queries["feature_name"].nunique()),
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
        "historical_or_proxy_queries": int((query_diag["intervention_class"] == "historical_or_proxy").sum()),
        "direct_intervention_queries": int((query_diag["intervention_class"] == "direct_intervention").sum()),
        "excluded_prior_policy_tokens": list(PRIOR_POLICY_TOKENS),
    }

    return CounterfactualArtifacts(
        submission=submission,
        query_diagnostics=query_diag,
        feature_diagnostics=feature_diag,
        metrics=metrics,
    )


def write_counterfactual_outputs(artifacts: CounterfactualArtifacts) -> None:
    artifacts.submission.to_csv(SUBMISSION_DIR / "submission_C_counterfactuals.csv", index=False)
    artifacts.query_diagnostics.to_csv(REPORT_DIR / "deliverable_c_query_diagnostics.csv", index=False)
    artifacts.feature_diagnostics.to_csv(REPORT_DIR / "deliverable_c_feature_diagnostics.csv", index=False)
    (REPORT_DIR / "deliverable_c_summary.json").write_text(json.dumps(artifacts.metrics, indent=2))
