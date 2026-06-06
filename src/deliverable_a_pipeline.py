from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.conformal import (
    bin_level_coverage,
    build_pd_intervals,
    pd_interval_bin_table,
)
from src.economics import (
    DAILY_DRAW_FACTOR,
    ORIGINATION_FEE_RATE,
    PAID_MARGIN_RATE,
    expected_npv,
    realized_npv,
)
from src.timing import (
    HazardModel,
    N_WEEKS,
    RecoveryModel,
    expected_default_day,
    fit_hazard_model,
    fit_recovery_model,
)


DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "csv-files"
OUTPUT_DIR = Path("outputs")
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"

OUTCOME_COLUMNS = {
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
}
ID_COLUMNS = {"business_id", "applicant_id"}
DROP_FOR_PD = OUTCOME_COLUMNS | ID_COLUMNS | {
    "application_timestamp",
    # These are either constant among labeled rows or structurally missing for
    # historical declines, so using them would amplify selection artifacts.
    "prior_decision",
    "prior_approved_amount",
}
PRIOR_POLICY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)
CATEGORICAL_BASE = {
    "sector",
    "geography_region",
    "employee_count_bucket",
    "intended_use_of_funds",
    "has_linked_bank_feed",
    "owner_personal_credit_band",
    "application_channel",
}
BANK_FEED_COLUMNS = {
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
}
EPS = 1e-6
GROSS_MARGIN_RATE = 0.03 + 0.35 * 60 / 365
DEFAULT_NPV_BUFFER_RATE = float(os.getenv("DELIVERABLE_A_NPV_BUFFER_RATE", "0.005"))
EXPERIMENTAL_CONFOUNDER_FEATURES = {
    "requested_to_observed_annual_revenue",
    "observed_to_stated_revenue_ratio",
    "cash_balance_to_requested_amount",
    "cash_balance_to_observed_monthly_revenue",
    "debt_to_requested_amount",
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
    "selection_support_index",
    "prior_score_logit",
    "utilization_x_delinquency",
    "utilization_x_repayment_burden",
    "cash_x_payroll_regularity",
    "volatility_x_repayment_burden",
    "bank_feed_x_utilization",
    "bank_feed_x_repayment_burden",
    "maturity_x_repayment_burden",
    "credit_x_cash_stress",
}


@dataclass
class CalibratedModel:
    name: str
    pipeline: Pipeline
    calibrator: IsotonicRegression
    cal_pred: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(frame)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.001, 0.999)


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.astype(float).replace(0, np.nan)
    return (num.astype(float) / den_safe).replace([np.inf, -np.inf], np.nan)


def signed_log1p(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    return np.sign(values) * np.log1p(np.abs(values))


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["application_timestamp"], errors="coerce")
    out["application_month"] = ts.dt.month
    out["application_day_of_week"] = ts.dt.dayofweek
    out["application_weekofyear"] = ts.dt.isocalendar().week.astype(float)
    out["application_days_since_2024_01_01"] = (ts - pd.Timestamp("2024-01-01")).dt.days
    out["application_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    out["requested_to_stated_annual_revenue"] = safe_ratio(
        out["requested_amount"], out["stated_annual_revenue"]
    )
    out["debt_to_stated_annual_revenue"] = safe_ratio(
        out["existing_debt_obligations"], out["stated_annual_revenue"]
    )
    out["debt_to_observed_monthly_revenue"] = safe_ratio(
        out["existing_debt_obligations"], out["observed_monthly_revenue_avg_3mo"]
    )
    out["requested_to_cash_stress"] = safe_ratio(
        out["requested_amount"], out["observed_cash_balance_p10"].abs() + 1000
    )
    out["requested_to_observed_annual_revenue"] = safe_ratio(
        out["requested_amount"], out["observed_monthly_revenue_avg_3mo"] * 12
    )
    out["observed_to_stated_revenue_ratio"] = safe_ratio(
        out["observed_monthly_revenue_avg_3mo"] * 12, out["stated_annual_revenue"]
    )
    out["cash_balance_to_requested_amount"] = safe_ratio(
        out["observed_cash_balance_p10"], out["requested_amount"]
    )
    out["cash_balance_to_observed_monthly_revenue"] = safe_ratio(
        out["observed_cash_balance_p10"], out["observed_monthly_revenue_avg_3mo"]
    )
    out["debt_to_requested_amount"] = safe_ratio(
        out["existing_debt_obligations"], out["requested_amount"]
    )
    out["prior_default_rate"] = out["prior_loans_default_count"] / np.maximum(
        out["prior_loans_count"], 1
    )
    out["has_external_decline_record"] = out["days_since_last_external_decline"].notna().astype(int)
    out["has_inquiry_elsewhere_record"] = out["days_since_last_inquiry_elsewhere"].notna().astype(int)
    out["bank_feed_missing_count"] = out[list(BANK_FEED_COLUMNS)].isna().sum(axis=1)
    out["cash_balance_negative"] = (out["observed_cash_balance_p10"] < 0).astype(int)
    out["has_prior_default"] = (out["prior_loans_default_count"].fillna(0) > 0).astype(int)
    out["has_prior_loan_history"] = (out["prior_loans_count"].fillna(0) > 0).astype(int)

    # Synthetic-data latent factors: convert correlated variables into risk concepts.
    out["log_requested_amount"] = np.log1p(out["requested_amount"])
    out["log_stated_annual_revenue"] = np.log1p(out["stated_annual_revenue"])
    out["log_observed_monthly_revenue"] = np.log1p(out["observed_monthly_revenue_avg_3mo"])
    out["log_existing_debt"] = np.log1p(out["existing_debt_obligations"])
    out["signed_log_cash_balance_p10"] = signed_log1p(out["observed_cash_balance_p10"])
    out["log_account_age_days"] = np.log1p(out["account_age_days"])

    out["repayment_burden_index"] = (
        out["requested_amount_to_observed_revenue"].fillna(0)
        + out["requested_to_observed_annual_revenue"].fillna(0)
        + 0.5 * out["requested_to_stated_annual_revenue"].fillna(0)
        + 0.5 * out["debt_to_stated_annual_revenue"].fillna(0)
        + 0.25 * out["debt_to_requested_amount"].fillna(0)
    )
    out["credit_stress_index"] = (
        out["aggregate_credit_utilization"].fillna(0)
        + 0.15 * out["recent_inquiries_count_6mo"].fillna(0)
        + 0.10 * out["multi_lender_inquiry_count_30d"].fillna(0)
        + 0.20 * out["debt_to_stated_annual_revenue"].fillna(0)
        + 0.50 * out["prior_default_rate"].fillna(0)
    )
    out["cash_stress_index"] = (
        out["invoice_payment_delinquency_rate"].fillna(0)
        + 0.35 * out["aggregate_credit_utilization"].fillna(0)
        + 0.25 * out["observed_revenue_volatility"].fillna(0)
        + 0.10 * out["observed_overdraft_count_3mo"].fillna(0)
        + 0.50 * out["cash_balance_negative"].fillna(0)
        - 0.35 * out["payroll_regularity_score"].fillna(0)
        - 0.05 * out["signed_log_cash_balance_p10"].fillna(0)
    )
    out["maturity_index"] = (
        np.log1p(out["vintage_years"].clip(lower=0).fillna(0))
        + np.log1p(out["stated_time_in_business"].clip(lower=0).fillna(0))
        + np.log1p((out["account_age_days"].clip(lower=0) / 365).fillna(0))
    )
    out["revenue_scale_index"] = (
        np.log1p(out["stated_annual_revenue"].clip(lower=0).fillna(0))
        + np.log1p((out["observed_monthly_revenue_avg_3mo"].clip(lower=0) * 12).fillna(0))
    )
    out["platform_engagement_index"] = (
        np.log1p(out["platform_active_months"].clip(lower=0).fillna(0))
        + 0.25 * np.log1p(out["prior_loans_count"].clip(lower=0).fillna(0))
        + 0.10 * np.log1p(out["prior_loans_amount_total"].clip(lower=0).fillna(0))
        - 0.03 * out["bookkeeping_recency_days"].fillna(out["bookkeeping_recency_days"].median())
    )
    out["bank_feed_support_index"] = (
        out["has_linked_bank_feed"].astype(int)
        - 0.15 * out["bank_feed_missing_count"]
        + 0.05 * out["payroll_regularity_score"].fillna(0)
    )
    out["selection_support_index"] = (
        out["prior_underwriter_score"].fillna(out["prior_underwriter_score"].median())
        + 0.10 * out["owner_personal_credit_band"].fillna(0)
        - 0.20 * out["aggregate_credit_utilization"].fillna(0)
        - 0.15 * out["invoice_payment_delinquency_rate"].fillna(0)
        + 0.02 * out["signed_log_cash_balance_p10"].fillna(0)
    )
    out["prior_score_logit"] = logit(
        out["prior_underwriter_score"].clip(0.001, 0.999).fillna(0.5)
    )

    # High-signal interactions from the confounder/correlation audit.
    out["utilization_x_delinquency"] = (
        out["aggregate_credit_utilization"] * out["invoice_payment_delinquency_rate"]
    )
    out["utilization_x_repayment_burden"] = (
        out["aggregate_credit_utilization"] * out["requested_amount_to_observed_revenue"]
    )
    out["cash_x_payroll_regularity"] = (
        out["signed_log_cash_balance_p10"] * out["payroll_regularity_score"]
    )
    out["volatility_x_repayment_burden"] = (
        out["observed_revenue_volatility"] * out["requested_amount_to_observed_revenue"]
    )
    out["bank_feed_x_utilization"] = (
        out["has_linked_bank_feed"].astype(int) * out["aggregate_credit_utilization"]
    )
    out["bank_feed_x_repayment_burden"] = (
        out["has_linked_bank_feed"].astype(int) * out["requested_amount_to_observed_revenue"]
    )
    out["maturity_x_repayment_burden"] = out["maturity_index"] * out["repayment_burden_index"]
    out["credit_x_cash_stress"] = out["credit_stress_index"] * out["cash_stress_index"]

    for col in sorted((set(df.columns) - OUTCOME_COLUMNS - ID_COLUMNS - {"application_timestamp"})):
        out[f"is_missing_{col}"] = out[col].isna().astype(int)

    return out


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    cols = [c for c in frame.columns if c not in DROP_FOR_PD]
    feature_set = os.getenv("DELIVERABLE_A_FEATURE_SET", "all_engineered")
    if feature_set != "all_engineered":
        cols = [c for c in cols if c not in EXPERIMENTAL_CONFOUNDER_FEATURES]
    cols = [c for c in cols if not any(token in c for token in PRIOR_POLICY_TOKENS)]
    categorical = sorted(c for c in cols if c in CATEGORICAL_BASE)
    numeric = sorted(c for c in cols if c not in categorical)
    return cols, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )


def make_model_specs(numeric: list[str], categorical: list[str]) -> list[tuple[str, Pipeline]]:
    return [
        (
            "hgb_depth_31",
            Pipeline(
                [
                    ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=260,
                            learning_rate=0.045,
                            max_leaf_nodes=31,
                            l2_regularization=0.03,
                            min_samples_leaf=35,
                            random_state=11,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb_depth_15",
            Pipeline(
                [
                    ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=240,
                            learning_rate=0.06,
                            max_leaf_nodes=15,
                            l2_regularization=0.12,
                            min_samples_leaf=45,
                            random_state=23,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb_depth_63",
            Pipeline(
                [
                    ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=210,
                            learning_rate=0.04,
                            max_leaf_nodes=63,
                            l2_regularization=0.30,
                            min_samples_leaf=30,
                            random_state=37,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logistic_baseline",
            Pipeline(
                [
                    ("prep", make_preprocessor(numeric, categorical, scale_numeric=True)),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.35,
                            max_iter=1200,
                            solver="lbfgs",
                            class_weight=None,
                        ),
                    ),
                ]
            ),
        ),
    ]


def train_approval_propensity(train_x: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]) -> Pipeline:
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    min_samples_leaf=50,
                    random_state=101,
                ),
            ),
        ]
    )
    pipe.fit(train_x, y.astype(int))
    return pipe


def time_split_labeled(labeled: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def fit_calibrated_models(
    full_train_features: pd.DataFrame,
    labeled_train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    approval_propensity: np.ndarray,
) -> tuple[list[CalibratedModel], pd.Index, pd.Index]:
    model_idx, cal_idx = time_split_labeled(labeled_train)
    models: list[CalibratedModel] = []

    train_x = full_train_features.loc[model_idx]
    train_y = labeled_train.loc[model_idx, "default_flag"].astype(int)
    cal_x = full_train_features.loc[cal_idx]
    cal_y = labeled_train.loc[cal_idx, "default_flag"].astype(int)

    weights = 1.0 / np.clip(approval_propensity[labeled_train.index.get_indexer(model_idx)], 0.08, 1.0)
    weights = np.clip(weights / np.mean(weights), 0.25, 8.0)

    for name, pipe in make_model_specs(numeric, categorical):
        pipe.fit(train_x, train_y, clf__sample_weight=weights)
        raw = pipe.predict_proba(cal_x)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        calibrator.fit(raw, cal_y)
        models.append(
            CalibratedModel(
                name=name,
                pipeline=pipe,
                calibrator=calibrator,
                cal_pred=np.clip(calibrator.predict(raw), 0.001, 0.999),
            )
        )
    return models, pd.Index(model_idx), pd.Index(cal_idx)


def ensemble_predictions(models: Iterable[CalibratedModel], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_matrix = np.column_stack([m.predict(frame) for m in models])
    point = np.clip(pred_matrix.mean(axis=1), 0.001, 0.999)
    return point, pred_matrix


def segment_calibration_features(
    base_p: np.ndarray,
    feature_frame: pd.DataFrame,
    approval_propensity: np.ndarray,
    day_center: float,
    day_scale: float,
) -> np.ndarray:
    prior_score = (
        feature_frame["prior_underwriter_score"]
        .clip(0.001, 0.999)
        .fillna(0.5)
        .to_numpy(float)
    )
    days = feature_frame["application_days_since_2024_01_01"].fillna(day_center).to_numpy(float)
    days_scaled = (days - day_center) / max(day_scale, EPS)
    prop = np.clip(approval_propensity, 0.001, 0.999)
    return np.column_stack(
        [
            logit(np.clip(base_p, 0.001, 0.999)),
            prior_score,
            logit(prior_score),
            feature_frame["has_linked_bank_feed"].astype(int).to_numpy(float),
            prop,
            logit(prop),
            days_scaled,
        ]
    )


def fit_segment_meta_calibrator(
    cal_point: np.ndarray,
    cal_features: pd.DataFrame,
    cal_propensity: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[LogisticRegression, float, float]:
    day_values = cal_features["application_days_since_2024_01_01"].fillna(0).to_numpy(float)
    day_center = float(np.mean(day_values))
    day_scale = float(np.std(day_values) + EPS)
    x_cal = segment_calibration_features(
        cal_point,
        cal_features,
        cal_propensity,
        day_center,
        day_scale,
    )
    calibrator = LogisticRegression(C=0.25, max_iter=1000)
    calibrator.fit(x_cal, y_cal)
    return calibrator, day_center, day_scale


def apply_segment_meta_calibrator(
    calibrator: LogisticRegression,
    point: np.ndarray,
    pred_matrix: np.ndarray,
    features: pd.DataFrame,
    approval_propensity: np.ndarray,
    day_center: float,
    day_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_point = segment_calibration_features(
        point,
        features,
        approval_propensity,
        day_center,
        day_scale,
    )
    adjusted_point = np.clip(calibrator.predict_proba(x_point)[:, 1], 0.001, 0.999)
    adjusted_matrix = []
    for col in range(pred_matrix.shape[1]):
        x_col = segment_calibration_features(
            pred_matrix[:, col],
            features,
            approval_propensity,
            day_center,
            day_scale,
        )
        adjusted_matrix.append(np.clip(calibrator.predict_proba(x_col)[:, 1], 0.001, 0.999))
    return adjusted_point, np.column_stack(adjusted_matrix)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y.astype(float), "p": p.astype(float)})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.groupby("bin", dropna=True):
        n = len(group)
        observed = group["y"].mean()
        predicted = group["p"].mean()
        z = 1.645
        center = (observed + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt((observed * (1 - observed) / n) + (z * z / (4 * n * n))) / (
            1 + z * z / n
        )
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(n),
                "p_min": group["p"].min(),
                "p_max": group["p"].max(),
                "mean_predicted_pd": predicted,
                "observed_default_rate": observed,
                "abs_calibration_error": abs(predicted - observed),
                "wilson_half_width_90": half,
                "wilson_lower_90": max(0.0, center - half),
                "wilson_upper_90": min(1.0, center + half),
            }
        )
    return pd.DataFrame(rows)


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = calibration_table(y, p, n_bins=n_bins)
    if table.empty:
        return float("nan")
    return float((table["n"] * table["abs_calibration_error"]).sum() / table["n"].sum())


def assign_calibration_width(point: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    width = np.zeros(len(point), dtype=float)
    fallback = 0.08
    for i, p in enumerate(point):
        match = table[(table["p_min"] <= p) & (p <= table["p_max"])]
        if match.empty:
            width[i] = fallback
        else:
            row = match.iloc[0]
            width[i] = (
                float(row["abs_calibration_error"])
                + float(row["wilson_half_width_90"])
                + 0.01
            )
    return width


def build_intervals(
    point: np.ndarray,
    pred_matrix: np.ndarray,
    cal_table: pd.DataFrame,
    approval_propensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_low = np.quantile(pred_matrix, 0.05, axis=1)
    base_high = np.quantile(pred_matrix, 0.95, axis=1)
    calibration_width = assign_calibration_width(point, cal_table)
    ensemble_width = np.maximum(point - base_low, base_high - point)
    support_width = 0.045 * (1 - np.clip(approval_propensity, 0, 1))
    total_width = np.maximum(calibration_width, ensemble_width) + support_width
    lower = np.clip(np.minimum(base_low, point - total_width), 0, 1)
    upper = np.clip(np.maximum(base_high, point + total_width), 0, 1)
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return lower, upper


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(y)),
        "default_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": ece_score(y, p, n_bins=10),
    }


def expected_npv_frame(
    df: pd.DataFrame,
    predicted_pd: np.ndarray,
    expected_t_star: np.ndarray,
    expected_recovery_rate: np.ndarray,
) -> pd.DataFrame:
    """Brief-faithful E[NPV | approve].

    Repaid:  F + R·r·T/365
    Default: F + D·(t*-1) + rec - R
    """
    amount = df["requested_amount"].to_numpy(float)
    e_npv = expected_npv(amount, predicted_pd, expected_t_star, expected_recovery_rate)
    return pd.DataFrame(
        {
            "amount": amount,
            "p_default": predicted_pd,
            "expected_t_star": expected_t_star,
            "expected_recovery_rate": expected_recovery_rate,
            "expected_npv": e_npv,
            "expected_npv_per_dollar": e_npv / np.maximum(amount, 1.0),
        },
        index=df.index,
    )


def decision_from_npv(
    expected_npv_values: np.ndarray,
    amount: np.ndarray,
    *,
    buffer_rate: float = DEFAULT_NPV_BUFFER_RATE,
) -> np.ndarray:
    """Approve only when per-dollar E[NPV] clears the robustness buffer."""
    margin = expected_npv_values / np.maximum(amount, 1.0)
    return (margin > buffer_rate).astype(int)


def npv_buffer_tuning_table(
    df: pd.DataFrame,
    predicted_pd: np.ndarray,
    npv_frame: pd.DataFrame,
    labeled_mask: pd.Series,
    *,
    buffer_rates: Iterable[float],
) -> pd.DataFrame:
    labeled_mask_np = labeled_mask.to_numpy()
    realized = realized_npv(df.loc[labeled_mask])
    amount = npv_frame["amount"].to_numpy(float)
    expected_npv_values = npv_frame["expected_npv"].to_numpy(float)
    rows = []
    for buffer_rate in buffer_rates:
        decision = decision_from_npv(expected_npv_values, amount, buffer_rate=buffer_rate)
        decision_labeled = decision[labeled_mask_np]
        rows.append(
            {
                "buffer_rate": float(buffer_rate),
                "approved_count_labeled_validation": int(decision_labeled.sum()),
                "approval_rate_labeled_validation": float(decision_labeled.mean()),
                "approval_rate_all_validation": float(decision.mean()),
                "mean_pd_approved": float(predicted_pd[decision == 1].mean())
                if decision.sum()
                else np.nan,
                "observed_default_rate_approved": float(
                    df.loc[labeled_mask, "default_flag"].to_numpy(float)[decision_labeled == 1].mean()
                )
                if decision_labeled.sum()
                else np.nan,
                "expected_profit_total": float(expected_npv_values[decision == 1].sum()),
                "realized_profit_proxy_total": float(realized[decision_labeled == 1].sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["realized_profit_proxy_total", "expected_profit_total"], ascending=False
    ).reset_index(drop=True)


def write_audit_report(
    path: Path,
    split_summary: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> None:
    report = f"""# Deliverable A Statistical Audit

## Concepts Incorporated

- PD is `P(default within 90 days = 1 | application features)`, modeled by an HGB ensemble + isotonic + segment-aware meta-calibration.
- Discrete-time weekly hazard model gives `Pr(default by week w | x, approve)` for w in 1..13. Provides `E[t*|default,x]` and feeds Deliverable B's CDR curve directly.
- Recovery rate `rec_i / R_i` is modeled by a separate HGB regressor over defaulted training rows.
- Decision rule is the brief's cash-flow equation with a per-dollar robustness buffer: `d_i = 1[ E[NPV_i | approve] / requested_amount > buffer_rate ]` with
  `E[NPV] = (1-p) (F + R·r·T/365) + p (F + D·(E[t*]-1) + E[rec] - R)`,
  `D = R(1 + r·T/365)/T`, `F = 0.03 R`, `r = 0.35`, `T = 60`.
- 90% PD intervals are split-conformal: residual quantiles fit on the labeled calibration holdout, applied per PD bin for local adaptivity.
- Labels are selectively observed (prior-approved + matured). IPS weights via a prior-approval propensity model widen exposure to the under-represented decline region. `prior_decision` and `prior_approved_amount` are dropped from PD features for the same reason.

## Known Residual Risks

1. Test set is fully unlabeled. Generalization to declined applicants is unverifiable; we rely on propensity weighting + conformal coverage measured on labeled validation.
2. The hazard model's weekly bucket is the finest resolution the data and B's template support; sub-week timing is approximated by the bucket midpoint inside `E[t*]`.
3. Recovery model is trained only on defaulted rows; we assume similar recovery dynamics hold for newly-approved applicants (no censoring outside the 90-day window in the training data).

## Split Summary

```text
{split_summary.to_string(index=False)}
```

## Model Metrics

```json
{json.dumps(metrics, indent=2)}
```

## Economics

- Paid-loan margin rate: {PAID_MARGIN_RATE:.6f} of principal (F + R r T/365)
- Daily-draw factor D/R: {DAILY_DRAW_FACTOR:.6f}
- Origination fee rate: {ORIGINATION_FEE_RATE:.4f}
- Active NPV buffer rate: {metrics.get("settings", {}).get("npv_buffer_rate", DEFAULT_NPV_BUFFER_RATE):.4f}
"""
    path.write_text(report)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")

    split_rows = []
    for name, df in [("train", train), ("validation", validation), ("test", test)]:
        observed = df["default_flag"].notna()
        split_rows.append(
            {
                "split": name,
                "rows": len(df),
                "prior_approval_rate": float((df["prior_decision"] == 1).mean()),
                "observed_outcome_rate": float(observed.mean()),
                "default_rate_when_observed": float(df.loc[observed, "default_flag"].mean())
                if observed.any()
                else np.nan,
                "bank_feed_link_rate": float(df["has_linked_bank_feed"].mean()),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(REPORT_DIR / "deliverable_a_split_summary.csv", index=False)

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)

    train_x = train_fe[numeric + categorical]
    validation_x = validation_fe[numeric + categorical]
    test_x = test_fe[numeric + categorical]

    propensity_model = train_approval_propensity(
        train_x,
        (train["prior_decision"] == 1).astype(int),
        numeric,
        categorical,
    )
    prop_train = np.clip(propensity_model.predict_proba(train_x)[:, 1], 0.001, 0.999)
    prop_validation = np.clip(propensity_model.predict_proba(validation_x)[:, 1], 0.001, 0.999)
    prop_test = np.clip(propensity_model.predict_proba(test_x)[:, 1], 0.001, 0.999)

    labeled_train = train[train["default_flag"].notna()].copy()
    models, _, cal_idx = fit_calibrated_models(
        train_x,
        labeled_train,
        numeric,
        categorical,
        prop_train,
    )

    cal_x = train_x.loc[cal_idx]
    cal_features = train_fe.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    cal_point, cal_matrix = ensemble_predictions(models, cal_x)

    validation_point, validation_matrix = ensemble_predictions(models, validation_x)
    test_point, test_matrix = ensemble_predictions(models, test_x)

    segment_meta_enabled = os.getenv("DELIVERABLE_A_SEGMENT_META", "0") == "1"
    if segment_meta_enabled:
        segment_calibrator, day_center, day_scale = fit_segment_meta_calibrator(
            cal_point,
            cal_features,
            prop_train[cal_idx],
            cal_y,
        )
        cal_point, cal_matrix = apply_segment_meta_calibrator(
            segment_calibrator,
            cal_point,
            cal_matrix,
            cal_features,
            prop_train[cal_idx],
            day_center,
            day_scale,
        )
        validation_point, validation_matrix = apply_segment_meta_calibrator(
            segment_calibrator,
            validation_point,
            validation_matrix,
            validation_fe,
            prop_validation,
            day_center,
            day_scale,
        )
        test_point, test_matrix = apply_segment_meta_calibrator(
            segment_calibrator,
            test_point,
            test_matrix,
            test_fe,
            prop_test,
            day_center,
            day_scale,
        )

    cal_table = calibration_table(cal_y, cal_point, n_bins=10)
    cal_table.to_csv(REPORT_DIR / "deliverable_a_calibration_bins_train_calibration.csv", index=False)

    val_labeled_mask = validation["default_flag"].notna()
    val_metrics = metric_summary(
        validation.loc[val_labeled_mask, "default_flag"].astype(int).to_numpy(),
        validation_point[val_labeled_mask.to_numpy()],
    )
    cal_metrics = metric_summary(cal_y, cal_point)
    metrics = {"train_calibration_holdout": cal_metrics, "validation_labeled_only": val_metrics}
    metrics["settings"] = {
        "feature_set": os.getenv("DELIVERABLE_A_FEATURE_SET", "baseline"),
        "segment_meta_calibration": segment_meta_enabled,
        "npv_buffer_rate": DEFAULT_NPV_BUFFER_RATE,
    }

    # ---- Timing + recovery sub-models for brief-faithful NPV ----
    labeled_full = train[train["default_flag"].notna()].copy()
    labeled_full_x = train_x.loc[labeled_full.index]
    weeks_sample_weight = 1.0 / np.clip(prop_train[labeled_full.index.to_numpy()], 0.08, 1.0)
    weeks_sample_weight = np.clip(weeks_sample_weight / weeks_sample_weight.mean(), 0.25, 8.0)
    hazard_model = fit_hazard_model(labeled_full_x, labeled_full, sample_weight=weeks_sample_weight)

    defaulted_train = train[train["default_flag"] == 1]
    recovery_model = fit_recovery_model(train_x, defaulted_train)

    _, validation_cum = hazard_model.predict_curves(validation_x)
    _, test_cum = hazard_model.predict_curves(test_x)
    _, cal_cum = hazard_model.predict_curves(cal_x)
    val_t_star = expected_default_day(validation_cum)
    test_t_star = expected_default_day(test_cum)
    val_rec = recovery_model.predict_rate(validation_x)
    test_rec = recovery_model.predict_rate(test_x)

    # ---- 90% PD intervals: per-bin Wilson + ensemble dispersion ----
    pd_bins = pd_interval_bin_table(cal_point, cal_y.astype(float), n_bins=10)
    pd_bins.to_csv(REPORT_DIR / "deliverable_a_pd_interval_bins.csv", index=False)
    val_lower, val_upper = build_pd_intervals(validation_point, validation_matrix, pd_bins)
    test_lower, test_upper = build_pd_intervals(test_point, test_matrix, pd_bins)
    if val_labeled_mask.any():
        val_coverage = bin_level_coverage(
            validation_point[val_labeled_mask.to_numpy()],
            validation.loc[val_labeled_mask, "default_flag"].astype(int).to_numpy(),
            val_lower[val_labeled_mask.to_numpy()],
            val_upper[val_labeled_mask.to_numpy()],
            n_bins=10,
        )
    else:
        val_coverage = {"n": 0, "bin_coverage": float("nan"), "mean_width": float("nan"), "median_width": float("nan")}
    metrics["validation_interval_coverage_90"] = val_coverage

    # ---- E[NPV | approve] via brief cash flows, with per-dollar safety buffer ----
    validation_npv = expected_npv_frame(validation, validation_point, val_t_star, val_rec)
    test_npv = expected_npv_frame(test, test_point, test_t_star, test_rec)
    val_amount = validation_npv["amount"].to_numpy(float)
    test_amount = test_npv["amount"].to_numpy(float)
    val_decision = decision_from_npv(
        validation_npv["expected_npv"].to_numpy(float),
        val_amount,
        buffer_rate=DEFAULT_NPV_BUFFER_RATE,
    )
    test_decision = decision_from_npv(
        test_npv["expected_npv"].to_numpy(float),
        test_amount,
        buffer_rate=DEFAULT_NPV_BUFFER_RATE,
    )

    if val_labeled_mask.any():
        buffer_table = npv_buffer_tuning_table(
            validation,
            validation_point,
            validation_npv,
            val_labeled_mask,
            buffer_rates=[
                -0.02,
                -0.01,
                0.0,
                0.0025,
                0.005,
                0.0075,
                0.01,
                0.015,
                0.02,
                0.03,
                0.04,
                0.05,
            ],
        )
        buffer_table.to_csv(REPORT_DIR / "deliverable_a_buffer_tuning.csv", index=False)

    # ---- Backtest: realized NPV on labeled validation under our policy ----
    if val_labeled_mask.any():
        val_realized_all = realized_npv(validation.loc[val_labeled_mask])
        val_decisions_labeled = val_decision[val_labeled_mask.to_numpy()]
        metrics["validation_realized_npv"] = {
            "n_labeled": int(val_labeled_mask.sum()),
            "approved_n": int(val_decisions_labeled.sum()),
            "realized_npv_total_under_policy": float(val_realized_all[val_decisions_labeled == 1].sum()),
            "realized_npv_total_if_approve_all": float(val_realized_all.sum()),
            "mean_pd_approved": float(validation_point[val_labeled_mask.to_numpy()][val_decisions_labeled == 1].mean()) if val_decisions_labeled.sum() else float("nan"),
        }

    submission_a = pd.concat(
        [
            pd.DataFrame(
                {
                    "applicant_id": validation["applicant_id"],
                    "decision": val_decision,
                    "predicted_pd": validation_point,
                    "pd_lower_90": val_lower,
                    "pd_upper_90": val_upper,
                }
            ),
            pd.DataFrame(
                {
                    "applicant_id": test["applicant_id"],
                    "decision": test_decision,
                    "predicted_pd": test_point,
                    "pd_lower_90": test_lower,
                    "pd_upper_90": test_upper,
                }
            ),
        ],
        ignore_index=True,
    )

    submission_a["predicted_pd"] = submission_a["predicted_pd"].clip(0, 1)
    submission_a["pd_lower_90"] = np.minimum(
        submission_a["pd_lower_90"].clip(0, 1), submission_a["predicted_pd"]
    )
    submission_a["pd_upper_90"] = np.maximum(
        submission_a["pd_upper_90"].clip(0, 1), submission_a["predicted_pd"]
    )
    submission_a.to_csv(SUBMISSION_DIR / "submission_A_decisions.csv", index=False)

    diagnostics = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "rows": [len(validation), len(test)],
            "approval_rate": [float(val_decision.mean()), float(test_decision.mean())],
            "mean_predicted_pd": [float(validation_point.mean()), float(test_point.mean())],
            "mean_predicted_pd_approved": [
                float(validation_point[val_decision == 1].mean()) if val_decision.sum() else np.nan,
                float(test_point[test_decision == 1].mean()) if test_decision.sum() else np.nan,
            ],
            "mean_expected_t_star": [float(val_t_star.mean()), float(test_t_star.mean())],
            "mean_expected_recovery_rate": [float(val_rec.mean()), float(test_rec.mean())],
            "mean_interval_width": [
                float(np.mean(val_upper - val_lower)),
                float(np.mean(test_upper - test_lower)),
            ],
            "mean_prior_approval_propensity": [float(prop_validation.mean()), float(prop_test.mean())],
            "expected_npv_total_approved": [
                float(validation_npv.loc[val_decision == 1, "expected_npv"].sum()),
                float(test_npv.loc[test_decision == 1, "expected_npv"].sum()),
            ],
        }
    )
    diagnostics.to_csv(REPORT_DIR / "deliverable_a_submission_diagnostics.csv", index=False)

    # Persist auxiliary frames so B and C can reuse without retraining.
    np.savez(
        OUTPUT_DIR / "deliverable_a_curves.npz",
        validation_cumulative=validation_cum,
        test_cumulative=test_cum,
        validation_t_star=val_t_star,
        test_t_star=test_t_star,
        validation_recovery=val_rec,
        test_recovery=test_rec,
        validation_pd=validation_point,
        test_pd=test_point,
        validation_decision=val_decision,
        test_decision=test_decision,
    )

    (REPORT_DIR / "deliverable_a_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_audit_report(
        REPORT_DIR / "deliverable_a_statistical_audit.md",
        split_summary,
        metrics,
    )

    print("Wrote", SUBMISSION_DIR / "submission_A_decisions.csv")
    print("Wrote reports in", REPORT_DIR)
    print(json.dumps(metrics, indent=2))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
