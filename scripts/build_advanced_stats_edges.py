#!/usr/bin/env python3
"""Advanced statistical edge diagnostics for Deliverables A/B/C/D.

Outputs focus on the writeup-critical risks that are easy to miss:
leakage, train/validation/test drift, recency weighting, hazard timing,
expected-profit underwriting, and reject-inference sensitivity.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from scipy.stats import ks_2samp
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
RANDOM_SEED = 2026

OUTCOME_COLUMNS = {
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
}
ID_COLUMNS = {"business_id", "applicant_id"}
FORBIDDEN_POLICY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(y)),
        "default_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_pd": float(np.mean(p)),
    }


def psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)
    expected = expected / max(expected.sum(), 1.0)
    actual = actual / max(actual.sum(), 1.0)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_drift(train_s: pd.Series, other_s: pd.Series) -> dict[str, float]:
    train_v = train_s.dropna().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    other_v = other_s.dropna().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_v) < 20 or len(other_v) < 20:
        return {"psi": np.nan, "ks": np.nan, "std_mean_diff": np.nan, "train_missing": float(train_s.isna().mean()), "other_missing": float(other_s.isna().mean())}
    quantiles = np.unique(np.nanquantile(train_v, np.linspace(0, 1, 11)))
    if len(quantiles) < 3:
        bins = np.array([train_v.min() - 1e-9, train_v.max() + 1e-9])
    else:
        bins = quantiles.copy()
        bins[0] = -np.inf
        bins[-1] = np.inf
    train_counts, _ = np.histogram(train_v, bins=bins)
    other_counts, _ = np.histogram(other_v, bins=bins)
    pooled_sd = np.sqrt((train_v.var(ddof=1) + other_v.var(ddof=1)) / 2.0)
    return {
        "psi": psi_from_counts(train_counts, other_counts),
        "ks": float(ks_2samp(train_v, other_v).statistic),
        "std_mean_diff": float((other_v.mean() - train_v.mean()) / pooled_sd) if pooled_sd > 0 else 0.0,
        "train_missing": float(train_s.isna().mean()),
        "other_missing": float(other_s.isna().mean()),
    }


def categorical_drift(train_s: pd.Series, other_s: pd.Series) -> dict[str, float]:
    train_v = train_s.astype(str).fillna("__missing__")
    other_v = other_s.astype(str).fillna("__missing__")
    cats = sorted(set(train_v.unique()) | set(other_v.unique()))
    train_counts = train_v.value_counts().reindex(cats, fill_value=0).to_numpy()
    other_counts = other_v.value_counts().reindex(cats, fill_value=0).to_numpy()
    train_p = train_counts / max(train_counts.sum(), 1)
    other_p = other_counts / max(other_counts.sum(), 1)
    return {
        "psi": psi_from_counts(train_counts, other_counts),
        "ks": np.nan,
        "std_mean_diff": np.nan,
        "total_variation": float(0.5 * np.abs(train_p - other_p).sum()),
        "train_missing": float(train_s.isna().mean()),
        "other_missing": float(other_s.isna().mean()),
    }


def build_drift(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skip = OUTCOME_COLUMNS | ID_COLUMNS
    for col in [c for c in train.columns if c not in skip]:
        for split_name, other in [("validation", validation), ("test", test)]:
            if col not in other.columns:
                continue
            if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_bool_dtype(train[col]):
                stats = numeric_drift(train[col], other[col])
                kind = "numeric"
            else:
                stats = categorical_drift(train[col], other[col])
                kind = "categorical"
            rows.append({"feature": col, "kind": kind, "comparison": f"train_vs_{split_name}", **stats})
    out = pd.DataFrame(rows)
    return out.sort_values(["psi", "ks"], ascending=False, na_position="last")


def leakage_audit(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    forbidden_cols = [
        col
        for col in feature_cols
        if col in OUTCOME_COLUMNS
        or col in ID_COLUMNS
        or col == "application_timestamp"
        or any(token in col for token in FORBIDDEN_POLICY_TOKENS)
    ]
    labeled = train[train["default_flag"].notna()].copy()
    model_idx, cal_idx = time_split_labeled(train)
    model_business = set(train.loc[model_idx, "business_id"])
    cal_business = set(train.loc[cal_idx, "business_id"])

    rows = []
    y = labeled["default_flag"].astype(int).to_numpy()
    for col in feature_cols:
        s = add_application_features(labeled)[col] if col not in labeled.columns else labeled[col]
        if s.nunique(dropna=True) < 2:
            continue
        try:
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                x = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(s.astype(float).median()).to_numpy()
            else:
                x = pd.Categorical(s.astype(str).fillna("__missing__")).codes
            auc = roc_auc_score(y, x)
            auc = max(float(auc), float(1.0 - auc))
            if auc >= 0.90:
                rows.append({"feature": col, "single_feature_abs_auc": auc})
        except Exception:
            continue
    suspicious = pd.DataFrame(rows, columns=["feature", "single_feature_abs_auc"])
    if not suspicious.empty:
        suspicious = suspicious.sort_values("single_feature_abs_auc", ascending=False)

    audit = {
        "feature_count": int(len(feature_cols)),
        "forbidden_feature_count": int(len(forbidden_cols)),
        "forbidden_features": forbidden_cols,
        "business_overlap_model_calibration": int(len(model_business & cal_business)),
        "business_overlap_train_validation": int(len(set(train["business_id"]) & set(validation["business_id"]))),
        "business_overlap_train_test": int(len(set(train["business_id"]) & set(test["business_id"]))),
        "business_overlap_validation_test": int(len(set(validation["business_id"]) & set(test["business_id"]))),
        "suspicious_single_feature_auc_count": int(len(suspicious)),
    }
    return audit, suspicious


def prepare_feature_matrices(train_fe: pd.DataFrame, validation_fe: pd.DataFrame, test_fe: pd.DataFrame) -> tuple[list[str], list[str], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, numeric, categorical = feature_columns(train_fe)
    feature_cols = numeric + categorical
    x_train = train_fe[feature_cols].copy()
    x_val = validation_fe[feature_cols].copy()
    x_test = test_fe[feature_cols].copy()
    for col in categorical:
        x_train[col] = x_train[col].astype("category")
        x_val[col] = x_val[col].astype("category")
        x_test[col] = x_test[col].astype("category")
    return feature_cols, numeric, categorical, x_train, x_val, x_test


def fit_lgbm_variant(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    categorical: list[str],
    *,
    recency_weighted: bool,
) -> np.ndarray:
    model_idx, cal_idx = time_split_labeled(train)
    y_fit = train.loc[model_idx, "default_flag"].astype(int)
    y_cal = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    fit_x = x_train.loc[model_idx].copy()
    cal_x = x_train.loc[cal_idx].copy()
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
        random_state=RANDOM_SEED + int(recency_weighted),
        verbosity=-1,
    )
    fit_weight = None
    cal_weight = None
    if recency_weighted:
        ts = pd.to_datetime(train["application_timestamp"])
        max_ts = ts.max()
        age_days = (max_ts - ts).dt.days.astype(float)
        weights = np.power(0.5, age_days / 180.0)
        weights = weights / weights.loc[model_idx].mean()
        weights = weights.clip(0.25, 4.0)
        fit_weight = weights.loc[model_idx].to_numpy(float)
        cal_weight = weights.loc[cal_idx].to_numpy(float)
    model.fit(fit_x, y_fit, sample_weight=fit_weight, categorical_feature=categorical)
    raw_cal = model.predict_proba(cal_x)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(raw_cal, y_cal, sample_weight=cal_weight)
    return np.clip(iso.predict(model.predict_proba(x_val)[:, 1]), 0.001, 0.999)


def recency_weighting_experiment(train: pd.DataFrame, validation: pd.DataFrame, x_train: pd.DataFrame, x_val: pd.DataFrame, categorical: list[str], curves: np.lib.npyio.NpzFile) -> pd.DataFrame:
    labeled = validation["default_flag"].notna().to_numpy()
    y_val = validation.loc[labeled, "default_flag"].astype(int).to_numpy()
    realized = realized_npv(validation.loc[labeled])
    rows = []
    for name, weighted in [("unweighted_lightgbm", False), ("recency_weighted_lightgbm", True)]:
        p = fit_lgbm_variant(train, x_train, x_val, categorical, recency_weighted=weighted)
        enpv = expected_npv(
            validation["requested_amount"].to_numpy(float),
            p,
            curves["validation_t_star"],
            curves["validation_recovery"],
        )
        margin = enpv / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
        candidates = np.unique(np.r_[np.linspace(-0.05, 0.08, 53), np.quantile(margin[labeled], np.linspace(0.01, 0.99, 79))])
        best = {"threshold": np.nan, "labeled_validation_npv": -np.inf, "approved": 0}
        for threshold in candidates:
            decision = margin[labeled] > threshold
            if decision.sum() == 0:
                continue
            npv = float(realized[decision].sum())
            if npv > best["labeled_validation_npv"]:
                best = {"threshold": float(threshold), "labeled_validation_npv": npv, "approved": int(decision.sum())}
        rows.append(
            {
                "variant": name,
                **metric_summary(y_val, p[labeled]),
                "best_threshold": best["threshold"],
                "best_labeled_validation_npv": best["labeled_validation_npv"],
                "best_labeled_approved": best["approved"],
            }
        )
    return pd.DataFrame(rows)


def fit_lgbm_with_calibration(
    fit_x: pd.DataFrame,
    fit_y: np.ndarray,
    cal_x: pd.DataFrame,
    cal_y: np.ndarray,
    score_x: pd.DataFrame,
    categorical: list[str],
    *,
    sample_weight: np.ndarray | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, LGBMClassifier, IsotonicRegression]:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=520,
        learning_rate=0.035,
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
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(raw_cal, cal_y)
    pred = np.clip(iso.predict(model.predict_proba(score_x)[:, 1]), 0.001, 0.999)
    return pred, model, iso


def score_model(model: LGBMClassifier, iso: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    return np.clip(iso.predict(model.predict_proba(x)[:, 1]), 0.001, 0.999)


def best_npv_policy(validation: pd.DataFrame, p: np.ndarray, curves: np.lib.npyio.NpzFile) -> dict[str, float]:
    labeled = validation["default_flag"].notna().to_numpy()
    realized = realized_npv(validation.loc[labeled])
    enpv = expected_npv(
        validation["requested_amount"].to_numpy(float),
        p,
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    margin = enpv / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
    candidates = np.unique(np.r_[np.linspace(-0.05, 0.08, 53), np.quantile(margin[labeled], np.linspace(0.01, 0.99, 79))])
    best = {"threshold": np.nan, "labeled_validation_npv": -np.inf, "labeled_approved": 0, "labeled_default_rate_approved": np.nan}
    for threshold in candidates:
        decision = margin[labeled] > threshold
        if decision.sum() == 0:
            continue
        npv = float(realized[decision].sum())
        if npv > best["labeled_validation_npv"]:
            best = {
                "threshold": float(threshold),
                "labeled_validation_npv": npv,
                "labeled_approved": int(decision.sum()),
                "labeled_default_rate_approved": float(np.mean(realized[decision] < 0)),
            }
    return best


def reject_augmentation_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    categorical: list[str],
    curves: np.lib.npyio.NpzFile,
) -> pd.DataFrame:
    """Evaluate screenshot-style simple and fuzzy reject inference.

    Simple augmentation assigns hard pseudo-good/bad labels to historical
    rejects using a score cutoff. Fuzzy augmentation duplicates each reject
    into good/bad rows with probability weights. These are diagnostics: hidden
    reject outcomes remain unidentified.
    """
    model_idx, cal_idx = time_split_labeled(train)
    reject_idx = train.index[train["default_flag"].isna()].to_numpy()
    y_fit = train.loc[model_idx, "default_flag"].astype(int).to_numpy()
    y_cal = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    y_val = validation.loc[validation["default_flag"].notna(), "default_flag"].astype(int).to_numpy()
    base_val, base_model, base_iso = fit_lgbm_with_calibration(
        x_train.loc[model_idx],
        y_fit,
        x_train.loc[cal_idx],
        y_cal,
        x_val,
        categorical,
        random_state=RANDOM_SEED + 31,
    )
    base_reject = score_model(base_model, base_iso, x_train.loc[reject_idx])
    accept_bad_rate = float(train.loc[model_idx, "default_flag"].mean())
    eval_prior_declined = pd.concat([validation["prior_decision"], test["prior_decision"]], ignore_index=True).to_numpy() == 0

    rows = []
    variants = [("accept_only", "none", 1.0)]
    variants.extend((f"simple_gamma_{g:g}", "simple", g) for g in [1.25, 1.5, 2.0, 3.0])
    variants.extend((f"fuzzy_gamma_{g:g}", "fuzzy", g) for g in [1.25, 1.5, 2.0, 3.0])

    for variant_name, method, gamma in variants:
        if method == "none":
            val_p = base_val
            test_p = score_model(base_model, base_iso, x_test)
        else:
            reject_p = expit(logit(np.clip(base_reject, 0.001, 0.999)) + np.log(gamma))
            reject_weight_scale = 0.35
            if method == "simple":
                target_bad_rate = min(max(accept_bad_rate * gamma, accept_bad_rate), 0.85)
                cutoff = np.quantile(reject_p, 1.0 - target_bad_rate)
                pseudo_y = (reject_p >= cutoff).astype(int)
                fit_x = pd.concat([x_train.loc[model_idx], x_train.loc[reject_idx]], ignore_index=True)
                fit_y = np.r_[y_fit, pseudo_y]
                weights = np.r_[np.ones(len(y_fit)), np.full(len(pseudo_y), reject_weight_scale)]
            else:
                reject_x = x_train.loc[reject_idx]
                fit_x = pd.concat([x_train.loc[model_idx], reject_x, reject_x], ignore_index=True)
                fit_y = np.r_[y_fit, np.ones(len(reject_x), dtype=int), np.zeros(len(reject_x), dtype=int)]
                weights = np.r_[
                    np.ones(len(y_fit)),
                    reject_weight_scale * reject_p,
                    reject_weight_scale * (1.0 - reject_p),
                ]
            val_p, model, iso = fit_lgbm_with_calibration(
                fit_x,
                fit_y,
                x_train.loc[cal_idx],
                y_cal,
                x_val,
                categorical,
                sample_weight=weights,
                random_state=RANDOM_SEED + int(gamma * 100) + (7 if method == "fuzzy" else 0),
            )
            test_p = score_model(model, iso, x_test)

        val_labeled = validation["default_flag"].notna().to_numpy()
        policy = best_npv_policy(validation, val_p, curves)
        eval_p = np.r_[val_p, test_p]
        rows.append(
            {
                "variant": variant_name,
                "method": method,
                "reject_odds_gamma": gamma,
                **metric_summary(y_val, val_p[val_labeled]),
                **policy,
                "mean_pd_prior_declined_eval": float(eval_p[eval_prior_declined].mean()),
                "p90_pd_prior_declined_eval": float(np.quantile(eval_p[eval_prior_declined], 0.90)),
                "mean_pd_prior_approved_eval": float(eval_p[~eval_prior_declined].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["labeled_validation_npv", "auroc"], ascending=False)


def eval_frame_with_economics(validation: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame, curves: np.lib.npyio.NpzFile) -> pd.DataFrame:
    frame = pd.concat([validation.assign(_split="validation"), test.assign(_split="test")], ignore_index=True)
    frame = frame.join(submission[["decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]])
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    amount = frame["requested_amount"].to_numpy(float)
    frame["expected_npv"] = expected_npv(amount, frame["predicted_pd"].to_numpy(float), t_star, recovery)
    frame["expected_npv_margin"] = frame["expected_npv"] / np.maximum(amount, 1.0)
    paid = npv_repaid(amount)
    default = npv_default(amount, t_star, recovery * amount)
    denom = paid - default
    frame["break_even_pd"] = np.clip(np.divide(paid, denom, out=np.full_like(paid, np.nan), where=denom > 0), 0, 1)
    frame["pd_headroom"] = frame["break_even_pd"] - frame["predicted_pd"]
    frame["realized_npv"] = np.nan
    val_realized = realized_npv(validation.fillna(np.nan))
    frame.loc[: len(validation) - 1, "realized_npv"] = np.where(validation["default_flag"].notna(), val_realized, np.nan)
    return frame


def expected_profit_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = frame["decision"] == 1
    ranked = frame.copy()
    ranked["profit_quintile"] = pd.qcut(ranked["expected_npv_margin"], 5, labels=False, duplicates="drop") + 1
    quintile = (
        ranked.groupby("profit_quintile", dropna=False)
        .agg(
            rows=("applicant_id", "size"),
            approved=("decision", "sum"),
            approval_rate=("decision", "mean"),
            mean_pd=("predicted_pd", "mean"),
            mean_expected_npv=("expected_npv", "mean"),
            expected_npv_total=("expected_npv", "sum"),
            mean_margin=("expected_npv_margin", "mean"),
            mean_headroom=("pd_headroom", "mean"),
        )
        .reset_index()
    )
    approved_profit = (
        frame.loc[approved]
        .assign(profit_decile=pd.qcut(frame.loc[approved, "expected_npv_margin"], 10, labels=False, duplicates="drop") + 1)
        .groupby("profit_decile", dropna=False)
        .agg(
            approved=("applicant_id", "size"),
            mean_pd=("predicted_pd", "mean"),
            expected_npv_total=("expected_npv", "sum"),
            mean_expected_npv=("expected_npv", "mean"),
            mean_margin=("expected_npv_margin", "mean"),
            prior_declined_approved=("prior_decision", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    return quintile, approved_profit


def reject_inference_sensitivity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_declined_approved = (frame["prior_decision"] == 0) & (frame["decision"] == 1)
    approved = frame["decision"] == 1
    rows = []
    p = frame["predicted_pd"].to_numpy(float)
    amount = frame["requested_amount"].to_numpy(float)
    paid = npv_repaid(amount)
    # Reconstruct default-side NPV from E[NPV] and p: E = (1-p)paid + p*default.
    default_side = np.divide(
        frame["expected_npv"].to_numpy(float) - (1.0 - p) * paid,
        np.maximum(p, 1e-6),
    )
    for odds_gamma in [1.0, 1.25, 1.50, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
        stressed_p = p.copy()
        idx = prior_declined_approved.to_numpy()
        stressed_p[idx] = expit(logit(np.clip(stressed_p[idx], 0.001, 0.999)) + np.log(odds_gamma))
        stressed_npv = (1.0 - stressed_p) * paid + stressed_p * default_side
        rows.append(
            {
                "reject_default_odds_gamma": odds_gamma,
                "approved_expected_npv_total": float(stressed_npv[approved.to_numpy()].sum()),
                "prior_declined_approved_expected_npv": float(stressed_npv[idx].sum()),
                "prior_declined_approved_mean_pd": float(stressed_p[idx].mean()) if idx.sum() else np.nan,
                "prior_declined_approved_predicted_defaults": float(stressed_p[idx].sum()) if idx.sum() else np.nan,
            }
        )
    table = pd.DataFrame(rows)

    focus = frame.loc[prior_declined_approved].copy()
    if focus.empty:
        return table, pd.DataFrame()
    # Per-row odds multiplier that would drive expected NPV to zero. This is an
    # unobserved-confounding sensitivity threshold, not an identified parameter.
    p0 = np.clip(focus["predicted_pd"].to_numpy(float), 0.001, 0.999)
    be = np.clip(focus["break_even_pd"].to_numpy(float), 0.001, 0.999)
    row_gamma = (be / (1.0 - be)) / (p0 / (1.0 - p0))
    focus["row_break_even_odds_gamma"] = row_gamma
    focus["gamma_tier"] = pd.cut(
        focus["row_break_even_odds_gamma"],
        bins=[0, 1.5, 2, 3, 5, 10, np.inf],
        labels=["<=1.5x", "1.5-2x", "2-3x", "3-5x", "5-10x", ">10x"],
        include_lowest=True,
    )
    gamma_summary = (
        focus.groupby("gamma_tier", dropna=False)
        .agg(
            prior_declined_approved=("applicant_id", "size"),
            mean_pd=("predicted_pd", "mean"),
            mean_break_even_pd=("break_even_pd", "mean"),
            expected_npv_total=("expected_npv", "sum"),
            mean_expected_npv_margin=("expected_npv_margin", "mean"),
        )
        .reset_index()
    )
    return table, gamma_summary


def life_table_hazard(
    validation: pd.DataFrame,
    submission: pd.DataFrame,
    curves: np.lib.npyio.NpzFile,
) -> pd.DataFrame:
    approved_labeled = (submission.iloc[: len(validation)]["decision"].to_numpy(int) == 1) & validation["default_flag"].notna().to_numpy()
    frame = validation.loc[approved_labeled].copy()
    active_pd = submission.iloc[: len(validation)]["predicted_pd"].to_numpy(float)[approved_labeled]
    raw_cum = curves["validation_cumulative"][approved_labeled]
    terminal = np.clip(raw_cum[:, [-1]], 1e-6, 1.0)
    shape = np.maximum.accumulate(np.clip(raw_cum / terminal, 0.0, 1.0), axis=1)
    pred_cum = np.maximum.accumulate(shape * np.clip(active_pd[:, None], 0.0, 1.0), axis=1)
    pred_prev = np.concatenate([np.zeros((len(frame), 1)), pred_cum[:, :-1]], axis=1)
    pred_hazard = np.clip((pred_cum - pred_prev) / np.clip(1.0 - pred_prev, 1e-9, None), 0.0, 1.0)

    default_week = np.where(
        frame["default_flag"].to_numpy() == 1,
        np.ceil(frame["days_to_default"].fillna(91).to_numpy(float) / 7),
        np.inf,
    )
    rows = []
    survival = 1.0
    for week in range(1, 14):
        at_risk = default_week >= week
        events = default_week == week
        r_k = int(at_risk.sum())
        d_k = int(events.sum())
        h_k = d_k / r_k if r_k else np.nan
        survival *= 1.0 - (h_k if np.isfinite(h_k) else 0.0)
        pred_h_k = float(pred_hazard[at_risk, week - 1].mean()) if r_k else np.nan
        rows.append(
            {
                "loan_age_week": week,
                "number_at_risk": r_k,
                "number_of_events": d_k,
                "number_censored": 0,
                "empirical_hazard": h_k,
                "plugin_survival": survival,
                "empirical_cumulative_default": 1.0 - survival,
                "mean_predicted_hazard_at_risk": pred_h_k,
                "hazard_error": pred_h_k - h_k if np.isfinite(h_k) else np.nan,
                "mean_predicted_cumulative_default": float(pred_cum[:, week - 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def hazard_timing_summary() -> dict[str, object]:
    summary_path = REPORT_DIR / "deliverable_b_validation_timing_summary.json"
    if not summary_path.exists():
        return {"status": "missing_deliverable_b_validation_timing_summary"}
    summary = json.loads(summary_path.read_text())
    diagnostics_path = REPORT_DIR / "deliverable_b_validation_timing_diagnostics.csv"
    if diagnostics_path.exists():
        diag = pd.read_csv(diagnostics_path)
        summary["mean_signed_error"] = float(diag["error"].mean())
        summary["p95_abs_error"] = float(diag["abs_error"].quantile(0.95))
        summary["max_abs_error"] = float(diag["abs_error"].max())
    return summary


def write_markdown(summary: dict[str, object]) -> None:
    best_aug = summary["reject_augmentation"].get("best_variant", {})
    lines = [
        "# Advanced Statistical Edges Audit",
        "",
        "## Main Read",
        "- The active policy is now full-engineered and prior-policy-proxy filtered.",
        "- The strongest remaining statistical risk is reject inference: locally observed labels still do not identify outcomes for prior-declined applicants.",
        "- Drift and recency diagnostics should be discussed in Deliverable D as validation controls, not as proof of hidden test performance.",
        "- The life-table section is a validation audit on this hackathon data only; it does not import the external example data from the screenshots.",
        "",
        "## Screenshot Concepts Applied",
        "- Reject workflow: prior-approved/labeled applications are the `accepted` population; prior-declined/unlabeled applications are the `rejected` population.",
        "- Simple augmentation: score rejects, assign hard pseudo-good/bad labels by cutoff, and refit a scorecard-style model.",
        "- Fuzzy augmentation: duplicate each reject into bad/good rows with probability weights, then refit.",
        "- Hazard audit: compute `h(k)=d_k/r_k` on approved labeled validation loans and compare it to the model-implied weekly hazard.",
        "",
        "## Key Numbers",
        f"- Forbidden model features found: {summary['leakage']['forbidden_feature_count']}",
        f"- Train/validation business overlap: {summary['leakage']['business_overlap_train_validation']}",
        f"- Train/test business overlap: {summary['leakage']['business_overlap_train_test']}",
        f"- Worst model-relevant train/test PSI: {summary['drift']['max_model_relevant_train_test_psi']:.3f}",
        f"- Worst model-relevant train/validation PSI: {summary['drift']['max_model_relevant_train_validation_psi']:.3f}",
        f"- Prior-declined approvals: {summary['reject_inference']['prior_declined_approved']:,}",
        f"- Prior-declined base expected NPV: ${summary['reject_inference']['base_prior_declined_expected_npv']:,.0f}",
        f"- Prior-declined expected NPV at 3x default odds: ${summary['reject_inference']['prior_declined_expected_npv_gamma_3x']:,.0f}",
        f"- Best reject-augmentation diagnostic: `{best_aug.get('variant', 'n/a')}` with labeled validation NPV ${best_aug.get('labeled_validation_npv', 0):,.0f}",
        f"- Life-table week-13 empirical CDR: {summary['hazard_timing']['life_table_week13_empirical_cdr']:.4f}",
        f"- Life-table week-13 predicted CDR: {summary['hazard_timing']['life_table_week13_predicted_cdr']:.4f}",
        "",
        "## Output Files",
        "- `advanced_leakage_audit.json`",
        "- `advanced_suspicious_single_feature_auc.csv`",
        "- `advanced_drift_diagnostics.csv`",
        "- `advanced_recency_weighting_experiment.csv`",
        "- `advanced_expected_profit_quintiles.csv`",
        "- `advanced_approved_profit_deciles.csv`",
        "- `advanced_reject_inference_sensitivity.csv`",
        "- `advanced_reject_break_even_gamma_by_tier.csv`",
        "- `advanced_reject_augmentation_experiment.csv`",
        "- `advanced_life_table_hazard_validation.csv`",
        "- `advanced_hazard_timing_summary.json`",
    ]
    (REPORT_DIR / "advanced_stats_edges_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission = pd.read_csv(SUBMISSION_A)
    curves = np.load(CURVES_PATH)

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    feature_cols, _, categorical, x_train, x_val, x_test = prepare_feature_matrices(train_fe, validation_fe, test_fe)

    leakage, suspicious_auc = leakage_audit(train, validation, test, feature_cols)
    drift = build_drift(train, validation, test)
    recency = recency_weighting_experiment(train, validation, x_train, x_val, categorical, curves)
    reject_aug = reject_augmentation_experiment(train, validation, test, x_train, x_val, x_test, categorical, curves)
    frame = eval_frame_with_economics(validation, test, submission, curves)
    profit_quintiles, approved_profit_deciles = expected_profit_tables(frame)
    reject_sensitivity, reject_gamma_tiers = reject_inference_sensitivity(frame)
    life_table = life_table_hazard(validation, submission, curves)
    hazard = hazard_timing_summary()

    leakage_path = REPORT_DIR / "advanced_leakage_audit.json"
    leakage_path.write_text(json.dumps(leakage, indent=2))
    suspicious_auc.to_csv(REPORT_DIR / "advanced_suspicious_single_feature_auc.csv", index=False)
    drift.to_csv(REPORT_DIR / "advanced_drift_diagnostics.csv", index=False)
    recency.to_csv(REPORT_DIR / "advanced_recency_weighting_experiment.csv", index=False)
    reject_aug.to_csv(REPORT_DIR / "advanced_reject_augmentation_experiment.csv", index=False)
    profit_quintiles.to_csv(REPORT_DIR / "advanced_expected_profit_quintiles.csv", index=False)
    approved_profit_deciles.to_csv(REPORT_DIR / "advanced_approved_profit_deciles.csv", index=False)
    reject_sensitivity.to_csv(REPORT_DIR / "advanced_reject_inference_sensitivity.csv", index=False)
    reject_gamma_tiers.to_csv(REPORT_DIR / "advanced_reject_break_even_gamma_by_tier.csv", index=False)
    life_table.to_csv(REPORT_DIR / "advanced_life_table_hazard_validation.csv", index=False)
    (REPORT_DIR / "advanced_hazard_timing_summary.json").write_text(json.dumps(hazard, indent=2))

    prior_declined_approved = (frame["prior_decision"] == 0) & (frame["decision"] == 1)
    model_relevant_drift = drift[
        (~drift["feature"].eq("application_timestamp"))
        & (~drift["feature"].apply(lambda c: any(token in str(c) for token in FORBIDDEN_POLICY_TOKENS)))
    ]
    summary = {
        "feature_set": os.getenv("DELIVERABLE_A_FEATURE_SET", "all_engineered"),
        "leakage": leakage,
        "drift": {
            "max_train_validation_psi": float(drift.loc[drift["comparison"] == "train_vs_validation", "psi"].max()),
            "max_train_test_psi": float(drift.loc[drift["comparison"] == "train_vs_test", "psi"].max()),
            "high_psi_feature_count": int((drift["psi"] >= 0.10).sum()),
            "max_model_relevant_train_validation_psi": float(
                model_relevant_drift.loc[model_relevant_drift["comparison"] == "train_vs_validation", "psi"].max()
            ),
            "max_model_relevant_train_test_psi": float(
                model_relevant_drift.loc[model_relevant_drift["comparison"] == "train_vs_test", "psi"].max()
            ),
            "high_model_relevant_psi_feature_count": int((model_relevant_drift["psi"] >= 0.10).sum()),
        },
        "recency_weighting": recency.to_dict(orient="records"),
        "reject_augmentation": {
            "best_variant": reject_aug.iloc[0].to_dict() if not reject_aug.empty else {},
            "active_style_accept_only": reject_aug.loc[reject_aug["variant"] == "accept_only"].iloc[0].to_dict()
            if "accept_only" in set(reject_aug["variant"])
            else {},
        },
        "reject_inference": {
            "prior_declined_approved": int(prior_declined_approved.sum()),
            "base_prior_declined_expected_npv": float(frame.loc[prior_declined_approved, "expected_npv"].sum()),
            "prior_declined_expected_npv_gamma_3x": float(
                reject_sensitivity.loc[reject_sensitivity["reject_default_odds_gamma"] == 3.0, "prior_declined_approved_expected_npv"].iloc[0]
            ),
            "prior_declined_expected_npv_gamma_6x": float(
                reject_sensitivity.loc[reject_sensitivity["reject_default_odds_gamma"] == 6.0, "prior_declined_approved_expected_npv"].iloc[0]
            ),
        },
        "hazard_timing": {
            "mean_abs_cdr_error": hazard.get("mean_abs_cdr_error"),
            "p95_abs_error": hazard.get("p95_abs_error"),
            "max_abs_error": hazard.get("max_abs_error"),
            "life_table_week13_empirical_cdr": float(life_table.loc[life_table["loan_age_week"] == 13, "empirical_cumulative_default"].iloc[0])
            if not life_table.empty
            else None,
            "life_table_week13_predicted_cdr": float(life_table.loc[life_table["loan_age_week"] == 13, "mean_predicted_cumulative_default"].iloc[0])
            if not life_table.empty
            else None,
        },
    }
    (REPORT_DIR / "advanced_stats_edges_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
