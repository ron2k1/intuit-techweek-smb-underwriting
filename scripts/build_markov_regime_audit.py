#!/usr/bin/env python3
"""Markov-style regime audit for the underwriting data.

This is the part of "regime shift" that is actually relevant to the supplied
screenshots: infer a small number of discrete weekly regimes from exogenous
application/economic features, estimate a transition matrix over those regimes,
and audit whether the active underwriting model behaves differently by state.

It is intentionally an audit/challenger, not a blind replacement of the active
submission. We only use outcome labels for evaluation after regimes are inferred
from non-outcome weekly features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.mixture import GaussianMixture
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_compact_feature_reject_bakeoff import odds_stress_pd  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_A = OUTPUT_DIR / "submission" / "submission_A_decisions.csv"
CURVES_PATH = OUTPUT_DIR / "deliverable_a_curves.npz"
RANDOM_SEED = 2026

REGIME_FEATURES = [
    "observed_revenue_trend_3mo",
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "requested_amount",
    "requested_amount_to_observed_revenue",
    "existing_debt_obligations",
    "aggregate_credit_utilization",
    "invoice_payment_delinquency_rate",
    "multi_lender_inquiry_count_30d",
    "has_linked_bank_feed",
]


def week_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.to_period("W-SUN").dt.start_time


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame["_split"] = name
        frame["_week_start"] = week_start(frame["application_timestamp"])
    return train, validation, test


def weekly_feature_frame(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, group in all_rows.groupby("_week_start", sort=True):
        row: dict[str, object] = {
            "week_start": week,
            "rows": int(len(group)),
            "train_rows": int((group["_split"] == "train").sum()),
            "validation_rows": int((group["_split"] == "validation").sum()),
            "test_rows": int((group["_split"] == "test").sum()),
        }
        for feature in REGIME_FEATURES:
            if feature not in group.columns:
                continue
            values = group[feature]
            if values.dtype == bool:
                values = values.astype(float)
            row[f"{feature}_mean"] = float(pd.to_numeric(values, errors="coerce").mean())
            row[f"{feature}_missing"] = float(values.isna().mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
    return out


def regime_matrix(weeks: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        col
        for col in weeks.columns
        if col.endswith("_mean") or col.endswith("_missing")
    ]
    x = weeks[feature_cols].copy()
    # Weekly means can still be missing if no applicant in a week linked a bank
    # feed. Use train-week medians so future rows do not define the imputation.
    train_week = weeks["train_rows"] > 0
    med = x.loc[train_week].median(numeric_only=True)
    x = x.fillna(med).fillna(0.0)
    return x, feature_cols


def fit_regimes(x: pd.DataFrame, weeks: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], GaussianMixture, StandardScaler]:
    train_week = weeks["train_rows"] > 0
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.loc[train_week])
    all_scaled = scaler.transform(x)

    candidates = []
    models: dict[int, GaussianMixture] = {}
    for k in range(2, 6):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=20,
            random_state=RANDOM_SEED + k,
            reg_covar=1e-4,
        )
        model.fit(x_scaled)
        bic = float(model.bic(x_scaled))
        candidates.append({"n_regimes": k, "bic": bic})
        models[k] = model
    bic_table = pd.DataFrame(candidates).sort_values("bic").reset_index(drop=True)
    best_k = int(bic_table.iloc[0]["n_regimes"])
    model = models[best_k]
    probs = model.predict_proba(all_scaled)
    states = probs.argmax(axis=1)

    out = weeks.copy()
    # Sort labels by mean revenue trend so names are stable/interpretable.
    tmp = out.assign(_raw_state=states)
    trend_col = "observed_revenue_trend_3mo_mean"
    order = (
        tmp.groupby("_raw_state")[trend_col]
        .mean()
        .sort_values(ascending=False)
        .index.to_list()
        if trend_col in tmp.columns
        else sorted(np.unique(states))
    )
    relabel = {raw: i for i, raw in enumerate(order)}
    out["regime"] = [relabel[s] for s in states]
    for raw, new in relabel.items():
        out[f"regime_prob_{new}"] = probs[:, raw]
    meta = {
        "selected_regime_count": best_k,
        "bic_table": bic_table.to_dict(orient="records"),
        "labeling": "regime 0 has highest weekly mean observed_revenue_trend_3mo; larger ids are weaker revenue-trend states",
    }
    return out, meta, model, scaler


def transition_matrix(regime_weeks: pd.DataFrame) -> pd.DataFrame:
    states = sorted(regime_weeks["regime"].unique())
    idx = {state: i for i, state in enumerate(states)}
    counts = np.ones((len(states), len(states)))  # Laplace smoothing.
    ordered = regime_weeks.sort_values("week_start")["regime"].to_numpy()
    for a, b in zip(ordered[:-1], ordered[1:]):
        counts[idx[a], idx[b]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, index=[f"from_{s}" for s in states], columns=[f"to_{s}" for s in states])


def attach_regime(frame: pd.DataFrame, regime_weeks: pd.DataFrame) -> pd.DataFrame:
    mapping = regime_weeks[["week_start", "regime"]].copy()
    out = frame.merge(mapping, left_on="_week_start", right_on="week_start", how="left")
    out = out.drop(columns=["week_start"])
    return out


def active_policy_audit(validation: pd.DataFrame, test: pd.DataFrame, regime_weeks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = pd.read_csv(SUBMISSION_A)
    curves = np.load(CURVES_PATH)
    n_val = len(validation)
    val_sub = sub.iloc[:n_val].reset_index(drop=True)
    test_sub = sub.iloc[n_val:].reset_index(drop=True)

    validation = attach_regime(validation, regime_weeks).join(val_sub[["decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]])
    test = attach_regime(test, regime_weeks).join(test_sub[["decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]])

    val_amount = validation["requested_amount"].to_numpy(float)
    test_amount = test["requested_amount"].to_numpy(float)
    validation["expected_npv"] = expected_npv(
        val_amount,
        validation["predicted_pd"].to_numpy(float),
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    test["expected_npv"] = expected_npv(
        test_amount,
        test["predicted_pd"].to_numpy(float),
        curves["test_t_star"],
        curves["test_recovery"],
    )
    validation["expected_npv_margin"] = validation["expected_npv"] / np.maximum(val_amount, 1.0)
    test["expected_npv_margin"] = test["expected_npv"] / np.maximum(test_amount, 1.0)

    validation_labeled = validation[validation["default_flag"].notna()].copy()
    validation_labeled["realized_npv"] = realized_npv(validation_labeled)

    rows = []
    for split_name, frame in [("validation_labeled", validation_labeled), ("validation_all", validation), ("test_all", test)]:
        for regime, group in frame.groupby("regime", dropna=False):
            decision = group["decision"].to_numpy(int).astype(bool)
            row = {
                "split": split_name,
                "regime": int(regime) if pd.notna(regime) else -1,
                "rows": int(len(group)),
                "approval_rate": float(decision.mean()) if len(group) else np.nan,
                "mean_pd": float(group["predicted_pd"].mean()),
                "mean_pd_approved": float(group.loc[decision, "predicted_pd"].mean()) if decision.sum() else np.nan,
                "expected_npv_approved": float(group.loc[decision, "expected_npv"].sum()),
                "expected_npv_margin_approved": float(group.loc[decision, "expected_npv_margin"].mean()) if decision.sum() else np.nan,
            }
            if split_name == "validation_labeled":
                y = group["default_flag"].astype(int).to_numpy()
                p = group["predicted_pd"].to_numpy(float)
                row.update(
                    {
                        "actual_default_rate": float(y.mean()),
                        "approved_default_rate": float(group.loc[decision, "default_flag"].mean()) if decision.sum() else np.nan,
                        "realized_npv_approved": float(group.loc[decision, "realized_npv"].sum()),
                        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
                        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) > 1 else np.nan,
                        "brier": float(brier_score_loss(y, p)),
                        "calibration_error": float(y.mean() - p.mean()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), pd.concat([validation, test], ignore_index=True)


def exact_threshold_policy(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    p_val: np.ndarray,
    p_test: np.ndarray,
    *,
    name: str,
) -> dict[str, object]:
    curves = np.load(CURVES_PATH)
    labeled = validation["default_flag"].notna().to_numpy()
    y = validation.loc[labeled, "default_flag"].astype(int).to_numpy()
    realized = realized_npv(validation.loc[labeled])
    val_amount = validation["requested_amount"].to_numpy(float)
    test_amount = test["requested_amount"].to_numpy(float)
    val_prior = validation["prior_decision"].to_numpy() == 0
    test_prior = test["prior_decision"].to_numpy() == 0

    val_npv = expected_npv(val_amount, p_val, curves["validation_t_star"], curves["validation_recovery"])
    test_npv = expected_npv(test_amount, p_test, curves["test_t_star"], curves["test_recovery"])
    val_margin = val_npv / np.maximum(val_amount, 1.0)
    test_margin = test_npv / np.maximum(test_amount, 1.0)

    val_stressed = p_val.copy()
    val_stressed[val_prior] = odds_stress_pd(val_stressed[val_prior], 3.0)
    val_stressed_margin = expected_npv(
        val_amount,
        val_stressed,
        curves["validation_t_star"],
        curves["validation_recovery"],
    ) / np.maximum(val_amount, 1.0)
    test_stressed = p_test.copy()
    test_stressed[test_prior] = odds_stress_pd(test_stressed[test_prior], 3.0)
    test_stressed_margin = expected_npv(
        test_amount,
        test_stressed,
        curves["test_t_star"],
        curves["test_recovery"],
    ) / np.maximum(test_amount, 1.0)

    val_eligible = (~val_prior) | ((val_margin > 0.03) & (val_stressed_margin > 0.0))
    test_eligible = (~test_prior) | ((test_margin > 0.03) & (test_stressed_margin > 0.0))
    real_full = np.full(len(validation), np.nan)
    real_full[labeled] = realized
    candidates = np.where(labeled & val_eligible)[0]
    order = candidates[np.argsort(val_margin[candidates])[::-1]]
    cumulative = np.cumsum(real_full[order])
    k = int(np.argmax(cumulative)) + 1
    threshold = float(val_margin[order[k - 1]] - 1e-12)
    val_decision = val_eligible & (val_margin > threshold)
    test_decision = test_eligible & (test_margin > threshold)
    labeled_decision = val_decision[labeled]
    all_decision = np.r_[val_decision, test_decision]
    all_npv = np.r_[val_npv, test_npv]
    return {
        "candidate": name,
        "threshold": threshold,
        "validation_labeled_realized_npv": float(realized[labeled_decision].sum()),
        "validation_labeled_approved": int(labeled_decision.sum()),
        "validation_labeled_default_rate_approved": float(y[labeled_decision].mean()) if labeled_decision.sum() else np.nan,
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "headline_expected_npv": float(all_npv[all_decision].sum()),
        "auroc": float(roc_auc_score(y, p_val[labeled])),
        "log_loss": float(log_loss(y, p_val[labeled], labels=[0, 1])),
        "brier": float(brier_score_loss(y, p_val[labeled])),
        "mean_pd": float(p_val[labeled].mean()),
    }


def regime_adjustment_sweep(validation: pd.DataFrame, test: pd.DataFrame, regime_weeks: pd.DataFrame) -> pd.DataFrame:
    sub = pd.read_csv(SUBMISSION_A)
    n_val = len(validation)
    p_val_base = np.clip(sub.iloc[:n_val]["predicted_pd"].to_numpy(float), 0.001, 0.999)
    p_test_base = np.clip(sub.iloc[n_val:]["predicted_pd"].to_numpy(float), 0.001, 0.999)
    validation = attach_regime(validation, regime_weeks)
    test = attach_regime(test, regime_weeks)

    rows = [exact_threshold_policy(validation, test, p_val_base, p_test_base, name="active_pd_exact_threshold")]
    regimes = sorted(validation["regime"].dropna().unique())
    # Single-regime odds offsets. This is the simplest switching-submodel test:
    # same base PD model, but state-specific intercepts.
    for regime in regimes:
        val_mask = (validation["regime"].to_numpy() == regime).astype(float)
        test_mask = (test["regime"].to_numpy() == regime).astype(float)
        for alpha in np.linspace(-0.35, 0.35, 15):
            p_val = expit(logit(p_val_base) + alpha * val_mask)
            p_test = expit(logit(p_test_base) + alpha * test_mask)
            rows.append(exact_threshold_policy(validation, test, p_val, p_test, name=f"regime_{int(regime)}_odds_{alpha:+.2f}"))
    return pd.DataFrame(rows).sort_values(["validation_labeled_realized_npv", "auroc"], ascending=[False, False]).reset_index(drop=True)


def write_report(summary: dict[str, object], paths: dict[str, Path]) -> None:
    lines = [
        "# Markov Regime Audit",
        "",
        "## Interpretation",
        "",
        "The relevant Markov-chain idea is discrete latent economic states with transition probabilities. "
        "This audit infers weekly states from non-outcome application/economic features, estimates a state transition matrix, "
        "and checks whether the active underwriting policy behaves differently by state.",
        "",
        "## Key Results",
        "",
        f"- Selected regimes: {summary['selected_regime_count']}",
        f"- Active variant rank in switching-intercept sweep: {summary['active_rank_in_candidate_sweep']}",
        f"- Best candidate: `{summary['best_candidate']['candidate']}`",
        f"- Best validation NPV: ${summary['best_candidate']['validation_labeled_realized_npv']:,.0f}",
        f"- Active exact-threshold validation NPV: ${summary['active_exact_threshold']['validation_labeled_realized_npv']:,.0f}",
        "",
        "## Files",
        "",
    ]
    for label, path in paths.items():
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}` ({label})")
    (REPORT_DIR / "markov_regime_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_frames()
    all_rows = pd.concat([train, validation, test], ignore_index=True)
    weeks = weekly_feature_frame(all_rows)
    x, feature_cols = regime_matrix(weeks)
    regime_weeks, regime_meta, _model, _scaler = fit_regimes(x, weeks)
    trans = transition_matrix(regime_weeks)
    policy_by_regime, scored_rows = active_policy_audit(validation, test, regime_weeks)
    candidate_sweep = regime_adjustment_sweep(validation, test, regime_weeks)

    regime_feature_summary = (
        regime_weeks.groupby("regime")
        .agg(
            weeks=("week_start", "size"),
            first_week=("week_start", "min"),
            last_week=("week_start", "max"),
            train_rows=("train_rows", "sum"),
            validation_rows=("validation_rows", "sum"),
            test_rows=("test_rows", "sum"),
            mean_revenue_trend=("observed_revenue_trend_3mo_mean", "mean"),
            mean_requested_amount=("requested_amount_mean", "mean"),
            mean_request_to_observed_revenue=("requested_amount_to_observed_revenue_mean", "mean"),
            mean_existing_debt=("existing_debt_obligations_mean", "mean"),
            mean_invoice_delinquency=("invoice_payment_delinquency_rate_mean", "mean"),
        )
        .reset_index()
    )

    paths = {
        "weekly regimes": REPORT_DIR / "markov_regime_weekly_states.csv",
        "transition matrix": REPORT_DIR / "markov_regime_transition_matrix.csv",
        "regime feature summary": REPORT_DIR / "markov_regime_feature_summary.csv",
        "active policy by regime": REPORT_DIR / "markov_regime_active_policy_by_regime.csv",
        "switching intercept candidates": REPORT_DIR / "markov_regime_candidate_sweep.csv",
    }
    regime_weeks.to_csv(paths["weekly regimes"], index=False)
    trans.to_csv(paths["transition matrix"])
    regime_feature_summary.to_csv(paths["regime feature summary"], index=False)
    policy_by_regime.to_csv(paths["active policy by regime"], index=False)
    candidate_sweep.to_csv(paths["switching intercept candidates"], index=False)

    active_exact = candidate_sweep[candidate_sweep["candidate"] == "active_pd_exact_threshold"].iloc[0].to_dict()
    best = candidate_sweep.iloc[0].to_dict()
    active_rank = int(candidate_sweep.index[candidate_sweep["candidate"].eq("active_pd_exact_threshold")][0]) + 1
    summary = {
        **regime_meta,
        "regime_features": feature_cols,
        "transition_matrix": trans.to_dict(orient="index"),
        "active_rank_in_candidate_sweep": active_rank,
        "active_exact_threshold": active_exact,
        "best_candidate": best,
        "outputs": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in paths.items()},
    }
    (REPORT_DIR / "markov_regime_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(summary, paths)
    print(json.dumps(
        {
            "selected_regime_count": summary["selected_regime_count"],
            "active_rank_in_candidate_sweep": active_rank,
            "best_candidate": best,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
