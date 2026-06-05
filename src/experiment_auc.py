#!/usr/bin/env python3
"""Can we push Deliverable A's AUC above 0.758? Honest CV comparison.

Measures AUC with 5-fold stratified CV on the 51,722 OBSERVED training rows
(NOT the 2,551 val rows — tuning to those would overfit). Compares:
  1. baseline features (current build_a pipeline)
  2. + domain-engineered features
  3. + engineered + tuned hyperparameters
and reports a held-out val AUC for the best config as a reality check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import (  # noqa: E402
    build_cat_dtypes, build_features, CATEGORICAL,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SEED = 17


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features. Inputs may be null (no bank feed); inf -> NaN (HistGBT ok)."""
    out = df.copy()
    obs_annual = df["observed_monthly_revenue_avg_3mo"] * 12

    out["eng_stated_obs_rev_ratio"] = df["stated_annual_revenue"] / obs_annual
    out["eng_stated_obs_rev_gap"] = df["stated_annual_revenue"] - obs_annual
    out["eng_prior_default_rate"] = (
        df["prior_loans_default_count"] / df["prior_loans_count"].replace(0, np.nan))
    out["eng_debt_to_obs_rev"] = df["existing_debt_obligations"] / obs_annual
    out["eng_debt_to_stated_rev"] = (
        df["existing_debt_obligations"] / df["stated_annual_revenue"].replace(0, np.nan))
    out["eng_loan_to_cash_p10"] = df["requested_amount"] / df["observed_cash_balance_p10"]
    out["eng_loan_to_obs_rev"] = df["requested_amount"] / obs_annual
    out["eng_req_to_prior_approved"] = (
        df["requested_amount"] / df["prior_approved_amount"].replace(0, np.nan))
    out["eng_total_inquiries"] = (
        df["recent_inquiries_count_6mo"].fillna(0)
        + df["multi_lender_inquiry_count_30d"].fillna(0))
    out["eng_overdraft_per_rev"] = (
        df["observed_overdraft_count_3mo"] / df["observed_monthly_revenue_avg_3mo"])

    eng_cols = [c for c in out.columns if c.startswith("eng_")]
    out[eng_cols] = out[eng_cols].replace([np.inf, -np.inf], np.nan)
    return out


def make_model(seed: int, tuned: bool) -> HistGradientBoostingClassifier:
    if tuned:
        params = dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=63,
                      min_samples_leaf=25, l2_regularization=0.5,
                      max_features=0.8)
    else:
        params = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                      min_samples_leaf=50, l2_regularization=1.0)
    return HistGradientBoostingClassifier(
        loss="log_loss", early_stopping=True, validation_fraction=0.1,
        categorical_features="from_dtype", random_state=seed, **params)


def cv_auc(X: pd.DataFrame, y: np.ndarray, tuned: bool) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        m = make_model(SEED, tuned)
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")

    obs = train["default_flag"].notna().to_numpy()
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()

    print("5-fold CV AUC on 51,722 observed rows (mean ± std):\n")

    # 1. baseline
    cats = build_cat_dtypes(train, val, test)
    Xb = build_features(train, cats).loc[obs]
    m, s = cv_auc(Xb, y, tuned=False)
    print(f"  1. baseline features              {m:.4f} ± {s:.4f}")

    # 2. + engineered
    tr_e = add_engineered(train); va_e = add_engineered(val); te_e = add_engineered(test)
    cats_e = build_cat_dtypes(tr_e, va_e, te_e)
    Xe = build_features(tr_e, cats_e).loc[obs]
    m2, s2 = cv_auc(Xe, y, tuned=False)
    print(f"  2. + engineered features          {m2:.4f} ± {s2:.4f}   (Δ {m2-m:+.4f})")

    # 3. + engineered + tuned
    m3, s3 = cv_auc(Xe, y, tuned=True)
    print(f"  3. + engineered + tuned model     {m3:.4f} ± {s3:.4f}   (Δ {m3-m:+.4f})")

    # Reality check: train best config on all observed, score held-out val.
    vmask = val["default_flag"].notna().to_numpy()
    yv = val.loc[vmask, "default_flag"].astype(int).to_numpy()
    Xv = build_features(va_e, cats_e).reindex(columns=Xe.columns)
    for c, dt in cats_e.items():
        if c in Xe.columns:
            Xv[c] = Xv[c].astype(dt)
    best = make_model(SEED, tuned=(m3 >= m2))
    best.fit(Xe, y)
    val_auc = roc_auc_score(yv, best.predict_proba(Xv.loc[vmask])[:, 1])
    print(f"\n  held-out validation AUC (best single model): {val_auc:.4f}")
    print("  (the deployed model is a 25× bootstrap ensemble, which ranks a bit higher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
