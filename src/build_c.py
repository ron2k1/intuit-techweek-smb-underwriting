#!/usr/bin/env python3
"""Deliverable C: do(feature=value) counterfactual PDs for the 900 queries.

Backdoor-restricted, monotone-constrained model-perturbation g-formula (verified
in src/wf3_causal_c.py, all 8 sign checks pass). Causal outcome model = A's spine
on a frame that DROPS the selection/collider nodes (prior_underwriter_score,
prior_decision, prior_approved_amount) and the self-report zero-effect fields
(stated_annual_revenue, stated_time_in_business, intended_use_of_funds), with a
clean observed-only leverage replacing the shipped engineered ratio. Monotone
constraints (HGB monotonic_cst + sign-constrained logistic) guarantee the correct
interventional SIGN on the 15 unambiguous economic drivers.

  do(F=v): set the raw parent, recompute engineered leverage, re-predict.
  do(self-report): column is excluded -> frame unchanged -> EXACT baseline (delta 0).

Run:  .venv/Scripts/python.exe -m src.build_c   ->  submissions/submission_C_counterfactuals.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.build_a import build_cat_dtypes, numeric_frame
from src.wf3_causal_c import (
    causal_feature_frame, make_causal_model, monotone_cst, SignLogit,
    apply_intervention, ZERO_EFFECT_FEATURES, COLLIDER_DROP, SELF_REPORT_ZERO,
    SHIPPED_RATIO, SEED,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
OUT = REPO / "submissions" / "submission_C_counterfactuals.csv"
N_BOOT = 25
OOD_MULT = 2.5
W_HGB = 0.4


def fit_causal_blend(train, val, cat_dtypes):
    """Fit the monotone causal blend; return (models, logit, iso, ref_cols)."""
    f_train = causal_feature_frame(train, cat_dtypes)
    ref_cols = list(f_train.columns)
    assert SHIPPED_RATIO not in ref_cols
    for c in COLLIDER_DROP + SELF_REPORT_ZERO:
        assert c not in ref_cols, f"{c} leaked into causal frame"
    mono = monotone_cst(ref_cols)
    f_val = causal_feature_frame(val, cat_dtypes)

    obs = train["default_flag"].notna().to_numpy()
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    X_obs = f_train.loc[obs]
    n = len(X_obs)
    rng = np.random.default_rng(SEED)
    models = []
    val_hgb = np.zeros((len(f_val), N_BOOT))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = make_causal_model(SEED + b + 1, mono).fit(X_obs.iloc[idx], y[idx])
        models.append(m)
        val_hgb[:, b] = m.predict_proba(f_val)[:, 1]
    logit = SignLogit(mono, C=0.5).fit(numeric_frame(X_obs).to_numpy(), y)
    logit_val = logit.predict_proba(numeric_frame(f_val).to_numpy())[:, 1]
    blend_val = W_HGB * val_hgb.mean(1) + (1 - W_HGB) * logit_val
    vmask = val["default_flag"].notna().to_numpy()
    yv = val.loc[vmask, "default_flag"].astype(int).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(blend_val[vmask], yv)
    print(f"[fit] causal-frame val AUC(blend)={roc_auc_score(yv, blend_val[vmask]):.4f} "
          f"({len(ref_cols)} features, {sum(s!=0 for s in mono)} monotone-constrained)")
    return models, logit, iso, ref_cols


def predict_with_interval(frame, models, logit, iso, ref_cols, cat_dtypes):
    """Return (point PD, lo05, hi95) calibrated, from the HGB ensemble spread."""
    f = frame.reindex(columns=ref_cols)
    for c, dt in cat_dtypes.items():
        if c in f.columns:
            f[c] = f[c].astype(dt)
    hgb = np.column_stack([m.predict_proba(f)[:, 1] for m in models])     # rows x N_BOOT
    lg = logit.predict_proba(numeric_frame(f).to_numpy())[:, 1]
    blend = W_HGB * hgb + (1 - W_HGB) * lg[:, None]
    point = np.clip(iso.predict(W_HGB * hgb.mean(1) + (1 - W_HGB) * lg), 1e-6, 1 - 1e-6)
    lo = np.clip(iso.predict(np.quantile(blend, 0.05, axis=1)), 1e-6, 1 - 1e-6)
    hi = np.clip(iso.predict(np.quantile(blend, 0.95, axis=1)), 1e-6, 1 - 1e-6)
    return point, np.minimum(lo, point), np.maximum(hi, point)


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    q = pd.read_csv(DATA / "intervention_queries.csv")
    q["intervention_value"] = pd.to_numeric(q["intervention_value"], errors="coerce")
    assert q["intervention_value"].notna().all(), "non-numeric intervention value"

    cat_dtypes = build_cat_dtypes(train, val, test)
    cat_features = set(cat_dtypes.keys())
    models, logit, iso, ref_cols = fit_causal_blend(train, val, cat_dtypes)

    raw = test.drop_duplicates("applicant_id").set_index("applicant_id")
    assert q["applicant_id"].isin(raw.index).all(), "query applicant missing from test"

    # OOD flag per applicant: never-labelled declined region OR no bank feed.
    score = pd.to_numeric(raw["prior_underwriter_score"], errors="coerce")
    feed = raw["has_linked_bank_feed"].astype(bool)
    ood_by_aid = ((score < 0.273) | (~feed)).to_dict()

    # ---- Build the 900-row intervened frame (batched) -------------------- #
    cf_rows, base_rows = [], []
    for _, qr in q.iterrows():
        rr = raw.loc[qr["applicant_id"]]
        base_rows.append(rr)
        v = qr["intervention_value"]
        if qr["feature_name"] in cat_features:   # categorical do-value -> int code
            v = int(round(v))
        cf_rows.append(apply_intervention(rr, qr["feature_name"], v))
    cf_frame = causal_feature_frame(pd.DataFrame(cf_rows).reset_index(drop=True), cat_dtypes)
    base_frame = causal_feature_frame(pd.DataFrame(base_rows).reset_index(drop=True), cat_dtypes)

    pd_cf, lo, hi = predict_with_interval(cf_frame, models, logit, iso, ref_cols, cat_dtypes)
    pd_base, _, _ = predict_with_interval(base_frame, models, logit, iso, ref_cols, cat_dtypes)

    # ---- OOD widening (2.5x) in the never-labelled / no-feed region ------- #
    ood = q["applicant_id"].map(ood_by_aid).to_numpy().astype(bool)
    half = np.maximum(hi - pd_cf, pd_cf - lo)
    half_w = np.where(ood, OOD_MULT * half, half)
    lo = np.clip(pd_cf - half_w, 0.0, pd_cf)
    hi = np.clip(pd_cf + half_w, pd_cf, 1.0)

    out = pd.DataFrame({
        "query_id": q["query_id"].to_numpy(),
        "predicted_pd_cf": pd_cf,
        "pd_cf_lower_90": lo,
        "pd_cf_upper_90": hi,
    })
    assert len(out) == 900 and out["query_id"].is_unique
    assert (out["pd_cf_lower_90"] <= out["predicted_pd_cf"] + 1e-9).all()
    assert (out["predicted_pd_cf"] <= out["pd_cf_upper_90"] + 1e-9).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[written] {OUT}  ({len(out)} rows)")

    # ---- Diagnostics (not in submission) --------------------------------- #
    delta = pd_cf - pd_base
    zero = q["feature_name"].isin(ZERO_EFFECT_FEATURES).to_numpy()
    print(f"\n[diag] self-report zero-effect queries: {int(zero.sum())} "
          f"(max |delta| = {np.abs(delta[zero]).max() if zero.any() else 0:.6f}  -> must be ~0)")
    print(f"[diag] lever queries: {int((~zero).sum())}  mean |delta| = "
          f"{np.abs(delta[~zero]).mean():.4f}  max = {np.abs(delta[~zero]).max():.4f}")
    print(f"[diag] OOD-widened queries: {int(ood.sum())} of 900  "
          f"(mean interval width {np.mean(hi-lo):.3f})")
    by = (pd.DataFrame({"feature": q["feature_name"], "delta": delta})
          .groupby("feature")["delta"].agg(["mean", lambda s: s.abs().mean()]))
    by.columns = ["mean_delta", "mean_abs_delta"]
    print("\n[diag] mean signed delta by feature (sign sanity):")
    print(by.sort_values("mean_abs_delta", ascending=False).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
