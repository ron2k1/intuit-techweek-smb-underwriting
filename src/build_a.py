#!/usr/bin/env python3
"""Deliverable A — approve/decline + calibrated PD + 90% interval.

Pipeline:
  1. Build features (drop outcome leakage + ids + raw timestamp; add missingness
     indicators; mark integer-coded categoricals as pandas category dtype).
  2. Reject inference via IPW: fit P(prior approved | X) on all train rows, weight
     the observed (approved+matured) rows by 1/p_hat so the PD model's training
     distribution matches the full applicant population it must score.
  3. PD model: bootstrap ENSEMBLE of HistGradientBoostingClassifier (native NaN),
     fit on train-observed with the IPW weights. Ensemble spread -> 90% interval.
  4. Calibrate the ensemble-mean score with isotonic regression on validation
     (which has outcomes).
  5. Decide: approve when calibrated PD < break-even PD (LGD-derived ~0.088).
  6. Write submissions/submission_A_decisions.csv for all 13,306 val+test rows.

Run:  .venv/Scripts/python.exe -m src.build_a
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
OUT = REPO / "submissions" / "submission_A_decisions.csv"

SEED = 17
N_BOOT = 25  # ensemble size for epistemic intervals

# ---- Column bookkeeping -------------------------------------------------- #
OUTCOME_COLS = [
    "default_flag", "days_to_default", "days_to_full_repayment",
    "repayment_status", "final_recovered_amount", "observation_status",
]
ID_COLS = ["business_id", "applicant_id"]
DROP_RAW = ["application_timestamp"]  # not used as a raw feature for A

# Integer-coded categoricals (native categorical handling in HistGBT).
CATEGORICAL = [
    "sector", "geography_region", "employee_count_bucket", "intended_use_of_funds",
    "owner_personal_credit_band", "application_channel", "prior_decision",
]
# Nullable columns that get an explicit missingness indicator (MNAR signal).
MISSING_IND = [
    "observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
    "observed_revenue_volatility", "observed_cash_balance_p10",
    "observed_overdraft_count_3mo", "payroll_regularity_score",
    "days_since_last_external_decline", "days_since_last_inquiry_elsewhere",
    "prior_approved_amount", "prior_underwriter_score", "stated_annual_revenue",
]
# ---- Loan economics ------------------------------------------------------ #
# Loans amortize via daily ACH draws over 60 days, so a default (mean
# days_to_default = 43) has already repaid ~70% of principal before failing.
# Realized loss-given-default, crediting those pre-default draws, averages ~0.30
# -- NOT the ~0.91 you get by (wrongly) treating final_recovered_amount as the
# only recovery. See src/check_lgd.py. This sets break-even PD ~0.224, not 0.088.
TERM_INT = 0.35 * 60 / 365           # interest over the 60-day term (~0.0575)
FEE = 0.03                           # origination fee
NET_MARGIN = TERM_INT + FEE          # ~0.0875 fully-repaid net
LGD = 0.30                           # mean realized LGD under amortization
BREAK_EVEN_PD = NET_MARGIN / (LGD + NET_MARGIN)  # ~0.224


def build_cat_dtypes(*frames: pd.DataFrame) -> dict[str, CategoricalDtype]:
    """Fixed category sets shared across splits so HistGBT codes stay aligned."""
    cats: dict[str, CategoricalDtype] = {}
    for c in CATEGORICAL:
        vals = pd.concat([f[c] for f in frames if c in f.columns])
        levels = sorted(v for v in vals.dropna().unique())
        cats[c] = CategoricalDtype(categories=levels)
    return cats


def build_features(df: pd.DataFrame, cat_dtypes: dict[str, CategoricalDtype]) -> pd.DataFrame:
    """Return a model-ready feature frame (no target, no leakage)."""
    drop = set(OUTCOME_COLS + ID_COLS + DROP_RAW)
    feat = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    # Missingness indicators (added BEFORE any imputation; HistGBT keeps NaN).
    for c in MISSING_IND:
        if c in df.columns:
            feat[f"{c}__isna"] = df[c].isna().astype("int8")

    # Native categorical dtype with a FIXED category set (consistent codes).
    for c, dt in cat_dtypes.items():
        if c in feat.columns:
            feat[c] = feat[c].astype(dt)

    return feat


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        categorical_features="from_dtype",
        random_state=seed,
    )


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")

    print(f"break-even PD = {BREAK_EVEN_PD:.4f} (net margin {NET_MARGIN:.4f}, LGD {LGD})")

    # Fixed category sets across all splits, then build aligned feature frames.
    cat_dtypes = build_cat_dtypes(train, val, test)
    f_train = build_features(train, cat_dtypes)
    f_val = build_features(val, cat_dtypes).reindex(columns=f_train.columns)
    f_test = build_features(test, cat_dtypes).reindex(columns=f_train.columns)
    # reindex can coerce category dtype back to object; restore the fixed dtypes.
    for c, dt in cat_dtypes.items():
        if c in f_train.columns:
            f_val[c] = f_val[c].astype(dt)
            f_test[c] = f_test[c].astype(dt)

    # Observed (approved + matured) training rows.
    obs_mask = train["default_flag"].notna().to_numpy()
    y = train.loc[obs_mask, "default_flag"].astype(int).to_numpy()
    X_obs = f_train.loc[obs_mask]
    print(f"[data] training on {obs_mask.sum():,} observed rows, default rate {y.mean():.4f}")

    # NOTE: reject inference by IPW is mathematically void here. Selection is
    # deterministic on prior_underwriter_score (approved iff score>=0.273, ZERO
    # overlap), so P(approved|X)=1 on every observed row -> 1/p_hat == 1 (a no-op)
    # and is undefined (1/0) in the declined region. The selection problem is
    # therefore handled at the DECISION layer (see policies below), not by
    # reweighting. We train unweighted on the observed support.

    # ---- Bootstrap ensemble -------------------------------------------- #
    n = len(X_obs)
    rng = np.random.default_rng(SEED)
    val_scores = np.zeros((len(f_val), N_BOOT))
    test_scores = np.zeros((len(f_test), N_BOOT))
    aucs = []
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)  # bootstrap resample
        Xb = X_obs.iloc[idx]
        yb = y[idx]
        m = make_model(SEED + b + 1)
        m.fit(Xb, yb)
        val_scores[:, b] = m.predict_proba(f_val)[:, 1]
        test_scores[:, b] = m.predict_proba(f_test)[:, 1]
        # OOB-ish AUC on val (val never used in fit).
        aucs.append(roc_auc_score(val.loc[val["default_flag"].notna(), "default_flag"],
                                  val_scores[val["default_flag"].notna().to_numpy(), b]))
    print(f"[ensemble] mean val AUC across {N_BOOT} models: {np.mean(aucs):.4f}")

    val_mean = val_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    # ---- Isotonic calibration on validation outcomes ------------------- #
    vmask = val["default_flag"].notna().to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_mean[vmask], val.loc[vmask, "default_flag"].astype(int).to_numpy())

    def calibrate(s: np.ndarray) -> np.ndarray:
        return np.clip(iso.predict(s), 1e-6, 1 - 1e-6)

    # Calibrated point PD + ensemble-based 90% interval (calibrate the quantiles too).
    def predict_block(scores: np.ndarray):
        mean = scores.mean(axis=1)
        lo = np.quantile(scores, 0.05, axis=1)
        hi = np.quantile(scores, 0.95, axis=1)
        pd_point = calibrate(mean)
        pd_lo = np.minimum(calibrate(lo), pd_point)
        pd_hi = np.maximum(calibrate(hi), pd_point)
        return pd_point, pd_lo, pd_hi

    val_pd, val_lo, val_hi = predict_block(val_scores)
    test_pd, test_lo, test_hi = predict_block(test_scores)

    # ---- Coverage sanity-check on validation --------------------------- #
    _coverage_report(val_pd[vmask], val_lo[vmask], val_hi[vmask],
                     val.loc[vmask, "default_flag"].astype(int).to_numpy())

    # ---- Prediction artifact (reusable; policies are post-processing) --- #
    def meta_block(df, pd_p, pd_l, pd_h):
        return pd.DataFrame({
            "applicant_id": df["applicant_id"].to_numpy(),
            "prior_decision": df["prior_decision"].to_numpy(),
            "prior_underwriter_score": df["prior_underwriter_score"].to_numpy(),
            "requested_amount": df["requested_amount"].to_numpy(),
            "default_flag": df["default_flag"].to_numpy() if "default_flag" in df else np.nan,
            "days_to_default": (df["days_to_default"].to_numpy()
                                if "days_to_default" in df else np.nan),
            "final_recovered_amount": (df["final_recovered_amount"].to_numpy()
                                       if "final_recovered_amount" in df else np.nan),
            "predicted_pd": pd_p,
            "pd_lower_90": pd_l,
            "pd_upper_90": pd_h,
        })

    preds = pd.concat([
        meta_block(val, val_pd, val_lo, val_hi),
        meta_block(test, test_pd, test_lo, test_hi),
    ], ignore_index=True)
    pred_path = REPO / "reports" / "a_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    print(f"\n[written] {pred_path}  (calibrated PD + intervals for all {len(preds):,})")

    # ---- Decision policies + comparison -------------------------------- #
    compare_policies(preds)
    write_submission(preds, policy=DEFAULT_POLICY)
    return 0


# --------------------------------------------------------------------------- #
# Decision policies (the selection problem lives here, not in reweighting)
# --------------------------------------------------------------------------- #

DEFAULT_POLICY = "conservative"


def _decide(preds: pd.DataFrame, policy: str) -> np.ndarray:
    """Return a 0/1 decision array under the chosen funding policy."""
    pd_p = preds["predicted_pd"].to_numpy()
    pd_hi = preds["pd_upper_90"].to_numpy()
    has_support = (preds["prior_decision"] == 1).to_numpy()  # observed region
    profitable = pd_p < BREAK_EVEN_PD
    if policy == "conservative":
        return (profitable & has_support).astype(int)
    if policy == "gated":          # fund declined region only if 90% upper < BE
        confident_decline = (~has_support) & (pd_hi < BREAK_EVEN_PD)
        return ((profitable & has_support) | confident_decline).astype(int)
    if policy == "trust":          # fund anywhere PD < BE (pure extrapolation)
        return profitable.astype(int)
    raise ValueError(policy)


def _portfolio_value(preds: pd.DataFrame, decision: np.ndarray, true_pd: np.ndarray
                     ) -> float:
    """Expected portfolio profit (in $) of the funded set under given true PDs."""
    amt = preds["requested_amount"].to_numpy()
    funded = decision == 1
    ev_per_dollar = (1 - true_pd) * NET_MARGIN - true_pd * LGD
    return float((ev_per_dollar[funded] * amt[funded]).sum())


def _realized_value(preds: pd.DataFrame, decision: np.ndarray) -> tuple[float, int]:
    """Realized $ profit on funded loans with a TRUE observed outcome (val).

    Amortizing economics: a defaulted loan still repaid ~days_to_default/60 of its
    principal via daily draws before failing, plus the fee, plus post-default
    recovery. Capped at the full-repayment margin.
    """
    amt = preds["requested_amount"].to_numpy()
    d = preds["default_flag"].to_numpy()
    dtd = preds["days_to_default"].to_numpy()
    rec = np.nan_to_num(preds["final_recovered_amount"].to_numpy())
    has_truth = ~np.isnan(d)
    funded = (decision == 1) & has_truth
    frac = np.clip(np.minimum(np.nan_to_num(dtd), 60) / 60.0, 0, 1)
    default_profit = np.minimum(amt * (FEE + frac * (1 + TERM_INT) - 1) + rec,
                                amt * NET_MARGIN)
    profit = np.where(d == 0, amt * NET_MARGIN, default_profit)
    return float(profit[funded].sum()), int(funded.sum())


def compare_policies(preds: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("POLICY COMPARISON  (break-even PD = {:.3f})".format(BREAK_EVEN_PD))
    print("=" * 74)
    declined = (preds["prior_decision"] == 0).to_numpy()
    model_pd = preds["predicted_pd"].to_numpy()
    # Pessimistic true-PD for the unobserved region: monotonic floor at the
    # cutoff-edge empirical rate (0.245); observed region keeps model PD.
    pessimistic = np.where(declined, np.maximum(model_pd, 0.245), model_pd)

    for policy in ("conservative", "gated", "trust"):
        dec = _decide(preds, policy)
        n_fund = int(dec.sum())
        n_fund_decl = int((dec == 1)[declined].sum())
        realized, n_real = _realized_value(preds, dec)
        ev_model = _portfolio_value(preds, dec, model_pd)
        ev_pess = _portfolio_value(preds, dec, pessimistic)
        decl_amt = preds.loc[(dec == 1) & declined, "requested_amount"].sum()
        print(f"\n[{policy}]")
        print(f"  funded total ............ {n_fund:,}  "
              f"(of which {n_fund_decl:,} in UNOBSERVED declined region)")
        print(f"  realized $ on val (observed funded={n_real:,}) ... {realized:,.0f}")
        print(f"  EV $ (model PDs, optimistic) ................... {ev_model:,.0f}")
        print(f"  EV $ (declined-region floored at 0.245) ....... {ev_pess:,.0f}")
        print(f"  principal at risk in declined region .......... ${decl_amt:,.0f}")
    print("\n> Realized val $ is IDENTICAL across policies (they agree on the only")
    print("> region with outcomes). The differentiator is the declined-region EV:")
    print("> note how 'trust' swings hugely between the optimistic and floored views.")


def write_submission(preds: pd.DataFrame, policy: str) -> None:
    dec = _decide(preds, policy)
    sub = pd.DataFrame({
        "applicant_id": preds["applicant_id"],
        "decision": dec,
        "predicted_pd": preds["predicted_pd"],
        "pd_lower_90": preds["pd_lower_90"],
        "pd_upper_90": preds["pd_upper_90"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    n_app = int(sub["decision"].sum())
    print(f"\n[written] {OUT}")
    print(f"  policy='{policy}', approve {n_app:,}/{len(sub):,} "
          f"({100*n_app/len(sub):.1f}%)")


def _coverage_report(pd_p, pd_lo, pd_hi, y) -> None:
    """Bin by predicted PD; check empirical default rate vs interval (S_cal proxy)."""
    print("\n[coverage] decile bins on validation (empirical default vs 90% interval):")
    order = np.argsort(pd_p)
    bins = np.array_split(order, 10)
    width = np.mean(pd_hi - pd_lo)
    covered = 0
    for i, idx in enumerate(bins):
        emp = y[idx].mean()
        lo, hi = pd_lo[idx].mean(), pd_hi[idx].mean()
        pt = pd_p[idx].mean()
        ok = lo <= emp <= hi
        covered += ok
        print(f"  bin {i:2d}: pred {pt:.3f}  emp {emp:.3f}  "
              f"[{lo:.3f},{hi:.3f}] {'OK' if ok else 'MISS'}")
    print(f"[coverage] {covered}/10 bins contained; mean interval width {width:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
