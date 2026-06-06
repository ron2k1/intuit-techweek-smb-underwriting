#!/usr/bin/env python3
"""wf4_calibration -- compare PD calibrators on the BLEND scores, honest 5-fold OOF.

Methods compared (all on the SAME cross-fit OOF blend scores, no leakage):
  (a) ISOTONIC         -- current baseline
  (b) PLATT / sigmoid  -- LogisticRegression on logit(p)  (1-param scale+shift)
  (c) BETA calibration -- LogisticRegression on [log p, log(1-p)] (a,b,c form)
  (d) RAW blend        -- uncalibrated

PROTOCOL (honest, no leakage):
  - 5-fold StratifiedKFold(shuffle, seed=17) on the 51,722 observed rows.
  - For each fold k, the BLEND model (0.4*HGB-ensemble + 0.6*L2-logit) is trained on
    the OTHER 4 folds and scores fold k. That yields OOF blend scores for every row,
    each produced by a model that never saw that row  ->  oof_blend.
  - CROSS-FIT CALIBRATION: to calibrate fold k's OOF scores without leakage, fit the
    calibrator on every OTHER fold's OOF scores+labels (leave-fold-out), then apply to
    fold k. Pooling all 5 folds' calibrated test predictions gives a fully honest OOF
    calibrated probability for all rows. (This matches build_a's reported cross-fit
    isotonic OOF: ECE 0.0018.)

METRICS on the pooled OOF calibrated probs:
  ECE(10-bin, equal-count), Brier, LogLoss, realized AMORTIZING NPV at break-even
  PD=0.259 (calibration moves which observed-train loans clear the cut), and a
  reliability decile table (pred vs obs) per method.

Run (from repo root, PYTHONPATH=.):
  .venv/Scripts/python.exe -m src.wf4_calibration
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(r"C:\Users\ayush\intuit-hackathon\intuit-techweek-smb-underwriting")
DATA = REPO / "dataset"
sys.path.insert(0, str(REPO))
from src.build_a import (  # noqa: E402
    build_cat_dtypes, build_features, make_model, make_logit, numeric_frame,
)

SEED = 17
N_SPLITS = 5
N_BOOT = 25          # ensemble size, matches deployed build_a
W_HGB = 0.4          # deployed blend weight

# ---- amortizing economics (exact, from build_a) ---- #
TERM_INT = 0.35 * 60 / 365
FEE = 0.03
NET_MARGIN = TERM_INT + FEE
DAILY_DRAW = (1 + TERM_INT) / 60
LGD = 0.25
BREAK_EVEN_PD = NET_MARGIN / (LGD + NET_MARGIN)   # ~0.259
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Calibrators (each: fit(s, y) on calib scores+labels -> predict(s) probs)
# --------------------------------------------------------------------------- #
def fit_isotonic(s, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y)
    return lambda x: np.clip(iso.predict(x), EPS, 1 - EPS)


def fit_platt(s, y):
    """Platt/sigmoid: logistic on logit(score). 1 slope + 1 intercept."""
    z = np.log(np.clip(s, EPS, 1 - EPS) / np.clip(1 - s, EPS, 1 - EPS)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000)  # ~unregularized scale+shift
    lr.fit(z, y)

    def pred(x):
        zx = np.log(np.clip(x, EPS, 1 - EPS) / np.clip(1 - x, EPS, 1 - EPS)).reshape(-1, 1)
        return np.clip(lr.predict_proba(zx)[:, 1], EPS, 1 - EPS)
    return pred


def fit_beta(s, y):
    """Beta calibration (Kull et al. 2017): logistic on [log p, log(1-p)].

    p_hat = sigmoid( a*log p - b*log(1-p) + c ).  We feed features f1=log(p),
    f2=log(1-p); the logistic learns coefficients (a, -b) and intercept c. This
    3-parameter family subsumes Platt and handles the [0,1] boundary shape that a
    single-logit sigmoid cannot.
    """
    p = np.clip(s, EPS, 1 - EPS)
    F = np.column_stack([np.log(p), np.log(1 - p)])
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(F, y)

    def pred(x):
        px = np.clip(x, EPS, 1 - EPS)
        Fx = np.column_stack([np.log(px), np.log(1 - px)])
        return np.clip(lr.predict_proba(Fx)[:, 1], EPS, 1 - EPS)
    return pred


CALIBRATORS = {
    "isotonic": fit_isotonic,
    "platt": fit_platt,
    "beta": fit_beta,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ece_equalcount(p, y, nbins=10):
    """Equal-count (10-bin) Expected Calibration Error."""
    order = np.argsort(p)
    bins = np.array_split(order, nbins)
    n = len(p)
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        conf = p[idx].mean()
        acc = y[idx].mean()
        ece += (len(idx) / n) * abs(conf - acc)
    return float(ece)


def ece_equalwidth(p, y, nbins=10):
    """Equal-width (10-bin) ECE -- reported as a robustness cross-check."""
    edges = np.linspace(0, 1, nbins + 1)
    n = len(p)
    ece = 0.0
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < nbins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(p, y, nbins=10):
    order = np.argsort(p)
    bins = np.array_split(order, nbins)
    rows = []
    for i, idx in enumerate(bins):
        rows.append((i, len(idx), p[idx].mean(), y[idx].mean(),
                     p[idx].mean() - y[idx].mean()))
    return rows


def realized_npv(p, amt, dflag, dtd, rec, thr=BREAK_EVEN_PD):
    """Realized amortizing $ profit funding rows with calibrated PD < thr.

    Exact official NPV: repaid -> amt*NET_MARGIN ; default -> F + D*(t*-1) + rec - R,
    capped at the repaid margin. Mirrors build_a._realized_value.
    """
    funded = p < thr
    rec = np.nan_to_num(rec)
    draws = amt * DAILY_DRAW * np.clip(np.nan_to_num(dtd) - 1, 0, None)
    default_profit = np.minimum(amt * FEE + draws + rec - amt, amt * NET_MARGIN)
    profit = np.where(dflag == 0, amt * NET_MARGIN, default_profit)
    return float(profit[funded].sum()), int(funded.sum())


# --------------------------------------------------------------------------- #
# Build cross-fit OOF blend scores
# --------------------------------------------------------------------------- #
def oof_blend_scores(Xo, y):
    """5-fold cross-fit: return (oof_blend, fold_id) where oof_blend[i] is produced
    by a blend model trained WITHOUT row i (its fold)."""
    n = len(y)
    oof = np.zeros(n)
    fold_id = np.full(n, -1)
    Xnum = numeric_frame(Xo)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        fold_id[te] = k
        # HGB bootstrap ensemble on train fold
        rng = np.random.default_rng(SEED + 100 * k)
        ntr = len(tr)
        hgb_te = np.zeros(len(te))
        for b in range(N_BOOT):
            bs = rng.integers(0, ntr, ntr)
            m = make_model(SEED + b + 1)
            m.fit(Xo.iloc[tr].iloc[bs], y[tr][bs])
            hgb_te += m.predict_proba(Xo.iloc[te])[:, 1]
        hgb_te /= N_BOOT
        # L2 logistic on train fold
        logit = make_logit().fit(Xnum.iloc[tr], y[tr])
        logit_te = logit.predict_proba(Xnum.iloc[te])[:, 1]
        oof[te] = W_HGB * hgb_te + (1 - W_HGB) * logit_te
        print(f"  [oof fold {k}] n_te={len(te):,}  "
              f"AUC blend={roc_auc_score(y[te], oof[te]):.4f}")
    return oof, fold_id


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)

    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    amt = train.loc[obs, "requested_amount"].to_numpy()
    dflag = train.loc[obs, "default_flag"].astype(int).to_numpy()
    dtd = train.loc[obs, "days_to_default"].to_numpy()
    rec = train.loc[obs, "final_recovered_amount"].to_numpy()
    print(f"[data] {obs.sum():,} observed rows, default rate {y.mean():.4f}")
    print(f"[econ] break-even PD = {BREAK_EVEN_PD:.4f}\n")

    print("Building cross-fit OOF blend scores (HGB-ens x{} + logit, 5 folds)...".format(N_BOOT))
    oof, fold_id = oof_blend_scores(Xo, y)
    print(f"[oof] pooled blend AUC = {roc_auc_score(y, oof):.4f}  "
          f"raw mean={oof.mean():.4f}\n")

    # ---- Cross-fit calibration: leave-fold-out per calibrator ---- #
    methods = ["raw", "isotonic", "platt", "beta"]
    cal_oof = {m: np.zeros(len(y)) for m in methods}
    cal_oof["raw"] = oof.copy()
    for m in ("isotonic", "platt", "beta"):
        fit_fn = CALIBRATORS[m]
        for k in range(N_SPLITS):
            te = fold_id == k
            tr = ~te                       # all OTHER folds' OOF scores
            cal = fit_fn(oof[tr], y[tr])
            cal_oof[m][te] = cal(oof[te])

    # ---- Theoretical NPV ceiling: fund by TRUE per-row outcome NPV>0 ---- #
    # (upper bound a calibrator's cut can reach on this realized set)
    draws = amt * DAILY_DRAW * np.clip(np.nan_to_num(dtd) - 1, 0, None)
    default_profit = np.minimum(amt * FEE + draws + np.nan_to_num(rec) - amt,
                                amt * NET_MARGIN)
    row_profit = np.where(dflag == 0, amt * NET_MARGIN, default_profit)
    oracle_npv = float(row_profit[row_profit > 0].sum())
    fund_all_npv = float(row_profit.sum())

    # ---- Score every method ---- #
    print("=" * 86)
    print("CALIBRATION METRICS  (honest 5-fold cross-fit OOF, pooled over all "
          f"{len(y):,} rows)")
    print("=" * 86)
    print(f"{'method':10s} {'ECE10(ec)':>10s} {'ECE10(ew)':>10s} {'Brier':>9s} "
          f"{'LogLoss':>9s} {'AUC':>7s} {'NPV@0.259':>13s} {'nFund':>8s}")
    results = {}
    for m in methods:
        p = np.clip(cal_oof[m], EPS, 1 - EPS)
        ec = ece_equalcount(p, y)
        ew = ece_equalwidth(p, y)
        br = brier_score_loss(y, p)
        ll = log_loss(y, p)
        au = roc_auc_score(y, p)
        npv, nf = realized_npv(p, amt, dflag, dtd, rec)
        results[m] = dict(ece=ec, ece_ew=ew, brier=br, logloss=ll, auc=au,
                          npv=npv, nfund=nf)
        print(f"{m:10s} {ec:10.4f} {ew:10.4f} {br:9.4f} {ll:9.4f} {au:7.4f} "
              f"{npv:13,.0f} {nf:8,d}")
    print(f"\n  [reference] fund-by-true-NPV>0 oracle = {oracle_npv:,.0f}  |  "
          f"fund-ALL = {fund_all_npv:,.0f}")
    print(f"  [reference] baseline (memory) cross-fit isotonic OOF: "
          f"Brier 0.1168, LogLoss 0.3818, ECE 0.0018")

    # ---- Reliability decile tables ---- #
    for m in methods:
        p = np.clip(cal_oof[m], EPS, 1 - EPS)
        print("\n" + "-" * 70)
        print(f"RELIABILITY (decile, equal-count) -- {m}")
        print(f"  {'bin':>3s} {'n':>7s} {'pred':>8s} {'obs':>8s} {'gap':>9s}")
        for i, nn, pm, ym, gap in reliability_table(p, y):
            print(f"  {i:3d} {nn:7,d} {pm:8.4f} {ym:8.4f} {gap:+9.4f}")

    # ---- Per-fold breakdown: ECE / Brier / LogLoss / NPV per OUTER fold ---- #
    print("\n" + "=" * 86)
    print("PER-FOLD OOF (each fold's calibrated test preds; calibrator fit on other folds)")
    print("=" * 86)
    perfold = {m: [] for m in methods}
    for k in range(N_SPLITS):
        te = fold_id == k
        yk = y[te]
        line = f"[fold {k}] n={te.sum():,}"
        for m in methods:
            p = np.clip(cal_oof[m][te], EPS, 1 - EPS)
            br = brier_score_loss(yk, p)
            ll = log_loss(yk, p, labels=[0, 1])
            ec = ece_equalcount(p, yk)
            npv, nf = realized_npv(p, amt[te], dflag[te], dtd[te], rec[te])
            perfold[m].append(dict(brier=br, logloss=ll, ece=ec, npv=npv, nfund=nf))
        print(line)
        print(f"   {'method':10s} {'ECE':>8s} {'Brier':>8s} {'LogLoss':>8s} "
              f"{'NPV@.259':>12s} {'nFund':>7s}")
        for m in methods:
            r = perfold[m][-1]
            print(f"   {m:10s} {r['ece']:8.4f} {r['brier']:8.4f} {r['logloss']:8.4f} "
                  f"{r['npv']:12,.0f} {r['nfund']:7,d}")

    # ---- Per-fold mean +- std summary + paired deltas vs isotonic ---- #
    print("\n" + "=" * 86)
    print("PER-FOLD MEAN +/- STD  and PAIRED DELTA vs ISOTONIC (per fold)")
    print("=" * 86)
    for metric in ("ece", "brier", "logloss", "npv"):
        print(f"\n  metric = {metric}")
        iso_vals = np.array([perfold["isotonic"][k][metric] for k in range(N_SPLITS)])
        for m in methods:
            vals = np.array([perfold[m][k][metric] for k in range(N_SPLITS)])
            d = vals - iso_vals
            tag = ""
            if m != "isotonic":
                # lower is better for ece/brier/logloss; higher better for npv
                if metric == "npv":
                    wins = int((d > 0).sum())
                else:
                    wins = int((d < 0).sum())
                tag = (f"  vs-iso mean d={d.mean():+.4f} "
                       f"({'better' if (d.mean()<0)==(metric!='npv') else 'worse'} "
                       f"{wins}/{N_SPLITS} folds)")
            print(f"    {m:10s} mean={vals.mean():12.4f} std={vals.std(ddof=1):11.4f}"
                  + tag)

    # ---- Save per-fold table ---- #
    rows = []
    for k in range(N_SPLITS):
        for m in methods:
            r = perfold[m][k]
            rows.append(dict(fold=k, method=m, **r))
    fdf = pd.DataFrame(rows)
    outp = REPO / "reports" / "wf4_calibration_folds.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(outp, index=False)
    print(f"\n[written] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
