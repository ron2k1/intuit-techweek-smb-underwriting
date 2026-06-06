#!/usr/bin/env python3
"""wf4_intervals_tier23.py -- TIER-2/3 PD-interval methods, honest 5-fold OOF.

Goal: find a CALIBRATED 90% PD interval (pd_lower_90 / pd_upper_90) at mean width
clearly < 0.12 (our bootstrap-ensemble 5/95 isotonic baseline), measured HONESTLY
out-of-fold on the 51,722 observed train rows.

PROTOCOL (the only valid honesty bar):
  5-fold StratifiedKFold(shuffle, seed=17) on observed train rows.
  For each fold:
    * train every interval machine ONLY on the 4 train folds,
    * predict point PD + [lo, hi] on the held-out fold,
    * measure coverage on the held-out fold's REAL default outcomes.
  Coverage = fraction of 10 decile bins (sorted by point PD) whose empirical
  default rate lies in [mean lower, mean upper] of the bin.  Also report a finer
  20-bin version and the mean interval width.  A method is adopted only if it
  HOLDS ~9-10/10 decile bins OOF at mean width clearly < 0.12.

Methods compared (all under identical OOF protocol):
  BASE : bootstrap-ensemble (15 HGB) 5/95 pct, isotonic-mapped  (current baseline)
  CQR  : conformalized quantile regression -- GradientBoostingRegressor quantile
         loss @0.05/0.95 on PD, conformal additive adjustment on a calib split.
  GBQ  : gradient-boosting quantiles ALONE (no conformal adjustment).
  BB   : Bayesian Beta-Binomial band -- each PD a Beta posterior shrinking to a
         local default-rate prior (kNN-in-PD-space neighbourhood), 5/95 of Beta.

Run:
  set PYTHONPATH=C:\\Users\\ayush\\intuit-hackathon\\intuit-techweek-smb-underwriting
  .venv\\Scripts\\python.exe -m src.wf4_intervals_tier23
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(r"C:\Users\ayush\intuit-hackathon\intuit-techweek-smb-underwriting")
DATA = REPO / "dataset"
sys.path.insert(0, str(REPO))
from src.build_a import (  # noqa: E402
    build_cat_dtypes,
    build_features,
    make_logit,
    make_model,
    numeric_frame,
)

SEED = 17
N_BOOT = 15
TARGET = 0.90


# --------------------------------------------------------------------------- #
# Coverage metrics
# --------------------------------------------------------------------------- #
def binned_cov(pt, lo, hi, y, nbins):
    """Fraction of bins (sorted by point PD) whose empirical default rate is in
    [mean lo, mean hi].  Returns (coverage_fraction, rows)."""
    order = np.argsort(pt)
    bins = np.array_split(order, nbins)
    covered = 0
    rows = []
    for i, idx in enumerate(bins):
        emp = y[idx].mean()
        L, H = lo[idx].mean(), hi[idx].mean()
        ok = L <= emp <= H
        covered += ok
        rows.append((i, pt[idx].mean(), emp, L, H, ok))
    return covered / nbins, rows


# --------------------------------------------------------------------------- #
# Point-PD model = the SAME calibrated blend the baseline ranks with, refit OOF.
# We need an honest point PD on the held-out fold to form decile bins, plus a
# cross-fit calibrator. We fit blend on the train folds, isotonic-calibrate via
# an internal cross-fit on those train folds, apply to held-out fold.
# --------------------------------------------------------------------------- #
def fit_point_pd(X_tr, y_tr, X_te, seed=SEED):
    """Return calibrated point PD on X_te (blend 0.4 HGB-ens + 0.6 logit, isotonic).

    Calibrator is cross-fit WITHIN the train folds so it is honest, then applied
    to the held-out fold."""
    n = len(X_tr)
    rng = np.random.default_rng(seed)
    te_scores = np.zeros((len(X_te), N_BOOT))
    Z_tr = numeric_frame(X_tr)
    Z_te = numeric_frame(X_te)

    # Calibrator inputs: a SINGLE internal holdout (honest within train folds),
    # cheap 8-boot ensemble + logit -> blend on the internal-holdout rows. The
    # point PD only needs to be a good RANKER to form decile bins, so we do not
    # need the full 45-fit cross-fit; the isotonic just maps blend->prob.
    n_cal_boot = 8
    skf_in = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    itr, ite = next(iter(skf_in.split(X_tr, y_tr)))
    sub_scores = np.zeros((len(ite), n_cal_boot))
    rng_i = np.random.default_rng(seed + 1)
    ni = len(itr)
    for b in range(n_cal_boot):
        idx = rng_i.integers(0, ni, ni)
        m = make_model(seed + b + 1)
        m.fit(X_tr.iloc[itr].iloc[idx], y_tr[itr][idx])
        sub_scores[:, b] = m.predict_proba(X_tr.iloc[ite])[:, 1]
    lg = make_logit().fit(Z_tr.iloc[itr], y_tr[itr])
    lg_p = lg.predict_proba(Z_tr.iloc[ite])[:, 1]
    blend_cal = 0.4 * sub_scores.mean(axis=1) + 0.6 * lg_p
    iso_pt = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_pt.fit(blend_cal, y_tr[ite])

    # full ensemble on all train folds -> held-out fold AND back onto train rows
    # (the train raw_blend is needed to fit the INTERVAL isotonic for BASE and the
    # BB neighbourhood point PDs). We score train rows in-sample here (acceptable:
    # the interval isotonic is a calibration map, not a coverage estimate).
    tr_scores = np.zeros((n, N_BOOT))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = make_model(seed + b + 1)
        m.fit(X_tr.iloc[idx], y_tr[idx])
        te_scores[:, b] = m.predict_proba(X_te)[:, 1]
        tr_scores[:, b] = m.predict_proba(X_tr)[:, 1]
    lg = make_logit().fit(Z_tr, y_tr)
    lg_te = lg.predict_proba(Z_te)[:, 1]
    lg_tr = lg.predict_proba(Z_tr)[:, 1]
    raw_blend_te = 0.4 * te_scores.mean(axis=1) + 0.6 * lg_te
    raw_blend_train = 0.4 * tr_scores.mean(axis=1) + 0.6 * lg_tr
    pt = np.clip(iso_pt.predict(raw_blend_te), 1e-6, 1 - 1e-6)
    return pt, te_scores, iso_pt, raw_blend_train, raw_blend_te


# --------------------------------------------------------------------------- #
# BASELINE interval: bootstrap 5/95 pct, isotonic(interval)-mapped.
# --------------------------------------------------------------------------- #
def base_interval(te_scores, raw_blend_train, y_tr, raw_blend_te, pt):
    # interval isotonic is fit on the HGB-ensemble-mean scale in build_a; here we
    # fit one isotonic on the blend train scores (same calibrator family) and map
    # raw 5/95 percentiles of the ensemble through it.
    iso_iv = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_iv.fit(raw_blend_train, y_tr)
    lo_raw = np.quantile(te_scores, 0.05, axis=1)
    hi_raw = np.quantile(te_scores, 0.95, axis=1)
    lo = np.minimum(np.clip(iso_iv.predict(lo_raw), 1e-6, 1 - 1e-6), pt)
    hi = np.maximum(np.clip(iso_iv.predict(hi_raw), 1e-6, 1 - 1e-6), pt)
    return lo, hi


# --------------------------------------------------------------------------- #
# (b) GBQ: gradient-boosting quantile regression directly on the 0/1 outcome.
# Targets 0.05 / 0.95 quantile of y|X. For a Bernoulli, low quantile -> 0,
# high -> mostly 0 unless local rate high; produces a [lo,hi] band.
# --------------------------------------------------------------------------- #
def _gbq_model(alpha, seed):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, learning_rate=0.05, max_iter=300,
        max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=1.0,
        categorical_features="from_dtype", random_state=seed)


def fit_gbq(Xtr, ytr, Xte, alpha_lo=0.05, alpha_hi=0.95, seed=SEED):
    glo = _gbq_model(alpha_lo, seed).fit(Xtr, ytr)
    ghi = _gbq_model(alpha_hi, seed).fit(Xtr, ytr)
    lo = np.clip(glo.predict(Xte), 0, 1)
    hi = np.clip(ghi.predict(Xte), 0, 1)
    lo = np.minimum(lo, hi)
    return lo, hi, glo, ghi


# --------------------------------------------------------------------------- #
# (a) CQR: GBQ quantiles + conformal additive adjustment on a held-out calib
# split of the train folds. Conformity score E = max(qlo - y, y - qhi); the band
# becomes [qlo - Q, qhi + Q] with Q the (1-alpha) quantile of E on calib.
# --------------------------------------------------------------------------- #
def fit_cqr(Xtr, ytr, Xte, target=TARGET, seed=SEED):
    # split train folds into proper-train + calib
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_idx, cal_idx = next(iter(skf.split(Xtr, ytr)))
    lo_c, hi_c, glo, ghi = fit_gbq(Xtr.iloc[tr_idx], ytr[tr_idx],
                                   Xtr.iloc[cal_idx], seed=seed)
    ycal = ytr[cal_idx]
    E = np.maximum(lo_c - ycal, ycal - hi_c)
    n = len(E)
    q_level = min(1.0, np.ceil((n + 1) * target) / n)
    Q = np.quantile(E, q_level, method="higher")
    lo_te = np.clip(glo.predict(Xte) - Q, 0, 1)
    hi_te = np.clip(ghi.predict(Xte) + Q, 0, 1)
    lo_te = np.minimum(lo_te, hi_te)
    return lo_te, hi_te, Q


# --------------------------------------------------------------------------- #
# (c) Bayesian Beta-Binomial band. For each held-out row, find its k nearest
# neighbours IN POINT-PD SPACE among the train-fold rows, form a Beta posterior
# Beta(a0 + sum y_nbr, b0 + sum (1-y_nbr)) with a weak prior centred on the row's
# own point PD, and take the 5/95 quantiles of that Beta as the interval.
# This is a local-default-rate posterior; width shrinks where data is dense.
# --------------------------------------------------------------------------- #
def fit_betabinom(pt_tr, y_tr, pt_te, k=400, m0=20.0):
    """Beta-Binomial 5/95 band around each held-out point PD.

    Prior: Beta(m0*p_row, m0*(1-p_row)) (weak, centred on the row's own PD).
    Likelihood: the k nearest train rows in PD space (their 0/1 outcomes).
    """
    order = np.argsort(pt_tr)
    pt_tr_sorted = pt_tr[order]
    y_tr_sorted = y_tr[order]
    cumy = np.concatenate([[0], np.cumsum(y_tr_sorted)])
    lo = np.empty(len(pt_te))
    hi = np.empty(len(pt_te))
    n_tr = len(pt_tr_sorted)
    pos = np.searchsorted(pt_tr_sorted, pt_te)
    for i in range(len(pt_te)):
        p = pt_te[i]
        c = pos[i]
        a = max(0, c - k // 2)
        b = min(n_tr, a + k)
        a = max(0, b - k)
        s = cumy[b] - cumy[a]          # positives among neighbours
        n_nbr = b - a
        alpha = m0 * p + s
        betap = m0 * (1 - p) + (n_nbr - s)
        lo[i] = beta_dist.ppf(0.05, alpha, betap)
        hi[i] = beta_dist.ppf(0.95, alpha, betap)
    return lo, hi


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")

    cats = build_cat_dtypes(train, val, test)
    f_train = build_features(train, cats)

    obs = train["default_flag"].notna().to_numpy()
    X = f_train.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    print(f"[data] observed rows={obs.sum():,}  default rate={y.mean():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    methods = ["BASE", "GBQ", "CQR", "BB"]
    res = {m: {"d10": [], "d20": [], "w": [], "auc": []} for m in methods}
    # store one fold's decile tables for display
    sample_rows = {}

    for fi, (tr, te) in enumerate(skf.split(X, y)):
        print(f"\n===== FOLD {fi} (train {len(tr):,} / test {len(te):,}) =====")
        X_tr, X_te = X.iloc[tr], X.iloc[te].reset_index(drop=True)
        y_tr, y_te = y[tr], y[te]

        pt, te_scores, iso_pt, raw_blend_train, raw_blend_te = fit_point_pd(
            X_tr, y_tr, X_te, seed=SEED)
        auc = roc_auc_score(y_te, pt)

        # point-PD for the train rows (for BB neighbourhood) -- reuse iso on blend
        # train scores already calibrated (raw_blend_train is cross-fit raw).
        pt_tr = np.clip(iso_pt.predict(raw_blend_train), 1e-6, 1 - 1e-6)

        # --- BASE ---
        lo, hi = base_interval(te_scores, raw_blend_train, y_tr, raw_blend_te, pt)
        for tag, nb in (("d10", 10), ("d20", 20)):
            c, rows = binned_cov(pt, lo, hi, y_te, nb)
            res["BASE"][tag].append(c)
            if fi == 0 and nb == 10:
                sample_rows[("BASE", 10)] = rows
        res["BASE"]["w"].append(float(np.mean(hi - lo)))
        res["BASE"]["auc"].append(auc)
        print(f"  BASE: d10={res['BASE']['d10'][-1]:.2f} "
              f"d20={res['BASE']['d20'][-1]:.2f} width={res['BASE']['w'][-1]:.4f}")

        # --- GBQ ---
        lo, hi, _, _ = fit_gbq(X_tr, y_tr.astype(float), X_te, seed=SEED)
        for tag, nb in (("d10", 10), ("d20", 20)):
            c, rows = binned_cov(pt, lo, hi, y_te, nb)
            res["GBQ"][tag].append(c)
            if fi == 0 and nb == 10:
                sample_rows[("GBQ", 10)] = rows
        res["GBQ"]["w"].append(float(np.mean(hi - lo)))
        print(f"  GBQ : d10={res['GBQ']['d10'][-1]:.2f} "
              f"d20={res['GBQ']['d20'][-1]:.2f} width={res['GBQ']['w'][-1]:.4f}")

        # --- CQR ---
        lo, hi, Q = fit_cqr(X_tr, y_tr.astype(float), X_te, seed=SEED)
        for tag, nb in (("d10", 10), ("d20", 20)):
            c, rows = binned_cov(pt, lo, hi, y_te, nb)
            res["CQR"][tag].append(c)
            if fi == 0 and nb == 10:
                sample_rows[("CQR", 10)] = rows
        res["CQR"]["w"].append(float(np.mean(hi - lo)))
        print(f"  CQR : d10={res['CQR']['d10'][-1]:.2f} "
              f"d20={res['CQR']['d20'][-1]:.2f} width={res['CQR']['w'][-1]:.4f} (Q={Q:.3f})")

        # --- BB ---
        lo, hi = fit_betabinom(pt_tr, y_tr, pt, k=400, m0=20.0)
        for tag, nb in (("d10", 10), ("d20", 20)):
            c, rows = binned_cov(pt, lo, hi, y_te, nb)
            res["BB"][tag].append(c)
            if fi == 0 and nb == 10:
                sample_rows[("BB", 10)] = rows
        res["BB"]["w"].append(float(np.mean(hi - lo)))
        print(f"  BB  : d10={res['BB']['d10'][-1]:.2f} "
              f"d20={res['BB']['d20'][-1]:.2f} width={res['BB']['w'][-1]:.4f}")

    # --------- summary ---------
    print("\n" + "=" * 74)
    print("HONEST 5-FOLD OOF SUMMARY (decile coverage / 20-bin coverage / width)")
    print("=" * 74)
    print(f"{'method':6s} | {'d10 mean(+/-)':16s} | {'d20 mean(+/-)':16s} | "
          f"{'meanWidth(+/-)':16s}")
    for m in methods:
        d10 = np.array(res[m]["d10"]); d20 = np.array(res[m]["d20"])
        w = np.array(res[m]["w"])
        print(f"{m:6s} | {d10.mean():.2f} (+/-{d10.std():.2f})     | "
              f"{d20.mean():.2f} (+/-{d20.std():.2f})     | "
              f"{w.mean():.4f} (+/-{w.std():.4f})")
    print(f"\nper-fold d10: " + " | ".join(
        f"{m}={['%.1f'%x for x in res[m]['d10']]}" for m in methods))
    print(f"per-fold width: " + " | ".join(
        f"{m}={['%.3f'%x for x in res[m]['w']]}" for m in methods))
    print(f"point-PD OOF AUC (mean): {np.mean(res['BASE']['auc']):.4f}")

    print("\n--- FOLD 0 decile tables (pred / emp / [lo,hi]) ---")
    for m in methods:
        rows = sample_rows.get((m, 10))
        if rows is None:
            continue
        print(f"\n[{m}]")
        for i, ptm, emp, L, H, ok in rows:
            print(f"  bin {i:2d}: pred {ptm:.3f}  emp {emp:.3f}  "
                  f"[{L:.3f},{H:.3f}]  {'OK' if ok else 'MISS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
