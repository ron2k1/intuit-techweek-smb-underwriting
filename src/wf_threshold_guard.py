#!/usr/bin/env python3
"""WALK-FORWARD anti-overfitting guard for the Deliverable-A funding cut.

Question set:
  (1) Sort observed rows by application_timestamp; do EXPANDING-WINDOW walk-forward
      (train on all past, evaluate the next time block; ~5 evaluated blocks).
  (2) Per block report OOF AUC and REALIZED amortizing profit at:
        (i)  the flat closed-form break-even cut 0.226 (calibrated PD), and
        (ii) the IN-SAMPLE validation-argmax threshold computed on the PRIOR block
             (a profit-maximizing cut fit on the previous block's own outcomes) to
             show it is overfit / unstable out of sample.
  (3) Confirm: is 0.226 inside the profit-FLAT region each block (profit within a
      small tolerance of that block's own oracle-best cut)?  Does the prior-block
      argmax threshold swing across blocks?
  (4) Does adding RECENCY sample-weights (exp decay, ~6-month half-life) to the
      model fit change the LATEST-block AUC or profit?

All thresholds act on the CALIBRATED PD. Calibration is fit OUT-OF-FOLD: for each
evaluated block we calibrate the model score using an isotonic map trained on the
held-out tail of the train window (never on the evaluated block).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import build_cat_dtypes, build_features  # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path(__file__).resolve().parent.parent / "dataset"
SEED = 17
TERM_INT, FEE = 0.35 * 60 / 365, 0.03
NET_MARGIN = TERM_INT + FEE
LGD = 0.30
BE = NET_MARGIN / (NET_MARGIN + LGD)   # ~0.2257
N_BLOCKS = 6                           # 1 warmup + 5 evaluated blocks
HALF_LIFE_DAYS = 182.5                 # ~6-month half-life for recency weights


def realized_profit(amt, dflag, dtd, rec):
    """Per-loan REALIZED amortizing profit (matches build_a)."""
    rec = np.nan_to_num(rec)
    frac = np.clip(np.minimum(np.nan_to_num(dtd), 60) / 60.0, 0, 1)
    dp = np.minimum(amt * (FEE + frac * (1 + TERM_INT) - 1) + rec, amt * NET_MARGIN)
    return np.where(dflag == 0, amt * NET_MARGIN, dp)


def clf(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        categorical_features="from_dtype", random_state=seed)


def argmax_threshold(cal_pd, prof, grid):
    """In-sample profit-maximizing constant cut on calibrated PD."""
    best_t, best_p = grid[0], -1e18
    for t in grid:
        p = prof[cal_pd < t].sum()
        if p > best_p:
            best_t, best_p = t, p
    return best_t, best_p


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)

    # Sort ALL train rows by application_timestamp (test is the future).
    ts = pd.to_datetime(train["application_timestamp"])
    order = np.argsort(ts.to_numpy(), kind="mergesort")
    train = train.iloc[order].reset_index(drop=True)
    ts = ts.iloc[order].reset_index(drop=True)
    day_idx = (ts - ts.min()).dt.total_seconds().to_numpy() / 86400.0

    X = build_features(train, cats)
    obs = train["default_flag"].notna().to_numpy()
    obs_idx = np.where(obs)[0]
    y = train["default_flag"].to_numpy()
    amt = train["requested_amount"].to_numpy()
    prof = realized_profit(amt, y, train["days_to_default"].to_numpy(),
                           train["final_recovered_amount"].to_numpy())

    grid = np.round(np.linspace(0.10, 0.40, 61), 4)  # threshold search grid

    print(f"break-even (closed-form) BE = {BE:.4f}")
    print(f"observed rows = {obs_idx.size:,}; time range "
          f"{ts.min().date()} .. {ts.max().date()}")

    # Expanding-window blocks over OBSERVED rows (chronological).
    blocks = np.array_split(obs_idx, N_BLOCKS)
    for b in range(N_BLOCKS):
        sub = blocks[b]
        print(f"  block {b}: n={sub.size:5d}  "
              f"time {ts.iloc[sub].min().date()}..{ts.iloc[sub].max().date()}  "
              f"default_rate={y[sub].mean():.4f}")

    # --- Per-block walk-forward -------------------------------------------- #
    print("\n" + "=" * 100)
    print("EXPANDING-WINDOW WALK-FORWARD  (train=past observed, eval=next block)")
    print("=" * 100)
    header = (f"{'blk':>3} {'n_eval':>6} {'AUC':>7} {'oracle_t':>8} "
              f"{'p@oracle':>11} {'p@BE0.226':>11} {'gap_BE%':>8} "
              f"{'prior_argmax_t':>14} {'p@prior_t':>11} {'gap_prior%':>10}")
    print(header)

    prior_argmax_t = None       # argmax cut learned on the PRIOR block
    rows = []
    for b in range(1, N_BLOCKS):
        tr = np.concatenate(blocks[:b])   # all past observed
        te = blocks[b]
        ytr, yte = y[tr], y[te]
        if len(np.unique(yte)) < 2:
            continue

        # Hold out the most-recent tail of train for isotonic calibration so the
        # calibration map is fit out-of-fold w.r.t. the evaluated block.
        n_cal = max(2000, int(0.15 * tr.size))
        cal_idx = tr[-n_cal:]
        fit_idx = tr[:-n_cal]

        m = clf(SEED)
        m.fit(X.iloc[fit_idx], y[fit_idx])
        s_cal = m.predict_proba(X.iloc[cal_idx])[:, 1]
        s_te = m.predict_proba(X.iloc[te])[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(s_cal, y[cal_idx])
        cal_te = np.clip(iso.predict(s_te), 1e-6, 1 - 1e-6)

        auc = roc_auc_score(yte, s_te)

        # Block's own oracle-best cut (upper bound; uses block's OWN outcomes).
        oracle_t, p_oracle = argmax_threshold(cal_te, prof[te], grid)
        p_be = prof[te][cal_te < BE].sum()
        gap_be = 100.0 * (p_oracle - p_be) / abs(p_oracle) if p_oracle else 0.0

        # In-sample validation-argmax threshold = argmax fit on the PRIOR block,
        # applied here (out of sample) to expose overfit/instability.
        if prior_argmax_t is not None:
            p_prior = prof[te][cal_te < prior_argmax_t].sum()
            gap_prior = 100.0 * (p_oracle - p_prior) / abs(p_oracle) if p_oracle else 0.0
            pt_str = f"{prior_argmax_t:14.4f}"
            pp_str = f"{p_prior:11,.0f}"
            gp_str = f"{gap_prior:10.2f}"
        else:
            p_prior, gap_prior = np.nan, np.nan
            pt_str, pp_str, gp_str = f"{'--':>14}", f"{'--':>11}", f"{'--':>10}"

        print(f"{b:>3d} {te.size:>6d} {auc:7.4f} {oracle_t:8.3f} "
              f"{p_oracle:11,.0f} {p_be:11,.0f} {gap_be:8.2f} "
              f"{pt_str} {pp_str} {gp_str}")
        rows.append(dict(blk=b, auc=auc, oracle_t=oracle_t, p_oracle=p_oracle,
                         p_be=p_be, gap_be=gap_be, prior_t=prior_argmax_t,
                         p_prior=p_prior, gap_prior=gap_prior))

        # Compute THIS block's in-sample argmax cut to carry to the next block.
        prior_argmax_t, _ = argmax_threshold(cal_te, prof[te], grid)

    # --- Summary of the threshold guard ------------------------------------ #
    print("\n" + "-" * 100)
    oracle_ts = [r["oracle_t"] for r in rows]
    prior_ts = [r["prior_t"] for r in rows if r["prior_t"] is not None]
    gaps_be = [r["gap_be"] for r in rows]
    gaps_prior = [r["gap_prior"] for r in rows if not np.isnan(r["gap_prior"])]
    print(f"per-block oracle-best cut : {['%.3f' % t for t in oracle_ts]}  "
          f"(min {min(oracle_ts):.3f}, max {max(oracle_ts):.3f}, "
          f"swing {max(oracle_ts)-min(oracle_ts):.3f})")
    print(f"prior-block argmax cut    : {['%.3f' % t for t in prior_ts]}  "
          f"(min {min(prior_ts):.3f}, max {max(prior_ts):.3f}, "
          f"swing {max(prior_ts)-min(prior_ts):.3f})")
    print(f"BE-0.226 profit gap vs oracle (per block %): "
          f"{['%.2f' % g for g in gaps_be]}  -> mean {np.mean(gaps_be):.2f}%, "
          f"max {max(gaps_be):.2f}%")
    if gaps_prior:
        print(f"prior-argmax profit gap vs oracle (per block %): "
              f"{['%.2f' % g for g in gaps_prior]}  -> mean {np.mean(gaps_prior):.2f}%, "
              f"max {max(gaps_prior):.2f}%")

    # --- Flat-region check around 0.226 on the LATEST block ---------------- #
    print("\n" + "=" * 100)
    print("PROFIT-FLAT REGION around the cut, LATEST evaluated block (most test-like)")
    print("=" * 100)
    tr = np.concatenate(blocks[:N_BLOCKS - 1]); te = blocks[N_BLOCKS - 1]
    n_cal = max(2000, int(0.15 * tr.size))
    cal_idx, fit_idx = tr[-n_cal:], tr[:-n_cal]
    m = clf(SEED); m.fit(X.iloc[fit_idx], y[fit_idx])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(X.iloc[cal_idx])[:, 1], y[cal_idx])
    cal_te = np.clip(iso.predict(m.predict_proba(X.iloc[te])[:, 1]), 1e-6, 1 - 1e-6)
    p_oracle = max(prof[te][cal_te < t].sum() for t in grid)
    for t in [0.180, 0.200, 0.210, 0.2257, 0.240, 0.260, 0.280, 0.300]:
        p = prof[te][cal_te < t].sum()
        gap = 100.0 * (p_oracle - p) / abs(p_oracle)
        flag = "  <-- BE 0.226" if abs(t - 0.2257) < 1e-6 else ""
        print(f"  cut={t:.4f}  funded={int((cal_te<t).sum()):5d}  "
              f"profit={p:12,.0f}  gap_vs_oracle={gap:6.2f}%{flag}")

    # --- Recency weighting on the latest block ----------------------------- #
    print("\n" + "=" * 100)
    print("RECENCY SAMPLE-WEIGHTS (exp decay, ~6mo half-life) on the LATEST block")
    print("=" * 100)
    print(f"  weight = 0.5 ** (age_days / {HALF_LIFE_DAYS:.0f})")
    for b in range(1, N_BLOCKS):
        tr = np.concatenate(blocks[:b]); te = blocks[b]
        if len(np.unique(y[te])) < 2:
            continue
        n_cal = max(2000, int(0.15 * tr.size))
        cal_idx, fit_idx = tr[-n_cal:], tr[:-n_cal]

        def fit_eval(weighted):
            m = clf(SEED)
            if weighted:
                age = day_idx[fit_idx].max() - day_idx[fit_idx]
                w = 0.5 ** (age / HALF_LIFE_DAYS)
                m.fit(X.iloc[fit_idx], y[fit_idx], sample_weight=w)
            else:
                m.fit(X.iloc[fit_idx], y[fit_idx])
            s_cal = m.predict_proba(X.iloc[cal_idx])[:, 1]
            s_te = m.predict_proba(X.iloc[te])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(s_cal, y[cal_idx])
            cte = np.clip(iso.predict(s_te), 1e-6, 1 - 1e-6)
            return roc_auc_score(y[te], s_te), prof[te][cte < BE].sum()

        a0, p0 = fit_eval(False)
        a1, p1 = fit_eval(True)
        tag = "  <-- LATEST" if b == N_BLOCKS - 1 else ""
        print(f"  block {b}: AUC base={a0:.4f} recency={a1:.4f} (d={a1-a0:+.4f})  "
              f"profit@BE base={p0:12,.0f} recency={p1:12,.0f} "
              f"(d={p1-p0:+,.0f}){tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
