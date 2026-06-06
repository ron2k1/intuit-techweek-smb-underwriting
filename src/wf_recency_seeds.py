#!/usr/bin/env python3
"""Seed-robustness check for recency weighting on the latest (test-like) block.
Is the small AUC/profit bump from ~6mo half-life recency weights real or 1-seed noise?
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
TERM_INT, FEE = 0.35 * 60 / 365, 0.03
NET_MARGIN = TERM_INT + FEE
LGD = 0.30
BE = NET_MARGIN / (NET_MARGIN + LGD)
N_BLOCKS = 6
HALF_LIFE_DAYS = 182.5


def realized_profit(amt, dflag, dtd, rec):
    rec = np.nan_to_num(rec)
    frac = np.clip(np.minimum(np.nan_to_num(dtd), 60) / 60.0, 0, 1)
    dp = np.minimum(amt * (FEE + frac * (1 + TERM_INT) - 1) + rec, amt * NET_MARGIN)
    return np.where(dflag == 0, amt * NET_MARGIN, dp)


def clf(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        categorical_features="from_dtype", random_state=seed)


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    ts = pd.to_datetime(train["application_timestamp"])
    order = np.argsort(ts.to_numpy(), kind="mergesort")
    train = train.iloc[order].reset_index(drop=True)
    ts = ts.iloc[order].reset_index(drop=True)
    day_idx = (ts - ts.min()).dt.total_seconds().to_numpy() / 86400.0
    X = build_features(train, cats)
    obs_idx = np.where(train["default_flag"].notna().to_numpy())[0]
    y = train["default_flag"].to_numpy()
    amt = train["requested_amount"].to_numpy()
    prof = realized_profit(amt, y, train["days_to_default"].to_numpy(),
                           train["final_recovered_amount"].to_numpy())
    blocks = np.array_split(obs_idx, N_BLOCKS)
    tr = np.concatenate(blocks[:N_BLOCKS - 1]); te = blocks[N_BLOCKS - 1]
    n_cal = max(2000, int(0.15 * tr.size))
    cal_idx, fit_idx = tr[-n_cal:], tr[:-n_cal]
    age = day_idx[fit_idx].max() - day_idx[fit_idx]
    w = 0.5 ** (age / HALF_LIFE_DAYS)

    print("LATEST block, recency vs base across seeds (~6mo half-life):")
    dA, dP = [], []
    for seed in [17, 23, 42, 101, 2024]:
        def fe(weighted):
            m = clf(seed)
            m.fit(X.iloc[fit_idx], y[fit_idx], sample_weight=(w if weighted else None))
            sc = m.predict_proba(X.iloc[cal_idx])[:, 1]
            stx = m.predict_proba(X.iloc[te])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(sc, y[cal_idx])
            cte = np.clip(iso.predict(stx), 1e-6, 1 - 1e-6)
            return roc_auc_score(y[te], stx), prof[te][cte < BE].sum()
        a0, p0 = fe(False); a1, p1 = fe(True)
        dA.append(a1 - a0); dP.append(p1 - p0)
        print(f"  seed {seed:>4}: dAUC={a1-a0:+.4f}  dProfit={p1-p0:+,.0f}")
    print(f"\n  mean dAUC={np.mean(dA):+.4f} (std {np.std(dA):.4f})  "
          f"mean dProfit={np.mean(dP):+,.0f} (std {np.std(dP):,.0f})")
    print(f"  dAUC sign: {sum(d>0 for d in dA)}/5 positive; "
          f"dProfit sign: {sum(d>0 for d in dP)}/5 positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
