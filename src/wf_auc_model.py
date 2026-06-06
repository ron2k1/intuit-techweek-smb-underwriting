#!/usr/bin/env python3
"""wf_auc_model — validate AUC-maximization levers with 5-fold OOF.

Compares, all via 5-fold stratified OOF on the OBSERVED train rows:
  (a) baseline HistGBT (deployed recipe)
  (b) regularized Logistic (median-impute + StandardScaler + LogisticRegression)
  (c) blends HGB*w + Logit*(1-w) for w in {0.4..0.9}
  (d) HGB WITHOUT prior_underwriter_score (drop the column + its __isna)

Reports per-fold OOF AUC for every variant and the best blend weight.
Run: .venv/Scripts/python.exe -m src.wf_auc_model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import build_cat_dtypes, build_features  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SEED = 17
N_SPLITS = 5


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
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


def to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Category cols -> .cat.codes (treating -1/NaN); numeric kept. Returns float frame.
    Median imputation done downstream per-fold (fit on train fold only)."""
    out = pd.DataFrame(index=X.index)
    for c in X.columns:
        col = X[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            codes = col.cat.codes.astype("float64")
            codes[codes < 0] = np.nan  # NaN category -> impute later
            out[c] = codes
        else:
            out[c] = pd.to_numeric(col, errors="coerce").astype("float64")
    return out


def fit_logit_oof(Xnum_tr, y_tr, Xnum_te):
    """Median-impute (fit on train) + StandardScaler + LogisticRegression(C=0.5)."""
    med = Xnum_tr.median(axis=0)
    Xtr = Xnum_tr.fillna(med).to_numpy()
    Xte = Xnum_te.fillna(med).to_numpy()
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xte = sc.transform(Xte)
    clf = LogisticRegression(C=0.5, max_iter=3000)
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)[:, 1]


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)

    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    print(f"[data] {obs.sum():,} observed rows, default rate {y.mean():.4f}, "
          f"{Xo.shape[1]} features")

    # Drop-score variant feature frame: remove prior_underwriter_score AND its isna ind.
    drop_cols = [c for c in Xo.columns if c.startswith("prior_underwriter_score")]
    Xo_nops = Xo.drop(columns=drop_cols)
    print(f"[drop-score] dropping {drop_cols} -> {Xo_nops.shape[1]} features")

    # Numeric frame for logistic.
    Xnum = to_numeric_frame(Xo)

    blend_ws = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # OOF prediction accumulators.
    oof_hgb = np.zeros(len(y))
    oof_logit = np.zeros(len(y))
    oof_hgb_nops = np.zeros(len(y))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_rows = []
    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        # (a) baseline HGB
        m = make_hgb(SEED + k)
        m.fit(Xo.iloc[tr], y[tr])
        p_hgb = m.predict_proba(Xo.iloc[te])[:, 1]
        oof_hgb[te] = p_hgb

        # (d) HGB without prior_underwriter_score
        m2 = make_hgb(SEED + k)
        m2.fit(Xo_nops.iloc[tr], y[tr])
        p_hgb_nops = m2.predict_proba(Xo_nops.iloc[te])[:, 1]
        oof_hgb_nops[te] = p_hgb_nops

        # (b) logistic
        p_logit = fit_logit_oof(Xnum.iloc[tr], y[tr], Xnum.iloc[te])
        oof_logit[te] = p_logit

        yte = y[te]
        a_hgb = roc_auc_score(yte, p_hgb)
        a_logit = roc_auc_score(yte, p_logit)
        a_nops = roc_auc_score(yte, p_hgb_nops)
        row = {"fold": k, "n_te": len(te), "hgb": a_hgb, "logit": a_logit,
               "hgb_nops": a_nops}
        for w in blend_ws:
            blend = w * p_hgb + (1 - w) * p_logit
            row[f"blend_{w}"] = roc_auc_score(yte, blend)
        fold_rows.append(row)
        msg = (f"[fold {k}] n={len(te):,}  HGB={a_hgb:.4f}  Logit={a_logit:.4f}  "
               f"HGB_noPS={a_nops:.4f}  " +
               "  ".join(f"b{w}={row[f'blend_{w}']:.4f}" for w in blend_ws))
        print(msg)

    fdf = pd.DataFrame(fold_rows)

    # Global OOF AUC (pooled across all folds — the headline metric).
    print("\n" + "=" * 78)
    print("POOLED OOF AUC (single AUC over all OOF predictions):")
    g_hgb = roc_auc_score(y, oof_hgb)
    g_logit = roc_auc_score(y, oof_logit)
    g_nops = roc_auc_score(y, oof_hgb_nops)
    print(f"  HGB baseline ............... {g_hgb:.4f}")
    print(f"  Logistic ................... {g_logit:.4f}")
    print(f"  HGB without prior_uw_score . {g_nops:.4f}   (delta vs HGB: {g_nops-g_hgb:+.4f})")
    pooled_blend = {}
    for w in blend_ws:
        gb = roc_auc_score(y, w * oof_hgb + (1 - w) * oof_logit)
        pooled_blend[w] = gb
        print(f"  Blend w={w} (HGB*w+Logit*(1-w)) {gb:.4f}   (delta vs HGB: {gb-g_hgb:+.4f})")

    # Per-fold mean +- std for each variant.
    print("\n" + "=" * 78)
    print("PER-FOLD MEAN +- STD AUC:")
    cols = ["hgb", "logit", "hgb_nops"] + [f"blend_{w}" for w in blend_ws]
    for c in cols:
        print(f"  {c:14s} mean={fdf[c].mean():.4f}  std={fdf[c].std(ddof=1):.4f}  "
              f"min={fdf[c].min():.4f}  max={fdf[c].max():.4f}")

    # Best blend by per-fold mean, and paired delta vs HGB per fold.
    best_w = max(blend_ws, key=lambda w: fdf[f"blend_{w}"].mean())
    print("\n" + "=" * 78)
    print(f"BEST BLEND by per-fold mean: w={best_w}")
    paired = fdf[f"blend_{best_w}"] - fdf["hgb"]
    print(f"  per-fold (blend_{best_w} - HGB): "
          + ", ".join(f"{v:+.4f}" for v in paired))
    print(f"  mean paired delta = {paired.mean():+.5f}  std = {paired.std(ddof=1):.5f}  "
          f"min = {paired.min():+.5f}")
    print(f"  ALL folds positive? {bool((paired > 0).all())}")

    drop_paired = fdf["hgb_nops"] - fdf["hgb"]
    print("\nDROP prior_underwriter_score, per-fold (HGB_noPS - HGB): "
          + ", ".join(f"{v:+.4f}" for v in drop_paired))
    print(f"  mean = {drop_paired.mean():+.5f}  -> dropping {'HURTS' if drop_paired.mean()<0 else 'helps'}")

    # Save per-fold table.
    outp = REPO / "reports" / "wf_auc_model_folds.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(outp, index=False)
    print(f"\n[written] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
