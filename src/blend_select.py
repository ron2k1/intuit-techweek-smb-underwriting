"""Walk-forward selection of the A-blend weights -- chooses the convex combination
of {HGB, LightGBM, logistic} that maximizes OUT-OF-TIME AUC, never the val set.

Run:  python -m src.blend_select

WHY a third leg. The competitive review found us strongest on economics / threshold /
draw-capping / OOD design / A+B+C completeness, and trailing on exactly ONE axis: raw
point AUC-ROC. AUC is pure rank discrimination, so the no-regret way to lift it is to
add an algorithmically-distinct booster (LightGBM, leaf-wise) to the existing
HGB(depth-wise)+logistic blend: the two trees make different ranking errors, so a convex
average ranks better than either alone.

HOW we avoid overfitting the weights (the whole point of this file):
  * Weights are chosen on the SAME anchored walk-forward folds as src/backtest.py
    (fit on all labelled rows strictly before test month m, score month m), NOT on the
    val book. The val set stays untouched so it remains an honest final check.
  * AUC is rank-invariant, so per fold we fit each leg ONCE on its raw probabilities
    (no per-fold isotonic needed) and the chosen weights transfer to the calibrated
    production legs unchanged -- isotonic is monotone, it preserves each leg's ranking.
  * Selection criterion = MEAN fold AUC, with a NO-REGRET guard: the winning weights
    must not lose AUC versus the current 0.5/0.5 HGB+logit blend in ANY single fold.
    A lever that helps on average but hurts a fold is drift-fragile -> rejected.

Output: the recommended BLEND_W = (w_hgb, w_lgb, w_logit) to paste into build_a, plus
the per-fold AUC of the current blend vs the recommended blend so the lift is auditable.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data as D
from . import model as M
from . import build_a as A
from .backtest import N_FOLDS, MIN_TEST

# convex-weight grid step over the 2-simplex (w_logit = 1 - w_hgb - w_lgb)
GRID_STEP = 0.05
# the current shipped blend, for the no-regret comparison
CURRENT_W = (0.5, 0.0, 0.5)   # (hgb, lgb, logit)


def _fold_predictions() -> list[dict]:
    """For each walk-forward fold, the per-leg probabilities + truth on the held-out
    month. One fit per leg per fold (raw probs; AUC is rank-based)."""
    tr, feats_all = D.load_features("train")
    feats = A.feats_for_a(feats_all)
    cat_idx = D.categorical_indices(feats)
    ts = pd.to_datetime(tr["application_timestamp"], errors="coerce")
    y = D.target_vector(tr)
    lab = y.notna().to_numpy()
    X = D.to_model_matrix(tr, feats).to_numpy()
    month = ts.dt.to_period("M")
    lab_months = sorted(month[lab].dropna().unique())
    test_months = lab_months[-N_FOLDS:]

    folds = []
    for m in test_months:
        tr_mask = lab & (month < m).to_numpy()
        te_mask = lab & (month == m).to_numpy()
        if te_mask.sum() < MIN_TEST or tr_mask.sum() < 2000:
            continue
        Xtr, ytr = X[tr_mask], y[tr_mask].astype(int).to_numpy()
        Xte = X[te_mask]
        yte = y[te_mask].astype(int).to_numpy()
        p_hgb = M.hgb(A.SEED, cat_idx, scoring="roc_auc").fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        p_lgb = M.lgbm(A.SEED, cat_idx).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        p_lg = A.logit_pipe().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        folds.append({"month": str(m), "y": yte,
                      "hgb": p_hgb, "lgb": p_lgb, "logit": p_lg})
        print(f"  fold {str(m)}  n={te_mask.sum():5d}  "
              f"AUC hgb={roc_auc_score(yte, p_hgb):.4f} "
              f"lgb={roc_auc_score(yte, p_lgb):.4f} logit={roc_auc_score(yte, p_lg):.4f}")
    return folds


def _blend_auc(folds: list[dict], w: tuple[float, float, float]) -> np.ndarray:
    """Per-fold AUC of the blend w=(hgb,lgb,logit)."""
    wh, wl, wg = w
    return np.array([roc_auc_score(f["y"], wh * f["hgb"] + wl * f["lgb"] + wg * f["logit"])
                     for f in folds])


def main() -> None:
    print(f"walk-forward blend selection ({N_FOLDS} folds, grid step {GRID_STEP}):")
    folds = _fold_predictions()
    if not folds:
        print("no usable folds")
        return

    base_auc = _blend_auc(folds, CURRENT_W)
    base_mean = base_auc.mean()
    print(f"\ncurrent blend {CURRENT_W}: mean fold AUC={base_mean:.4f} "
          f"per-fold={np.round(base_auc, 4).tolist()}")

    steps = [round(i * GRID_STEP, 4) for i in range(int(1 / GRID_STEP) + 1)]
    cands = []
    for wh, wl in itertools.product(steps, steps):
        wg = round(1.0 - wh - wl, 4)
        if wg < -1e-9:
            continue
        wg = max(wg, 0.0)
        auc = _blend_auc(folds, (wh, wl, wg))
        cands.append({"w": (wh, wl, wg), "mean": auc.mean(),
                      "min_lift": (auc - base_auc).min(), "per_fold": auc})

    # rank by mean fold AUC; report the best, and the best that is ALSO no-regret
    cands.sort(key=lambda c: c["mean"], reverse=True)
    best = cands[0]
    noregret = next((c for c in cands if c["min_lift"] >= 0.0), None)

    print(f"\nbest mean AUC      : w={best['w']} mean={best['mean']:.4f} "
          f"(+{best['mean']-base_mean:.4f})  worst-fold lift={best['min_lift']:+.4f}")
    if noregret is not None:
        print(f"best NO-REGRET     : w={noregret['w']} mean={noregret['mean']:.4f} "
              f"(+{noregret['mean']-base_mean:.4f})  worst-fold lift={noregret['min_lift']:+.4f}")
        print(f"  per-fold AUC     : {np.round(noregret['per_fold'], 4).tolist()}")
        print(f"\nRECOMMEND BLEND_W = {noregret['w']}  (hgb, lgb, logit)  "
              f"-- maximizes mean OOF AUC subject to no per-fold regret")
    else:
        print("no no-regret weighting beats the current blend in every fold; KEEP current.")

    # top-5 for visibility
    print("\ntop-5 by mean fold AUC:")
    for c in cands[:5]:
        print(f"  w={c['w']}  mean={c['mean']:.4f}  worst-fold lift={c['min_lift']:+.4f}")


if __name__ == "__main__":
    main()
