"""Walk-forward (rolling-origin) backtest -- the overfitting + drift diagnostic.

Run:  python -m src.backtest

WHY this exists (verified against the real data in scratch/probe_temporal.py):
  * train = 2024-01..2025-06 (the PAST, 18 months); val/test = the FUTURE 13-week
    cohort window. So train->val is ALREADY an out-of-time split -- classic
    memorization is not the risk (model is regularized; no business_id leakage).
  * BUT the train default rate DRIFTS up monotonically 15.2% (2024Q1) -> 21.1%
    (2025Q2). A model fit on the full-history AVERAGE (~17.5%) under-prices risk
    in the high-default deployment window. That is the real "overfitting": fitting
    a stale regime, not noise.

What this measures (the two things a single val split cannot tell you):
  1. STABILITY of the profit-max threshold tau across time. build_a picks tau as a
     single argmax on ~2.5k val rows. If the walk-forward tau swings, that argmax is
     overfit and a robust aggregate (median across folds) is the honest choice.
  2. CALIBRATION DRIFT per fold: mean predicted PD vs realized default rate on the
     held-out NEXT block. Systematic under-prediction in later folds = the drift
     miscalibration, and tells us how much to correct for deployment.

Design: anchored expanding window. For each forward test month m in the last
N_FOLDS months, fit on ALL labelled rows strictly before m, score month m. Then a
final DEPLOYMENT fold = fit all train, score the labelled val book (the truest
single backtest of what we actually ship). A RESUB row (fit all, score the same
rows) bounds the memorization gap from above.

Speed: uses the plain regularized booster (model.hgb), not the calibrated blend, so
~1 fit/fold. AUC is rank-based (calibration-invariant); for the tau/calibration
readouts the booster's log_loss probabilities are accurate enough to show direction
and magnitude. Production A still uses the calibrated HGB+logit blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data as D
from . import model as M
from . import build_a as A

N_FOLDS = 9                # test on each of the last 9 train months (2024-10..2025-06)
MIN_TEST = 500             # skip a fold if the test month has too few labelled rows


def best_tau(pd_hat: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    """Profit-max threshold on one fold: the tau that maximizes summed realized
    value over approved (pd_hat < tau) loans. Mirrors build_a's argmax exactly."""
    cands = np.unique(np.round(pd_hat, 4))
    best, tau = -np.inf, float(cands[-1]) + 1e-6
    for tc in cands:
        tot = float(np.nansum(value[pd_hat < tc]))
        if tot > best:
            best, tau = tot, float(tc)
    return tau, best


def main() -> None:
    tr, feats_all = D.load_features("train")
    va, _ = D.load_features("val")
    feats = A.feats_for_a(feats_all)
    cat_idx = D.categorical_indices(feats)

    ts = pd.to_datetime(tr["application_timestamp"], errors="coerce")
    y = D.target_vector(tr)
    lab = y.notna().to_numpy()
    X = D.to_model_matrix(tr, feats).to_numpy()
    amt = tr["requested_amount"].to_numpy()
    dd = pd.to_numeric(tr["days_to_default"], errors="coerce").to_numpy()
    month = ts.dt.to_period("M")

    # forward test months = the last N_FOLDS distinct months that have labels
    lab_months = sorted(month[lab].dropna().unique())
    test_months = lab_months[-N_FOLDS:]
    tau_th = A.theoretical_tau()
    print(f"walk-forward: {len(test_months)} folds  test months "
          f"{test_months[0]}..{test_months[-1]}  closed-form break-even tau={tau_th:.4f}\n")
    print(f"{'fold(test)':>10} {'train_n':>8} {'test_n':>7} {'AUC':>6} "
          f"{'pred_pd':>8} {'real_dr':>8} {'calib_gap':>10} {'tau*':>7}")

    rows = []
    for m in test_months:
        tr_mask = lab & (month < m).to_numpy()
        te_mask = lab & (month == m).to_numpy()
        if te_mask.sum() < MIN_TEST or tr_mask.sum() < 2000:
            continue
        clf = M.hgb(A.SEED, cat_idx, scoring="roc_auc").fit(X[tr_mask], y[tr_mask].astype(int).to_numpy())
        p = clf.predict_proba(X[te_mask])[:, 1]
        yt = y[te_mask].astype(int).to_numpy()
        auc = float(roc_auc_score(yt, p))
        npv = A.realized_npv(amt[te_mask], yt.astype(float), dd[te_mask])
        tau, _ = best_tau(p, npv)
        pred_pd, real_dr = float(p.mean()), float(yt.mean())
        rows.append({"month": str(m), "auc": auc, "pred_pd": pred_pd,
                     "real_dr": real_dr, "calib_gap": pred_pd - real_dr, "tau": tau})
        print(f"{str(m):>10} {tr_mask.sum():>8} {te_mask.sum():>7} {auc:>6.3f} "
              f"{pred_pd:>8.3f} {real_dr:>8.3f} {pred_pd-real_dr:>+10.3f} {tau:>7.4f}")

    R = pd.DataFrame(rows)

    # --- DEPLOYMENT fold: fit all train, score the labelled val book (truest backtest) ---
    yva = D.target_vector(va)
    lva = yva.notna().to_numpy()
    Xva = D.to_model_matrix(va, feats).to_numpy()
    clf = M.hgb(A.SEED, cat_idx, scoring="roc_auc").fit(X[lab], y[lab].astype(int).to_numpy())
    pv = clf.predict_proba(Xva[lva])[:, 1]
    yv = yva[lva].astype(int).to_numpy()
    auc_dep = roc_auc_score(yv, pv)
    amt_va = va["requested_amount"].to_numpy()[lva]
    dd_va = pd.to_numeric(va["days_to_default"], errors="coerce").to_numpy()[lva]
    npv_va = A.realized_npv(amt_va, yv.astype(float), dd_va)
    tau_dep, _ = best_tau(pv, npv_va)
    auc_dep = float(auc_dep)

    # --- RESUB: fit all, score same labelled train (upper bound on memorization) ---
    auc_resub = float(roc_auc_score(y[lab].astype(int).to_numpy(), clf.predict_proba(X[lab])[:, 1]))

    print("\n=== SUMMARY ===")
    print(f"walk-forward AUC      : mean={R.auc.mean():.4f}  std={R.auc.std():.4f}  "
          f"min={R.auc.min():.4f}  max={R.auc.max():.4f}")
    print(f"deployment AUC (val)  : {auc_dep:.4f}   resub AUC (train): {auc_resub:.4f}   "
          f"memorization gap (resub - walkfwd) = {auc_resub - R.auc.mean():+.4f}")
    print(f"calibration drift     : early folds gap={R.calib_gap.iloc[:3].mean():+.3f}  "
          f"late folds gap={R.calib_gap.iloc[-3:].mean():+.3f}  "
          f"(negative = model UNDER-predicts risk -> drift miscalibration)")
    print(f"tau* stability        : mean={R.tau.mean():.4f}  median={R.tau.median():.4f}  "
          f"std={R.tau.std():.4f}  range=[{R.tau.min():.4f}, {R.tau.max():.4f}]")
    print(f"tau* deployment(val)  : {tau_dep:.4f}   closed-form break-even: {tau_th:.4f}")
    print(f"\nVERDICT:")
    gap = auc_resub - R.auc.mean()
    print(f"  - memorization: resub-vs-walkforward AUC gap = {gap:+.4f} "
          f"({'LOW (well-regularized)' if gap < 0.03 else 'WATCH'})")
    print(f"  - threshold overfit: build_a single-argmax tau vs walk-forward median "
          f"-> use median={R.tau.median():.4f} if it differs materially from a single fold")
    drift = R.calib_gap.iloc[-3:].mean()
    print(f"  - drift: late-fold calibration gap = {drift:+.3f} "
          f"({'UNDER-prices risk; recency-weight or drift-correct calibration' if drift < -0.01 else 'OK'})")


if __name__ == "__main__":
    main()
