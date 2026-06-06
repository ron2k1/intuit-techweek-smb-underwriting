#!/usr/bin/env python3
"""wf_intervals.py -- validate a CONFORMAL / tighter 90% PD interval for Deliverable A.

LEVER: tighten the 90% PD interval (currently bootstrap 5th/95th pct, mean width
~0.131 on val) WITHOUT losing coverage, via a conformal half-width lambda on the
bootstrap-ensemble std, and an OOD widening term.

Two coverage notions are reported because the task names both:
  (1) PER-ROW 0/1 coverage  = fraction of labeled rows whose realized default
      (0 or 1) lies in [lower, upper].  This is DEGENERATE for a PD interval
      (a probability band ~[0.05,0.4] almost never contains a 0/1 point), so we
      quantify exactly how degenerate it is and what width it would force.
  (2) DECILE-BINNED coverage = sort by point PD, split into 10 bins, check whether
      the bin's empirical default rate lies in [mean lower, mean upper].  This is
      build_a's existing report and the operationally meaningful calibration check.

We conformalize the half-width with HONEST estimation: lambda (and the OOD term)
are chosen on out-of-fold residuals via 5-fold CV over the labeled validation rows
AND via a time-ordered split-conformal, so coverage is measured on rows never used
to pick lambda. We report PER-FOLD numbers.

Run:
  .venv/Scripts/python.exe src/wf_intervals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(r"C:\Users\ayush\intuit-hackathon\intuit-techweek-smb-underwriting")
DATA = REPO / "dataset"
sys.path.insert(0, str(REPO / "src"))
from build_a import build_cat_dtypes, build_features, make_model  # noqa: E402

SEED = 17
N_BOOT = 15          # task asks for ~15
TARGET_COV = 0.90
OOD_CUT = 0.273


def fit_ensemble(X_obs, y, f_eval_list, n_boot=N_BOOT, seed=SEED):
    """Return list of (n_eval, n_boot) score matrices, one per eval frame."""
    n = len(X_obs)
    rng = np.random.default_rng(seed)
    out = [np.zeros((len(f), n_boot)) for f in f_eval_list]
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m = make_model(seed + b + 1)
        m.fit(X_obs.iloc[idx], y[idx])
        for k, f in enumerate(f_eval_list):
            out[k][:, b] = m.predict_proba(f)[:, 1]
    return out


def calibrated(iso, s):
    return np.clip(iso.predict(s), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- #
# Coverage metrics
# --------------------------------------------------------------------------- #
def perrow_cov(lo, hi, y):
    return float(((y >= lo) & (y <= hi)).mean())


def decile_cov(pt, lo, hi, y, nbins=10):
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


def conformal_lambda_perrow(pt, std, y, target=TARGET_COV):
    """Smallest lambda so per-row 0/1 coverage >= target on the calib set."""
    # residual to nearest reachable: outcome y in {0,1}. interval = pt +/- lam*std
    # covered iff |y - pt| <= lam*std  ->  lam >= |y-pt|/std.
    r = np.abs(y - pt) / np.maximum(std, 1e-9)
    q = np.quantile(r, target, method="higher")
    return float(q)


def conformal_lambda_decile(pt, std, y, target=TARGET_COV, nbins=10):
    """Pick smallest lambda (grid) achieving >= target decile-bin coverage on calib."""
    order = np.argsort(pt)
    bins = np.array_split(order, nbins)
    binstat = []
    for idx in bins:
        emp = y[idx].mean()
        binstat.append((pt[idx].mean(), std[idx].mean(), emp))
    binstat = np.array(binstat)  # (nbins, 3): mean_pt, mean_std, emp
    grid = np.linspace(0.0, 5.0, 1001)
    for lam in grid:
        lo = np.clip(binstat[:, 0] - lam * binstat[:, 1], 0, 1)
        hi = np.clip(binstat[:, 0] + lam * binstat[:, 1], 0, 1)
        cov = np.mean((lo <= binstat[:, 2]) & (binstat[:, 2] <= hi))
        if cov >= target - 1e-9:
            return float(lam)
    return float(grid[-1])


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")

    cats = build_cat_dtypes(train, val, test)
    f_train = build_features(train, cats)
    f_val = build_features(val, cats).reindex(columns=f_train.columns)
    for c, dt in cats.items():
        if c in f_train.columns:
            f_val[c] = f_val[c].astype(dt)

    obs = train["default_flag"].notna().to_numpy()
    X_obs = f_train.loc[obs].reset_index(drop=True)
    y_tr = train.loc[obs, "default_flag"].astype(int).to_numpy()

    vmask = val["default_flag"].notna().to_numpy()
    f_val_lab = f_val.loc[vmask].reset_index(drop=True)
    y_val = val.loc[vmask, "default_flag"].astype(int).to_numpy()
    val_lab = val.loc[vmask].reset_index(drop=True)
    score_ood = val_lab["prior_underwriter_score"].to_numpy()
    is_ood = (score_ood < OOD_CUT) | np.isnan(score_ood)
    print(f"[data] train obs={obs.sum():,}  labeled val={vmask.sum():,}  "
          f"val default rate={y_val.mean():.4f}  OOD labeled rows={is_ood.sum()}")

    # ----- ensemble on full train-observed, score labeled val ----- #
    (val_scores,) = fit_ensemble(X_obs, y_tr, [f_val_lab])
    aucs = [roc_auc_score(y_val, val_scores[:, b]) for b in range(N_BOOT)]
    print(f"[ensemble] {N_BOOT} models, mean val AUC {np.mean(aucs):.4f}")

    # calibrate ensemble-mean with isotonic fit on ALL labeled val (matches build_a)
    raw_mean = val_scores.mean(axis=1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_mean, y_val)
    pt = calibrated(iso, raw_mean)
    # ensemble std on the calibrated scale (calibrate each member then std)
    cal_members = np.column_stack([calibrated(iso, val_scores[:, b]) for b in range(N_BOOT)])
    std = cal_members.std(axis=1)
    print(f"[spread] calibrated ensemble std: mean {std.mean():.4f} "
          f"median {np.median(std):.4f} max {std.max():.4f}")

    # ============================================================= #
    # BASELINE: current build_a recipe (5/95 pct of raw, calibrated)
    # ============================================================= #
    lo_raw = np.quantile(val_scores, 0.05, axis=1)
    hi_raw = np.quantile(val_scores, 0.95, axis=1)
    base_lo = np.minimum(calibrated(iso, lo_raw), pt)
    base_hi = np.maximum(calibrated(iso, hi_raw), pt)
    base_w = base_hi - base_lo
    print("\n" + "=" * 70)
    print("BASELINE (current build_a 5/95 bootstrap-pct interval)")
    print("=" * 70)
    print(f"  mean width {base_w.mean():.4f}  median {np.median(base_w):.4f}")
    print(f"  PER-ROW 0/1 coverage : {perrow_cov(base_lo, base_hi, y_val):.4f}")
    bcov, brows = decile_cov(pt, base_lo, base_hi, y_val)
    print(f"  DECILE-bin coverage  : {bcov:.2f}  ({int(bcov*10)}/10 bins)")

    # ============================================================= #
    # (A) PER-ROW 0/1 conformal -- show what width 90% really costs
    # ============================================================= #
    print("\n" + "=" * 70)
    print("(A) PER-ROW 0/1 COVERAGE conformal (literal task metric)")
    print("=" * 70)
    # honest: 5-fold OOF lambda
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_cov, fold_w, fold_lam = [], [], []
    for fi, (tr, te) in enumerate(kf.split(pt)):
        lam = conformal_lambda_perrow(pt[tr], std[tr], y_val[tr])
        lo = np.clip(pt[te] - lam * std[te], 0, 1)
        hi = np.clip(pt[te] + lam * std[te], 0, 1)
        c = perrow_cov(lo, hi, y_val[te])
        w = float(np.mean(hi - lo))
        fold_cov.append(c); fold_w.append(w); fold_lam.append(lam)
        print(f"  fold {fi}: lambda={lam:7.3f}  OOF per-row cov={c:.3f}  mean width={w:.3f}")
    print(f"  >> mean OOF per-row coverage {np.mean(fold_cov):.3f} "
          f"(+/-{np.std(fold_cov):.3f})  mean width {np.mean(fold_w):.3f} "
          f"(+/-{np.std(fold_w):.3f})  mean lambda {np.mean(fold_lam):.2f}")
    print("  INTERPRETATION: to contain a 0/1 outcome 90% of the time the band must")
    print("  blow up to ~full [0,1]; width >> 0.10 target. This metric is degenerate")
    print("  for a probability interval and should NOT be the design target.")

    # ============================================================= #
    # (B) DECILE-bin conformal -- meaningful, tighten toward width<0.10
    # ============================================================= #
    print("\n" + "=" * 70)
    print("(B) DECILE-BINNED coverage conformal  (operational metric)")
    print("=" * 70)
    fold_cov, fold_w, fold_med, fold_lam = [], [], [], []
    for fi, (tr, te) in enumerate(kf.split(pt)):
        lam = conformal_lambda_decile(pt[tr], std[tr], y_val[tr])
        lo = np.clip(pt[te] - lam * std[te], 0, 1)
        hi = np.clip(pt[te] + lam * std[te], 0, 1)
        c, _ = decile_cov(pt[te], lo, hi, y_val[te])
        w = float(np.mean(hi - lo)); med = float(np.median(hi - lo))
        fold_cov.append(c); fold_w.append(w); fold_med.append(med); fold_lam.append(lam)
        print(f"  fold {fi}: lambda={lam:6.3f}  OOF decile cov={c:.2f}  "
              f"mean width={w:.3f}  median={med:.3f}")
    print(f"  >> mean OOF decile coverage {np.mean(fold_cov):.2f} "
          f"(+/-{np.std(fold_cov):.2f})  mean width {np.mean(fold_w):.3f} "
          f"(+/-{np.std(fold_w):.3f})  median width {np.mean(fold_med):.3f}  "
          f"mean lambda {np.mean(fold_lam):.2f}")

    # full-fit lambda (deploy value) on all labeled val, then report on all
    lam_full = conformal_lambda_decile(pt, std, y_val)
    lo = np.clip(pt - lam_full * std, 0, 1)
    hi = np.clip(pt + lam_full * std, 0, 1)
    c_all, rows = decile_cov(pt, lo, hi, y_val)
    w_all = float(np.mean(hi - lo)); med_all = float(np.median(hi - lo))
    print(f"\n  [deploy lambda={lam_full:.3f}] in-sample decile cov={c_all:.2f}  "
          f"mean width={w_all:.4f}  median width={med_all:.4f}")
    print("  decile table (deploy lambda):")
    for i, ptm, emp, L, H, ok in rows:
        print(f"    bin {i:2d}: pred {ptm:.3f}  emp {emp:.3f}  "
              f"[{L:.3f},{H:.3f}] {'OK' if ok else 'MISS'}")

    # ============================================================= #
    # (C) TIME-ORDERED split-conformal (walk-forward proxy)
    # ============================================================= #
    print("\n" + "=" * 70)
    print("(C) TIME-ORDERED split-conformal (calib=earlier half, test=later half)")
    print("=" * 70)
    ts = val_lab["application_timestamp"].to_numpy()
    order = np.argsort(ts)
    half = len(order) // 2
    cal_idx, te_idx = order[:half], order[half:]
    lam_t = conformal_lambda_decile(pt[cal_idx], std[cal_idx], y_val[cal_idx])
    lo = np.clip(pt[te_idx] - lam_t * std[te_idx], 0, 1)
    hi = np.clip(pt[te_idx] + lam_t * std[te_idx], 0, 1)
    c_t, _ = decile_cov(pt[te_idx], lo, hi, y_val[te_idx])
    w_t = float(np.mean(hi - lo)); med_t = float(np.median(hi - lo))
    pr_t = perrow_cov(lo, hi, y_val[te_idx])
    print(f"  lambda(earlier half)={lam_t:.3f}  -> LATER half: decile cov={c_t:.2f}  "
          f"per-row 0/1 cov={pr_t:.3f}  mean width={w_t:.4f}  median={med_t:.4f}")

    # ============================================================= #
    # (D) OOD additive widening -- width impact only (no labeled OOD to score)
    # ============================================================= #
    print("\n" + "=" * 70)
    print("(D) OOD additive widening term (prior_score<0.273 or no bank feed)")
    print("=" * 70)
    print(f"  labeled OOD rows available to MEASURE coverage: {is_ood.sum()} "
          f"-> cannot validate OOD coverage on labeled val.")
    # measure how much it would widen TEST OOD rows (illustrative)
    f_test = build_features(test, cats).reindex(columns=f_train.columns)
    for c, dt in cats.items():
        if c in f_test.columns:
            f_test[c] = f_test[c].astype(dt)
    (test_scores,) = fit_ensemble(X_obs, y_tr, [f_test])
    t_mean = calibrated(iso, test_scores.mean(axis=1))
    t_members = np.column_stack([calibrated(iso, test_scores[:, b]) for b in range(N_BOOT)])
    t_std = t_members.std(axis=1)
    ts_score = test["prior_underwriter_score"].to_numpy()
    t_ood = (ts_score < OOD_CUT) | np.isnan(ts_score)
    print(f"  test rows={len(test):,}  OOD test rows={t_ood.sum():,} "
          f"({100*t_ood.mean():.1f}%)")
    # additive support term: widen by delta on each side in OOD region.
    # principled delta = gap between cutoff-edge empirical default and model PD floor.
    for dlt in (0.05, 0.10, 0.15):
        lo = np.clip(t_mean - lam_full * t_std - np.where(t_ood, dlt, 0), 0, 1)
        hi = np.clip(t_mean + lam_full * t_std + np.where(t_ood, dlt, 0), 0, 1)
        w_in = float(np.mean((hi - lo)[~t_ood]))
        w_ood = float(np.mean((hi - lo)[t_ood]))
        print(f"  delta={dlt:.2f}: in-support mean width {w_in:.3f}  "
              f"OOD mean width {w_ood:.3f}")
    print("  NOTE: OOD widening cannot be coverage-validated (no labeled OOD); it is")
    print("  an honesty heuristic only. Width grows by 2*delta in OOD, by design.")

    # ============================================================= #
    # (E) HONEST lambda->coverage frontier (5-fold OOF, decile + per-row)
    #     Fix lambda GLOBALLY (not chosen in-fold) and measure OOF coverage,
    #     so we see the true width needed to HOLD 90% decile coverage.
    # ============================================================= #
    print("\n" + "=" * 70)
    print("(E) HONEST frontier: fixed lambda, 5-fold OOF decile coverage + width")
    print("=" * 70)
    print("  lambda | OOFdecileCov(+/-) | OOFperRow0/1 | meanW  medW")
    for lam in (0.485, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        dcs, prs, ws, ms = [], [], [], []
        for _, te in kf.split(pt):
            lo = np.clip(pt[te] - lam * std[te], 0, 1)
            hi = np.clip(pt[te] + lam * std[te], 0, 1)
            c, _ = decile_cov(pt[te], lo, hi, y_val[te])
            dcs.append(c); prs.append(perrow_cov(lo, hi, y_val[te]))
            ws.append(float(np.mean(hi - lo))); ms.append(float(np.median(hi - lo)))
        print(f"  {lam:5.3f}  | {np.mean(dcs):.2f} (+/-{np.std(dcs):.2f})    | "
              f"{np.mean(prs):.3f}       | {np.mean(ws):.3f}  {np.mean(ms):.3f}")
    print("  (baseline 5/95-pct width=0.123 holds 10/10 decile bins; this shows the")
    print("   lambda needed for std-scaled bands to HOLD decile coverage out-of-fold.)")

    # ----- summary line for the structured report ----- #
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"baseline: mean width {base_w.mean():.3f}, median {np.median(base_w):.3f}, "
          f"decile cov {bcov:.2f}, per-row 0/1 cov "
          f"{perrow_cov(base_lo, base_hi, y_val):.3f}")
    print(f"conformal(decile,deploy lam={lam_full:.2f}): mean width {w_all:.3f}, "
          f"median {med_all:.3f}, in-sample decile cov {c_all:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
