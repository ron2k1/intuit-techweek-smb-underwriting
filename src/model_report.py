#!/usr/bin/env python3
"""Standalone model report card for Deliverable A (READ-ONLY: does not modify the
pipeline, model, or submission). Produces a comparison-ready metric sheet:

  - AUC-ROC, Brier score, Log loss, Calibration (ECE + reliability)
  - PD -> expected profit (official amortizing economics)
  - Approve when expected profit > 0  ->  portfolio NPV (expected + realized)

Two evaluation views:
  (A) VALIDATION (2,551 labeled rows) -- the standard cross-team comparison set.
      NOTE: the deployed PD is calibrated ON validation, so its CALIBRATION metrics
      are in-sample (optimistic). AUC is unaffected by calibration.
  (B) HONEST 5-fold OUT-OF-FOLD on the 51,722 observed train rows -- the unbiased
      numbers (model + isotonic both cross-fit), the fair figures to quote.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import (  # noqa: E402  (imports only; nothing is modified)
    build_cat_dtypes, build_features, numeric_frame, make_model, make_logit,
    W_HGB, NET_MARGIN, DAILY_DRAW, FEE, LGD, BREAK_EVEN_PD,
)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
PREDS = REPO / "reports" / "a_predictions.csv"
SEED = 17


def ece(pd_hat, y, n_bins=10):
    """Expected Calibration Error via equal-frequency (decile) bins."""
    order = np.argsort(pd_hat)
    bins = np.array_split(order, n_bins)
    e, rows = 0.0, []
    for b in bins:
        conf, acc = pd_hat[b].mean(), y[b].mean()
        e += (len(b) / len(y)) * abs(conf - acc)
        rows.append((conf, acc, len(b)))
    return e, rows


def card(name, pd_hat, y):
    auc = roc_auc_score(y, pd_hat)
    brier = brier_score_loss(y, pd_hat)
    ll = log_loss(y, np.clip(pd_hat, 1e-6, 1 - 1e-6))
    e, rows = ece(pd_hat, y)
    print(f"\n--- {name}  (n={len(y):,}, base default={y.mean():.4f}) ---")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  Log loss  : {ll:.4f}")
    print(f"  ECE (10)  : {e:.4f}")
    print("  reliability (pred vs observed by decile):")
    for i, (c, a, nb) in enumerate(rows):
        print(f"    d{i:>2}: pred {c:.3f}  obs {a:.3f}  n={nb}")
    return dict(auc=auc, brier=brier, log_loss=ll, ece=e)


def exact_realized_npv(amt, dflag, dtd, rec):
    rec = np.nan_to_num(rec)
    draws = amt * DAILY_DRAW * np.clip(np.nan_to_num(dtd) - 1, 0, None)
    default = np.minimum(amt * FEE + draws + rec - amt, amt * NET_MARGIN)
    return np.where(dflag == 1, default, amt * NET_MARGIN)


def main() -> int:
    print("=" * 70)
    print("DELIVERABLE A -- PD MODEL REPORT CARD  (read-only; pipeline unchanged)")
    print(f"economics: NET_MARGIN={NET_MARGIN:.4f}  LGD={LGD}  break-even PD={BREAK_EVEN_PD:.4f}")
    print("=" * 70)

    # ---------- (A) VALIDATION metrics from the deployed predictions -------- #
    preds = pd.read_csv(PREDS)
    vmask = preds["default_flag"].notna().to_numpy()
    yv = preds.loc[vmask, "default_flag"].astype(int).to_numpy()
    pv = preds.loc[vmask, "predicted_pd"].to_numpy()
    print("\n### (A) VALIDATION SET -- deployed model "
          "(calibration in-sample; AUC is honest) ###")
    card("validation", pv, yv)

    # ---------- (B) HONEST 5-fold OOF metrics on observed train ------------- #
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)
    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    Zo = numeric_frame(Xo)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_raw = np.zeros(len(y))
    for tr, te in skf.split(Xo, y):
        h = make_model(SEED).fit(Xo.iloc[tr], y[tr])
        lg = make_logit().fit(Zo.iloc[tr], y[tr])
        oof_raw[te] = (W_HGB * h.predict_proba(Xo.iloc[te])[:, 1]
                       + (1 - W_HGB) * lg.predict_proba(Zo.iloc[te])[:, 1])
    # cross-fit isotonic so calibration metrics stay honest (no in-sample leak)
    oof_cal = np.zeros(len(y))
    for tr, te in skf.split(oof_raw.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(oof_raw[tr], y[tr])
        oof_cal[te] = np.clip(iso.predict(oof_raw[te]), 1e-6, 1 - 1e-6)
    print("\n### (B) HONEST 5-fold OUT-OF-FOLD (51,722 train) -- the fair numbers ###")
    card("OOF (calibrated, cross-fit)", oof_cal, y)

    # ---------- PD -> expected profit -> approve EV>0 -> portfolio NPV ------ #
    amt = preds["requested_amount"].to_numpy()
    pd_all = preds["predicted_pd"].to_numpy()
    declined = (preds["prior_decision"] == 0).to_numpy()
    ev_per_dollar = (1 - pd_all) * NET_MARGIN - pd_all * LGD   # expected profit / $
    approve_ev = ev_per_dollar > 0                              # == pd < break-even
    cons = approve_ev & ~declined                              # deployed conservative

    print("\n" + "=" * 70)
    print("EXPECTED PROFIT -> APPROVE (EV>0) -> PORTFOLIO NPV")
    print("=" * 70)
    print(f"  per-loan rule: approve iff (1-pd)*{NET_MARGIN:.4f} - pd*{LGD} > 0  "
          f"<=>  pd < {BREAK_EVEN_PD:.4f}")

    realized = exact_realized_npv(
        amt, preds["default_flag"].to_numpy(), preds["days_to_default"].to_numpy(),
        preds["final_recovered_amount"].to_numpy())
    has_truth = preds["default_flag"].notna().to_numpy()

    for label, mask in [("CONSERVATIVE (prior-approved only) [deployed]", cons),
                        ("EV>0 on ALL applicants (incl. declined region)", approve_ev)]:
        exp_npv = (ev_per_dollar[mask] * amt[mask]).sum()
        prin = amt[mask].sum()
        fr = mask & has_truth
        real = realized[fr].sum()
        print(f"\n  [{label}]")
        print(f"    approved        : {int(mask.sum()):,} / {len(mask):,} "
              f"({100*mask.mean():.1f}%)")
        print(f"    principal       : ${prin:,.0f}")
        print(f"    EXPECTED NPV    : ${exp_npv:,.0f}   (ROIC {100*exp_npv/prin:.2f}%)")
        print(f"    REALIZED NPV    : ${real:,.0f}  (on {int(fr.sum()):,} labeled approved)")
    print("\n(Realized NPV is only measurable on prior-approved+matured rows; the")
    print(" declined region is unlabeled, so its 'expected' figure is unverifiable.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
