#!/usr/bin/env python3
"""Decisive test: under the OFFICIAL exact NPV formula (from the brief), what
funding threshold maximizes realized portfolio NPV, and is it robustly above the
flat-LGD break-even 0.226? Uses ACTUAL outcomes (y, t*, rec) on observed rows.

Official per-loan NPV:
  repaid  (y=0): F + R*r*T/365            = R*(0.03 + 0.35*60/365)         (~0.0875*R)
  default (y=1): F + D*(t*-1) + rec - R   D = R*(1 + r*T/365)/T            (amortizing)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import build_cat_dtypes, build_features, numeric_frame, make_model, make_logit, W_HGB  # noqa: E402
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path(__file__).resolve().parent.parent / "dataset"
SEED = 17
r, T, F_RATE = 0.35, 60, 0.03
MARGIN = F_RATE + r * T / 365.0          # repaid margin per $ (~0.0875)
D_FACTOR = (1 + r * T / 365.0) / T       # daily draw per $ (~0.017625)
BREAK_EVEN_FLAT = MARGIN / (MARGIN + 0.30)   # 0.226


def exact_npv(R, y, tstar, rec, cap=False):
    """Official per-$ NPV * R. cap=True clips default NPV at the repaid margin."""
    repaid = R * MARGIN
    draws = R * D_FACTOR * np.clip(np.nan_to_num(tstar) - 1, 0, None)
    default = R * F_RATE + draws + np.nan_to_num(rec) - R
    if cap:
        default = np.minimum(default, repaid)
    return np.where(y == 1, default, repaid)


def oof_blend_pd(train):
    """5-fold OOF calibrated blend PD for observed train rows."""
    cats = build_cat_dtypes(train, train, train)
    X = build_features(train, cats)
    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    Zo = numeric_frame(Xo)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(Xo, y):
        h = make_model(SEED).fit(Xo.iloc[tr], y[tr])
        lg = make_logit().fit(Zo.iloc[tr], y[tr])
        s = W_HGB * h.predict_proba(Xo.iloc[te])[:, 1] + (1 - W_HGB) * lg.predict_proba(Zo.iloc[te])[:, 1]
        oof[te] = s
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit_transform(oof, y)
    return obs, np.clip(cal, 1e-6, 1 - 1e-6), y


def sweep(pd_hat, npv, label):
    best_t, best = None, -1e18
    grid = np.round(np.arange(0.05, 0.601, 0.005), 3)
    for t in grid:
        tot = npv[pd_hat < t].sum()
        if tot > best:
            best, best_t = tot, t
    at_be = npv[pd_hat < BREAK_EVEN_FLAT].sum()
    print(f"  [{label}] argmax tau={best_t:.3f} NPV=${best:,.0f} | @0.226 ${at_be:,.0f} "
          f"| gain ${best-at_be:,.0f} ({100*(best-at_be)/abs(at_be):.2f}%)")
    return best_t


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    print(f"break-even (flat LGD 0.30) = {BREAK_EVEN_FLAT:.4f}\n")

    obs, pd_hat, y = oof_blend_pd(train)
    o = train.loc[obs].reset_index(drop=True)
    R = o["requested_amount"].to_numpy()
    tstar = o["days_to_default"].to_numpy()
    rec = o["final_recovered_amount"].to_numpy()

    npv_unc = exact_npv(R, y, tstar, rec, cap=False)   # exact, uncapped (brief literal)
    npv_cap = exact_npv(R, y, tstar, rec, cap=True)     # capped at repaid margin (safer)

    # sanity: how often does uncapped default-NPV exceed the repaid margin?
    dmask = y == 1
    over = (npv_unc[dmask] > R[dmask] * MARGIN).mean()
    print(f"defaults with uncapped NPV > repaid margin (t*>~60 artifact): {over:.1%}")
    print(f"mean realized NPV/$ on defaults: uncapped {((npv_unc[dmask])/R[dmask]).mean():+.3f} "
          f"capped {((npv_cap[dmask])/R[dmask]).mean():+.3f}  "
          f"(=> effective LGD ~ {-((npv_cap[dmask])/R[dmask]).mean():.3f})\n")

    print("RANDOM OOF threshold sweep (exact official NPV, full 51,722 observed):")
    sweep(pd_hat, npv_unc, "uncapped/brief")
    sweep(pd_hat, npv_cap, "capped")

    # chronological walk-forward: optimal threshold per time block (stability check)
    ts = pd.to_datetime(o["application_timestamp"]).to_numpy()
    order = np.argsort(ts)
    blocks = np.array_split(order, 6)
    print("\nWalk-forward optimal threshold per chronological block (uncapped):")
    for i, blk in enumerate(blocks):
        if len(np.unique(y[blk])) < 2:
            continue
        sweep(pd_hat[blk], npv_unc[blk], f"block{i}")
    print("\n> If argmax tau is robustly >> 0.226 across blocks AND on the LATEST block,")
    print("> raising the threshold (lower effective LGD) is a real, test-relevant gain.")
    print("> If it jumps around / collapses to ~0.226 on recent blocks, keep 0.226.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
