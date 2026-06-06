#!/usr/bin/env python3
"""Portfolio NPV with CORRECT amortizing economics, + threshold-optimality proof.

Loans amortize via daily ACH draws over 60 days. Realized profit:

  repaid : amount * NET_MARGIN                                   (~0.0875)
  default: fee + (draws collected before default) + recovery - principal
         = amount*[0.03 + min(dtd,60)/60 * (1+TERM_INT) - 1] + final_recovered

Mean realized loss-given-default under this model is ~0.30 (NOT 0.91), so the
break-even PD is ~0.224 (NOT 0.088). We sweep the decision threshold against
REALIZED amortizing profit on validation to find the true optimum.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
PREDS = REPO / "reports" / "a_predictions.csv"

TERM_INT = 0.35 * 60 / 365         # interest over the 60-day term (~0.0575)
FEE = 0.03
NET_MARGIN = TERM_INT + FEE        # ~0.0875
DAILY_DRAW = (1 + TERM_INT) / 60   # D per $ = R(1+rT/365)/T
LGD = 0.25                         # effective LGD measured under the exact NPV formula
BREAK_EVEN = NET_MARGIN / (NET_MARGIN + LGD)   # ~0.259


def realized_profit(amt, dflag, dtd, rec):
    """Realized $ per loan under the OFFICIAL NPV formula (brief).

    repaid : amt*NET_MARGIN ;  default: F + D*(t*-1) + rec - R (capped at margin).
    """
    rec = np.nan_to_num(rec)
    draws = amt * DAILY_DRAW * np.clip(np.nan_to_num(dtd) - 1, 0, None)
    default_profit = np.minimum(amt * FEE + draws + rec - amt, amt * NET_MARGIN)
    return np.where(dflag == 0, amt * NET_MARGIN, default_profit)


def main() -> int:
    df = pd.read_csv(PREDS)  # already has days_to_default / outcomes for val rows

    amt = df["requested_amount"].to_numpy()
    pd_p = df["predicted_pd"].to_numpy()
    declined = (df["prior_decision"] == 0).to_numpy()

    # --- Deployed NPV: conservative funded set at the CORRECTED threshold --- #
    funded = (pd_p < BREAK_EVEN) & ~declined
    ev_per_dollar = (1 - pd_p) * NET_MARGIN - pd_p * LGD
    npv = (ev_per_dollar[funded] * amt[funded]).sum()
    principal = amt[funded].sum()
    print("=" * 66)
    print(f"DEPLOYED NPV  (conservative, LGD={LGD}, break-even={BREAK_EVEN:.3f})")
    print("=" * 66)
    print(f"  loans funded ................ {funded.sum():,}")
    print(f"  principal deployed .......... ${principal:,.0f}")
    print(f"  expected NPV ................ ${npv:,.0f}")
    print(f"  NPV / principal (ROIC) ...... {100*npv/principal:.2f}%")

    # For reference: NPV at the OLD (wrong) threshold, same corrected economics.
    funded_old = (pd_p < 0.088) & ~declined
    npv_old = (ev_per_dollar[funded_old] * amt[funded_old]).sum()
    print(f"\n  (for contrast) NPV if we kept the old 0.088 cut: ${npv_old:,.0f} "
          f"on {funded_old.sum():,} loans — leaves ${npv-npv_old:,.0f} on the table)")

    # --- Threshold sweep vs REALIZED amortizing profit (val, observed) --- #
    obs = df[df["default_flag"].notna() & ~declined].copy()
    a = obs["requested_amount"].to_numpy()
    p = obs["predicted_pd"].to_numpy()
    prof = realized_profit(a, obs["default_flag"].to_numpy(),
                           obs["days_to_default"].to_numpy(),
                           obs["final_recovered_amount"].to_numpy())
    print("\n" + "=" * 66)
    print("THRESHOLD SWEEP vs realized amortizing profit (val, observed)")
    print("=" * 66)
    print(f"  {'thr':>6} {'funded':>7} {'realized $':>13} {'$/loan':>8}")
    best = (None, -1e18)
    grid = [0.05, 0.088, 0.12, 0.15, 0.18, 0.20, BREAK_EVEN, 0.25, 0.28, 0.32, 0.40, 1.0]
    for thr in sorted(set(grid)):
        f = p < thr
        tot = prof[f].sum()
        per = tot / max(f.sum(), 1)
        tag = "  <- break-even" if abs(thr - BREAK_EVEN) < 1e-6 else ""
        if tot > best[1]:
            best = (thr, tot)
        print(f"  {thr:6.3f} {f.sum():7d} {tot:13,.0f} {per:8,.0f}{tag}")
    print(f"\n  realized-profit-maximizing threshold in grid: {best[0]:.3f} "
          f"(break-even = {BREAK_EVEN:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
