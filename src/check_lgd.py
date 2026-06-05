#!/usr/bin/env python3
"""Re-examine loss-given-default: does my LGD=0.91 ignore pre-default draws?

Loans amortize via daily ACH draws over a 60-day term. A loan that defaults late
has already repaid most principal. Realistic per-default loss:

    remaining_balance ~ principal * (1 - days_to_default/60)   (linear amortization)
    loss = remaining_balance - final_recovered_amount
    LGD  = loss / principal

This is far lower than 1 - recovery/principal for late defaults. Quantify it.
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
NET_MARGIN = 0.35 * 60 / 365 + 0.03


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    d = train[train["default_flag"] == 1].copy()
    amt = d["requested_amount"].to_numpy()
    dtd = d["days_to_default"].to_numpy()
    rec = np.nan_to_num(d["final_recovered_amount"].to_numpy())

    print(f"defaulted loans: {len(d):,}")
    print("\ndays_to_default distribution:")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q*100):02d} = {np.nanquantile(dtd, q):.0f} days")
    print(f"  mean = {np.nanmean(dtd):.1f} days")

    # --- Model A: my original (total write-off minus recovery) --- #
    lgd_A = np.clip(1 - rec / amt, 0, 1)

    # --- Model B: amortizing — credit pre-default draws --- #
    remaining_frac = np.clip(1 - dtd / 60.0, 0, 1)   # principal still owed at default
    remaining_bal = remaining_frac * amt
    loss_B = np.clip(remaining_bal - rec, 0, amt)
    lgd_B = loss_B / amt

    print("\n              mean LGD   median LGD   implied break-even PD")
    for name, lgd in [("A write-off (mine)", lgd_A), ("B amortizing   ", lgd_B)]:
        be = NET_MARGIN / (NET_MARGIN + lgd.mean())
        print(f"  {name}   {lgd.mean():.3f}      {np.median(lgd):.3f}        {be:.3f}")

    print("\nrecovery fraction (final_recovered/principal) vs days_to_default:")
    dd = pd.DataFrame({"dtd": dtd, "rec_frac": rec / amt, "lgd_B": lgd_B})
    dd["bucket"] = pd.cut(dd["dtd"], [0, 15, 30, 45, 60, 90])
    print(dd.groupby("bucket", observed=True).agg(
        n=("dtd", "size"), mean_rec_frac=("rec_frac", "mean"),
        mean_lgd_B=("lgd_B", "mean")).to_string())

    print("\n> If recovery fraction is flat & low across days_to_default, then")
    print("> final_recovered_amount is POST-default only and Model B (crediting the")
    print("> amortizing draws) is the economically correct loss — LGD well below 0.91.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
