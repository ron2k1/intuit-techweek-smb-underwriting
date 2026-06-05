#!/usr/bin/env python3
"""Follow-up investigation of two EDA surprises before building Deliverable A:

  (5) Is `requested_amount_to_observed_revenue` corrupted, or just computed
      against a different denominator than observed_monthly_revenue_avg_3mo?
  (6) Is self-report inflation real but conditional (vs the ~1.0 median we saw)?

Pure read-only analysis; prints findings.
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


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")

    print("=" * 70)
    print("INVESTIGATION 1: requested_amount_to_observed_revenue")
    print("=" * 70)
    col = "requested_amount_to_observed_revenue"
    sub = train[train[col].notna() & (train[col] != 0)].copy()
    print(f"non-null & non-zero rows: {len(sub):,}")
    print(f"ratio stats: min={sub[col].min():.4f} median={sub[col].median():.4f} "
          f"max={sub[col].max():.4f}")

    # Implied denominator = requested_amount / ratio. Compare to candidate revenues.
    sub["implied_denom"] = sub["requested_amount"] / sub[col]
    candidates = {
        "observed_monthly_revenue_avg_3mo": sub.get("observed_monthly_revenue_avg_3mo"),
        "observed_monthly*12 (annual)": sub.get("observed_monthly_revenue_avg_3mo") * 12,
        "stated_annual_revenue": sub.get("stated_annual_revenue"),
        "stated_annual/12 (monthly)": sub.get("stated_annual_revenue") / 12,
    }
    for name, denom in candidates.items():
        if denom is None:
            continue
        m = sub["implied_denom"].notna() & denom.notna() & (denom != 0)
        if m.sum() == 0:
            continue
        rel = ((sub.loc[m, "implied_denom"] - denom[m]).abs() / denom[m].abs())
        frac_match = (rel < 0.01).mean()
        print(f"  implied_denom vs {name:35s}: "
              f"median_rel_err={rel.median():.3f}  frac<1%={frac_match:.3f}")

    # Does the ratio correlate with default at all (is it still useful)?
    obs = sub[sub["default_flag"].notna()]
    if len(obs):
        q = pd.qcut(obs[col], 5, duplicates="drop")
        print("\n  default rate by ratio quintile:")
        print(obs.groupby(q, observed=True)["default_flag"].agg(["mean", "count"]).to_string())

    print()
    print("=" * 70)
    print("INVESTIGATION 2: self-report inflation (stated vs observed)")
    print("=" * 70)
    s = train[
        train["stated_annual_revenue"].notna()
        & train["observed_monthly_revenue_avg_3mo"].notna()
        & (train["observed_monthly_revenue_avg_3mo"] != 0)
    ].copy()
    s["obs_annual"] = s["observed_monthly_revenue_avg_3mo"] * 12
    s["ratio"] = s["stated_annual_revenue"] / s["obs_annual"]
    print(f"rows compared: {len(s):,}")
    print(f"ratio quantiles: "
          + ", ".join(f"p{int(q*100)}={s['ratio'].quantile(q):.2f}"
                      for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
    print(f"frac stated > observed (ratio>1.05): {(s['ratio'] > 1.05).mean():.3f}")
    print(f"frac stated < observed (ratio<0.95): {(s['ratio'] < 0.95).mean():.3f}")

    # Is inflation correlated with default? (inflators may be riskier)
    if s["default_flag"].notna().any():
        o = s[s["default_flag"].notna()]
        o = o.assign(infl=pd.qcut(o["ratio"], 5, duplicates="drop"))
        print("\n  default rate by stated/observed-revenue ratio quintile:")
        print(o.groupby("infl", observed=True)["default_flag"].agg(["mean", "count"]).to_string())

    # Inflation conditional on whether a bank feed exists (no-feed can't be checked).
    print(f"\n  NOTE: inflation is only checkable for the {len(s):,} feed-linked rows; "
          f"the {train['observed_monthly_revenue_avg_3mo'].isna().sum():,} no-feed rows "
          f"have no observed anchor — stated values there are unverifiable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
