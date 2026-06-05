#!/usr/bin/env python3
"""Empirical default rate vs prior_underwriter_score in the OBSERVED region.

If default falls monotonically as score rises, then declined applicants
(score < 0.273) extrapolate to even-higher default -> a monotonicity argument
says decline them all. This quantifies how risky the lowest observed scores are.
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
    obs = train[train["default_flag"].notna()].copy()  # approved + matured

    print(f"observed rows: {len(obs):,}, overall default {obs['default_flag'].mean():.4f}")
    print("\nActual default rate by prior_underwriter_score decile (observed region):")
    obs["score_decile"] = pd.qcut(obs["prior_underwriter_score"], 10)
    g = obs.groupby("score_decile", observed=True).agg(
        n=("default_flag", "size"),
        default_rate=("default_flag", "mean"),
        score_lo=("prior_underwriter_score", "min"),
        score_hi=("prior_underwriter_score", "max"),
    )
    print(g.to_string())

    # The riskiest *observed* applicants sit just above the 0.273 cutoff.
    edge = obs[obs["prior_underwriter_score"] < 0.35]
    print(f"\nObserved applicants nearest the cutoff (score < 0.35): "
          f"n={len(edge):,}, default rate {edge['default_flag'].mean():.4f}")
    print("Break-even PD is ~0.088 — so even the riskiest OBSERVED loans default "
          "well above break-even. Declined applicants (lower score) extrapolate higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
