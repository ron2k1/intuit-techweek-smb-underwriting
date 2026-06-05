#!/usr/bin/env python3
"""Diagnose the selection mechanism: what determines prior_decision?

If approval is a deterministic threshold on prior_underwriter_score, then the
declined region has NO outcome support (positivity violation) and IPW reject
inference is inapplicable -- a key writeup finding.
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
    appr = train["prior_decision"] == 1
    s = train["prior_underwriter_score"]

    print("prior_underwriter_score by prior_decision:")
    print(train.groupby("prior_decision")["prior_underwriter_score"]
          .describe()[["min", "25%", "50%", "75%", "max"]].to_string())

    print(f"\nscore range — approved: [{s[appr].min():.4f}, {s[appr].max():.4f}]")
    print(f"score range — declined: [{s[~appr].min():.4f}, {s[~appr].max():.4f}]")
    overlap_lo = max(s[appr].min(), s[~appr].min())
    overlap_hi = min(s[appr].max(), s[~appr].max())
    print(f"score overlap region: [{overlap_lo:.4f}, {overlap_hi:.4f}]")
    n_overlap = ((s >= overlap_lo) & (s <= overlap_hi)).sum()
    print(f"rows in overlap: {n_overlap:,} of {len(train):,}")

    # Is there a clean threshold? Sort by score, see if decision flips once.
    t = train[["prior_underwriter_score", "prior_decision"]].dropna().sort_values(
        "prior_underwriter_score")
    # Best single threshold accuracy.
    best_acc, best_thr = 0.0, None
    for q in np.linspace(0.01, 0.99, 99):
        thr = t["prior_underwriter_score"].quantile(q)
        pred = (t["prior_underwriter_score"] >= thr).astype(int)
        acc = (pred == t["prior_decision"]).mean()
        acc = max(acc, 1 - acc)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"\nbest single-threshold accuracy on score: {best_acc:.4f} at thr~{best_thr:.4f}")

    # How many approved fall in the declined score range and vice versa?
    misordered = ((s[appr].values[:, None] < s[~appr].min())).sum()
    print(f"approved rows below the min declined score: "
          f"{(s[appr] < s[~appr].min()).sum():,}")
    print(f"declined rows above the max approved score: "
          f"{(s[~appr] > s[appr].max()).sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
