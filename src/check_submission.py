#!/usr/bin/env python3
"""Inspect submission_A against the positivity violation:

How does our PD model behave in the OBSERVED (prior-approved, score>=0.273) vs
UNOBSERVED (prior-declined, score<0.273) regions? Do we fund loans in the
no-support region purely by extrapolation, and are intervals wider there?
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
SUB = REPO / "submissions" / "submission_A_decisions.csv"


def main() -> int:
    sub = pd.read_csv(SUB)
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    meta = pd.concat([
        val[["applicant_id", "prior_decision", "prior_underwriter_score"]],
        test[["applicant_id", "prior_decision", "prior_underwriter_score"]],
    ], ignore_index=True)
    df = sub.merge(meta, on="applicant_id", how="left")
    df["region"] = np.where(df["prior_decision"] == 1, "approved(observed)",
                            "declined(UNOBSERVED)")

    print("By prior-decision region:")
    g = df.groupby("region").agg(
        n=("applicant_id", "size"),
        we_approve=("decision", "sum"),
        approve_rate=("decision", "mean"),
        mean_pd=("predicted_pd", "mean"),
        median_pd=("predicted_pd", "median"),
        mean_width=("pd_upper_90", lambda s: (df.loc[s.index, "pd_upper_90"]
                                              - df.loc[s.index, "pd_lower_90"]).mean()),
    )
    print(g.to_string())

    funded = df[df["decision"] == 1]
    print(f"\nTotal funded: {len(funded):,}")
    print("Funded book composition by region:")
    print(funded["region"].value_counts().to_string())

    # PD vs score, bucketed, to see the extrapolation behavior.
    print("\nPredicted PD by score bucket (does the model push declined-region PD up?):")
    df["score_bucket"] = pd.cut(df["prior_underwriter_score"],
                                [0, 0.1, 0.2, 0.273, 0.4, 0.6, 0.8, 1.001])
    print(df.groupby("score_bucket", observed=True).agg(
        n=("applicant_id", "size"),
        mean_pd=("predicted_pd", "mean"),
        approve_rate=("decision", "mean"),
        mean_width=("pd_lower_90", lambda s: (df.loc[s.index, "pd_upper_90"]
                                              - df.loc[s.index, "pd_lower_90"]).mean()),
    ).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
