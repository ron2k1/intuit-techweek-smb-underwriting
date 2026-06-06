#!/usr/bin/env python3
"""Deliverable B preview: the 13x13 cohort-week x loan-age cumulative-default-rate
grid for OUR Deliverable-A approved set. READ-ONLY (does not change the pipeline).

Factorization (validated: default-day shape is ~proportional across risk strata):
    CDR(cohort c, age a) = mean_{i in approved & cohort c} PD_i  x  G(a)
where G(a) = cumulative fraction of a defaulting loan's mass realized by loan-age
week a, estimated from the absorbing-Markov default-day distribution (days_to_default
on matured train loans). PD_i = our calibrated blend PD; G(a) = shared timing shape.
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
WEEK1_START = pd.Timestamp("2025-06-30")
N = 13


def timing_curve() -> np.ndarray:
    """G(a), a=1..13: cumulative share of defaults occurring by loan-age week a.

    From the absorbing chain's observed absorption day (days_to_default on matured
    defaults). Week of default = ceil(day/7); G(13)=1 (day-90 balance check)."""
    train = pd.read_csv(DATA / "train.csv")
    dtd = train.loc[train["default_flag"] == 1, "days_to_default"].dropna().to_numpy()
    wk = np.ceil(dtd / 7.0).astype(int).clip(1, N)
    G = np.array([(wk <= a).mean() for a in range(1, N + 1)])
    return G


def main() -> int:
    sub = pd.read_csv(SUB)  # applicant_id, decision, predicted_pd, ...
    val = pd.read_csv(DATA / "validation.csv")[["applicant_id", "application_timestamp"]]
    test = pd.read_csv(DATA / "test.csv")[["applicant_id", "application_timestamp"]]
    ts = pd.concat([val, test], ignore_index=True)
    df = sub.merge(ts, on="applicant_id", how="left")

    # cohort_week from application_timestamp (consecutive 7-day windows from week 1).
    days = (pd.to_datetime(df["application_timestamp"]) - WEEK1_START).dt.days
    df["cohort_week"] = (days // 7 + 1).clip(1, N).astype(int)

    appr = df[df["decision"] == 1].copy()
    G = timing_curve()

    # per-cohort approved count + mean PD (the week-13 asymptote of each row)
    grp = appr.groupby("cohort_week")["predicted_pd"]
    mean_pd = grp.mean().reindex(range(1, N + 1))
    n_appr = grp.size().reindex(range(1, N + 1)).fillna(0).astype(int)

    # 13x13 grid: CDR(c,a) = mean_pd_c * G(a)
    grid = np.outer(mean_pd.to_numpy(), G)   # rows = cohort 1..13, cols = age 1..13

    # ---- print the matrix ------------------------------------------------ #
    print("DELIVERABLE B  -- 13x13 cumulative default-rate grid (our approved set)")
    print("rows = cohort_week (origination),  cols = loan_age_weeks\n")
    header = "coh \\ age " + "".join(f"{a:6d}" for a in range(1, N + 1))
    print(header)
    print("-" * len(header))
    for c in range(1, N + 1):
        row = "".join(f"{grid[c-1, a]:6.3f}" for a in range(N))
        print(f"  c{c:<2d} ({n_appr[c]:4d})  {row}")

    print("\nper-cohort summary:")
    print(f"  {'cohort':>6} {'approved':>9} {'mean_PD(=wk13)':>15}")
    for c in range(1, N + 1):
        print(f"  {c:>6} {n_appr[c]:>9} {mean_pd[c]:>15.4f}")

    print("\ndefault-timing curve G(a) (cumulative share of defaults by loan-age week):")
    print("  " + "  ".join(f"w{a}:{G[a-1]:.3f}" for a in range(1, N + 1)))
    print("\nReading the grid: each ROW is one origination cohort's expected cumulative")
    print("default rate as the book ages left->right; it rises from ~0 (weeks 1-2 = the")
    print("3-consecutive-miss floor) and plateaus weeks 9-12 (the days 61-89 dead zone),")
    print("then jumps at week 13 (the day-90 balance check) to that cohort's mean PD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
