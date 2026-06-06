"""Deliverable B: cohort x loan-age cumulative default trajectory (a vintage curve).

Run AFTER build_a:  python -m src.build_b -> submissions/submission_B_trajectory.csv

TIED TO A. The trajectory is decomposed as

    CDR(cohort c, age t) = PD_c * G(t)

  * PD_c  = the ultimate default rate of cohort c = mean predicted_pd (from A's
            output) over the applicants A APPROVED (decision==1) in that cohort.
            This is the literal A->B link: A's per-applicant PD is the t->inf
            asymptote of B's trajectory (G(13)=1).
  * G(t)  = the cumulative default-TIMING curve estimated from the fully-matured
            training loans: the fraction of a cohort's ultimate defaults that have
            occurred by loan age t weeks. Non-decreasing, G(13)=1. It carries the
            structural day-90 step (flat weeks 10-12, jump at week 13).

90% bands are bootstrapped over BOTH sources of uncertainty (the cohort mean PD
and the timing curve). Because every bootstrap path is monotone in age, so are the
percentile bands -- the validator's monotonicity gate is satisfied by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB = REPO_ROOT / "submissions"
N_WEEKS = D.N_COHORT_WEEKS          # 13
N_BOOT = 500
SEED = 20260605


def timing_curve(tr: pd.DataFrame, ytr: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """G(t) for t=1..13 from matured train defaults, plus the per-default week ids.

    default week = ceil(days_to_default / 7), clamped to [1, 13]; G(t) is the
    cumulative share of defaults with week <= t. Returns (G, weeks).
    """
    d = (ytr == 1).to_numpy()
    dtd = pd.to_numeric(tr["days_to_default"], errors="coerce").to_numpy()
    m = d & ~np.isnan(dtd)
    weeks = np.clip(np.ceil(dtd[m] / 7.0), 1, N_WEEKS).astype(int)
    G = np.array([(weeks <= t).mean() for t in range(1, N_WEEKS + 1)])
    return G, weeks


def cohort_mean_pd() -> dict[int, np.ndarray]:
    """For each cohort 1..13, the array of A's predicted_pd over APPROVED applicants."""
    a = pd.read_csv(SUB / "submission_A_decisions.csv")
    a["applicant_id"] = a["applicant_id"].astype(str)
    # cohort_week per applicant comes from the engineered val+test frames
    frames = []
    for split in ("val", "test"):
        df = D.add_engineered_features(D.load_raw(split))
        frames.append(df[["applicant_id", "cohort_week"]])
    cw = pd.concat(frames, ignore_index=True)
    cw["applicant_id"] = cw["applicant_id"].astype(str)
    m = a.merge(cw, on="applicant_id", how="left")
    approved = m[(m["decision"] == 1) & m["cohort_week"].between(1, N_WEEKS)]
    return {c: g["predicted_pd"].to_numpy()
            for c, g in approved.groupby("cohort_week")}


def main() -> None:
    SUB.mkdir(exist_ok=True)
    tr, _ = D.load_features("train")
    ytr = D.target_vector(tr)
    G, weeks = timing_curve(tr, ytr)
    pd_by_cohort = cohort_mean_pd()
    all_pd = np.concatenate([v for v in pd_by_cohort.values()]) if pd_by_cohort else np.array([0.2])
    global_mean = float(all_pd.mean())
    rng = np.random.default_rng(SEED)

    # bootstrap timing curves once, reused across cohorts
    Gboot = np.empty((N_BOOT, N_WEEKS))
    for b in range(N_BOOT):
        w = rng.choice(weeks, size=len(weeks), replace=True)
        Gboot[b] = [(w <= t).mean() for t in range(1, N_WEEKS + 1)]

    rows = []
    print(f"G(t)={np.round(G,3).tolist()}  cohorts with approvals="
          f"{sorted(pd_by_cohort)}  global_mean_PD={global_mean:.4f}")
    for c in range(1, N_WEEKS + 1):
        vals = pd_by_cohort.get(c, np.array([global_mean]))
        pd_c = float(vals.mean())
        # bootstrap PD_c (resample cohort applicants) x bootstrap G
        pd_boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                            for _ in range(N_BOOT)])
        for ti, t in enumerate(range(1, N_WEEKS + 1)):
            cdr = pd_c * G[ti]
            reps = pd_boot * Gboot[:, ti]
            lo, hi = np.percentile(reps, [5, 95])
            rows.append((c, t,
                         np.clip(cdr, 0, 1),
                         np.clip(min(lo, cdr), 0, 1),
                         np.clip(max(hi, cdr), 0, 1)))

    b = pd.DataFrame(rows, columns=["cohort_week", "loan_age_weeks",
                                    "cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"])
    # enforce exact monotonicity in age per cohort (guards float jitter)
    b = b.sort_values(["cohort_week", "loan_age_weeks"]).reset_index(drop=True)
    for col in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
        b[col] = b.groupby("cohort_week")[col].cummax()
    b.to_csv(SUB / "submission_B_trajectory.csv", index=False)
    asy = b[b.loan_age_weeks == N_WEEKS]["cumulative_default_rate"]
    print(f"[B] {len(b)} rows  asymptote(week13) mean={asy.mean():.4f} "
          f"[{asy.min():.3f}, {asy.max():.3f}]  (= per-cohort mean PD from A)")


if __name__ == "__main__":
    main()
