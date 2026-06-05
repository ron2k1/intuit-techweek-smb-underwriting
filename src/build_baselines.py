"""Baseline STUBS for Deliverable B and C -- SAFETY NETS so validate_submission.py
passes end-to-end from hour one. These are NOT the real deliverables:

  * B (cohort x age cumulative default trajectory) is owned by the DS Engineer and
    needs survival/hazard methods with right-censoring (audit_findings.md section 5).
  * C (do(feature=value) counterfactuals) is owned by the DS Engineer and needs
    backdoor adjustment / do-calculus, NOT naive re-prediction (section 6).

Both are clearly-labelled placeholders that produce VALID-shaped output so the team
always has a submittable bundle while the real models are built.

Run AFTER build_a (C reuses submission_A_decisions.csv's predicted_pd):
  python -m src.build_baselines
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB = REPO_ROOT / "submissions"
DATA = REPO_ROOT / "dataset"


def build_b_stub() -> None:
    """169-row grid, cumulative default rate non-decreasing in loan age per cohort.

    PLACEHOLDER curve: linear ramp to an asymptote = observed approved-validation
    default rate. Ignores right-censoring and cohort shift on purpose -- that is
    exactly what the DS Engineer's survival model replaces.
    """
    yva = D.target_vector(D.load_features("val")[0])
    asymptote = float(yva.mean())  # ~0.206
    rows = []
    for cohort in range(1, 14):
        for age in range(1, 14):
            cdr = asymptote * (age / 13.0)  # strictly non-decreasing in age
            rows.append((cohort, age, cdr,
                         max(cdr * 0.6, 0.0),
                         min(cdr * 1.4 + 0.02, 1.0)))
    b = pd.DataFrame(rows, columns=["cohort_week", "loan_age_weeks",
                                    "cumulative_default_rate",
                                    "cdr_lower_90", "cdr_upper_90"])
    b.to_csv(SUB / "submission_B_trajectory.csv", index=False)
    print(f"[B stub] {len(b)} rows  asymptote={asymptote:.3f}  "
          f"(PLACEHOLDER -> DS Eng survival model)")


def build_c_stub() -> None:
    """One row per intervention query.

    PLACEHOLDER: counterfactual PD = the applicant's BASELINE predicted_pd from A
    (i.e. the naive 'intervention has no effect' answer). This is deliberately the
    answer the challenge says is insufficient -- it just keeps the file valid until
    the DS Engineer implements true do() adjustment.
    """
    q = pd.read_csv(DATA / "intervention_queries.csv")
    base = pd.read_csv(SUB / "submission_A_decisions.csv").set_index("applicant_id")["predicted_pd"]
    glob = float(base.mean())
    pdcf = q["applicant_id"].map(base).fillna(glob).to_numpy()
    c = pd.DataFrame({
        "query_id": q["query_id"],
        "predicted_pd_cf": np.clip(pdcf, 0.0, 1.0),
        "pd_cf_lower_90": np.clip(pdcf - 0.10, 0.0, 1.0),
        "pd_cf_upper_90": np.clip(pdcf + 0.10, 0.0, 1.0),
    })
    c.to_csv(SUB / "submission_C_counterfactuals.csv", index=False)
    miss = int(q["applicant_id"].map(base).isna().sum())
    print(f"[C stub] {len(c)} rows  (naive baseline=predicted_pd, {miss} global-mean "
          f"fallback) -> DS Eng backdoor adjustment")


if __name__ == "__main__":
    SUB.mkdir(exist_ok=True)
    build_b_stub()
    build_c_stub()
