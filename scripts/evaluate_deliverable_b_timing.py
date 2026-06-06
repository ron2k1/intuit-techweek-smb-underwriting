#!/usr/bin/env python3
"""Backtest Deliverable B timing on labeled validation rows.

This is a partial diagnostic because labels are only observed for prior-approved
validation rows. It checks whether the cumulative default curve tracks
`days_to_default`, not just final PD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


N_WEEKS = 13


def assign_cohort_week(frame: pd.DataFrame, cohorts: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(frame["application_timestamp"], errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for row in cohorts.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        mask = (ts.dt.normalize() >= start) & (ts.dt.normalize() <= end)
        out.loc[mask] = int(row.cohort_week)
    return out.astype("Int64")


def normalize_curves_to_pd(raw_cumulative: np.ndarray, pd_point: np.ndarray) -> np.ndarray:
    terminal = np.clip(raw_cumulative[:, [-1]], 1e-6, 1.0)
    shape = np.maximum.accumulate(np.clip(raw_cumulative / terminal, 0.0, 1.0), axis=1)
    curves = shape * np.clip(pd_point[:, None], 0.0, 1.0)
    return np.maximum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)


def main() -> None:
    root = Path(".")
    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(root / "data" / "csv-files" / "validation.csv")
    cohorts = pd.read_csv(root / "data" / "cohort_week_definitions.csv")
    submission_a = pd.read_csv(root / "outputs" / "submission" / "submission_A_decisions.csv")
    submission_b = pd.read_csv(root / "outputs" / "submission" / "submission_B_trajectory.csv")

    validation["cohort_week"] = assign_cohort_week(validation, cohorts)
    decisions = submission_a.iloc[: len(validation)]["decision"].to_numpy(int)

    labeled_approved = (validation["default_flag"].notna().to_numpy()) & (decisions == 1)
    frame = validation.loc[labeled_approved].copy()
    rows = []
    for cohort_week in range(1, N_WEEKS + 1):
        cohort_mask = frame["cohort_week"].to_numpy(dtype=float) == cohort_week
        if not cohort_mask.any():
            continue
        cohort = frame.loc[cohort_mask]
        default_flag = cohort["default_flag"].fillna(0).to_numpy(float)
        days_to_default = cohort["days_to_default"].to_numpy(float)

        for age_week in range(1, N_WEEKS + 1):
            cutoff = 7 * age_week
            actual = ((default_flag == 1) & (days_to_default <= cutoff)).mean()
            rows.append(
                {
                    "cohort_week": cohort_week,
                    "loan_age_weeks": age_week,
                    "n_labeled_approved": int(len(cohort)),
                    "actual_cdr": float(actual),
                }
            )

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.merge(
        submission_b,
        on=["cohort_week", "loan_age_weeks"],
        how="left",
        validate="one_to_one",
    )
    diagnostics = diagnostics.rename(columns={"cumulative_default_rate": "predicted_cdr"})
    diagnostics["error"] = diagnostics["predicted_cdr"] - diagnostics["actual_cdr"]
    diagnostics["abs_error"] = diagnostics["error"].abs()
    diagnostics["interval_hit"] = (
        (diagnostics["actual_cdr"] >= diagnostics["cdr_lower_90"])
        & (diagnostics["actual_cdr"] <= diagnostics["cdr_upper_90"])
    )
    diagnostics["interval_width"] = diagnostics["cdr_upper_90"] - diagnostics["cdr_lower_90"]
    diagnostics.to_csv(report_dir / "deliverable_b_validation_timing_diagnostics.csv", index=False)

    summary = {
        "n_labeled_validation": int(validation["default_flag"].notna().sum()),
        "n_labeled_approved_under_current_policy": int(labeled_approved.sum()),
        "mean_abs_cdr_error": float(diagnostics["abs_error"].mean()),
        "week13_mean_actual_cdr": float(
            diagnostics.loc[diagnostics["loan_age_weeks"] == N_WEEKS, "actual_cdr"].mean()
        ),
        "week13_mean_predicted_cdr": float(
            diagnostics.loc[diagnostics["loan_age_weeks"] == N_WEEKS, "predicted_cdr"].mean()
        ),
        "interval_coverage": float(diagnostics["interval_hit"].mean()),
        "mean_interval_width": float(diagnostics["interval_width"].mean()),
        "median_interval_width": float(diagnostics["interval_width"].median()),
        "worst_rows": diagnostics.sort_values("abs_error", ascending=False).head(10).to_dict("records"),
    }
    (report_dir / "deliverable_b_validation_timing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
