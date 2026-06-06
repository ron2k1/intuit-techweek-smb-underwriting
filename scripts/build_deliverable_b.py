#!/usr/bin/env python3
"""Build Deliverable B from the final Deliverable A approved set.

This script follows the carry-forward notes in `deliverable_A_learnings.md`:

- use the team's actual A decisions, not all applicants;
- optimize the A policy by expected NPV per dollar, not a fixed PD threshold;
- convert applicant-level PD into cumulative weekly default curves;
- aggregate only approved applicants by origination cohort;
- add finite-sample/model uncertainty and enforce monotonic cumulative rates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.economics import expected_npv, realized_npv


N_WEEKS = 13
Z_90 = 1.645


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
    """Use survival-model shape but make week-13 cumulative default match A PD."""
    terminal = np.clip(raw_cumulative[:, [-1]], 1e-6, 1.0)
    shape = np.clip(raw_cumulative / terminal, 0.0, 1.0)
    shape = np.maximum.accumulate(shape, axis=1)
    curves = shape * np.clip(pd_point[:, None], 0.0, 1.0)
    return np.maximum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)


def tune_policy_buffer(validation: pd.DataFrame, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    """Backtest expected-NPV-per-dollar buffers on labeled validation rows."""
    val_npv = expected_npv(
        validation["requested_amount"].to_numpy(float),
        curves["validation_pd"],
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    labeled = validation["default_flag"].notna().to_numpy()
    realized = realized_npv(validation.loc[labeled])
    amount = validation.loc[labeled, "requested_amount"].to_numpy(float)
    val_npv_labeled = val_npv[labeled]
    pd_labeled = curves["validation_pd"][labeled]

    buffers = np.array(
        [
            -0.02,
            -0.01,
            0.00,
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.0125,
            0.015,
            0.02,
            0.025,
            0.03,
            0.04,
            0.05,
        ],
        dtype=float,
    )
    rows = []
    for buffer in buffers:
        decision = (val_npv_labeled / np.maximum(amount, 1.0)) > buffer
        rows.append(
            {
                "buffer_per_dollar": buffer,
                "approved_labeled": int(decision.sum()),
                "approval_rate_labeled": float(decision.mean()),
                "realized_npv_labeled": float(realized[decision].sum()),
                "expected_npv_labeled": float(val_npv_labeled[decision].sum()),
                "mean_pd_approved_labeled": float(pd_labeled[decision].mean())
                if decision.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("realized_npv_labeled", ascending=False)


def policy_decision(frame: pd.DataFrame, pd_point: np.ndarray, t_star: np.ndarray, recovery: np.ndarray, buffer: float) -> tuple[np.ndarray, np.ndarray]:
    e_npv = expected_npv(
        frame["requested_amount"].to_numpy(float),
        pd_point,
        t_star,
        recovery,
    )
    amount = frame["requested_amount"].to_numpy(float)
    decision = (e_npv / np.maximum(amount, 1.0) > buffer).astype(int)
    return decision, e_npv


def aggregate_b(
    eval_frame: pd.DataFrame,
    curves: np.ndarray,
    approved: np.ndarray,
    template: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved_frame = eval_frame.loc[approved == 1].copy()
    approved_curves = curves[approved == 1]
    global_curve = np.maximum.accumulate(approved_curves.mean(axis=0)) if len(approved_curves) else np.zeros(N_WEEKS)

    rows = []
    diagnostics = []
    for cohort_week in range(1, N_WEEKS + 1):
        mask = (approved_frame["cohort_week"].to_numpy(int) == cohort_week) if len(approved_frame) else np.array([], dtype=bool)
        cohort_curves = approved_curves[mask]
        n = len(cohort_curves)
        if n == 0:
            point = global_curve.copy()
            interval_n = max(len(approved_curves), 1)
            sparse_penalty = 0.08
        else:
            raw_point = cohort_curves.mean(axis=0)
            # Mild empirical-Bayes shrinkage for sparse cohorts.
            shrink = min(0.35, 20.0 / (n + 20.0))
            point = (1.0 - shrink) * raw_point + shrink * global_curve
            interval_n = n
            sparse_penalty = 0.03 / np.sqrt(n)

        point = np.maximum.accumulate(np.clip(point, 0.0, 1.0))
        diagnostics.append(
            {
                "cohort_week": cohort_week,
                "approved_count": int(n),
                "mean_cdr_week13": float(point[-1]),
            }
        )

        for age_week in range(1, N_WEEKS + 1):
            p_i = cohort_curves[:, age_week - 1] if n else approved_curves[:, age_week - 1]
            p = float(point[age_week - 1])
            if len(p_i):
                se = float(np.sqrt(np.sum(np.clip(p_i, 0, 1) * (1 - np.clip(p_i, 0, 1))) / (len(p_i) ** 2)))
            else:
                se = 0.0
            # Local validation coverage for the submitted 13x13 grid was just
            # under 90%; this small calibration buffer clears the coverage target
            # without changing the point forecast used for Straj.
            model_buffer = 0.015 + sparse_penalty
            half_width = Z_90 * se + model_buffer
            rows.append(
                {
                    "cohort_week": cohort_week,
                    "loan_age_weeks": age_week,
                    "cumulative_default_rate": p,
                    "cdr_lower_90": max(0.0, p - half_width),
                    "cdr_upper_90": min(1.0, p + half_width),
                }
            )

    submission = template[["cohort_week", "loan_age_weeks"]].merge(
        pd.DataFrame(rows),
        on=["cohort_week", "loan_age_weeks"],
        how="left",
        validate="one_to_one",
    )
    for col in ["cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"]:
        submission[col] = submission[col].fillna(0.0).clip(0.0, 1.0)

    # Enforce cumulative monotonicity by cohort and valid interval ordering.
    fixed = []
    for _, group in submission.groupby("cohort_week", sort=True):
        group = group.sort_values("loan_age_weeks").copy()
        group["cumulative_default_rate"] = np.maximum.accumulate(group["cumulative_default_rate"].to_numpy())
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_lower_90"] = np.maximum.accumulate(group["cdr_lower_90"].to_numpy())
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum(group["cdr_upper_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum.accumulate(group["cdr_upper_90"].to_numpy())
        group["cdr_upper_90"] = np.clip(group["cdr_upper_90"], 0.0, 1.0)
        fixed.append(group)
    submission = pd.concat(fixed, ignore_index=True)
    return submission, pd.DataFrame(diagnostics)


def labeled_validation_cdr(validation: pd.DataFrame, val_decision: np.ndarray) -> pd.DataFrame:
    """Observed CDR grid for labeled validation rows approved by A.

    This is not a hard target for B because validation labels cover only the
    prior-approved/matured subset. It is useful as a conservative calibration
    signal for cohort-age tail misses.
    """
    labeled_approved = validation["default_flag"].notna().to_numpy() & (val_decision == 1)
    frame = validation.loc[labeled_approved].copy()
    rows = []
    for cohort_week in range(1, N_WEEKS + 1):
        cohort = frame.loc[frame["cohort_week"].astype(float) == cohort_week]
        n = len(cohort)
        if n == 0:
            continue
        default_flag = cohort["default_flag"].fillna(0).to_numpy(float)
        days_to_default = cohort["days_to_default"].to_numpy(float)
        for age_week in range(1, N_WEEKS + 1):
            cutoff = 7 * age_week
            actual = ((default_flag == 1) & (days_to_default <= cutoff)).mean()
            rows.append(
                {
                    "cohort_week": cohort_week,
                    "loan_age_weeks": age_week,
                    "n_labeled_approved": int(n),
                    "actual_cdr_labeled": float(actual),
                }
            )
    return pd.DataFrame(rows)


def calibrate_b_to_labeled_tail(
    submission: pd.DataFrame,
    validation: pd.DataFrame,
    val_decision: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend B point forecasts toward labeled-validation CDRs.

    The blend is intentionally partial. It addresses local cohort-tail misses
    while acknowledging that labeled validation is selected by the old lender.
    """
    actual = labeled_validation_cdr(validation, val_decision)
    if actual.empty:
        return submission, actual

    out = submission.merge(
        actual,
        on=["cohort_week", "loan_age_weeks"],
        how="left",
        validate="one_to_one",
    )
    out["pre_calibration_cdr"] = out["cumulative_default_rate"]
    residual = out["actual_cdr_labeled"] - out["pre_calibration_cdr"]
    n = out["n_labeled_approved"].fillna(0).to_numpy(float)
    age = out["loan_age_weeks"].to_numpy(float)
    credibility = n / (n + 180.0)
    tail_weight = np.where(age >= 8, 0.45, np.where(age >= 5, 0.25, 0.10))
    blend_weight = np.clip(credibility * tail_weight, 0.0, 0.28)
    adjustment = np.where(out["actual_cdr_labeled"].notna(), blend_weight * residual, 0.0)
    out["calibration_weight"] = blend_weight
    out["calibration_adjustment"] = adjustment
    out["cumulative_default_rate"] = np.clip(out["pre_calibration_cdr"] + adjustment, 0.0, 1.0)

    fixed = []
    for _, group in out.groupby("cohort_week", sort=True):
        group = group.sort_values("loan_age_weeks").copy()
        group["cumulative_default_rate"] = np.maximum.accumulate(group["cumulative_default_rate"].to_numpy())
        # Recenter intervals around the calibrated point, retaining at least the
        # original half-width plus an uncertainty surcharge for using selected labels.
        old_half = np.maximum(
            group["pre_calibration_cdr"] - group["cdr_lower_90"],
            group["cdr_upper_90"] - group["pre_calibration_cdr"],
        ).to_numpy(float)
        surcharge = 0.006 * (group["calibration_weight"].to_numpy(float) > 0)
        half = old_half + surcharge
        point = group["cumulative_default_rate"].to_numpy(float)
        group["cdr_lower_90"] = np.clip(point - half, 0.0, 1.0)
        group["cdr_upper_90"] = np.clip(point + half, 0.0, 1.0)
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_lower_90"] = np.maximum.accumulate(group["cdr_lower_90"].to_numpy())
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum(group["cdr_upper_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum.accumulate(group["cdr_upper_90"].to_numpy())
        group["cdr_upper_90"] = np.clip(group["cdr_upper_90"], 0.0, 1.0)
        fixed.append(group)

    calibrated = pd.concat(fixed, ignore_index=True)
    diagnostics = calibrated[
        [
            "cohort_week",
            "loan_age_weeks",
            "n_labeled_approved",
            "actual_cdr_labeled",
            "pre_calibration_cdr",
            "cumulative_default_rate",
            "calibration_weight",
            "calibration_adjustment",
        ]
    ].copy()
    official_cols = [
        "cohort_week",
        "loan_age_weeks",
        "cumulative_default_rate",
        "cdr_lower_90",
        "cdr_upper_90",
    ]
    return calibrated[official_cols], diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-per-dollar", type=float, default=0.005)
    parser.add_argument("--write-updated-a", action="store_true", default=True)
    parser.add_argument(
        "--use-existing-a-decisions",
        action="store_true",
        help="Use the current submission_A_decisions.csv instead of recomputing A from the NPV buffer.",
    )
    args = parser.parse_args()

    root = Path(".")
    data_dir = root / "data"
    csv_dir = data_dir / "csv-files"
    output_dir = root / "outputs"
    report_dir = output_dir / "reports"
    submission_dir = output_dir / "submission"
    archive_dir = report_dir / "archive"
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(csv_dir / "validation.csv")
    test = pd.read_csv(csv_dir / "test.csv")
    cohorts = pd.read_csv(data_dir / "cohort_week_definitions.csv")
    template = pd.read_csv(data_dir / "submission_B_template.csv")
    submission_a = pd.read_csv(submission_dir / "submission_A_decisions.csv")
    npz = np.load(output_dir / "deliverable_a_curves.npz")
    curves = {key: npz[key] for key in npz.files}

    buffer_tuning = tune_policy_buffer(validation, curves)
    buffer_tuning.to_csv(report_dir / "deliverable_b_policy_buffer_tuning.csv", index=False)
    active_val_pd = submission_a.iloc[: len(validation)]["predicted_pd"].to_numpy(float)
    active_test_pd = submission_a.iloc[len(validation) :]["predicted_pd"].to_numpy(float)

    if args.use_existing_a_decisions:
        val_decision = submission_a.iloc[: len(validation)]["decision"].to_numpy(int)
        test_decision = submission_a.iloc[len(validation) :]["decision"].to_numpy(int)
        val_npv = expected_npv(
            validation["requested_amount"].to_numpy(float),
            active_val_pd,
            curves["validation_t_star"],
            curves["validation_recovery"],
        )
        test_npv = expected_npv(
            test["requested_amount"].to_numpy(float),
            active_test_pd,
            curves["test_t_star"],
            curves["test_recovery"],
        )
    else:
        val_decision, val_npv = policy_decision(
            validation,
            curves["validation_pd"],
            curves["validation_t_star"],
            curves["validation_recovery"],
            args.buffer_per_dollar,
        )
        test_decision, test_npv = policy_decision(
            test,
            curves["test_pd"],
            curves["test_t_star"],
            curves["test_recovery"],
            args.buffer_per_dollar,
        )

    if args.write_updated_a and not args.use_existing_a_decisions:
        backup = archive_dir / "submission_A_decisions_no_buffer_backup.csv"
        if not backup.exists():
            submission_a.to_csv(backup, index=False)
        updated_a = submission_a.copy()
        n_val = len(validation)
        updated_a.loc[: n_val - 1, "decision"] = val_decision
        updated_a.loc[n_val:, "decision"] = test_decision
        updated_a.to_csv(submission_dir / "submission_A_decisions.csv", index=False)

    validation["cohort_week"] = assign_cohort_week(validation, cohorts)
    test["cohort_week"] = assign_cohort_week(test, cohorts)
    eval_frame = pd.concat([validation, test], ignore_index=True)
    approved = np.concatenate([val_decision, test_decision])

    pd_for_b_validation = active_val_pd if args.use_existing_a_decisions else curves["validation_pd"]
    pd_for_b_test = active_test_pd if args.use_existing_a_decisions else curves["test_pd"]
    validation_curves = normalize_curves_to_pd(curves["validation_cumulative"], pd_for_b_validation)
    test_curves = normalize_curves_to_pd(curves["test_cumulative"], pd_for_b_test)
    eval_curves = np.vstack([validation_curves, test_curves])

    submission_b, cohort_diag = aggregate_b(eval_frame, eval_curves, approved, template)
    submission_b, b_calibration_diag = calibrate_b_to_labeled_tail(
        submission_b,
        validation,
        val_decision,
    )
    submission_b.to_csv(submission_dir / "submission_B_trajectory.csv", index=False)
    b_calibration_diag.to_csv(report_dir / "deliverable_b_tail_calibration.csv", index=False)

    cohort_extra = (
        eval_frame.assign(approved=approved, predicted_pd=np.concatenate([pd_for_b_validation, pd_for_b_test]))
        .groupby("cohort_week")
        .agg(
            rows=("applicant_id", "size"),
            approved_count=("approved", "sum"),
            approval_rate=("approved", "mean"),
            mean_pd=("predicted_pd", "mean"),
            mean_pd_approved=("predicted_pd", lambda s: float(s[approved[s.index] == 1].mean()) if np.any(approved[s.index] == 1) else np.nan),
            mean_prior_underwriter_score=("prior_underwriter_score", "mean"),
        )
        .reset_index()
    )
    cohort_diag = cohort_diag.merge(cohort_extra, on="cohort_week", how="left")
    cohort_diag = cohort_diag.rename(columns={"approved_count_x": "approved_count"})
    if "approved_count_y" in cohort_diag:
        cohort_diag = cohort_diag.drop(columns=["approved_count_y"])
    cohort_diag.to_csv(report_dir / "deliverable_b_cohort_diagnostics.csv", index=False)

    labeled = validation["default_flag"].notna().to_numpy()
    realized = realized_npv(validation.loc[labeled])
    val_policy_labeled = val_decision[labeled]
    summary = {
        "buffer_per_dollar": args.buffer_per_dollar,
        "validation_labeled_realized_npv": float(realized[val_policy_labeled == 1].sum()),
        "validation_labeled_approved": int(val_policy_labeled.sum()),
        "validation_labeled_approval_rate": float(val_policy_labeled.mean()),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_eval_total": int(approved.sum()),
        "submission_b_rows": int(len(submission_b)),
        "submission_b_week13_mean": float(
            submission_b.loc[submission_b["loan_age_weeks"] == N_WEEKS, "cumulative_default_rate"].mean()
        ),
    }
    (report_dir / "deliverable_b_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("Wrote", submission_dir / "submission_B_trajectory.csv")
    if args.write_updated_a and not args.use_existing_a_decisions:
        print("Updated", submission_dir / "submission_A_decisions.csv")


if __name__ == "__main__":
    main()
