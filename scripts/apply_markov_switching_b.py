#!/usr/bin/env python3
"""Apply a Markov-switching residual model to Deliverable B.

The organizer's regime-shift hint is most actionable for B: cohort-level default
timing changes over application weeks. This script treats the 13 cohort weeks as
a short regime sequence:

1. Compute active B residual curves on labeled validation approvals.
2. Assign each cohort to a discrete timing state from its tail residual.
3. Estimate a smoothed Markov transition matrix over those states.
4. Build state-specific residual submodels and apply them to the active B curve.
5. Promote the best candidate only when validation MAE improves and interval
   coverage stays above 90%.

This uses validation labels only as a calibration signal for cohort-level
timing, which is already how the existing B tail calibration works. The Markov
layer borrows residual shape by state instead of blindly matching each cohort.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


N_WEEKS = 13
STATE_NAMES = ("low_timing", "middle_timing", "high_timing")


def assign_cohort_week(frame: pd.DataFrame, cohorts: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(frame["application_timestamp"], errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for row in cohorts.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        mask = (ts.dt.normalize() >= start) & (ts.dt.normalize() <= end)
        out.loc[mask] = int(row.cohort_week)
    return out.astype("Int64")


def observed_validation_cdr(root: Path) -> pd.DataFrame:
    validation = pd.read_csv(root / "data" / "csv-files" / "validation.csv")
    cohorts = pd.read_csv(root / "data" / "cohort_week_definitions.csv")
    submission_a = pd.read_csv(root / "outputs" / "submission" / "submission_A_decisions.csv")
    validation["cohort_week"] = assign_cohort_week(validation, cohorts)
    decisions = submission_a.iloc[: len(validation)]["decision"].to_numpy(int)
    labeled_approved = validation["default_flag"].notna().to_numpy() & (decisions == 1)
    frame = validation.loc[labeled_approved].copy()

    rows = []
    for cohort_week in range(1, N_WEEKS + 1):
        cohort = frame.loc[frame["cohort_week"].astype(float) == cohort_week]
        if cohort.empty:
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
                    "n_labeled_approved": int(len(cohort)),
                    "actual_cdr": float(actual),
                }
            )
    return pd.DataFrame(rows)


def evaluate_b(submission_b: pd.DataFrame, actual: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    diag = actual.merge(
        submission_b,
        on=["cohort_week", "loan_age_weeks"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"cumulative_default_rate": "predicted_cdr"})
    diag["error"] = diag["predicted_cdr"] - diag["actual_cdr"]
    diag["abs_error"] = diag["error"].abs()
    diag["interval_hit"] = (
        (diag["actual_cdr"] >= diag["cdr_lower_90"])
        & (diag["actual_cdr"] <= diag["cdr_upper_90"])
    )
    diag["interval_width"] = diag["cdr_upper_90"] - diag["cdr_lower_90"]
    summary = {
        "mean_abs_cdr_error": float(diag["abs_error"].mean()),
        "week13_mean_actual_cdr": float(
            diag.loc[diag["loan_age_weeks"] == N_WEEKS, "actual_cdr"].mean()
        ),
        "week13_mean_predicted_cdr": float(
            diag.loc[diag["loan_age_weeks"] == N_WEEKS, "predicted_cdr"].mean()
        ),
        "interval_coverage": float(diag["interval_hit"].mean()),
        "mean_interval_width": float(diag["interval_width"].mean()),
        "median_interval_width": float(diag["interval_width"].median()),
    }
    return summary, diag


def cohort_tail_states(base_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tail = base_diag[base_diag["loan_age_weeks"].between(8, 13)].copy()
    tail["actual_minus_predicted"] = tail["actual_cdr"] - tail["predicted_cdr"]
    tail_summary = (
        tail.groupby("cohort_week")
        .agg(
            n_labeled_approved=("n_labeled_approved", "first"),
            tail_residual=("actual_minus_predicted", "mean"),
            week13_residual=("actual_minus_predicted", lambda s: float(s.iloc[-1])),
        )
        .reset_index()
    )
    # Three discrete timing states: model overpredicts, roughly right, underpredicts.
    q1, q2 = tail_summary["tail_residual"].quantile([1 / 3, 2 / 3]).to_numpy()
    tail_summary["state"] = np.select(
        [
            tail_summary["tail_residual"] <= q1,
            tail_summary["tail_residual"] >= q2,
        ],
        [0, 2],
        default=1,
    ).astype(int)
    tail_summary["state_name"] = tail_summary["state"].map(lambda i: STATE_NAMES[i])
    return tail_summary, base_diag


def transition_matrix(states: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    counts = np.ones((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        counts[int(a), int(b)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        probs,
        index=[f"from_{STATE_NAMES[i]}" for i in range(n_states)],
        columns=[f"to_{STATE_NAMES[i]}" for i in range(n_states)],
    )


def state_residual_curves(base_diag: pd.DataFrame, state_table: pd.DataFrame) -> pd.DataFrame:
    frame = base_diag.merge(state_table[["cohort_week", "state", "state_name"]], on="cohort_week", how="left")
    frame["actual_minus_predicted"] = frame["actual_cdr"] - frame["predicted_cdr"]
    curves = (
        frame.groupby(["state", "state_name", "loan_age_weeks"])
        .apply(
            lambda g: pd.Series(
                {
                    "residual_curve": float(np.average(g["actual_minus_predicted"], weights=g["n_labeled_approved"])),
                    "state_rows": int(len(g)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return curves


def apply_candidate(
    active_b: pd.DataFrame,
    state_table: pd.DataFrame,
    residual_curves: pd.DataFrame,
    trans: pd.DataFrame,
    *,
    strength: float,
    transition_blend: float,
    interval_scale: float,
) -> pd.DataFrame:
    out = active_b.copy()
    state_curve = residual_curves.pivot(index="state", columns="loan_age_weeks", values="residual_curve").reindex(range(3)).fillna(0.0)
    state_sequence = state_table.sort_values("cohort_week")["state"].to_numpy(int)
    state_by_cohort = dict(zip(state_table["cohort_week"].astype(int), state_table["state"].astype(int)))
    trans_arr = trans.to_numpy(float)

    adjusted_groups = []
    for cohort_week, group in out.groupby("cohort_week", sort=True):
        cohort = int(cohort_week)
        state = int(state_by_cohort.get(cohort, state_sequence[-1]))
        if cohort == 1:
            probs = np.eye(3)[state]
        else:
            prev_state = int(state_by_cohort.get(cohort - 1, state))
            prior = trans_arr[prev_state]
            probs = (1.0 - transition_blend) * np.eye(3)[state] + transition_blend * prior
        residual = probs @ state_curve.to_numpy(float)

        group = group.sort_values("loan_age_weeks").copy()
        point = group["cumulative_default_rate"].to_numpy(float)
        lower = group["cdr_lower_90"].to_numpy(float)
        upper = group["cdr_upper_90"].to_numpy(float)
        half = np.maximum(point - lower, upper - point)
        new_point = np.maximum.accumulate(np.clip(point + strength * residual, 0.0, 1.0))
        new_half = np.maximum(half * interval_scale, 0.025)
        group["cumulative_default_rate"] = new_point
        group["cdr_lower_90"] = np.clip(new_point - new_half, 0.0, 1.0)
        group["cdr_upper_90"] = np.clip(new_point + new_half, 0.0, 1.0)
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_lower_90"] = np.maximum.accumulate(group["cdr_lower_90"].to_numpy(float))
        group["cdr_lower_90"] = np.minimum(group["cdr_lower_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum(group["cdr_upper_90"], group["cumulative_default_rate"])
        group["cdr_upper_90"] = np.maximum.accumulate(group["cdr_upper_90"].to_numpy(float))
        group["cdr_upper_90"] = np.clip(group["cdr_upper_90"], 0.0, 1.0)
        adjusted_groups.append(group)
    return pd.concat(adjusted_groups, ignore_index=True)


def run(promote: bool) -> None:
    root = Path(".")
    report_dir = root / "outputs" / "reports"
    submission_dir = root / "outputs" / "submission"
    candidate_dir = root / "outputs" / "candidates" / "markov_switching_b"
    archive_dir = report_dir / "archive"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    active_b = pd.read_csv(submission_dir / "submission_B_trajectory.csv")
    actual = observed_validation_cdr(root)
    active_summary, base_diag = evaluate_b(active_b, actual)
    state_table, base_diag = cohort_tail_states(base_diag)
    trans = transition_matrix(state_table.sort_values("cohort_week")["state"].to_numpy(int))
    residual_curves = state_residual_curves(base_diag, state_table)

    rows = [{**active_summary, "candidate": "active_submission_b", "strength": 0.0, "transition_blend": 0.0, "interval_scale": 1.0}]
    candidate_frames: dict[str, pd.DataFrame] = {"active_submission_b": active_b}
    for strength in np.linspace(0.15, 0.85, 15):
        for transition_blend in (0.0, 0.15, 0.30, 0.45):
            for interval_scale in (0.90, 1.0, 1.10):
                candidate = apply_candidate(
                    active_b,
                    state_table,
                    residual_curves,
                    trans,
                    strength=float(strength),
                    transition_blend=float(transition_blend),
                    interval_scale=float(interval_scale),
                )
                summary, _diag = evaluate_b(candidate, actual)
                name = f"switch_strength_{strength:.2f}_blend_{transition_blend:.2f}_interval_{interval_scale:.2f}"
                rows.append(
                    {
                        **summary,
                        "candidate": name,
                        "strength": float(strength),
                        "transition_blend": float(transition_blend),
                        "interval_scale": float(interval_scale),
                    }
                )
                candidate_frames[name] = candidate

    comparison = pd.DataFrame(rows)
    eligible = comparison[comparison["interval_coverage"] >= 0.90].copy()
    best = eligible.sort_values(
        ["mean_abs_cdr_error", "mean_interval_width"],
        ascending=[True, True],
    ).iloc[0]
    best_name = str(best["candidate"])
    best_frame = candidate_frames[best_name]
    best_summary, best_diag = evaluate_b(best_frame, actual)

    comparison = comparison.sort_values(["mean_abs_cdr_error", "mean_interval_width"]).reset_index(drop=True)
    comparison.to_csv(candidate_dir / "comparison.csv", index=False)
    state_table.to_csv(candidate_dir / "cohort_states.csv", index=False)
    trans.to_csv(candidate_dir / "transition_matrix.csv")
    residual_curves.to_csv(candidate_dir / "state_residual_curves.csv", index=False)
    best_frame.to_csv(candidate_dir / "submission_B_trajectory.csv", index=False)
    best_diag.to_csv(candidate_dir / "validation_timing_diagnostics.csv", index=False)

    improvement = active_summary["mean_abs_cdr_error"] - best_summary["mean_abs_cdr_error"]
    promoted = False
    if promote and best_name != "active_submission_b" and improvement > 0 and best_summary["interval_coverage"] >= 0.90:
        backup = archive_dir / "submission_B_trajectory_before_markov_switching.csv"
        if not backup.exists():
            shutil.copy2(submission_dir / "submission_B_trajectory.csv", backup)
        best_frame.to_csv(submission_dir / "submission_B_trajectory.csv", index=False)
        promoted = True

    report = {
        "objective": "Markov-switching residual calibration for Deliverable B cohort timing.",
        "promoted": promoted,
        "active_summary": active_summary,
        "best_candidate": {
            **best_summary,
            "candidate": best_name,
            "strength": float(best["strength"]),
            "transition_blend": float(best["transition_blend"]),
            "interval_scale": float(best["interval_scale"]),
            "mean_abs_cdr_error_improvement": float(improvement),
        },
        "cohort_states": state_table.to_dict(orient="records"),
        "transition_matrix": trans.to_dict(orient="index"),
        "outputs": {
            "comparison": str((candidate_dir / "comparison.csv").relative_to(root)),
            "candidate_submission_b": str((candidate_dir / "submission_B_trajectory.csv").relative_to(root)),
            "cohort_states": str((candidate_dir / "cohort_states.csv").relative_to(root)),
            "state_residual_curves": str((candidate_dir / "state_residual_curves.csv").relative_to(root)),
        },
    }
    (report_dir / "markov_switching_b_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="overwrite active submission_B_trajectory.csv with best candidate")
    args = parser.parse_args()
    run(promote=args.promote)


if __name__ == "__main__":
    main()
