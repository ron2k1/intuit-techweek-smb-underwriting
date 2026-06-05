"""Deliverable B: cohort × loan-age cumulative default fraction.

CDR_{w,a} = |{ i in A_w : t_i <= 7a }| / |A_w|

A_w = applicants from the team's own approved set (Deliverable A decision == 1)
whose application date falls in cohort_week w.

Estimator: average per-row Pr(default by week a | x, approve) from the hazard
model (Deliverable A side-output). 90% intervals combine row-level hazard
dispersion across the ensemble PD models with a cohort-size Wilson half-width.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import _wilson_half_width
from src.deliverable_a_pipeline import (
    CSV_DIR,
    DEFAULT_NPV_BUFFER_RATE,
    REPORT_DIR,
    SUBMISSION_DIR,
    add_application_features,
    decision_from_npv,
    ensemble_predictions,
    feature_columns,
    fit_calibrated_models,
    fit_segment_meta_calibrator,
    apply_segment_meta_calibrator,
    train_approval_propensity,
)
from src.timing import (
    N_WEEKS,
    expected_default_day,
    fit_hazard_model,
    fit_recovery_model,
)
from src.economics import expected_npv
from src.conformal import bin_level_coverage, build_pd_intervals, pd_interval_bin_table


COHORT_WEEKS = list(range(1, 14))


def load_cohort_definitions(path: Path) -> pd.DataFrame:
    cohort_def = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return cohort_def


def assign_cohort_week(application_timestamp: pd.Series, cohort_def: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(application_timestamp, errors="coerce")
    out = pd.Series(index=ts.index, dtype="Int64")
    for _, row in cohort_def.iterrows():
        mask = (ts >= row["start_date"]) & (ts <= row["end_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        out[mask] = int(row["cohort_week"])
    return out


def cdr_with_intervals(
    cumulative_per_row: np.ndarray,
    cohort_index: np.ndarray,
    cohort_week: int,
    age_weeks: int,
) -> tuple[float, float, float, int]:
    """Estimator + Wilson interval for CDR_{w,a}.

    cumulative_per_row[i, a-1] = Pr(default by week a | x_i, approve)
    """
    rows = np.where(cohort_index == cohort_week)[0]
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    per_row = cumulative_per_row[rows, age_weeks - 1]
    point = float(per_row.mean())
    # Sampling variance of mean from i.i.d. Bernoulli draws with per-row prob p_i.
    # Var(mean) = sum_i p_i(1-p_i) / n^2 ; half-width = z * sqrt(var).
    var_mean = float((per_row * (1.0 - per_row)).sum() / max(n * n, 1))
    z = 1.6448536269514722
    half_sampling = z * np.sqrt(var_mean)
    half_wilson = _wilson_half_width(point, n)
    half = max(half_sampling, half_wilson)
    lower = max(0.0, point - half)
    upper = min(1.0, point + half)
    return point, lower, upper, n


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    train_x = train_fe[numeric + categorical]
    validation_x = validation_fe[numeric + categorical]
    test_x = test_fe[numeric + categorical]

    # Match A's pipeline for the PD ensemble + propensity, then add hazard model.
    prop_pipe = train_approval_propensity(train_x, (train["prior_decision"] == 1).astype(int), numeric, categorical)
    prop_train = np.clip(prop_pipe.predict_proba(train_x)[:, 1], 0.001, 0.999)
    prop_val = np.clip(prop_pipe.predict_proba(validation_x)[:, 1], 0.001, 0.999)
    prop_test = np.clip(prop_pipe.predict_proba(test_x)[:, 1], 0.001, 0.999)

    labeled = train[train["default_flag"].notna()].copy()
    models, _, cal_idx = fit_calibrated_models(train_x, labeled, numeric, categorical, prop_train)
    cal_x = train_x.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    cal_point, cal_matrix = ensemble_predictions(models, cal_x)

    val_point, val_matrix = ensemble_predictions(models, validation_x)
    test_point, test_matrix = ensemble_predictions(models, test_x)

    seg_cal, day_center, day_scale = fit_segment_meta_calibrator(
        cal_point, train_fe.loc[cal_idx], prop_train[cal_idx], cal_y
    )
    cal_point, cal_matrix = apply_segment_meta_calibrator(
        seg_cal, cal_point, cal_matrix, train_fe.loc[cal_idx], prop_train[cal_idx], day_center, day_scale
    )
    val_point, val_matrix = apply_segment_meta_calibrator(
        seg_cal, val_point, val_matrix, validation_fe, prop_val, day_center, day_scale
    )
    test_point, test_matrix = apply_segment_meta_calibrator(
        seg_cal, test_point, test_matrix, test_fe, prop_test, day_center, day_scale
    )

    # Hazard model gives Pr(default by week a | x, approve).
    labeled_x = train_x.loc[labeled.index]
    sw = 1.0 / np.clip(prop_train[labeled.index.to_numpy()], 0.08, 1.0)
    sw = np.clip(sw / sw.mean(), 0.25, 8.0)
    hazard = fit_hazard_model(labeled_x, labeled, sample_weight=sw)
    rec_model = fit_recovery_model(train_x, train[train["default_flag"] == 1])

    _, val_cum = hazard.predict_curves(validation_x)
    _, test_cum = hazard.predict_curves(test_x)
    val_t_star = expected_default_day(val_cum)
    test_t_star = expected_default_day(test_cum)
    val_rec = rec_model.predict_rate(validation_x)
    test_rec = rec_model.predict_rate(test_x)

    # Compute decisions using brief NPV.
    val_amount = validation["requested_amount"].to_numpy(float)
    test_amount = test["requested_amount"].to_numpy(float)
    val_enpv = expected_npv(val_amount, val_point, val_t_star, val_rec)
    test_enpv = expected_npv(test_amount, test_point, test_t_star, test_rec)
    val_decision = decision_from_npv(val_enpv, val_amount, buffer_rate=DEFAULT_NPV_BUFFER_RATE)
    test_decision = decision_from_npv(test_enpv, test_amount, buffer_rate=DEFAULT_NPV_BUFFER_RATE)

    # Prefer the exact A submission decisions if A has already been generated.
    # This keeps B conditional on the same approved book that will be submitted.
    submission_a_path = SUBMISSION_DIR / "submission_A_decisions.csv"
    if submission_a_path.exists():
        submission_a = pd.read_csv(submission_a_path, usecols=["applicant_id", "decision"])
        val_decision = (
            validation[["applicant_id"]]
            .merge(submission_a, on="applicant_id", how="left")["decision"]
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        test_decision = (
            test[["applicant_id"]]
            .merge(submission_a, on="applicant_id", how="left")["decision"]
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

    # Build cohort × age CDR table for submission.
    cohort_def = load_cohort_definitions(Path("data") / "cohort_week_definitions.csv")
    val_cohort = assign_cohort_week(validation["application_timestamp"], cohort_def).to_numpy()
    test_cohort = assign_cohort_week(test["application_timestamp"], cohort_def).to_numpy()

    # Pool approved rows from validation and test for cohort aggregation.
    pooled_cum = np.vstack([val_cum, test_cum])
    pooled_cohort = np.concatenate([val_cohort, test_cohort])
    pooled_decision = np.concatenate([val_decision, test_decision])
    approved_mask = pooled_decision == 1
    approved_cum = pooled_cum[approved_mask]
    approved_cohort = pooled_cohort[approved_mask]

    rows = []
    for w in COHORT_WEEKS:
        for a in COHORT_WEEKS:
            point, lo, hi, n_w = cdr_with_intervals(approved_cum, approved_cohort, w, a)
            rows.append(
                {
                    "cohort_week": w,
                    "loan_age_weeks": a,
                    "cumulative_default_rate": point,
                    "cdr_lower_90": lo,
                    "cdr_upper_90": hi,
                    "n_in_cohort": n_w,
                }
            )
    submission = pd.DataFrame(rows)
    submission_full = submission.copy()
    official_b = submission[[
        "cohort_week",
        "loan_age_weeks",
        "cumulative_default_rate",
        "cdr_lower_90",
        "cdr_upper_90",
    ]]
    official_b.to_csv(SUBMISSION_DIR / "submission_B_trajectory.csv", index=False)
    # Keep the legacy/debug filename for notebooks that may still reference it.
    official_b.to_csv(SUBMISSION_DIR / "submission_B_cohort_default_rates.csv", index=False)
    submission_full.to_csv(REPORT_DIR / "deliverable_b_with_counts.csv", index=False)

    diagnostics = {
        "n_approved_val": int(val_decision.sum()),
        "n_approved_test": int(test_decision.sum()),
        "n_approved_total": int(approved_mask.sum()),
        "by_cohort_counts": {int(w): int((approved_cohort == w).sum()) for w in COHORT_WEEKS},
        "cdr_week13_age13_mean": float(submission.loc[(submission["cohort_week"] == 13) & (submission["loan_age_weeks"] == 13), "cumulative_default_rate"].mean()),
        "cdr_age13_avg_across_cohorts": float(submission.loc[submission["loan_age_weeks"] == 13, "cumulative_default_rate"].mean()),
        "cdr_age1_avg_across_cohorts": float(submission.loc[submission["loan_age_weeks"] == 1, "cumulative_default_rate"].mean()),
        "mean_interval_width": float((submission["cdr_upper_90"] - submission["cdr_lower_90"]).mean()),
        "npv_buffer_rate": float(DEFAULT_NPV_BUFFER_RATE),
        "used_submission_a_decisions": bool(submission_a_path.exists()),
    }
    (REPORT_DIR / "deliverable_b_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    # Save curves so Deliverable C can reuse without retraining.
    np.savez(
        Path("outputs") / "deliverable_b_curves.npz",
        validation_cumulative=val_cum,
        test_cumulative=test_cum,
        validation_pd=val_point,
        test_pd=test_point,
        validation_t_star=val_t_star,
        test_t_star=test_t_star,
        validation_recovery=val_rec,
        test_recovery=test_rec,
        validation_decision=val_decision,
        test_decision=test_decision,
        validation_cohort=val_cohort,
        test_cohort=test_cohort,
    )

    print("Wrote", SUBMISSION_DIR / "submission_B_trajectory.csv")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
