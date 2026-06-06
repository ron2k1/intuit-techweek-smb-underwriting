#!/usr/bin/env python3
"""Audit the economic middle/frontier of the active A policy.

The "difficulty in the middle" is the approval frontier: rows close to
break-even, especially prior-declined rows with no local repayment labels. This
report makes that region explicit so the final policy is not justified by a
single global model metric.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.economics import expected_npv, npv_default, npv_repaid, realized_npv  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
SUBMISSION_A = PROJECT_ROOT / "outputs" / "submission" / "submission_A_decisions.csv"
CURVES_PATH = PROJECT_ROOT / "outputs" / "deliverable_a_curves.npz"


def odds(p: np.ndarray) -> np.ndarray:
    return p / np.maximum(1.0 - p, 1e-9)


def odds_stress_pd(p: np.ndarray, gamma: float) -> np.ndarray:
    stressed_odds = odds(p) * gamma
    return np.clip(stressed_odds / (1.0 + stressed_odds), 0.0, 1.0)


def q(values: np.ndarray, probs: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {f"q{int(p * 100):02d}": float("nan") for p in probs}
    return {f"q{int(p * 100):02d}": float(v) for p, v in zip(probs, np.quantile(values, probs))}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission = pd.read_csv(SUBMISSION_A)
    curves = np.load(CURVES_PATH)

    frame = pd.concat(
        [validation.assign(_split="validation"), test.assign(_split="test")],
        ignore_index=True,
    ).merge(submission, on="applicant_id", how="left", validate="one_to_one")
    if frame["decision"].isna().any():
        raise ValueError("submission_A_decisions.csv does not cover every validation/test applicant")

    amount = frame["requested_amount"].to_numpy(float)
    pd_point = frame["predicted_pd"].to_numpy(float)
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    enpv = expected_npv(amount, pd_point, t_star, recovery)
    margin = enpv / np.maximum(amount, 1.0)
    paid = npv_repaid(amount)
    default = npv_default(amount, t_star, recovery * amount)
    break_even_pd = np.clip(paid / np.maximum(paid - default, 1e-9), 0.0, 1.0)
    break_even_gamma = odds(break_even_pd) / np.maximum(odds(pd_point), 1e-9)

    approved = frame["decision"].to_numpy(int).astype(bool)
    prior_declined = frame["prior_decision"].to_numpy() == 0
    labeled = frame["_split"].eq("validation").to_numpy() & frame["default_flag"].notna().to_numpy()
    realized = np.full(len(frame), np.nan)
    validation_mask = frame["_split"].eq("validation").to_numpy()
    realized[validation_mask] = realized_npv(frame.loc[validation_mask])

    margin_bins = [
        (-np.inf, -0.02, "negative_lt_minus_2pct"),
        (-0.02, 0.0, "near_negative_minus_2pct_to_0"),
        (0.0, 0.01, "thin_positive_0_to_1pct"),
        (0.01, 0.03, "low_positive_1_to_3pct"),
        (0.03, 0.06, "middle_positive_3_to_6pct"),
        (0.06, np.inf, "strong_positive_gt_6pct"),
    ]
    margin_rows = []
    for lo, hi, name in margin_bins:
        idx = (margin > lo) & (margin <= hi)
        idx_approved = idx & approved
        idx_labeled_approved = idx_approved & labeled
        margin_rows.append(
            {
                "margin_bin": name,
                "rows": int(idx.sum()),
                "approved": int(idx_approved.sum()),
                "prior_declined_approved": int((idx_approved & prior_declined).sum()),
                "mean_pd_approved": float(np.nanmean(pd_point[idx_approved])) if idx_approved.any() else np.nan,
                "expected_npv_approved": float(enpv[idx_approved].sum()),
                "labeled_validation_approved": int(idx_labeled_approved.sum()),
                "labeled_validation_default_rate_approved": float(frame.loc[idx_labeled_approved, "default_flag"].mean())
                if idx_labeled_approved.any()
                else np.nan,
                "labeled_validation_realized_npv_approved": float(np.nansum(realized[idx_labeled_approved])),
            }
        )

    gamma_bins = [
        (0.0, 2.0, "lte_2x"),
        (2.0, 3.0, "2x_to_3x"),
        (3.0, 5.0, "3x_to_5x"),
        (5.0, 10.0, "5x_to_10x"),
        (10.0, np.inf, "gt_10x"),
    ]
    gamma_rows = []
    for lo, hi, name in gamma_bins:
        idx = approved & prior_declined & (break_even_gamma > lo) & (break_even_gamma <= hi)
        gamma_rows.append(
            {
                "break_even_odds_gamma_bin": name,
                "prior_declined_approved": int(idx.sum()),
                "mean_pd": float(np.nanmean(pd_point[idx])) if idx.any() else np.nan,
                "mean_margin": float(np.nanmean(margin[idx])) if idx.any() else np.nan,
                "expected_npv": float(enpv[idx].sum()),
                "mean_prior_underwriter_score": float(frame.loc[idx, "prior_underwriter_score"].mean())
                if idx.any()
                else np.nan,
                "mean_owner_credit_band": float(frame.loc[idx, "owner_personal_credit_band"].mean())
                if idx.any()
                else np.nan,
                "mean_utilization": float(frame.loc[idx, "aggregate_credit_utilization"].mean()) if idx.any() else np.nan,
                "mean_invoice_delinquency": float(frame.loc[idx, "invoice_payment_delinquency_rate"].mean())
                if idx.any()
                else np.nan,
            }
        )

    stress_rows = []
    for gamma in [1.0, 2.0, 3.0, 6.0, 10.0]:
        stressed_pd = pd_point.copy()
        stressed_pd[prior_declined] = odds_stress_pd(stressed_pd[prior_declined], gamma)
        stressed_enpv = expected_npv(amount, stressed_pd, t_star, recovery)
        stress_rows.append(
            {
                "prior_declined_default_odds_gamma": gamma,
                "headline_expected_npv": float(stressed_enpv[approved].sum()),
                "prior_declined_expected_npv": float(stressed_enpv[approved & prior_declined].sum()),
            }
        )

    margin_table = pd.DataFrame(margin_rows)
    gamma_table = pd.DataFrame(gamma_rows)
    stress_table = pd.DataFrame(stress_rows)
    margin_table.to_csv(REPORT_DIR / "middle_frontier_margin_bins.csv", index=False)
    gamma_table.to_csv(REPORT_DIR / "middle_frontier_prior_declined_gamma_bins.csv", index=False)
    stress_table.to_csv(REPORT_DIR / "middle_frontier_reject_stress.csv", index=False)

    summary = {
        "approved_total": int(approved.sum()),
        "prior_declined_approved": int((approved & prior_declined).sum()),
        "headline_expected_npv": float(enpv[approved].sum()),
        "labeled_validation_realized_npv": float(np.nansum(np.where(approved & labeled, realized, 0.0))),
        "labeled_validation_approved": int((approved & labeled).sum()),
        "margin_quantiles_all": q(margin, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
        "margin_quantiles_approved": q(margin[approved], [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
        "prior_declined_break_even_gamma_min_approved": float(np.nanmin(break_even_gamma[approved & prior_declined])),
        "prior_declined_break_even_gamma_q25_approved": float(np.nanquantile(break_even_gamma[approved & prior_declined], 0.25)),
        "prior_declined_break_even_gamma_median_approved": float(np.nanquantile(break_even_gamma[approved & prior_declined], 0.50)),
        "recommendation": (
            "Keep LightGBM as the global PD backbone, but govern the middle with economics, "
            "prior-declined odds-stress, and support diagnostics rather than a separate "
            "high-variance local model."
        ),
    }
    (REPORT_DIR / "middle_frontier_policy_audit.json").write_text(json.dumps(summary, indent=2))

    report = [
        "# Middle Frontier Policy Audit",
        "",
        "The active policy keeps LightGBM as the global calibrated PD backbone, then applies an economic decision layer.",
        "The hard region is not all applicants; it is the break-even frontier and especially prior-declined approvals with no local labels.",
        "",
        "## Summary",
        f"- Approved validation+test applicants: {summary['approved_total']:,}",
        f"- Prior-declined approvals: {summary['prior_declined_approved']:,}",
        f"- Headline expected NPV: ${summary['headline_expected_npv']:,.0f}",
        f"- Labeled-validation realized NPV: ${summary['labeled_validation_realized_npv']:,.0f}",
        f"- Labeled-validation approved: {summary['labeled_validation_approved']:,}",
        f"- Minimum prior-declined approved break-even odds gamma: {summary['prior_declined_break_even_gamma_min_approved']:.2f}x",
        "",
        "## Interpretation",
        "- Prior-approved near-frontier validation approvals are locally labeled and were not harmed by the tightened guardrail.",
        "- Prior-declined frontier approvals remain unlabeled, so the guardrail now removes approvals that do not survive a 3x default-odds stress.",
        "- This is a segmentation layer on top of LightGBM, not a replacement for the backbone model.",
        "",
        "## Output Tables",
        "- `middle_frontier_margin_bins.csv`",
        "- `middle_frontier_prior_declined_gamma_bins.csv`",
        "- `middle_frontier_reject_stress.csv`",
    ]
    (REPORT_DIR / "middle_frontier_policy_audit.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
