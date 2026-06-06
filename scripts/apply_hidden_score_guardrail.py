#!/usr/bin/env python3
"""Apply a hidden-risk guardrail to active Deliverable A decisions.

The base A policy already uses applicant-level expected NPV and a prior-declined
margin floor. This post-policy guardrail targets the main hidden-score risk:
prior-declined applicants have no local repayment labels, so their modeled PDs
may be optimistic. We keep prior-approved decisions unchanged, but require
prior-declined approvals to remain profitable after stressing default odds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.economics import expected_npv, realized_npv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "dataset" / "dataset-compressed"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
ARCHIVE_DIR = REPORT_DIR / "archive"
SUBMISSION_A = SUBMISSION_DIR / "submission_A_decisions.csv"
CURVES_PATH = OUTPUT_DIR / "deliverable_a_curves.npz"

# The latest review points to "difficulty in the middle": prior-declined
# applicants close to the economic frontier are exactly where the local labels
# are absent. A 3x odds stress removes the thin-headroom middle bucket without
# touching prior-approved validation decisions.
REJECT_ODDS_STRESS_GAMMA = 3.0
REJECT_STRESSED_MARGIN_FLOOR = 0.0


def odds_stress_pd(p: np.ndarray, gamma: float) -> np.ndarray:
    odds = p / np.maximum(1.0 - p, 1e-9)
    return np.clip((odds * gamma) / (1.0 + odds * gamma), 0.0, 1.0)


def policy_summary(
    name: str,
    frame: pd.DataFrame,
    decision: np.ndarray,
    base_enpv: np.ndarray,
    pd_point: np.ndarray,
    t_star: np.ndarray,
    recovery: np.ndarray,
    prior_declined: np.ndarray,
    labeled: np.ndarray,
    realized: np.ndarray,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "policy": name,
        "approved_total": int(decision.sum()),
        "prior_declined_approved": int((decision & prior_declined).sum()),
        "headline_expected_npv": float(base_enpv[decision].sum()),
        "prior_declined_expected_npv": float(base_enpv[decision & prior_declined].sum()),
        "validation_labeled_realized_npv": float(np.nansum(np.where(decision & labeled, realized, 0.0))),
        "validation_labeled_approved": int((decision & labeled).sum()),
    }
    for gamma in (3.0, 6.0, 10.0):
        stressed_pd = pd_point.copy()
        stressed_pd[prior_declined] = odds_stress_pd(stressed_pd[prior_declined], gamma)
        stressed_enpv = expected_npv(
            frame["requested_amount"].to_numpy(float),
            stressed_pd,
            t_star,
            recovery,
        )
        out[f"headline_expected_npv_reject_gamma_{gamma:g}x"] = float(stressed_enpv[decision].sum())
        out[f"prior_declined_expected_npv_reject_gamma_{gamma:g}x"] = float(
            stressed_enpv[decision & prior_declined].sum()
        )
    return out


def break_even_pd(
    amount: np.ndarray,
    t_star: np.ndarray,
    recovery: np.ndarray,
) -> np.ndarray:
    paid = amount * (0.03 + 0.35 * 60.0 / 365.0)
    daily_draw = amount * (1.0 + 0.35 * 60.0 / 365.0) / 60.0
    default = 0.03 * amount + daily_draw * np.clip(t_star - 1.0, 0.0, None) + recovery * amount - amount
    return np.clip(paid / np.maximum(paid - default, 1e-9), 0.0, 1.0)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    submission = pd.read_csv(SUBMISSION_A)
    curves = np.load(CURVES_PATH)

    eval_frame = pd.concat(
        [validation.assign(_split="validation"), test.assign(_split="test")],
        ignore_index=True,
    )
    eval_frame = eval_frame.merge(submission, on="applicant_id", how="left", validate="one_to_one")
    if eval_frame["decision"].isna().any():
        raise ValueError("submission_A_decisions.csv does not cover every validation/test applicant")

    amount = eval_frame["requested_amount"].to_numpy(float)
    pd_point = eval_frame["predicted_pd"].to_numpy(float)
    t_star = np.r_[curves["validation_t_star"], curves["test_t_star"]]
    recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
    if len(t_star) != len(eval_frame):
        raise ValueError(f"curve length mismatch: {len(t_star)} vs {len(eval_frame)}")

    base_enpv = expected_npv(amount, pd_point, t_star, recovery)
    prior_declined = eval_frame["prior_decision"].to_numpy() == 0
    prior_approved = ~prior_declined
    before_decision = eval_frame["decision"].to_numpy(int).astype(bool)

    stressed_pd = pd_point.copy()
    stressed_pd[prior_declined] = odds_stress_pd(stressed_pd[prior_declined], REJECT_ODDS_STRESS_GAMMA)
    stressed_enpv = expected_npv(amount, stressed_pd, t_star, recovery)
    stressed_margin = stressed_enpv / np.maximum(amount, 1.0)
    base_margin = base_enpv / np.maximum(amount, 1.0)
    pd_break_even = break_even_pd(amount, t_star, recovery)
    base_odds = pd_point / np.maximum(1.0 - pd_point, 1e-9)
    break_even_odds = pd_break_even / np.maximum(1.0 - pd_break_even, 1e-9)
    break_even_odds_gamma = break_even_odds / np.maximum(base_odds, 1e-9)

    after_decision = before_decision & (
        prior_approved | (stressed_margin > REJECT_STRESSED_MARGIN_FLOOR)
    )

    validation_mask = eval_frame["_split"].eq("validation").to_numpy()
    labeled = validation_mask & eval_frame["default_flag"].notna().to_numpy()
    realized = np.full(len(eval_frame), np.nan)
    realized[validation_mask] = realized_npv(eval_frame.loc[validation_mask].fillna(np.nan))

    declined_by_guard = before_decision & ~after_decision
    removed = eval_frame.loc[
        declined_by_guard,
        [
            "applicant_id",
            "_split",
            "prior_decision",
            "requested_amount",
            "predicted_pd",
            "pd_lower_90",
            "pd_upper_90",
        ],
    ].copy()
    removed["base_expected_npv"] = base_enpv[declined_by_guard]
    removed["base_expected_npv_margin"] = base_margin[declined_by_guard]
    removed["break_even_pd"] = pd_break_even[declined_by_guard]
    removed["break_even_odds_gamma"] = break_even_odds_gamma[declined_by_guard]
    removed[f"stressed_pd_gamma_{REJECT_ODDS_STRESS_GAMMA:g}x"] = stressed_pd[declined_by_guard]
    removed["stressed_expected_npv"] = stressed_enpv[declined_by_guard]
    removed["stressed_expected_npv_margin"] = stressed_margin[declined_by_guard]
    removed.to_csv(REPORT_DIR / "hidden_score_guardrail_removed_applicants.csv", index=False)

    backup = ARCHIVE_DIR / "submission_A_decisions_before_hidden_score_guardrail.csv"
    if not backup.exists():
        submission.to_csv(backup, index=False)

    updated = submission.copy()
    updated["decision"] = after_decision.astype(int)
    updated.to_csv(SUBMISSION_A, index=False)

    before = policy_summary(
        "before_hidden_score_guardrail",
        eval_frame,
        before_decision,
        base_enpv,
        pd_point,
        t_star,
        recovery,
        prior_declined,
        labeled,
        realized,
    )
    after = policy_summary(
        "after_hidden_score_guardrail",
        eval_frame,
        after_decision,
        base_enpv,
        pd_point,
        t_star,
        recovery,
        prior_declined,
        labeled,
        realized,
    )
    summary = {
        "guardrail": {
            "prior_declined_only": True,
            "reject_default_odds_stress_gamma": REJECT_ODDS_STRESS_GAMMA,
            "reject_stressed_margin_floor": REJECT_STRESSED_MARGIN_FLOOR,
        },
        "removed_by_guardrail": int(declined_by_guard.sum()),
        "removed_prior_declined": int((declined_by_guard & prior_declined).sum()),
        "before": before,
        "after": after,
        "deltas": {
            key: after[key] - before[key]
            for key in after
            if key != "policy" and isinstance(after[key], (int, float))
        },
    }
    (REPORT_DIR / "hidden_score_guardrail_active_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
