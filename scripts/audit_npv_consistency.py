#!/usr/bin/env python3
"""Audit active A decisions against the NPV rule and validation realized NPV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.economics import (  # noqa: E402
    DAILY_DRAW_FACTOR,
    ORIGINATION_FEE_RATE,
    PAID_MARGIN_RATE,
    expected_npv,
    realized_npv,
)


def main() -> None:
    root = Path(".")
    validation = pd.read_csv(root / "data" / "csv-files" / "validation.csv")
    test = pd.read_csv(root / "data" / "csv-files" / "test.csv")
    submission_a = pd.read_csv(root / "outputs" / "submission" / "submission_A_decisions.csv")
    curves = np.load(root / "outputs" / "deliverable_a_curves.npz")

    val_decision = submission_a.iloc[: len(validation)]["decision"].to_numpy(int)
    test_decision = submission_a.iloc[len(validation) :]["decision"].to_numpy(int)
    labeled = validation["default_flag"].notna().to_numpy()
    realized = realized_npv(validation.loc[labeled])

    val_enpv = expected_npv(
        validation["requested_amount"].to_numpy(float),
        curves["validation_pd"],
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    test_enpv = expected_npv(
        test["requested_amount"].to_numpy(float),
        curves["test_pd"],
        curves["test_t_star"],
        curves["test_recovery"],
    )

    rows = []
    for buffer in [-0.02, -0.01, 0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05]:
        vd = (val_enpv / np.maximum(validation["requested_amount"].to_numpy(float), 1.0) > buffer).astype(int)
        td = (test_enpv / np.maximum(test["requested_amount"].to_numpy(float), 1.0) > buffer).astype(int)
        all_decision = np.r_[vd, td]
        rows.append(
            {
                "buffer_per_dollar": buffer,
                "matches_active_submission": int((all_decision == submission_a["decision"].to_numpy(int)).sum()),
                "approved_total": int(all_decision.sum()),
                "validation_labeled_approved": int(vd[labeled].sum()),
                "validation_labeled_realized_npv": float(realized[vd[labeled] == 1].sum()),
            }
        )

    table = pd.DataFrame(rows).sort_values("validation_labeled_realized_npv", ascending=False)
    out_dir = root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "npv_consistency_buffer_audit.csv", index=False)

    summary = {
        "formula_constants": {
            "paid_margin_rate": PAID_MARGIN_RATE,
            "origination_fee_rate": ORIGINATION_FEE_RATE,
            "daily_draw_factor": DAILY_DRAW_FACTOR,
        },
        "active_submission": {
            "rows": int(len(submission_a)),
            "approved_total": int(submission_a["decision"].sum()),
            "approval_rate": float(submission_a["decision"].mean()),
            "validation_approval_rate": float(val_decision.mean()),
            "test_approval_rate": float(test_decision.mean()),
            "validation_labeled_realized_npv": float(realized[val_decision[labeled] == 1].sum()),
        },
        "best_buffer_by_labeled_validation_npv": table.iloc[0].to_dict(),
        "buffer_matching_active_submission": table.sort_values("matches_active_submission", ascending=False).iloc[0].to_dict(),
    }
    (out_dir / "npv_consistency_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

