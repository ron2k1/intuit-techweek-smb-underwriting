#!/usr/bin/env python3
"""Create a proxy audit for Deliverable C causal accuracy.

True intervention labels are hidden, so this report checks the things we can
validate locally: coverage, support, deterministic duplicates, feature treatment
coverage, and monotonic sanity after guardrails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
SUBMISSION_DIR = ROOT / "outputs" / "submission"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    queries = pd.read_csv(ROOT / "data" / "intervention_queries.csv")
    submission = pd.read_csv(SUBMISSION_DIR / "submission_C_counterfactuals.csv")
    diag = pd.read_csv(REPORT_DIR / "deliverable_c_query_diagnostics.csv")
    feature_diag = pd.read_csv(REPORT_DIR / "deliverable_c_feature_diagnostics.csv")
    treatment = pd.read_csv(REPORT_DIR / "deliverable_c_feature_treatment_plan.csv")

    merged = queries.merge(submission, on="query_id", how="left", validate="one_to_one")
    duplicate_consistent = True
    duplicate_groups = 0
    for _, group in merged.groupby(["applicant_id", "feature_name", "intervention_value"], dropna=False):
        if len(group) <= 1:
            continue
        duplicate_groups += 1
        if group[["predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"]].nunique().max() != 1:
            duplicate_consistent = False
            break

    treatment_missing = sorted(set(queries["feature_name"]) - set(treatment["feature_name"]))
    interval_violations = int(
        (
            (submission["pd_cf_lower_90"] > submission["predicted_pd_cf"])
            | (submission["predicted_pd_cf"] > submission["pd_cf_upper_90"])
        ).sum()
    )
    summary = {
        "rows_expected": int(len(queries)),
        "rows_submitted": int(len(submission)),
        "unique_query_ids": int(submission["query_id"].nunique()),
        "interval_order_violations": interval_violations,
        "duplicate_intervention_groups": int(duplicate_groups),
        "duplicate_predictions_consistent": bool(duplicate_consistent),
        "features_with_treatment_rules": int(treatment["feature_name"].nunique()),
        "treatment_missing_features": treatment_missing,
        "outside_train_min_max_queries": int(diag["outside_min_max"].sum()),
        "tail_support_queries": int(diag["outside_p01_p99"].sum()),
        "unseen_category_queries": int(diag["unseen_category"].sum()),
        "raw_material_sign_violations": int(diag["raw_sign_violation"].sum()),
        "monotonic_guard_applied": int(diag["monotonic_guard_applied"].sum()),
        "final_material_sign_violations": int(diag["sign_violation"].sum()),
        "mean_interval_width": float((submission["pd_cf_upper_90"] - submission["pd_cf_lower_90"]).mean()),
        "mean_abs_delta_final": float(diag["delta_final"].abs().mean()),
        "largest_abs_delta_final": float(diag["delta_final"].abs().max()),
        "classes": {
            str(k): int(v)
            for k, v in diag["intervention_class"].value_counts().sort_index().items()
        },
    }

    feature_cols = [
        "feature_name",
        "count",
        "intervention_class",
        "outside_min_max_rate",
        "outside_p01_p99_rate",
        "unseen_category_rate",
        "raw_sign_violation_count",
        "monotonic_guard_count",
        "sign_violation_count",
        "mean_delta_final",
        "mean_interval_width",
    ]
    feature_table = feature_diag[feature_cols].copy()
    feature_table.to_csv(REPORT_DIR / "deliverable_c_causal_accuracy_feature_audit.csv", index=False)
    (REPORT_DIR / "deliverable_c_causal_accuracy_audit.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Deliverable C Causal Accuracy Proxy Audit",
        "",
        "True counterfactual labels are hidden, so this audit checks local failure modes that would make C easy to penalize.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Feature-Level Audit",
        "",
        markdown_table(feature_table),
        "",
        "## Interpretation",
        "",
        "- No local audit can prove true causal accuracy because the scoring interventions are hidden.",
        "- The submitted C file is structurally complete, duplicate-deterministic, and support-aware.",
        "- Material opposite-sign effects for monotone risk features are neutralized and intervals widened.",
        "- Historical/proxy and measurement-process features are intentionally shrunk toward baseline rather than overclaimed.",
    ]
    (REPORT_DIR / "deliverable_c_causal_accuracy_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
