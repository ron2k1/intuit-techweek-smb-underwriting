#!/usr/bin/env python3
"""Build Deliverable C counterfactual PD predictions.

Implements the DAG/tricks memo's Phase C guidance:
- train a causal-safe calibrated PD model without prior-underwriter artifacts;
- answer every do(feature=value) query deterministically;
- shrink historical/proxy deltas toward baseline;
- widen 90% intervals for weak support, policy artifacts, and low support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counterfactuals import build_counterfactuals, write_counterfactual_outputs


def main() -> None:
    artifacts = build_counterfactuals()
    write_counterfactual_outputs(artifacts)
    print(json.dumps(artifacts.metrics, indent=2))
    print("Wrote outputs/submission/submission_C_counterfactuals.csv")
    print("Wrote outputs/reports/deliverable_c_query_diagnostics.csv")
    print("Wrote outputs/reports/deliverable_c_feature_diagnostics.csv")


if __name__ == "__main__":
    main()
