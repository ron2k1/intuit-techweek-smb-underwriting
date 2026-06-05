#!/usr/bin/env python3
"""AUC-ROC of the Deliverable A PD model, from the saved predictions artifact.

AUC is a ranking metric on the PD model, so it is the SAME for all three funding
policies (conservative/gated/trust) -- they share the model and only differ in
the decision cut. Labels exist only in the observed (prior-approved) region.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
PREDS = REPO / "reports" / "a_predictions.csv"


def auc_line(label, y, p):
    if len(np.unique(y)) < 2:
        print(f"  {label:42s} n={len(y):5d}  (single class — AUC undefined)")
        return
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print(f"  {label:42s} n={len(y):5d}  AUC={auc:.4f}  AP={ap:.4f}  "
          f"base_rate={y.mean():.3f}")


def main() -> int:
    df = pd.read_csv(PREDS)
    obs = df[df["default_flag"].notna()].copy()  # validation observed rows only
    y = obs["default_flag"].astype(int).to_numpy()
    p = obs["predicted_pd"].to_numpy()

    print("AUC-ROC of the PD model (identical across all 3 funding policies)")
    print("Labels available ONLY in the observed/prior-approved region:\n")
    auc_line("validation (all observed rows)", y, p)

    # By prior-score band within the observed region (where does ranking hold up?).
    print("\nWithin observed region, by prior_underwriter_score band:")
    for lo, hi in [(0.273, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (obs["prior_underwriter_score"] >= lo) & (obs["prior_underwriter_score"] < hi)
        auc_line(f"  score [{lo:.2f}, {hi:.2f})", y[m.to_numpy()], p[m.to_numpy()])

    print("\nNote: the declined region (prior_decision=0) has NO labels, so its")
    print("AUC is unmeasurable — the positivity violation again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
