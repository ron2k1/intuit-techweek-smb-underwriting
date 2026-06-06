#!/usr/bin/env python3
"""Apples-to-apples branch comparison: score every team's COMMITTED submission_A
predictions on the SAME validation outcomes, SAME metrics, SAME exact-NPV economics.
READ-ONLY. Self-reported numbers are ignored; everything here is recomputed.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SUB_A = REPO / "submissions" / "submission_A_decisions.csv"

# Exact official economics (same for all teams).
TERM_INT, FEE = 0.35 * 60 / 365, 0.03
NET_MARGIN = TERM_INT + FEE
DAILY_DRAW = (1 + TERM_INT) / 60

BRANCHES = {
    "ayush (ours)": ("LOCAL", str(SUB_A)),
    "ronil":        ("GIT", "origin/ronil:submissions/submission_A_decisions.csv"),
    "steven":       ("GIT", "origin/steven:outputs/submission/submission_A_decisions.csv"),
}


def load(kind, ref):
    if kind == "LOCAL":
        return pd.read_csv(ref)
    out = subprocess.run(["git", "-C", str(REPO), "show", ref],
                         capture_output=True, text=True)
    return pd.read_csv(io.StringIO(out.stdout))


def ece(p, y, n=10):
    o = np.argsort(p)
    e = 0.0
    for b in np.array_split(o, n):
        e += len(b) / len(y) * abs(p[b].mean() - y[b].mean())
    return e


def realized_npv(amt, dflag, dtd, rec):
    rec = np.nan_to_num(rec)
    draws = amt * DAILY_DRAW * np.clip(np.nan_to_num(dtd) - 1, 0, None)
    default = np.minimum(amt * FEE + draws + rec - amt, amt * NET_MARGIN)
    return np.where(dflag == 1, default, amt * NET_MARGIN)


def main() -> int:
    val = pd.read_csv(DATA / "validation.csv")
    lab = val[val["default_flag"].notna()].copy()
    lab_npv_all = realized_npv(lab["requested_amount"].to_numpy(),
                               lab["default_flag"].to_numpy(),
                               lab["days_to_default"].to_numpy(),
                               lab["final_recovered_amount"].to_numpy())
    lab = lab.assign(_npv=lab_npv_all)
    keep = lab[["applicant_id", "default_flag", "_npv"]].rename(columns={"default_flag": "y"})

    print(f"Scoring on {len(lab):,} labeled validation rows (base default "
          f"{lab['default_flag'].mean():.4f}); economics: exact NPV, NET_MARGIN "
          f"{NET_MARGIN:.4f}.\n")
    hdr = (f"{'branch':<14}{'AUC':>7}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}"
           f"{'appr%(all)':>11}{'appr%(val)':>11}{'val-def%':>9}{'realNPV(val)':>14}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for name, (kind, ref) in BRANCHES.items():
        sub = load(kind, ref)[["applicant_id", "decision", "predicted_pd"]]
        m = sub.merge(keep, on="applicant_id", how="inner")
        y = m["y"].astype(int).to_numpy()
        p = np.clip(m["predicted_pd"].to_numpy(), 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y, p); br = brier_score_loss(y, p)
        ll = log_loss(y, p); ec = ece(p, y)
        appr_all = sub["decision"].mean()
        funded = m["decision"] == 1
        appr_val = funded.mean()
        val_def = y[funded.to_numpy()].mean() if funded.sum() else float("nan")
        npv = m.loc[funded, "_npv"].sum()
        print(f"{name:<14}{auc:>7.4f}{br:>8.4f}{ll:>9.4f}{ec:>8.4f}"
              f"{100*appr_all:>10.1f}%{100*appr_val:>10.1f}%{100*val_def:>8.1f}%"
              f"{npv:>14,.0f}")
        rows.append(name)
    print("\nNotes:")
    print(" - AUC/Brier/LogLoss/ECE: PD QUALITY on the identical 2,551 labeled val rows.")
    print(" - realNPV(val): realized $ under the SAME exact economics on each team's")
    print("   val-approved+labeled loans (policy+model; declined rows are unlabeled).")
    print(" - Abhimanyu: NO committed predictions (notebook needs uninstalled libs) -> omitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
