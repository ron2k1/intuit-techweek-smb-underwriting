#!/usr/bin/env python3
"""Deliverable B: cohort_week x loan_age cumulative default-rate trajectory of our
Deliverable-A approved set, with 90% predictive intervals.

Model (absorbing-Markov / discrete-time survival framing):
  The default definition IS an absorbing chain (3 consecutive / 6 cumulative missed
  daily ACH draws / positive balance at day 90). days_to_default is the observed
  absorption day, so its distribution over matured loans is the chain's absorption-
  time CDF. We use:
      CDR(cohort c, age a) = mean_{i in approved & cohort c} PD_i  x  G(a)
  PD_i  = our calibrated blend PD (Deliverable A); sets each loan's total default mass.
  G(a)  = shared absorption-time CDF: cumulative share of a defaulting loan's mass
          realized by loan-age week a (validated: timing shape ~proportional across
          risk strata, so it factors out of the cohort mean).

90% band = predictive interval over the REALIZED cohort fraction, combining:
  (1) PD/model uncertainty on the cohort mean (per-loan ensemble interval),
  (2) timing-curve uncertainty (bootstrap of the absorption days),
  (3) binomial sampling of n_c approved outcomes at the expected rate.

Run:  .venv/Scripts/python.exe -m src.build_b   ->  submissions/submission_B_trajectory.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SUB_A = REPO / "submissions" / "submission_A_decisions.csv"
OUT = REPO / "submissions" / "submission_B_trajectory.csv"
WEEK1_START = pd.Timestamp("2025-06-30")
N = 13
SEED = 17
N_MC = 4000
Z = 1.6448536269514722  # 90% one-sided normal


def absorption_cdf_boot(rng, n_boot):
    """G(a), a=1..13 (point) + n_boot bootstrap curves, from the chain's observed
    absorption day (days_to_default on matured train defaults). week = ceil(day/7)."""
    train = pd.read_csv(DATA / "train.csv")
    dtd = train.loc[train["default_flag"] == 1, "days_to_default"].dropna().to_numpy()
    wk = np.ceil(dtd / 7.0).astype(int).clip(1, N)
    G = np.array([(wk <= a).mean() for a in range(1, N + 1)])
    boots = np.empty((n_boot, N))
    m = len(wk)
    for b in range(n_boot):
        s = wk[rng.integers(0, m, m)]
        boots[b] = [(s <= a).mean() for a in range(1, N + 1)]
    return G, boots


def main() -> int:
    rng = np.random.default_rng(SEED)
    sub = pd.read_csv(SUB_A)  # applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90
    val = pd.read_csv(DATA / "validation.csv")[["applicant_id", "application_timestamp"]]
    test = pd.read_csv(DATA / "test.csv")[["applicant_id", "application_timestamp"]]
    ts = pd.concat([val, test], ignore_index=True)
    df = sub.merge(ts, on="applicant_id", how="left")
    days = (pd.to_datetime(df["application_timestamp"]) - WEEK1_START).dt.days
    df["cohort_week"] = (days // 7 + 1).clip(1, N).astype(int)
    appr = df[df["decision"] == 1].copy()

    G, Gboot = absorption_cdf_boot(rng, N_MC)
    print(f"approved={len(appr):,}  timing G(a): "
          + " ".join(f"w{a}:{G[a-1]:.3f}" for a in range(1, N + 1)))

    rows = []
    for c in range(1, N + 1):
        g = appr[appr["cohort_week"] == c]
        n_c = len(g)
        if n_c == 0:                       # no approvals in cohort -> 0 with trivial band
            for a in range(1, N + 1):
                rows.append((c, a, 0.0, 0.0, 0.0))
            continue
        pd_i = g["predicted_pd"].to_numpy()
        sig_i = (g["pd_upper_90"].to_numpy() - g["pd_lower_90"].to_numpy()) / (2 * Z)
        mu = pd_i.mean()
        se_mu = np.sqrt(np.sum(sig_i ** 2)) / n_c          # model uncertainty on cohort mean PD
        mu_draws = np.clip(mu + rng.normal(0, max(se_mu, 1e-9), N_MC), 0, 1)
        for a in range(1, N + 1):
            r = np.clip(mu_draws * Gboot[:, a - 1], 0, 1)   # expected rate draws (PD x timing)
            frac = rng.binomial(n_c, r) / n_c               # realized-fraction sampling
            point = float(mu * G[a - 1])
            lo = float(min(np.quantile(frac, 0.05), point))
            hi = float(max(np.quantile(frac, 0.95), point))
            rows.append((c, a, point, lo, hi))

    out = pd.DataFrame(rows, columns=["cohort_week", "loan_age_weeks",
                                      "cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"])
    # enforce non-decreasing point within each cohort (G is non-decreasing, so this
    # is a safety cummax) and clip to [0,1].
    out["cumulative_default_rate"] = (
        out.groupby("cohort_week")["cumulative_default_rate"].cummax())
    for col in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
        out[col] = out[col].clip(0, 1)
    out["cdr_lower_90"] = np.minimum(out["cdr_lower_90"], out["cumulative_default_rate"])
    out["cdr_upper_90"] = np.maximum(out["cdr_upper_90"], out["cumulative_default_rate"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[written] {OUT}  ({len(out)} rows)")

    # ---- preview: point grid + mean band width ---------------------------- #
    piv = out.pivot(index="cohort_week", columns="loan_age_weeks",
                    values="cumulative_default_rate")
    print("\ncumulative_default_rate grid (cohort x loan_age):")
    print(piv.round(3).to_string())
    width = (out["cdr_upper_90"] - out["cdr_lower_90"]).mean()
    print(f"\nmean 90% band width: {width:.3f}  "
          f"(e.g. cohort1/age13: {out.iloc[12]['cumulative_default_rate']:.3f} "
          f"[{out.iloc[12]['cdr_lower_90']:.3f}, {out.iloc[12]['cdr_upper_90']:.3f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
