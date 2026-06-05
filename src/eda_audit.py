#!/usr/bin/env python3
"""EDA + trap audit for the SMB Underwriting Challenge (Deliverable A groundwork).

Goal: confirm *empirically* the known traps before any modeling, and collect
evidence for the Deliverable D writeup. This script does NOT model or decide
anything -- it only reads train/validation/test and prints a structured report.

Run:
    .venv/Scripts/python.exe -m src.eda_audit
    (or)  python src/eda_audit.py

It writes a markdown findings file to reports/eda_audit_findings.md and prints
the same content to stdout.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and choke on unicode arrows/symbols.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
OUT = REPO / "reports" / "eda_audit_findings.md"

OUTCOME_COLS = [
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
]

# Loan economics (from the dataset guide / README).
NET_MARGIN = 0.35 * 60 / 365 + 0.03  # ~0.0875 on a fully-repaid loan


class Report:
    """Collects markdown lines; prints and saves them."""

    def __init__(self) -> None:
        self.buf = io.StringIO()

    def __call__(self, line: str = "") -> None:
        print(line)
        self.buf.write(line + "\n")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.buf.getvalue(), encoding="utf-8")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    ddict = pd.read_csv(DATA / "data_dictionary.csv")
    return train, val, test, ddict


def pct(n: int, d: int) -> str:
    return f"{n:,} ({100 * n / d:.1f}%)" if d else f"{n:,} (n/a)"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def section_overview(r: Report, train, val, test, ddict) -> None:
    r("# EDA + Trap Audit — SMB Underwriting")
    r()
    r("## 0. Shapes & column groups")
    r()
    r(f"- train: {train.shape[0]:,} x {train.shape[1]}")
    r(f"- validation: {val.shape[0]:,} x {val.shape[1]}")
    r(f"- test: {test.shape[0]:,} x {test.shape[1]}")
    r(f"- Deliverable A universe (val+test): {val.shape[0] + test.shape[0]:,}")
    r()
    groups = ddict.groupby("group")["field"].apply(list)
    for g, fields in groups.items():
        r(f"- **{g}** ({len(fields)}): {', '.join(fields)}")
    r()


def section_selection_bias(r: Report, train, val, test) -> None:
    r("## 1. Selection bias / reject inference")
    r()
    r("Outcomes (`default_flag`) exist only for prior-approved & matured loans. "
      "If we train on observed rows and score the full population, PD is biased.")
    r()
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        n = len(df)
        has_out = df["default_flag"].notna().sum()
        r(f"### {name} (n={n:,})")
        if "prior_decision" in df:
            vc = df["prior_decision"].value_counts(dropna=False)
            r(f"- prior_decision: {vc.to_dict()}")
        r(f"- rows with default_flag observed: {pct(int(has_out), n)}")
        if "observation_status" in df:
            r(f"- observation_status: {df['observation_status'].value_counts(dropna=False).to_dict()}")
        if has_out:
            dr = df.loc[df["default_flag"].notna(), "default_flag"].mean()
            r(f"- **observed default rate: {dr:.4f}**")
        r()
    # Cross-tab: outcome availability vs prior_decision in train.
    if "prior_decision" in train:
        r("### Outcome availability by prior_decision (train)")
        t = train.copy()
        t["has_outcome"] = t["default_flag"].notna()
        ct = pd.crosstab(t["prior_decision"], t["has_outcome"])
        r("```")
        r(ct.to_string())
        r("```")
        r("> If outcomes appear ONLY under one prior_decision value, that confirms "
          "the selection node: naive dropna()+train learns P(default | approved), "
          "not P(default | applied).")
        r()


def section_leakage(r: Report, train, val, test) -> None:
    r("## 2. Outcome leakage")
    r()
    r("These columns are post-outcome and must never be features:")
    r(f"- {', '.join(OUTCOME_COLS)}")
    r()
    r("Null-rate of outcome columns per split (should be ~100% null in test):")
    r()
    rows = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        for c in OUTCOME_COLS:
            if c in df:
                null_rate = df[c].isna().mean()
                rows.append((name, c, f"{100*null_rate:.1f}%"))
    tab = pd.DataFrame(rows, columns=["split", "column", "null_rate"])
    piv = tab.pivot(index="column", columns="split", values="null_rate")
    r("```")
    r(piv.to_string())
    r("```")
    r()


def section_missingness(r: Report, train) -> None:
    r("## 3. MNAR missingness (bank-feed)")
    r()
    bank = [
        "observed_monthly_revenue_avg_3mo",
        "observed_revenue_trend_3mo",
        "observed_revenue_volatility",
        "observed_cash_balance_p10",
        "observed_overdraft_count_3mo",
        "payroll_regularity_score",
    ]
    bank = [c for c in bank if c in train]
    if "has_linked_bank_feed" in train:
        r(f"- has_linked_bank_feed: {train['has_linked_bank_feed'].value_counts(dropna=False).to_dict()}")
    r()
    r("Null rate of bank-feed columns (should track has_linked_bank_feed=False):")
    for c in bank:
        r(f"- {c}: {train[c].isna().mean():.3f} null")
    r()
    # Is missingness informative about default? Compare default rate by feed presence.
    obs = train[train["default_flag"].notna()]
    if "has_linked_bank_feed" in obs and len(obs):
        r("### Is bank-feed missingness informative about default? (observed rows only)")
        g = obs.groupby("has_linked_bank_feed")["default_flag"].agg(["mean", "count"])
        r("```")
        r(g.to_string())
        r("```")
        r("> A gap here ⇒ missingness is MNAR and informative ⇒ add a "
          "`has_linked_bank_feed` / per-column missing indicator instead of blind imputation.")
        r()


def section_self_report(r: Report, train) -> None:
    r("## 4. Self-report inflation (stated vs observed)")
    r()
    if {"stated_annual_revenue", "observed_monthly_revenue_avg_3mo"} <= set(train.columns):
        sub = train[
            train["stated_annual_revenue"].notna()
            & train["observed_monthly_revenue_avg_3mo"].notna()
        ].copy()
        sub["observed_annual_rev_est"] = sub["observed_monthly_revenue_avg_3mo"] * 12
        sub["inflation_ratio"] = sub["stated_annual_revenue"] / sub["observed_annual_rev_est"].replace(0, np.nan)
        r(f"- rows with both stated & observed revenue: {len(sub):,}")
        r(f"- stated / (observed*12) ratio — median: {sub['inflation_ratio'].median():.2f}, "
          f"mean: {sub['inflation_ratio'].mean():.2f}, "
          f"p90: {sub['inflation_ratio'].quantile(0.9):.2f}")
        r("> Ratio >> 1 ⇒ stated revenue is optimistically inflated. Important for "
          "Deliverable C: do(stated_revenue=X) likely has ~0 true causal effect.")
    r()


def section_integrity(r: Report, train, val, test) -> None:
    r("## 5. Planted integrity violations")
    r()
    full = pd.concat([train, val, test], ignore_index=True)

    # (a) prior_loans_default_count <= prior_loans_count
    if {"prior_loans_default_count", "prior_loans_count"} <= set(full.columns):
        bad = (full["prior_loans_default_count"] > full["prior_loans_count"]).sum()
        r(f"- prior_loans_default_count > prior_loans_count: **{bad:,}** rows")

    # (b) days_to_default in [1, 90]
    if "days_to_default" in full.columns:
        d = full["days_to_default"].dropna()
        bad = ((d < 1) | (d > 90)).sum()
        r(f"- days_to_default outside [1,90]: **{bad:,}** rows (of {len(d):,} non-null)")

    # (c) default_flag vs repayment_status consistency (train, observed)
    if {"default_flag", "repayment_status"} <= set(train.columns):
        obs = train[train["default_flag"].notna() & train["repayment_status"].notna()]
        if len(obs):
            ct = pd.crosstab(obs["repayment_status"], obs["default_flag"])
            r("- default_flag vs repayment_status (train, observed):")
            r("```")
            r(ct.to_string())
            r("```")

    # (d) business_id spanning splits
    if "business_id" in full.columns:
        bt = set(train["business_id"]); bv = set(val["business_id"]); bs = set(test["business_id"])
        r(f"- business_id overlap train∩val: {len(bt & bv):,}, "
          f"train∩test: {len(bt & bs):,}, val∩test: {len(bv & bs):,} (should be 0)")

    # (e) engineered ratio vs raw inputs
    if {"requested_amount_to_observed_revenue", "requested_amount",
        "observed_monthly_revenue_avg_3mo"} <= set(full.columns):
        sub = full[
            full["requested_amount_to_observed_revenue"].notna()
            & full["observed_monthly_revenue_avg_3mo"].notna()
            & (full["observed_monthly_revenue_avg_3mo"] != 0)
        ].copy()
        recomputed = sub["requested_amount"] / sub["observed_monthly_revenue_avg_3mo"]
        diff = (recomputed - sub["requested_amount_to_observed_revenue"]).abs()
        rel = diff / recomputed.abs().replace(0, np.nan)
        bad = (rel > 0.01).sum()
        r(f"- requested_amount_to_observed_revenue mismatch vs raw (>1% rel err): "
          f"**{bad:,}** of {len(sub):,} checked")
    r()


def section_economics(r: Report, train) -> None:
    r("## 6. Loan economics & break-even PD")
    r()
    r(f"- Net margin on a fully-repaid loan ≈ {NET_MARGIN:.4f} (35% APR × 60/365 + 3% fee)")
    r()
    # Estimate LGD from final_recovered_amount among defaults.
    obs = train[train["default_flag"] == 1]
    if {"final_recovered_amount", "requested_amount"} <= set(train.columns) and len(obs):
        rec = obs["final_recovered_amount"].fillna(0)
        amt = obs["requested_amount"]
        recovery_frac = (rec / amt.replace(0, np.nan)).clip(0, 1)
        lgd = 1 - recovery_frac
        r(f"- defaults in train: {len(obs):,}")
        r(f"- recovery fraction (recovered/amount) — median: {recovery_frac.median():.3f}, "
          f"mean: {recovery_frac.mean():.3f}")
        r(f"- implied LGD — median: {lgd.median():.3f}, mean: {lgd.mean():.3f}")
        for lgd_assume in (lgd.mean(), 1.0, 0.5):
            be = NET_MARGIN / (lgd_assume + NET_MARGIN)
            r(f"- break-even PD at LGD={lgd_assume:.2f}: **{be:.3f}**")
        r("> Approve when predicted_pd < break-even PD (≈8–15%), NOT < 0.5.")
    r()


def main() -> int:
    r = Report()
    train, val, test, ddict = load()
    section_overview(r, train, val, test, ddict)
    section_selection_bias(r, train, val, test)
    section_leakage(r, train, val, test)
    section_missingness(r, train)
    section_self_report(r, train)
    section_integrity(r, train, val, test)
    section_economics(r, train)
    r.save(OUT)
    print(f"\n[written] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
