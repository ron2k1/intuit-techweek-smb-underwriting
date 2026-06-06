"""Shared feature pipeline for the SMB Underwriting Challenge.

ONE place every deliverable (A / B / C) loads data from, so the whole team sees
an identical, audited feature space.

The dataset is arithmetically CLEAN (0 integrity violations -- see scratch/probe*.py).
The traps are STRUCTURAL, so this module's job is not "cleaning" in the usual
sense (no dropna / no blind impute). It is four principled moves:

  1. QUARANTINE post-outcome leakage columns (blank in test; using them cheats
     and is impossible at inference time).
  2. KEEP declined applicants. Outcomes exist only for prior_decision==1, but the
     population we must score is ~40% declines. Dropping them is the selection-bias
     trap. (The PD model in build_a trains on the labelled subset but the feature
     space here is identical for everyone, so declines are scored, never dropped.)
  3. ENCODE missingness as signal. Bank-feed nulls are MNAR ("no linked feed"),
     so we add *__isna indicators and let HistGradientBoosting read NaN natively.
  4. RECOMPUTE the planted trap feature ourselves. `requested_amount_to_observed_
     revenue` is SOURCE-MIXED: = requested/(observed_monthly*12) when a feed exists,
     but silently = requested/stated_annual when it does not (proven in probe2 --
     100% of no-feed rows use the gameable self-reported denominator, with no flag).
     We drop it and rebuild verified / stated leverage as separate, labelled cols.

See reports/audit_findings.md for the evidence behind each choice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# --- column taxonomy (pinned against the real 44-column schema) ---------------

# The label. Observed ONLY for prior-approved & matured loans (prior_decision==1).
TARGET = "default_flag"

# Post-outcome columns: known only after the loan resolves, blank in test.
# Using ANY as a feature is both leakage and impossible at inference.
LEAKAGE = [
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
    # prior_approved_amount: set only for approved (44% null) -- the PRIOR lender's
    # post-decision output, so it encodes the selection we must not lean on.
    "prior_approved_amount",
]

# Identifiers, not features. applicant_id is the output key; business_id is the
# entity (used only to confirm no entity spans splits -- it does not).
IDS = ["applicant_id", "business_id"]

# The planted source-mixing trap (see module docstring). Drop, rebuild ourselves.
TRAP_COLS = ["requested_amount_to_observed_revenue"]

# Raw timestamp -> we derive cohort_week, then drop the raw string.
RAW_DROP = ["application_timestamp"]

# Bank-feed block: present iff has_linked_bank_feed. ~37% missing = MNAR.
BANK_FEED = [
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
]

# Other informative-missing columns (~49% null): "never declined/inquired
# elsewhere" is itself a signal, so flag rather than impute.
INFORMATIVE_NULL = [
    "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere",
]

# Cohort window = the 13 weeks the val/test applicants span (2025-06-30..09-28).
COHORT_START = pd.Timestamp("2025-06-30")
N_COHORT_WEEKS = 13

# Nominal integer-coded categoricals (codes are arbitrary labels, NOT magnitudes).
# We declare these to the booster as categorical_features so it splits on SUBSETS
# of levels instead of on a meaningless code threshold. The other two coded fields
# (employee_count_bucket, owner_personal_credit_band) are ORDINAL -- "smaller code =
# fewer employees" / credit band ordering -- so we keep them numeric on purpose.
CATEGORICAL_NOMINAL = [
    "sector",
    "geography_region",
    "intended_use_of_funds",
    "application_channel",
]

# --- causal taxonomy (Deliverable C) -----------------------------------------
# C must estimate an INTERVENTIONAL effect, so its outcome model must be backdoor
# valid: drop selection/collider nodes and drop self-report fields whose true causal
# effect on default is ~0 (the "self-report inflation" trap). A keeps all of these
# because A only needs PREDICTION. This list is exactly the A-vs-C difference.
COLLIDERS = [
    # prior lender's score + decision: the selection mechanism that created the
    # labelled sample. Conditioning on them for a causal effect opens a backdoor.
    "prior_underwriter_score",
    "prior_decision",
]
SELF_REPORT_CAUSAL_NULL = [
    # what the applicant *wrote* does not change whether their business repays;
    # the real driver is the bank-observed counterpart. do(stated_*) true effect ~0.
    "stated_annual_revenue",
    "stated_time_in_business",
]
SELF_REPORT_DERIVED = [
    # engineered from stated_* -> carry the same self-report content, not a cause.
    "lev_stated",
    "report_gap",
]
# cohort_week is a calendar index (constant 0 in train), not a cause -> drop for C.
CAUSAL_EXCLUDE = COLLIDERS + SELF_REPORT_CAUSAL_NULL + SELF_REPORT_DERIVED + ["cohort_week"]


def load_raw(split: str) -> pd.DataFrame:
    """Load one split verbatim. split in {train, val, test}."""
    fname = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}[split]
    return pd.read_csv(DATASET_DIR / fname, low_memory=False)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the trap feature honestly + add MNAR indicators + cohort week."""
    df = df.copy()
    obs_annual = df["observed_monthly_revenue_avg_3mo"] * 12.0

    # Leverage, split by source so the model can tell verified from gameable.
    # lev_verified is NaN when there is no feed -- an honest gap, not a fill.
    df["lev_verified"] = df["requested_amount"] / obs_annual
    df["lev_stated"] = df["requested_amount"] / df["stated_annual_revenue"]
    # Self-report inflation: stated revenue vs bank-observed (NaN when no feed).
    df["report_gap"] = df["stated_annual_revenue"] / obs_annual

    # Missingness-as-signal (MNAR). no_bank_feed duplicates has_linked_bank_feed
    # intentionally as an explicit risk flag; trees ignore the redundancy.
    df["no_bank_feed"] = (~df["has_linked_bank_feed"].astype(bool)).astype("int8")
    for c in INFORMATIVE_NULL:
        df[f"{c}__isna"] = df[c].isna().astype("int8")

    # Cohort week 1..13 from application date. Train predates the window -> 0
    # (constant in train, so it is a no-op feature for A but lets B group rows).
    ts = pd.to_datetime(df["application_timestamp"], errors="coerce")
    wk = ((ts - COHORT_START).dt.days // 7) + 1
    df["cohort_week"] = (
        wk.where((wk >= 1) & (wk <= N_COHORT_WEEKS)).fillna(0).astype("int16")
    )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every legitimate application-time predictor (post-engineering)."""
    drop = set(IDS + LEAKAGE + TRAP_COLS + RAW_DROP + [TARGET])
    return [c for c in df.columns if c not in drop]


def causal_feature_columns(df: pd.DataFrame) -> list[str]:
    """Backdoor-valid predictor set for Deliverable C.

    feature_columns() minus colliders/selection nodes and self-report fields.
    Intervening on an excluded feature therefore moves the prediction by exactly
    zero -- which is the correct interventional answer for the self-report trap.
    """
    drop = set(CAUSAL_EXCLUDE)
    return [c for c in feature_columns(df) if c not in drop]


def categorical_indices(feats: list[str]) -> list[int]:
    """Positions of the nominal categoricals within `feats`, for HGB's
    categorical_features= argument (codes are already non-negative ints)."""
    return [i for i, c in enumerate(feats) if c in set(CATEGORICAL_NOMINAL)]


def to_model_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Float matrix for HGB. NaN preserved on purpose (native split handling)."""
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) == "boolean":
            X[c] = X[c].astype("float64")
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype("float64")


def target_vector(df: pd.DataFrame) -> pd.Series:
    """0/1 default label; NaN where unobserved (declined or not yet matured)."""
    return pd.to_numeric(df[TARGET], errors="coerce")


def load_features(split: str) -> tuple[pd.DataFrame, list[str]]:
    """Public entry point: (dataframe with engineered features, feature list)."""
    df = add_engineered_features(load_raw(split))
    return df, feature_columns(df)


if __name__ == "__main__":  # smoke test: python -m src.data
    for split in ("train", "val", "test"):
        df, feats = load_features(split)
        y = target_vector(df)
        labelled = int(y.notna().sum())
        rate = float(y.mean()) if labelled else float("nan")
        print(
            f"{split:5s} rows={len(df):6d} feats={len(feats):3d} "
            f"labelled={labelled:6d} default_rate={rate:.4f}"
        )
    # show the rebuilt leverage cols vs the dropped trap, on train
    df, feats = load_features("train")
    print("\nengineered cols present:", [c for c in feats if c in
          ("lev_verified", "lev_stated", "report_gap", "no_bank_feed", "cohort_week")])
    print("trap col dropped:", "requested_amount_to_observed_revenue" not in feats)
    print("leakage dropped:", all(c not in feats for c in LEAKAGE + [TARGET]))
