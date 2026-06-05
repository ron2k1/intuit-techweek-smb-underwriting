"""
data.py — loading, splitting, and feature engineering.

Public API
----------
load_raw()              -> (train_raw, val_raw, test)
prepare_splits()        -> (train, val_all, val_labeled, val_labeled_positions)
engineer_features()     -> feature-enriched DataFrame
build_feature_cols()    -> (feature_cols, cat_indices)
build_matrices()        -> dict of numpy arrays ready for model training
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, EXCLUDE, BANK_FEED_COLS, CAT_FEATURE_NAMES


def load_raw(data_dir=DATA_DIR):
    """Read the three raw CSVs from disk. Returns (train_raw, val_raw, test)."""
    train_raw = pd.read_csv(data_dir / 'train.csv')
    val_raw   = pd.read_csv(data_dir / 'validation.csv')
    test      = pd.read_csv(data_dir / 'test.csv')
    return train_raw, val_raw, test


def prepare_splits(train_raw, val_raw):
    """
    Create clean, ready-to-use splits from the raw DataFrames.

    Train
        Rows with a missing default_flag (declined / immature loans) are dropped
        upfront. This avoids accidental leakage from later filtering steps.

    Val — two copies are returned:
        val_all     : all 4 489 rows, original index preserved.
                      Used ONLY for building the submission file.
        val_labeled : only rows where default_flag is known.
                      Used for calibration, conformal intervals, and optimisation.
        val_labeled_positions : integer row positions of val_labeled inside
                      val_all (i.e. val_labeled.index). Use these to slice
                      val_all prediction arrays down to the labeled subset.

    Note on reject inference
        Training on approved+matured loans only introduces selection bias.
        It is partially mitigated by including prior_underwriter_score and
        has_prior_approval as features so the model can self-correct, and by
        calibrating on the independent val set.
    """
    train = train_raw.dropna(subset=['default_flag']).reset_index(drop=True)

    val_all     = val_raw.copy()
    val_labeled = val_raw.dropna(subset=['default_flag']).copy()
    val_labeled_positions = val_labeled.index.tolist()

    return train, val_all, val_labeled, val_labeled_positions


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and missingness indicators.

    Features added
    ──────────────
    {col}__missing          : 1 if the bank-feed column is null (MNAR indicator)
    has_prior_approval      : 1 if the prior lender approved this applicant
    approved_to_requested   : prior approved amount / requested amount
    prior_default_rate      : prior_loans_default_count / prior_loans_count
    prior_default_rate__missing : 1 if no prior loans exist
    loan_to_annual_rev      : requested_amount / stated_annual_revenue
    has_external_decline    : 1 if days_since_last_external_decline is non-null
    has_inquiry_elsewhere   : 1 if days_since_last_inquiry_elsewhere is non-null

    Integrity fixes
    ───────────────
    prior_loans_default_count is clamped to prior_loans_count (planted violation).
    """
    df = df.copy()

    for col in BANK_FEED_COLS:
        df[f'{col}__missing'] = df[col].isna().astype(np.int8)

    df['has_prior_approval'] = df['prior_approved_amount'].notna().astype(np.int8)
    df['approved_to_requested'] = (
        df['prior_approved_amount'] / df['requested_amount'].clip(lower=1)
    )

    df['prior_default_rate'] = np.where(
        df['prior_loans_count'] > 0,
        df['prior_loans_default_count'] / df['prior_loans_count'],
        np.nan,
    )
    df['prior_default_rate__missing'] = df['prior_default_rate'].isna().astype(np.int8)

    df['loan_to_annual_rev'] = (
        df['requested_amount'] / df['stated_annual_revenue'].clip(lower=1)
    )

    df['has_external_decline']  = df['days_since_last_external_decline'].notna().astype(np.int8)
    df['has_inquiry_elsewhere'] = df['days_since_last_inquiry_elsewhere'].notna().astype(np.int8)

    # Clamp integrity violation
    df['prior_loans_default_count'] = np.minimum(
        df['prior_loans_default_count'], df['prior_loans_count']
    )

    return df


def build_feature_cols(train_fe: pd.DataFrame):
    """
    Derive the final feature list and categorical indices from the engineered
    training DataFrame. Returns (feature_cols: list[str], cat_indices: list[int]).
    """
    feature_cols = [
        c for c in train_fe.columns
        if c not in EXCLUDE and train_fe[c].dtype != object
    ]
    cat_indices = [
        i for i, c in enumerate(feature_cols) if c in CAT_FEATURE_NAMES
    ]
    return feature_cols, cat_indices


def build_matrices(train_fe, val_all_fe, val_labeled_fe, test_fe, feature_cols):
    """
    Convert engineered DataFrames to numpy arrays.

    Returns a dict with keys:
        X_train, y_train          — labeled training data
        X_val_all                 — all val rows (for submission predictions)
        X_val_labeled, y_val_labeled — labeled val rows (for calibration)
        X_test                    — test rows
    """
    return dict(
        X_train       = train_fe[feature_cols].values,
        y_train       = train_fe['default_flag'].values.astype(np.int32),
        X_val_all     = val_all_fe[feature_cols].values,
        X_val_labeled = val_labeled_fe[feature_cols].values,
        y_val_labeled = val_labeled_fe['default_flag'].values.astype(np.int32),
        X_test        = test_fe[feature_cols].values,
    )
