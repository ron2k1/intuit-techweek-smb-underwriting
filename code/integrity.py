"""
integrity.py — data quality and cross-split leakage checks.

The README lists seven known traps. This module checks for the four that
are detectable from the data alone (traps 1, 3, 4, 7 from the README).

Public API
----------
check_integrity(df, label)           -> prints a table of flagged issues
check_split_leakage(train, val, test) -> prints business_id overlap counts
"""

import pandas as pd


def check_integrity(df: pd.DataFrame, label: str = 'dataset') -> None:
    """
    Run structural consistency checks on a single split.

    Checks performed
    ────────────────
    prior_loans_default_count > prior_loans_count
        A business cannot have more defaults than loans taken. Rows that
        violate this are planted integrity violations (README trap 7).

    days_to_default outside [1, 90]
        Default day must fall within the 90-day loan term.

    default_flag vs repayment_status mismatch
        A defaulted loan should never be marked paid_in_full and vice versa.
    """
    issues = {}

    mask = df['prior_loans_default_count'] > df['prior_loans_count']
    issues['prior_loans_default_count > prior_loans_count'] = int(mask.sum())

    if df['days_to_default'].notna().any():
        bad = (df['days_to_default'] > 90) | (df['days_to_default'] < 1)
        issues['days_to_default out of [1, 90]'] = int(bad.sum())

    if 'repayment_status' in df.columns and df['repayment_status'].notna().any():
        mismatch = (
            ((df['default_flag'] == 1) & (df['repayment_status'] == 'paid_in_full')) |
            ((df['default_flag'] == 0) & (df['repayment_status'] == 'defaulted'))
        )
        issues['default_flag / repayment_status mismatch'] = int(mismatch.sum())

    print(f'--- Integrity: {label} ---')
    for description, count in issues.items():
        flag = '  *** FLAGGED' if count > 0 else ''
        print(f'  {description}: {count}{flag}')


def check_split_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Verify that no business_id appears in more than one split.
    Any overlap would mean the model has seen outcome data for a test applicant.
    """
    train_biz = set(train['business_id'])
    val_biz   = set(val['business_id'])
    test_biz  = set(test['business_id'])

    print('--- Split leakage (business_id) ---')
    print(f'  train ∩ val  : {len(train_biz & val_biz)}  (expected 0)')
    print(f'  train ∩ test : {len(train_biz & test_biz)}  (expected 0)')
    print(f'  val   ∩ test : {len(val_biz & test_biz)}  (expected 0)')
