"""
decision.py — NPV-based approval decisions and CDR matrix.

Loan economics (per the challenge spec)
----------------------------------------
    NPV_repay_i   = F_i + R_i * r * (T/365)
    NPV_default_i = F_i + D_i * (t*_i - 1) + rec_i - R_i

    F_i = origination fee = 0.03 * R_i
    D_i = daily draw      = R_i * (1 + r*T/365) / T
    r   = 0.35  (APR)
    T   = 60    (term in days)
    t*  = days_to_default  (1 .. T)
    rec = final_recovered_amount

Break-even condition  E[NPV | approve] = 0:
    (1 - PD) * E[NPV_repay] + PD * E[NPV_default] = 0
    => PD* = E[NPV_repay] / (E[NPV_repay] - E[NPV_default])

CDR matrix
----------
    CDR[segment, t] = P(t* = t | default, segment)
    Segment = owner_personal_credit_band (primary) × sector (secondary)
    Gives per-applicant E[t*] and E[rec/R], enabling applicant-specific thresholds.

Public API
----------
compute_breakeven_pd(train_raw)
    -> (breakeven_pd, avg_npv_default, avg_npv_repay)   global threshold

compute_cdr_matrix(train_raw, segment_cols)
    -> dict with keys: cdr, expected_t, expected_rec_rate, global_expected_t,
                       global_expected_rec_rate

compute_per_loan_breakeven(applicants_df, cdr_stats, segment_cols)
    -> ndarray of per-applicant break-even PD thresholds

make_decisions(all_pds, breakeven)
    -> decisions ndarray  (breakeven may be scalar or per-row array)
"""

import numpy as np
import pandas as pd

R_ANNUAL = 0.35
T_DAYS   = 60
ORIG_FEE = 0.03

# Default segmentation: credit band captures default-timing risk most cleanly
DEFAULT_SEGMENT_COLS = ['owner_personal_credit_band', 'sector']


# ── Global break-even PD (precise NPV formula) ───────────────────────────────

def compute_breakeven_pd(train_raw: pd.DataFrame):
    """
    Estimate a global break-even PD from the full NPV cash-flow model.

    Accounts for:
      - origination fee collected upfront (F)
      - partial daily repayments received before default: D * (t* - 1)
      - final recovery after default (rec)

    Returns
    -------
    breakeven_pd   : float   PD threshold; approve if predicted PD < this
    avg_npv_default: float   mean NPV per dollar for defaulted loans
    avg_npv_repay  : float   mean NPV per dollar for repaid loans
    """
    r, T = R_ANNUAL, T_DAYS

    defaults = train_raw[train_raw['default_flag'] == 1].copy()
    R_d  = defaults['requested_amount']
    t    = defaults['days_to_default'].fillna(T).clip(1, T)
    rec  = defaults['final_recovered_amount'].fillna(0)
    F_d  = ORIG_FEE * R_d
    D    = R_d * (1 + r * T / 365) / T
    npv_default = F_d + D * (t - 1) + rec - R_d

    repaid = train_raw[train_raw['default_flag'] == 0].copy()
    R_rep  = repaid['requested_amount']
    npv_repay = ORIG_FEE * R_rep + R_rep * r * T / 365

    avg_npv_repay   = float(npv_repay.mean())
    avg_npv_default = float(npv_default.mean())
    breakeven_pd    = avg_npv_repay / (avg_npv_repay - avg_npv_default)

    return breakeven_pd, avg_npv_default, avg_npv_repay


# ── CDR matrix ────────────────────────────────────────────────────────────────

def compute_cdr_matrix(
    train_raw: pd.DataFrame,
    segment_cols: list = DEFAULT_SEGMENT_COLS,
) -> dict:
    """
    Build a Conditional Default Rate (CDR) matrix from training defaults.

    CDR[segment, t] = P(t* = t | default, segment)
    — the empirical distribution of default timing within each origination segment.

    Two derived statistics drive the per-loan NPV:
        expected_t        : E[t* | segment]        — expected default day
        expected_rec_rate : E[rec/R | segment]      — expected recovery rate

    Both fall back to the global mean for unseen segments at prediction time.

    Parameters
    ----------
    train_raw     : raw training DataFrame (must contain days_to_default,
                    final_recovered_amount, requested_amount)
    segment_cols  : columns that define the origination segment

    Returns
    -------
    dict with keys:
        cdr                   : DataFrame  (n_segments, T_DAYS) — CDR distributions
        expected_t            : Series     E[t* | segment]
        expected_rec_rate     : Series     E[rec/R | segment]
        global_expected_t     : float      fallback for unseen segments
        global_expected_rec_rate : float   fallback for unseen segments
        segment_cols          : list       stored for use in compute_per_loan_breakeven
    """
    r, T = R_ANNUAL, T_DAYS
    days     = range(1, T + 1)
    days_arr = np.arange(1, T + 1, dtype=float)   # numpy array for arithmetic

    defaults = train_raw[train_raw['default_flag'] == 1].copy()
    defaults['t']        = defaults['days_to_default'].fillna(T).clip(1, T).astype(int)
    defaults['rec_rate'] = (
        defaults['final_recovered_amount'].fillna(0)
        / defaults['requested_amount'].clip(lower=1)
    ).clip(0, 1)

    # ── CDR distribution and derived stats per segment ────────────────────────
    cdr_rows     = {}
    exp_t_rows   = {}
    exp_rec_rows = {}

    for segment, grp in defaults.groupby(segment_cols):
        counts = grp['t'].value_counts().reindex(days, fill_value=0)
        total  = counts.sum()
        dist   = (counts / total).values if total > 0 else np.zeros(T)
        cdr_rows[segment]     = dist
        exp_t_rows[segment]   = float(np.dot(days_arr, dist))   # E[t*] = Σ t·P(t)
        exp_rec_rows[segment] = float(grp['rec_rate'].mean())

    idx = pd.MultiIndex.from_tuples(cdr_rows.keys(), names=segment_cols) \
          if len(segment_cols) > 1 else \
          pd.Index(cdr_rows.keys(), name=segment_cols[0])

    cdr = pd.DataFrame(
        list(cdr_rows.values()),
        index=idx,
        columns=list(days),
    )
    expected_t        = pd.Series(exp_t_rows, name='expected_t')
    expected_rec_rate = pd.Series(exp_rec_rows, name='expected_rec_rate')

    # Fix multi-index consistency
    expected_t.index        = cdr.index
    expected_rec_rate.index = cdr.index

    global_expected_t        = float(defaults['t'].mean())
    global_expected_rec_rate = float(defaults['rec_rate'].mean())

    print(f'CDR matrix: {len(cdr)} segments × {T} days')
    print(f'Global E[t*]={global_expected_t:.1f}d  '
          f'global E[rec/R]={global_expected_rec_rate:.4f}')

    return dict(
        cdr                      = cdr,
        expected_t               = expected_t,
        expected_rec_rate        = expected_rec_rate,
        global_expected_t        = global_expected_t,
        global_expected_rec_rate = global_expected_rec_rate,
        segment_cols             = segment_cols,
    )


# ── Per-applicant break-even PD ───────────────────────────────────────────────

def compute_per_loan_breakeven(
    applicants_df: pd.DataFrame,
    cdr_stats: dict,
) -> np.ndarray:
    """
    Compute a per-applicant break-even PD using the CDR matrix.

    Each applicant's break-even depends on their own loan size (which scales
    F and D uniformly, so it cancels) and their segment's expected default
    timing and recovery rate.

    Parameters
    ----------
    applicants_df : DataFrame with columns for requested_amount and
                    each column in cdr_stats['segment_cols']
    cdr_stats     : output of compute_cdr_matrix

    Returns
    -------
    breakeven_pds : ndarray of shape (n_applicants,)
    """
    r, T  = R_ANNUAL, T_DAYS
    scols = cdr_stats['segment_cols']

    exp_t_map   = cdr_stats['expected_t'].to_dict()
    exp_rec_map = cdr_stats['expected_rec_rate'].to_dict()
    fallback_t  = cdr_stats['global_expected_t']
    fallback_rec = cdr_stats['global_expected_rec_rate']

    R   = applicants_df['requested_amount'].values
    F   = ORIG_FEE * R
    D   = R * (1 + r * T / 365) / T

    # Look up segment stats; fall back to global for unseen combinations
    if len(scols) == 1:
        keys = applicants_df[scols[0]].values
    else:
        keys = list(zip(*[applicants_df[c].values for c in scols]))

    exp_t   = np.array([exp_t_map.get(k, fallback_t)   for k in keys])
    exp_rec = np.array([exp_rec_map.get(k, fallback_rec) for k in keys])

    # Per-applicant expected NPV under each outcome
    npv_repay   = F + R * r * T / 365                          # full repayment
    npv_default = F + D * (exp_t - 1) + exp_rec * R - R        # default at E[t*]

    # Break-even: PD* = NPV_repay_i / (NPV_repay_i - NPV_default_i)
    # denom is always positive in practice (repaying >> defaulting in value),
    # but guard degenerate rows by falling back to the portfolio mean.
    denom            = npv_repay - npv_default
    global_fallback  = float(npv_repay.mean() / denom.mean())
    breakeven_pds    = np.where(np.abs(denom) > 1e-8, npv_repay / denom, global_fallback)
    return np.clip(breakeven_pds, 0.01, 0.99)


# ── Binary decisions ─────────────────────────────────────────────────────────

def make_decisions(all_pds: np.ndarray, breakeven) -> np.ndarray:
    """
    Return binary decisions: 1 = approve, 0 = decline.

    breakeven may be a scalar (global threshold) or a per-row array
    (from compute_per_loan_breakeven).  Approve when predicted PD < threshold.
    """
    return (all_pds < np.asarray(breakeven)).astype(int)
