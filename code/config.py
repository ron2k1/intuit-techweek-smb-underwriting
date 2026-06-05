"""
config.py — all constants, paths, and hyper-parameters for Deliverable A.
Change things here; nothing else needs to be touched for a quick experiment.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path('dataset/dataset-compressed')
SUB_DIR  = Path('submissions')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Columns that can never be used as model features ─────────────────────────
OUTCOME_COLS = [
    'default_flag',
    'days_to_default',
    'days_to_full_repayment',
    'repayment_status',
    'final_recovered_amount',
    'observation_status',
]
ID_COLS = ['business_id', 'applicant_id', 'application_timestamp']

# prior_decision is a selection collider — conditioning on it biases causal estimates
EXCLUDE = set(OUTCOME_COLS + ID_COLS + ['prior_decision'])

# ── Bank-feed columns (null iff applicant has no linked feed → MNAR) ─────────
BANK_FEED_COLS = [
    'observed_monthly_revenue_avg_3mo',
    'observed_revenue_trend_3mo',
    'observed_revenue_volatility',
    'observed_cash_balance_p10',
    'observed_overdraft_count_3mo',
    'payroll_regularity_score',
]

# ── Integer-coded categorical columns (passed to CatBoost as cat_features) ───
CAT_FEATURE_NAMES = [
    'sector',
    'geography_region',
    'employee_count_bucket',
    'intended_use_of_funds',
    'application_channel',
    'owner_personal_credit_band',
]

# ── CatBoost hyper-parameters ─────────────────────────────────────────────────
CB_PARAMS = dict(
    iterations    = 1000,
    learning_rate = 0.05,
    depth         = 6,
    loss_function = 'Logloss',
    eval_metric   = 'AUC',
    random_seed   = SEED,
    od_type       = 'Iter',   # early-stopping trigger
    od_wait       = 50,       # wait this many rounds before stopping
    verbose       = 0,
)

N_FOLDS = 10

# ── Loan economics ────────────────────────────────────────────────────────────
# Fixed terms: 60-day term, 35 % APR, 3 % origination fee
# Revenue per dollar for a fully-repaid loan ≈ 0.0875
REVENUE_RATE = 0.35 * (60 / 365) + 0.03

# Conformal coverage target: intervals must contain truth with probability >= 1-ALPHA
ALPHA = 0.10
