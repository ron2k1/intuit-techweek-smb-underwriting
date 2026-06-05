"""
decision.py — profit-maximising approval threshold and decision logic.

The break-even PD is derived from the loan economics:

    Expected profit per dollar = (1 - PD) * REVENUE_RATE  -  PD * LGD  > 0
    => approve when  PD  <  REVENUE_RATE / (LGD + REVENUE_RATE)

LGD (Loss Given Default) = 1 - average recovery rate, estimated from training defaults.

README reference: "Approve below the profit-break-even PD, NOT below 0.5."

Public API
----------
compute_breakeven_pd(train_raw)  -> (breakeven_pd, lgd, avg_recovery)
make_decisions(all_pds, breakeven_pd)  -> decisions ndarray
"""

import numpy as np
import pandas as pd

from config import REVENUE_RATE


def compute_breakeven_pd(train_raw: pd.DataFrame):
    """
    Estimate break-even PD from observed recovery rates in training defaults.

    Parameters
    ----------
    train_raw : the *original* (unfiltered) training DataFrame so that
                final_recovered_amount is available for defaulted loans.

    Returns
    -------
    breakeven_pd  : float  PD threshold above which approving destroys value
    lgd           : float  1 - avg_recovery_rate
    avg_recovery  : float  mean(recovered / requested) across training defaults
    """
    defaults = train_raw[train_raw['default_flag'] == 1].copy()
    recovery_rates = (
        defaults['final_recovered_amount'].fillna(0) / defaults['requested_amount']
    ).clip(0, 1)

    avg_recovery = float(recovery_rates.mean())
    lgd          = 1.0 - avg_recovery
    breakeven_pd = REVENUE_RATE / (lgd + REVENUE_RATE)

    return breakeven_pd, lgd, avg_recovery


def make_decisions(all_pds: np.ndarray, breakeven_pd: float) -> np.ndarray:
    """
    Return a binary array: 1 = approve, 0 = decline.
    Approve when predicted PD is strictly below the profit break-even threshold.
    """
    return (all_pds < breakeven_pd).astype(int)
