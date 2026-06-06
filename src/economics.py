"""Brief-faithful NPV economics.

Repaid:    NPV = F + R · r · T / 365
Default:   NPV = F + D · (t* - 1) + rec - R
           D = R · (1 + r·T/365) / T,  F = 0.03·R, r = 0.35, T = 60.

Decision rule: d_i = 1[E[NPV_i | approve] > 0].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


APR = 0.35
TERM_DAYS = 60
ORIGINATION_FEE_RATE = 0.03
INTEREST_OVER_TERM = APR * TERM_DAYS / 365.0  # 0.057534
PAID_MARGIN_RATE = ORIGINATION_FEE_RATE + INTEREST_OVER_TERM  # 0.087534
GROSS_OWED_FACTOR = 1.0 + INTEREST_OVER_TERM
DAILY_DRAW_FACTOR = GROSS_OWED_FACTOR / TERM_DAYS  # D / R


def npv_repaid(amount: np.ndarray) -> np.ndarray:
    return amount * PAID_MARGIN_RATE


def npv_default(amount: np.ndarray, t_star: np.ndarray, recovery_amount: np.ndarray) -> np.ndarray:
    fee = ORIGINATION_FEE_RATE * amount
    draws = DAILY_DRAW_FACTOR * amount * np.clip(t_star - 1, 0, None)
    return fee + draws + recovery_amount - amount


def expected_npv(
    amount: np.ndarray,
    p_default: np.ndarray,
    expected_t_star: np.ndarray,
    expected_recovery_rate: np.ndarray,
) -> np.ndarray:
    """E[NPV | approve] = (1-p) · NPV_paid + p · NPV_default(t*, rec)."""
    paid = npv_repaid(amount)
    rec_amount = expected_recovery_rate * amount
    default = npv_default(amount, expected_t_star, rec_amount)
    return (1.0 - p_default) * paid + p_default * default


def realized_npv(df: pd.DataFrame) -> np.ndarray:
    """Truth-side NPV using observed columns; for backtest only."""
    amount = df["requested_amount"].to_numpy(float)
    default = df["default_flag"].fillna(0).to_numpy(float)
    t_star = df["days_to_default"].fillna(0).to_numpy(float)
    rec = df["final_recovered_amount"].fillna(0).to_numpy(float)
    paid = npv_repaid(amount)
    defaulted = npv_default(amount, t_star, rec)
    return np.where(default == 1, defaulted, paid)
