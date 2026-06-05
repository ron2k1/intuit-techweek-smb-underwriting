"""
calibration.py — probability calibration and conformal prediction intervals.

Calibration
    Raw CatBoost probabilities are calibrated with isotonic regression fitted
    on the labeled validation set (independent from training → no leakage).

Conformal intervals
    Split conformal prediction (Papadopoulos et al.) on the same labeled val set.
    Nonconformity score: s_i = |y_i - p_hat_i|.
    The corrected 90th-percentile of {s_i} gives a half-width q_90 such that
    the interval [p - q_90, p + q_90] has >= 90 % marginal coverage.

Public API
----------
fit_calibrator(val_raw_all, y_val_labeled, val_labeled_positions)  -> IsotonicRegression
apply_calibration(iso, val_raw_all, test_raw, val_labeled_positions)
    -> (val_cal_all, val_cal_labeled, test_cal)
conformal_intervals(y_val_labeled, val_cal_labeled, alpha)  -> (q_90, coverage)
build_intervals(all_pds, q_90)                              -> (lower, upper)
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import ALPHA


def fit_calibrator(
    val_raw_all: np.ndarray,
    y_val_labeled: np.ndarray,
    val_labeled_positions: list,
) -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator on the labeled val subset.

    val_raw_all           : raw (uncalibrated) ensemble mean probs for ALL val rows
    y_val_labeled         : true labels for labeled val rows only
    val_labeled_positions : integer row indices mapping labeled val → val_all
    """
    val_raw_labeled = val_raw_all[val_labeled_positions]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_raw_labeled, y_val_labeled)
    return iso


def apply_calibration(
    iso: IsotonicRegression,
    val_raw_all: np.ndarray,
    test_raw: np.ndarray,
    val_labeled_positions: list,
):
    """
    Apply the fitted calibrator to all val rows and test rows.

    Returns
    -------
    val_cal_all     (n_val_all,)     calibrated probs for ALL val rows → submission
    val_cal_labeled (n_val_labeled,) calibrated probs for labeled val  → conformal
    test_cal        (n_test,)        calibrated probs for test rows     → submission
    """
    val_cal_all     = iso.predict(val_raw_all)
    val_cal_labeled = val_cal_all[val_labeled_positions]
    test_cal        = iso.predict(test_raw)
    return val_cal_all, val_cal_labeled, test_cal


def conformal_intervals(
    y_val_labeled: np.ndarray,
    val_cal_labeled: np.ndarray,
    alpha: float = ALPHA,
):
    """
    Compute the conformal half-width q_90 from the labeled val set.

    Uses the finite-sample corrected quantile level ceil((n+1)*(1-alpha))/n
    which guarantees >= (1-alpha) marginal coverage.

    Returns (q_90: float, empirical_coverage: float).
    """
    conf_scores = np.abs(y_val_labeled.astype(float) - val_cal_labeled)
    n_cal   = len(conf_scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_90    = float(np.quantile(conf_scores, q_level))

    coverage = float(np.mean(
        (y_val_labeled >= val_cal_labeled - q_90) &
        (y_val_labeled <= val_cal_labeled + q_90)
    ))
    return q_90, coverage


def build_intervals(all_pds: np.ndarray, q_90: float):
    """
    Apply conformal half-width to an array of point estimates.

    Clips to [0, 1] then enforces lower <= pd <= upper.
    Returns (lower, upper) as numpy arrays of the same shape as all_pds.
    """
    lower = np.minimum(np.clip(all_pds - q_90, 0.0, 1.0), all_pds)
    upper = np.maximum(np.clip(all_pds + q_90, 0.0, 1.0), all_pds)
    return lower, upper
