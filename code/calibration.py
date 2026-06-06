"""
calibration.py — probability calibration and prediction intervals.

Public API
----------
bootstrap_calibrated_intervals(model, X_train, y_train, X_val_all,
                                y_val_labeled, val_labeled_positions,
                                X_test, cat_indices, ...)
    -> dict with val_cal_all, val_cal_labeled, test_cal,
               val_lower, val_upper, test_lower, test_upper, coverage

Legacy API (kept for reference)
--------------------------------
fit_calibrator(val_raw_all, y_val_labeled, val_labeled_positions)  -> IsotonicRegression
apply_calibration(iso, val_raw_all, test_raw, val_labeled_positions)
    -> (val_cal_all, val_cal_labeled, test_cal)
conformal_intervals(y_val_labeled, val_cal_labeled, alpha)  -> (q_90, coverage)
build_intervals(all_pds, q_90)                              -> (lower, upper)
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool

from config import ALPHA, CB_PARAMS, SEED


# ── Bootstrap calibrated prediction intervals ─────────────────────────────────

def bootstrap_calibrated_intervals(
    model: CatBoostClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_all: np.ndarray,
    y_val_labeled: np.ndarray,
    val_labeled_positions: list,
    X_test: np.ndarray,
    cat_indices: list,
    n_boot: int = 200,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> dict:
    """
    Bootstrap calibrated prediction intervals with conformal coverage correction.

    Steps 1–4 per Deliverable A spec
    ---------------------------------
    1. Resample X_train 200 times (with replacement); train a fresh CatBoost per sample.
    2. Calibrate each bootstrap model's val predictions (isotonic regression on
       val_labeled); apply the fitted calibrator to val_all and test.
    3. Point estimate  = bootstrap mean (axis=0); clips to [0,1].
       pd_lower_90     = 5th percentile of bootstrap calibrated predictions.
       pd_upper_90     = 95th percentile of bootstrap calibrated predictions.
    4. Conformal coverage check on val_labeled; if empirical coverage < (1-alpha),
       expand intervals using the conformal nonconformity quantile q_90.

    Parameters
    ----------
    model                 : reference CatBoostClassifier (tree count + params extracted)
    X_train, y_train      : full labeled training set for bootstrap resampling
    X_val_all             : all val rows (labeled + unlabeled) — for submission
    y_val_labeled         : true default labels for the labeled val subset
    val_labeled_positions : integer indices mapping labeled val rows → val_all
    X_test                : test rows — for submission
    cat_indices           : categorical feature positions in the feature array
    n_boot                : bootstrap iterations (default 200)
    seed                  : base random seed; bootstrap b uses seed+b
    alpha                 : target miscoverage level (default 0.10 → 90% intervals)

    Returns
    -------
    dict with keys:
        val_cal_all      (n_val_all,)     bootstrap-mean calibrated PD, all val rows
        val_cal_labeled  (n_val_labeled,) bootstrap-mean calibrated PD, labeled val
        test_cal         (n_test,)        bootstrap-mean calibrated PD, test rows
        val_lower        (n_val_all,)     pd_lower_90 for val rows
        val_upper        (n_val_all,)     pd_upper_90 for val rows
        test_lower       (n_test,)        pd_lower_90 for test rows
        test_upper       (n_test,)        pd_upper_90 for test rows
        coverage         float            final empirical coverage on val_labeled
    """
    rng       = np.random.default_rng(seed)
    n_train   = len(X_train)
    n_val_all = len(X_val_all)
    n_test    = len(X_test)

    # ── Step 1 — Bootstrap model params: same hypers as reference, no early stopping ─
    # Use the reference model's actual tree count so each bootstrap model trains
    # for the same number of rounds (comparable capacity, no eval-set dependency).
    boot_params = dict(
        iterations    = model.tree_count_,
        learning_rate = CB_PARAMS['learning_rate'],
        depth         = CB_PARAMS['depth'],
        loss_function = CB_PARAMS['loss_function'],
        verbose       = 0,
    )

    cal_boot_val_all = np.full((n_boot, n_val_all), np.nan)
    cal_boot_test    = np.full((n_boot, n_test),    np.nan)

    with tqdm(range(n_boot), desc='Bootstrap', unit='model') as pbar:
        for b in pbar:
            try:
                # Step 1 — resample and train
                idx  = rng.integers(0, n_train, size=n_train)
                X_b  = X_train[idx]
                y_b  = y_train[idx]
                boot_model = CatBoostClassifier(**boot_params, random_seed=seed + b)
                boot_model.fit(Pool(X_b, label=y_b, cat_features=cat_indices))

                raw_val_all = boot_model.predict_proba(X_val_all)[:, 1]
                raw_test    = boot_model.predict_proba(X_test)[:, 1]

                # Step 2 — isotonic calibration on labeled val, apply to val_all + test
                raw_val_labeled = raw_val_all[val_labeled_positions]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(raw_val_labeled, y_val_labeled)
                cal_boot_val_all[b] = iso.predict(raw_val_all)
                cal_boot_test[b]    = iso.predict(raw_test)

            except Exception:
                pass  # NaN row excluded from nanmean / nanpercentile

    # ── Step 3 — Point estimate and bootstrap intervals ───────────────────────
    val_cal_all = np.clip(np.nanmean(cal_boot_val_all, axis=0), 0.0, 1.0)
    test_cal    = np.clip(np.nanmean(cal_boot_test,    axis=0), 0.0, 1.0)

    val_cal_labeled = val_cal_all[val_labeled_positions]

    val_lower  = np.clip(np.nanpercentile(cal_boot_val_all,  5, axis=0), 0.0, 1.0)
    val_upper  = np.clip(np.nanpercentile(cal_boot_val_all, 95, axis=0), 0.0, 1.0)
    test_lower = np.clip(np.nanpercentile(cal_boot_test,     5, axis=0), 0.0, 1.0)
    test_upper = np.clip(np.nanpercentile(cal_boot_test,    95, axis=0), 0.0, 1.0)

    # Enforce lower <= point_estimate <= upper row-by-row
    val_lower  = np.minimum(val_lower,  val_cal_all)
    val_upper  = np.maximum(val_upper,  val_cal_all)
    test_lower = np.minimum(test_lower, test_cal)
    test_upper = np.maximum(test_upper, test_cal)

    # ── Step 4 — Conformal coverage check and correction ─────────────────────
    val_lower_labeled = val_lower[val_labeled_positions]
    val_upper_labeled = val_upper[val_labeled_positions]
    coverage = float(np.mean(
        (y_val_labeled >= val_lower_labeled) & (y_val_labeled <= val_upper_labeled)
    ))
    print(f'Bootstrap interval coverage on val: {coverage * 100:.1f}%  (target: {(1 - alpha) * 100:.0f}%)')

    if coverage < (1 - alpha):
        # Compute conformal nonconformity scores from the base model calibrated on val
        base_raw_val     = model.predict_proba(X_val_all)[:, 1]
        base_raw_labeled = base_raw_val[val_labeled_positions]
        iso_base = IsotonicRegression(out_of_bounds='clip')
        iso_base.fit(base_raw_labeled, y_val_labeled)
        base_cal_labeled = iso_base.predict(base_raw_labeled)

        conf_scores = np.abs(y_val_labeled.astype(float) - base_cal_labeled)
        n_cal   = len(conf_scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        q_90    = float(np.quantile(conf_scores, q_level))

        val_lower  = np.clip(val_lower  - q_90, 0.0, 1.0)
        val_upper  = np.clip(val_upper  + q_90, 0.0, 1.0)
        test_lower = np.clip(test_lower - q_90, 0.0, 1.0)
        test_upper = np.clip(test_upper + q_90, 0.0, 1.0)
        # Re-enforce monotonicity after expansion
        val_lower  = np.minimum(val_lower,  val_cal_all)
        val_upper  = np.maximum(val_upper,  val_cal_all)
        test_lower = np.minimum(test_lower, test_cal)
        test_upper = np.maximum(test_upper, test_cal)

        val_lower_labeled = val_lower[val_labeled_positions]
        val_upper_labeled = val_upper[val_labeled_positions]
        coverage = float(np.mean(
            (y_val_labeled >= val_lower_labeled) & (y_val_labeled <= val_upper_labeled)
        ))
        print(f'After conformal correction (q_90={q_90:.4f}): {coverage * 100:.1f}%')

    return dict(
        val_cal_all     = val_cal_all,
        val_cal_labeled = val_cal_labeled,
        test_cal        = test_cal,
        val_lower       = val_lower,
        val_upper       = val_upper,
        test_lower      = test_lower,
        test_upper      = test_upper,
        coverage        = coverage,
    )


# ── Legacy: single-model isotonic calibration + conformal intervals ───────────
# Kept for reference; the notebook now uses bootstrap_calibrated_intervals above.

def fit_calibrator(
    val_raw_all: np.ndarray,
    y_val_labeled: np.ndarray,
    val_labeled_positions: list,
) -> IsotonicRegression:
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
    val_cal_all     = iso.predict(val_raw_all)
    val_cal_labeled = val_cal_all[val_labeled_positions]
    test_cal        = iso.predict(test_raw)
    return val_cal_all, val_cal_labeled, test_cal


def conformal_intervals(
    y_val_labeled: np.ndarray,
    val_cal_labeled: np.ndarray,
    alpha: float = ALPHA,
):
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
    lower = np.minimum(np.clip(all_pds - q_90, 0.0, 1.0), all_pds)
    upper = np.maximum(np.clip(all_pds + q_90, 0.0, 1.0), all_pds)
    return lower, upper
