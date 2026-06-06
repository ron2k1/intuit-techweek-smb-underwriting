"""Shared PD modeling spine -- the literal tie between Deliverables A and C.

One calibrated gradient-boosted default model, one bootstrap uncertainty recipe,
one conformal interval calibrator. A and C import all three and differ ONLY in the
feature set they pass in:

    A  -> data.feature_columns(df)         (all predictors; pure prediction)
    C  -> data.causal_feature_columns(df)  (backdoor-valid; drops colliders + self-report)

Because the estimator / calibration / interval machinery is identical, C's baseline
(no-intervention) PD is the same quantity A reports, just measured on the causal
feature set -- so a do() effect is a clean, comparable delta on top of A's spine.

All formulas are explicit (no black-box wrappers around the uncertainty):
  * calibrated probability      : isotonic regression over internal CV folds
  * epistemic interval          : bootstrap-ensemble standard deviation
  * 90% coverage                : conformal multiplier lambda = 90th percentile of
                                  standardized bin residuals |r_bin - p_bin| / std_bin
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

# 90% two-sided z, kept for reference / fallbacks. The conformal lambda below is
# what actually sets width; Z90 is only used if a caller wants a normal interval.
Z90 = 1.6448536269514722
MIN_HALFWIDTH = 0.02     # floor: never claim an interval tighter than +/-2 pts of PD
DEFAULT_SEED = 20260605


def hgb(seed: int, cat_idx: list[int] | None = None,
        scoring: str = "loss") -> HistGradientBoostingClassifier:
    """The one booster everyone uses. Native NaN handling; native categoricals.

    Tuned conservatively (depth-limited, strong min_samples_leaf + L2) because the
    labelled sample is a SELECTED slice of the population -- we want a smooth,
    well-regularized surface that extrapolates sanely into the reject region, not a
    sharp interpolator of the approved book.

    `scoring` is the early-stopping monitor. Deliverable A passes scoring="roc_auc"
    so model selection maximizes RANK discrimination (the AUC-ROC objective);
    calibration then fixes the probability levels. Default "loss" (log_loss) is a
    proper scoring rule that rewards calibration directly -- used by C where the
    causal estimate cares about levels more than ranking.
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        scoring=scoring,
        categorical_features=(cat_idx if cat_idx else None),
        random_state=seed,
    )


def lgbm(seed: int, cat_idx: list[int] | None = None, scoring: str | None = None):
    """A SECOND, algorithmically-distinct gradient booster for the rank blend.

    LightGBM grows leaf-wise (best-first) where sklearn's HistGradientBoosting grows
    depth-first/level-balanced; on the same features they make DIFFERENT ranking errors,
    so a convex blend of the two lifts AUC-ROC beyond either alone (the one axis the
    competitive review found us trailing on). Regularized to MATCH the HGB leg
    (400 trees, lr 0.05, ~31 leaves, min_child 50, L2 1.0) so neither leg is sharper than
    the other -- the same "smooth surface that extrapolates sanely into the reject region"
    rationale as hgb(). `deterministic`+single-threaded so the shipped PD is reproducible.

    cat_idx/scoring are accepted for signature parity with hgb() but LightGBM reads the
    int-coded categoricals as numeric here (same as the logistic leg) -- declaring float
    categoricals to LightGBM errors, and the blend's diversity comes from the algorithm,
    not from re-handling the categoricals a third way.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def fit_calibrated(X: np.ndarray, y: np.ndarray, cat_idx: list[int] | None = None,
                   seed: int = DEFAULT_SEED, cv: int = 5,
                   scoring: str = "loss",
                   sample_weight: np.ndarray | None = None,
                   estimator=None) -> CalibratedClassifierCV:
    """Probability-calibrated PD model (isotonic, internal CV).

    Isotonic (not Platt) because the selection bias bends the reliability curve
    non-monotonically in shape but monotonically in rank; isotonic respects the
    ranking while free-form correcting the levels.

    `sample_weight` (optional) flows to BOTH the booster fit and the isotonic
    calibrator across every CV fold. Deliverable A passes recency (time-decay)
    weights so the calibrated LEVEL tracks the recent, higher-default regime the
    walk-forward backtest found drifting upward -- ranking is unchanged, only the
    probability level is corrected.

    `estimator` (optional) swaps in a different base learner (e.g. model.lgbm(...))
    while keeping the SAME isotonic-CV calibration wrapper, so every blend leg is
    calibrated identically and the convex average stays a genuine probability.
    """
    base = estimator if estimator is not None else hgb(seed, cat_idx, scoring)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    cal.fit(X, y, sample_weight=sample_weight)
    return cal


def bootstrap_pd(X: np.ndarray, y: np.ndarray, targets: list[np.ndarray],
                 cat_idx: list[int] | None = None, seed: int = DEFAULT_SEED,
                 n_models: int = 12, scoring: str = "loss",
                 sample_weight: np.ndarray | None = None) -> list[np.ndarray]:
    """Bootstrap-ensemble PD for each matrix in `targets`.

    Returns a list aligned with `targets`; element j is an (n_models, len(targets[j]))
    array of PD predictions. The across-model standard deviation is the epistemic
    uncertainty: where the labelled data pins the surface it is small, where we
    extrapolate (reject region) the resampled models disagree and it grows.

    `sample_weight` (optional) is carried through each bootstrap resample (weights
    indexed by the resampled rows) so the uncertainty band reflects the SAME recency
    regime as the point PD -- otherwise the intervals would be calibrated to the
    stale full-history mix while the point estimate tracks the recent regime.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    sw = None if sample_weight is None else np.asarray(sample_weight, float)
    out = [np.empty((n_models, len(t))) for t in targets]
    for k in range(n_models):
        idx = rng.integers(0, n, n)
        m = hgb(seed + 1 + k, cat_idx, scoring).fit(
            X[idx], y[idx], sample_weight=(None if sw is None else sw[idx]))
        for j, t in enumerate(targets):
            out[j][k] = m.predict_proba(t)[:, 1]
    return out


def conformal_lambda(pd_cal: np.ndarray, std_cal: np.ndarray, y_cal: np.ndarray,
                     coverage: float = 0.90, n_bins: int = 20) -> float:
    """Calibrate the interval-width multiplier for ~`coverage` bin-level coverage.

    Split-conformal on the labelled calibration set:
      1. bin rows into quantile bins of predicted PD,
      2. per bin b: empirical default rate r_b, mean predicted p_b, mean ensemble std s_b,
      3. standardized residual  z_b = |r_b - p_b| / s_b,
      4. lambda = the `coverage` quantile of {z_b}.
    Then half-width = lambda * std achieves ~`coverage` of bins inside the band.
    Self-correcting: an overconfident ensemble (tiny s_b) yields large z_b -> large
    lambda -> wider bands, exactly as needed.
    """
    pd_cal = np.asarray(pd_cal, float)
    std_cal = np.asarray(std_cal, float)
    y_cal = np.asarray(y_cal, float)
    order = np.argsort(pd_cal)
    bins = np.array_split(order, n_bins)
    z = []
    for b in bins:
        if len(b) < 10:
            continue
        r_b = y_cal[b].mean()
        p_b = pd_cal[b].mean()
        s_b = max(std_cal[b].mean(), 1e-6)
        z.append(abs(r_b - p_b) / s_b)
    if not z:
        return Z90
    return float(np.quantile(z, coverage))


def make_interval(pd_hat: np.ndarray, std: np.ndarray, lam: float,
                  ood: np.ndarray | None = None, ood_boost: float = 1.5,
                  min_hw: float = MIN_HALFWIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric 90% interval: half-width = max(lambda*std, floor) * (1 + boost*ood).

    `ood` (0/1) widens the band in the never-labelled region where even a calibrated
    point estimate is an extrapolation the data cannot vouch for.
    """
    hw = np.maximum(lam * np.asarray(std, float), min_hw)
    if ood is not None:
        hw = hw * (1.0 + ood_boost * np.asarray(ood, float))
    lo = np.clip(pd_hat - hw, 0.0, 1.0)
    hi = np.clip(pd_hat + hw, 0.0, 1.0)
    return lo, hi
