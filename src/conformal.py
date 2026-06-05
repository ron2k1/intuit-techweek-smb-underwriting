"""90% PD intervals over the latent default probability.

The brief asks for intervals [l, u] on the *probability of default*, not a
prediction set for the binary outcome y. Conformalizing |y - p_hat| on binary
labels degenerates because residuals concentrate near 0 and 1.

Instead we combine two principled sources of uncertainty about the latent rate:
  1. Per-bin Wilson 90% confidence interval on the observed default rate
     within the calibration bin (captures finite-sample uncertainty about the
     bin's true rate).
  2. Ensemble dispersion across the calibrated PD models (captures model
     disagreement at the row level).

The interval half-width is the maximum of (1) and (2). Coverage is verified
at the bin level: in each PD bin, the interval around the bin's mean
prediction should contain the empirical default rate ~90% of the time
across bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.6448536269514722  # standard normal 95th percentile (two-sided 90%)


def _wilson_half_width(rate: float, n: int, z: float = Z90) -> float:
    if n <= 0:
        return 0.5
    denom = 1.0 + z * z / n
    half = z * np.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return float(half)


def pd_interval_bin_table(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"p": p_cal.astype(float), "y": y_cal.astype(float)})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.groupby("bin", dropna=True):
        n = len(group)
        rate = float(group["y"].mean())
        mean_pred = float(group["p"].mean())
        wilson_half = _wilson_half_width(rate, n)
        miscalibration = abs(mean_pred - rate)
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(n),
                "p_min": float(group["p"].min()),
                "p_max": float(group["p"].max()),
                "mean_predicted_pd": mean_pred,
                "observed_default_rate": rate,
                "wilson_half_width": wilson_half,
                "miscalibration": miscalibration,
                "bin_width": wilson_half + miscalibration,
            }
        )
    table = pd.DataFrame(rows).sort_values("p_min").reset_index(drop=True)
    return table


def assign_bin_widths(point: np.ndarray, table: pd.DataFrame, fallback: float) -> np.ndarray:
    widths = np.full(len(point), fallback, dtype=float)
    for _, row in table.iterrows():
        mask = (point >= row["p_min"]) & (point <= row["p_max"])
        widths[mask] = row["bin_width"]
    return widths


def build_pd_intervals(
    point: np.ndarray,
    pred_matrix: np.ndarray,
    bin_table: pd.DataFrame,
    fallback_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-width = max(per-bin Wilson + miscalibration, ensemble 5-95 spread)."""
    if fallback_width is None:
        fallback_width = float(bin_table["bin_width"].median()) if len(bin_table) else 0.05
    bin_widths = assign_bin_widths(point, bin_table, fallback_width)
    if pred_matrix is not None and pred_matrix.shape[1] > 1:
        ensemble_low = np.quantile(pred_matrix, 0.05, axis=1)
        ensemble_high = np.quantile(pred_matrix, 0.95, axis=1)
        ensemble_half = np.maximum(point - ensemble_low, ensemble_high - point)
    else:
        ensemble_half = np.zeros_like(point)
    half = np.maximum(bin_widths, ensemble_half)
    lower = np.clip(point - half, 0.0, 1.0)
    upper = np.clip(point + half, 0.0, 1.0)
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return lower, upper


def bin_level_coverage(
    p_val: np.ndarray,
    y_val: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Coverage proxy: in each PD bin, the interval should contain the bin's
    observed default rate. Reports the fraction of bins that achieve this and
    the mean / median interval width."""
    if len(y_val) == 0:
        return {"n": 0, "bin_coverage": float("nan"), "mean_width": float("nan"), "median_width": float("nan")}
    frame = pd.DataFrame(
        {
            "p": p_val.astype(float),
            "y": y_val.astype(float),
            "lower": lower.astype(float),
            "upper": upper.astype(float),
        }
    )
    frame["bin"] = pd.qcut(frame["p"], q=min(n_bins, frame["p"].nunique()), labels=False, duplicates="drop")
    bin_hits = []
    for _, group in frame.groupby("bin", dropna=True):
        rate = float(group["y"].mean())
        lo = float(group["lower"].mean())
        hi = float(group["upper"].mean())
        bin_hits.append(int(lo <= rate <= hi))
    return {
        "n": int(len(y_val)),
        "bin_coverage": float(np.mean(bin_hits)) if bin_hits else float("nan"),
        "n_bins": int(len(bin_hits)),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
    }
