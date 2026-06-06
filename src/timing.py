"""Discrete-time weekly hazard model + recovery regressor.

Provides per-row cumulative default probability by week, expected default day,
and expected recovery rate. Powers Deliverable A NPV, Deliverable B CDR curves,
and Deliverable C counterfactuals.

Time grid:
- 13 weekly buckets covering days 1..91 (week w = days 7(w-1)+1 .. 7w).
- Default at day t* maps to bucket ceil(t*/7) in {1,..,13}.
- Repaid loans (days_to_full_repayment = 60) are treated as non-defaulters across
  all 13 buckets, since the brief defines default events through day 90.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


N_WEEKS = 13
WEEK_DAYS = 7


def day_to_week(day: float) -> int:
    return int(min(N_WEEKS, max(1, np.ceil(day / WEEK_DAYS))))


def build_person_period(labeled: pd.DataFrame, feature_x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Long-format reshape for discrete-time hazard.

    For each labeled row, emit one record per surviving week up to event or 13.
    Label = 1 only in the bucket where default actually occurred.
    """
    weeks_to_event = np.where(
        labeled["default_flag"].to_numpy() == 1,
        np.ceil(labeled["days_to_default"].fillna(N_WEEKS * WEEK_DAYS).to_numpy() / WEEK_DAYS),
        N_WEEKS,
    ).astype(int)
    weeks_to_event = np.clip(weeks_to_event, 1, N_WEEKS)
    defaulted = labeled["default_flag"].to_numpy() == 1

    row_indices = []
    weeks = []
    hazards = []
    for local_idx, (wte, dft) in enumerate(zip(weeks_to_event, defaulted)):
        for w in range(1, wte + 1):
            row_indices.append(local_idx)
            weeks.append(w)
            hazards.append(1 if (dft and w == wte) else 0)

    row_indices = np.array(row_indices, dtype=int)
    expanded = feature_x.iloc[row_indices].copy()
    expanded["week_bucket"] = weeks
    return expanded, np.asarray(hazards, dtype=int)


@dataclass
class HazardModel:
    pipeline: Pipeline
    feature_cols: list[str]

    def predict_hazard_matrix(self, feature_x: pd.DataFrame) -> np.ndarray:
        """Return (n_rows, N_WEEKS) matrix of per-week hazards."""
        n = len(feature_x)
        out = np.zeros((n, N_WEEKS), dtype=float)
        for w in range(1, N_WEEKS + 1):
            xw = feature_x.copy()
            xw["week_bucket"] = w
            out[:, w - 1] = self.pipeline.predict_proba(xw[self.feature_cols])[:, 1]
        return np.clip(out, 1e-6, 1 - 1e-6)

    def cumulative_default(self, hazards: np.ndarray) -> np.ndarray:
        """(n_rows, N_WEEKS) cumulative default probability by end of week w."""
        survival = np.cumprod(1.0 - hazards, axis=1)
        return 1.0 - survival

    def predict_curves(self, feature_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        hazards = self.predict_hazard_matrix(feature_x)
        cumulative = self.cumulative_default(hazards)
        return hazards, cumulative


def fit_hazard_model(
    feature_x: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    sample_weight: np.ndarray | None = None,
    random_state: int = 17,
) -> HazardModel:
    person_period, y = build_person_period(labeled, feature_x)
    feature_cols = list(person_period.columns)

    weights = None
    if sample_weight is not None:
        # Repeat each row's weight across its person-period entries
        person_period_idx = []
        weeks_to_event = np.where(
            labeled["default_flag"].to_numpy() == 1,
            np.ceil(labeled["days_to_default"].fillna(N_WEEKS * WEEK_DAYS).to_numpy() / WEEK_DAYS),
            N_WEEKS,
        ).astype(int)
        weeks_to_event = np.clip(weeks_to_event, 1, N_WEEKS)
        for local_idx, wte in enumerate(weeks_to_event):
            person_period_idx.extend([local_idx] * int(wte))
        weights = sample_weight[np.array(person_period_idx, dtype=int)]

    clf = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        min_samples_leaf=40,
        random_state=random_state,
    )
    pipe = Pipeline([("clf", clf)])
    fit_kwargs = {"clf__sample_weight": weights} if weights is not None else {}
    pipe.fit(person_period[feature_cols], y, **fit_kwargs)
    return HazardModel(pipeline=pipe, feature_cols=feature_cols)


@dataclass
class RecoveryModel:
    pipeline: Pipeline
    feature_cols: list[str]
    fallback_rate: float

    def predict_rate(self, feature_x: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            return np.full(len(feature_x), self.fallback_rate, dtype=float)
        x = feature_x[self.feature_cols]
        pred = self.pipeline.predict(x)
        return np.clip(pred, 0.0, 1.0)


def fit_recovery_model(
    feature_x: pd.DataFrame,
    defaulted_rows: pd.DataFrame,
) -> RecoveryModel:
    """Predict recovery_rate = final_recovered_amount / requested_amount in [0,1]."""
    feature_cols = list(feature_x.columns)
    if len(defaulted_rows) < 200:
        rate = float(
            (defaulted_rows["final_recovered_amount"] / defaulted_rows["requested_amount"])
            .clip(0, 1)
            .mean()
        )
        return RecoveryModel(pipeline=None, feature_cols=feature_cols, fallback_rate=rate or 0.10)

    train_x = feature_x.loc[defaulted_rows.index, feature_cols]
    rate = (defaulted_rows["final_recovered_amount"] / defaulted_rows["requested_amount"]).clip(0, 1).to_numpy()
    reg = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        min_samples_leaf=30,
        random_state=29,
    )
    pipe = Pipeline([("reg", reg)])
    pipe.fit(train_x, rate)
    fallback = float(rate.mean())
    return RecoveryModel(pipeline=pipe, feature_cols=feature_cols, fallback_rate=fallback)


def expected_default_day(cumulative: np.ndarray) -> np.ndarray:
    """E[t* | default] using week midpoints, weighted by bucket-conditional mass.

    cumulative: (n_rows, N_WEEKS), F_w(x) = Pr(default by end of week w | x).
    Conditional bucket mass m_w = (F_w - F_{w-1}) / F_13.
    Week midpoint day = 7w - 3.
    """
    n = cumulative.shape[0]
    f = cumulative
    prev = np.concatenate([np.zeros((n, 1)), f[:, :-1]], axis=1)
    bucket_mass = f - prev
    total = f[:, -1:].clip(min=1e-9)
    week_idx = np.arange(1, N_WEEKS + 1)
    midpoints = 7 * week_idx - 3
    conditional = bucket_mass / total
    return (conditional * midpoints).sum(axis=1)
