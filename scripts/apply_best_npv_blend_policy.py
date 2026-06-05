#!/usr/bin/env python3
"""Apply the best current direct-NPV blend policy to submission A.

Policy selected from `scripts/experiment_npv_policy.py`:

score = 0.3 * decomposed_npv_margin + 0.7 * direct_hgb_predicted_margin
approve if score > 0.005

This changes only `decision`; PD point estimates and intervals remain from the
calibrated A PD model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


ALPHA_DIRECT = 0.7
THRESHOLD = 0.005


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main() -> None:
    root = Path(".")
    report_dir = root / "outputs" / "reports"
    submission_dir = root / "outputs" / "submission"
    archive_dir = report_dir / "archive"
    report_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "data" / "csv-files" / "train.csv")
    validation = pd.read_csv(root / "data" / "csv-files" / "validation.csv")
    test = pd.read_csv(root / "data" / "csv-files" / "test.csv")
    submission_a = pd.read_csv(submission_dir / "submission_A_decisions.csv")
    curves = np.load(root / "outputs" / "deliverable_a_curves.npz")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    feature_cols = numeric + categorical

    labeled_train = train["default_flag"].notna()
    y_train_npv = realized_npv(train.loc[labeled_train])
    y_train_margin = y_train_npv / np.maximum(
        train.loc[labeled_train, "requested_amount"].to_numpy(float), 1.0
    )

    model = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical)),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_iter=260,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    loss="squared_error",
                    random_state=701,
                ),
            ),
        ]
    )
    model.fit(train_fe.loc[labeled_train, feature_cols], y_train_margin)

    val_direct = model.predict(validation_fe[feature_cols])
    test_direct = model.predict(test_fe[feature_cols])
    val_decomp = expected_npv(
        validation["requested_amount"].to_numpy(float),
        curves["validation_pd"],
        curves["validation_t_star"],
        curves["validation_recovery"],
    ) / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
    test_decomp = expected_npv(
        test["requested_amount"].to_numpy(float),
        curves["test_pd"],
        curves["test_t_star"],
        curves["test_recovery"],
    ) / np.maximum(test["requested_amount"].to_numpy(float), 1.0)

    val_score = (1 - ALPHA_DIRECT) * val_decomp + ALPHA_DIRECT * val_direct
    test_score = (1 - ALPHA_DIRECT) * test_decomp + ALPHA_DIRECT * test_direct
    val_decision = (val_score > THRESHOLD).astype(int)
    test_decision = (test_score > THRESHOLD).astype(int)

    backup = archive_dir / "submission_A_decisions_before_direct_npv_blend.csv"
    if not backup.exists():
        submission_a.to_csv(backup, index=False)

    updated = submission_a.copy()
    updated.loc[: len(validation) - 1, "decision"] = val_decision
    updated.loc[len(validation) :, "decision"] = test_decision
    updated.to_csv(submission_dir / "submission_A_decisions.csv", index=False)

    labeled_val = validation["default_flag"].notna().to_numpy()
    realized_val = realized_npv(validation.loc[labeled_val])
    summary = {
        "policy": "blend_decomp_direct_hgb",
        "alpha_direct": ALPHA_DIRECT,
        "threshold": THRESHOLD,
        "validation_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(val_decision.sum() + test_decision.sum()),
        "validation_labeled_approved": int(val_decision[labeled_val].sum()),
        "validation_labeled_realized_npv": float(realized_val[val_decision[labeled_val] == 1].sum()),
        "score_note": "Decision score is margin-like but not a calibrated PD.",
    }
    (report_dir / "direct_npv_blend_active_policy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

