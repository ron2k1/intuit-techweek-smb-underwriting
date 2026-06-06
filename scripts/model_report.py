#!/usr/bin/env python3
"""Generate a comparison-ready Deliverable A model report card.

This is intentionally read-only with respect to submission files. It reports the
currently selected LightGBM/no-prior policy in the same style as teammate model
cards: PD metrics, calibration deciles, and policy/NPV summary rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv-files"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SUBMISSION_DIR = OUTPUT_DIR / "submission"

DROP_PRIOR_SCORE_PROXIES = {
    "prior_underwriter_score",
    "prior_score_logit",
    "selection_support_index",
}
MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 900,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 55,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.35,
    "verbosity": -1,
}


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"
    return f"{sign}${value:.0f}"


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = calibration_table(y, p, n_bins=n_bins)
    return float((table["n"] * table["abs_error"]).sum() / table["n"].sum())


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(y)),
        "default_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10bin": ece_score(y, p, n_bins=10),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows = []
    for decile, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        pred = float(np.mean(p[idx]))
        obs = float(np.mean(y[idx]))
        rows.append(
            {
                "decile": decile + 1,
                "n": int(len(idx)),
                "p_min": float(np.min(p[idx])),
                "p_max": float(np.max(p[idx])),
                "mean_predicted_pd": pred,
                "observed_default_rate": obs,
                "abs_error": abs(pred - obs),
            }
        )
    return pd.DataFrame(rows)


def prepare_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    numeric = [c for c in numeric if c not in DROP_PRIOR_SCORE_PROXIES]
    categorical = [c for c in categorical if c not in DROP_PRIOR_SCORE_PROXIES]
    feature_cols = numeric + categorical

    frames = [train_fe, validation_fe, test_fe]
    for frame in frames:
        for col in categorical:
            frame[col] = frame[col].astype("category")
    return train, validation, test, feature_cols, categorical


def fit_predict_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    categorical: list[str],
) -> np.ndarray:
    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    for frame in [train_fe, validation_fe]:
        for col in categorical:
            frame[col] = frame[col].astype("category")

    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * 0.80)
    model_idx = ordered[:split_at]
    cal_idx = ordered[split_at:]

    model = LGBMClassifier(**MODEL_PARAMS, random_state=1103)
    model.fit(
        train_fe.loc[model_idx, feature_cols],
        train.loc[model_idx, "default_flag"].astype(int),
        categorical_feature=categorical,
    )
    raw_cal = model.predict_proba(train_fe.loc[cal_idx, feature_cols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, train.loc[cal_idx, "default_flag"].astype(int).to_numpy())
    raw_val = model.predict_proba(validation_fe[feature_cols])[:, 1]
    return np.clip(calibrator.predict(raw_val), 0.001, 0.999)


def out_of_fold_predictions(
    train: pd.DataFrame,
    feature_cols: list[str],
    categorical: list[str],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    train_fe = add_application_features(train)
    for col in categorical:
        train_fe[col] = train_fe[col].astype("category")

    labeled_idx = train.index[train["default_flag"].notna()].to_numpy()
    y = train.loc[labeled_idx, "default_flag"].astype(int).to_numpy()
    oof = np.full(len(labeled_idx), np.nan)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)

    for fold, (fit_pos, holdout_pos) in enumerate(splitter.split(labeled_idx, y), start=1):
        fit_idx_all = labeled_idx[fit_pos]
        holdout_idx = labeled_idx[holdout_pos]
        fit_ordered = (
            train.loc[fit_idx_all]
            .sort_values("application_timestamp")
            .index.to_numpy()
        )
        cal_start = int(len(fit_ordered) * 0.80)
        model_idx = fit_ordered[:cal_start]
        cal_idx = fit_ordered[cal_start:]

        model = LGBMClassifier(**MODEL_PARAMS, random_state=1103 + fold)
        model.fit(
            train_fe.loc[model_idx, feature_cols],
            train.loc[model_idx, "default_flag"].astype(int),
            categorical_feature=categorical,
        )
        raw_cal = model.predict_proba(train_fe.loc[cal_idx, feature_cols])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        calibrator.fit(raw_cal, train.loc[cal_idx, "default_flag"].astype(int).to_numpy())
        raw_holdout = model.predict_proba(train_fe.loc[holdout_idx, feature_cols])[:, 1]
        oof[holdout_pos] = np.clip(calibrator.predict(raw_holdout), 0.001, 0.999)

    if np.isnan(oof).any():
        raise RuntimeError("OOF prediction generation left missing predictions")
    return y, oof


def policy_rows(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_DIR / "submission_A_decisions.csv")
    curves = np.load(OUTPUT_DIR / "deliverable_a_curves.npz")
    val = submission.iloc[: len(validation)].copy()
    tst = submission.iloc[len(validation) :].copy()

    val_amount = validation["requested_amount"].to_numpy(float)
    test_amount = test["requested_amount"].to_numpy(float)
    val_margin = expected_npv(
        val_amount,
        val["predicted_pd"].to_numpy(float),
        curves["validation_t_star"],
        curves["validation_recovery"],
    ) / np.maximum(val_amount, 1.0)
    test_margin = expected_npv(
        test_amount,
        tst["predicted_pd"].to_numpy(float),
        curves["test_t_star"],
        curves["test_recovery"],
    ) / np.maximum(test_amount, 1.0)
    val_prior_declined = validation["prior_decision"].to_numpy() == 0
    test_prior_declined = test["prior_decision"].to_numpy() == 0

    policies = {
        "Selected/deployed": (
            val["decision"].to_numpy(int).astype(bool),
            tst["decision"].to_numpy(int).astype(bool),
        ),
        "EV > 0 on all": (
            val_margin > 0.0,
            test_margin > 0.0,
        ),
        "Raw NPV threshold": (
            val_margin > 0.009380087912607058,
            test_margin > 0.009380087912607058,
        ),
    }

    realized = realized_npv(validation.loc[validation["default_flag"].notna()])
    labeled = validation["default_flag"].notna().to_numpy()
    rows = []
    for name, (val_decision, test_decision) in policies.items():
        all_decision = np.r_[val_decision, test_decision]
        all_amount = np.r_[val_amount, test_amount]
        all_margin = np.r_[val_margin, test_margin]
        approved_amount = float(all_amount[all_decision].sum())
        expected_npv_total = float((all_margin[all_decision] * all_amount[all_decision]).sum())
        val_labeled_decision = val_decision[labeled]
        rows.append(
            {
                "policy": name,
                "approved": int(all_decision.sum()),
                "approval_rate": float(all_decision.mean()),
                "expected_npv": expected_npv_total,
                "roic": expected_npv_total / approved_amount if approved_amount else np.nan,
                "realized_npv_verifiable": float(realized[val_labeled_decision].sum()),
                "labeled_approved": int(val_labeled_decision.sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    metrics: dict[str, dict[str, float]],
    val_cal: pd.DataFrame,
    oof_cal: pd.DataFrame,
    policies: pd.DataFrame,
) -> None:
    report_path = REPORT_DIR / "selected_model_report_card.md"
    json_path = REPORT_DIR / "selected_model_report_card.json"
    csv_path = REPORT_DIR / "selected_model_policy_report.csv"
    cal_path = REPORT_DIR / "selected_model_calibration_deciles.csv"

    policies.to_csv(csv_path, index=False)
    pd.concat(
        [
            val_cal.assign(split="validation"),
            oof_cal.assign(split="honest_oof"),
        ],
        ignore_index=True,
    ).to_csv(cal_path, index=False)

    payload = {
        "model": "LightGBM/no-prior-score PD + brief-formula NPV policy",
        "metrics": metrics,
        "policies": policies.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    metric_rows = [
        ("AUC-ROC", "auroc"),
        ("Brier score", "brier"),
        ("Log loss", "log_loss"),
        ("Calibration ECE, 10-bin", "ece_10bin"),
    ]
    lines = [
        "# Selected Model Report Card",
        "",
        "Model: LightGBM/no-prior-score PD + brief-formula NPV policy.",
        "",
        "| Metric | Validation (2,551) | Honest OOF (51,722) |",
        "| --- | ---: | ---: |",
    ]
    for label, key in metric_rows:
        lines.append(
            f"| {label} | {metrics['validation'][key]:.4f} | {metrics['honest_oof'][key]:.4f} |"
        )

    lines.extend(
        [
            "",
            "Calibration deciles, validation:",
            "",
            "```text",
            "low  "
            + "  ".join(
                f"{r.mean_predicted_pd:.3f}->{r.observed_default_rate:.3f}"
                for r in val_cal.head(5).itertuples()
            ),
            "high "
            + "  ".join(
                f"{r.mean_predicted_pd:.3f}->{r.observed_default_rate:.3f}"
                for r in val_cal.tail(5).itertuples()
            ),
            "```",
            "",
            "Calibration deciles, honest OOF:",
            "",
            "```text",
            "low  "
            + "  ".join(
                f"{r.mean_predicted_pd:.3f}->{r.observed_default_rate:.3f}"
                for r in oof_cal.head(5).itertuples()
            ),
            "high "
            + "  ".join(
                f"{r.mean_predicted_pd:.3f}->{r.observed_default_rate:.3f}"
                for r in oof_cal.tail(5).itertuples()
            ),
            "```",
            "",
            "PD -> expected NPV -> decision",
            "",
            "Decision rule: use the brief cash-flow formula with predicted PD, expected default day, and expected recovery. Selected policy approves when NPV margin > 0.00938, with a 0.030 margin guardrail for prior-declined applicants.",
            "",
            "| Policy | Approved | Expected NPV | ROIC | Realized NPV (verifiable) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policies.itertuples():
        lines.append(
            "| "
            f"{row.policy} | "
            f"{row.approved:,} ({row.approval_rate:.1%}) | "
            f"{fmt_money(row.expected_npv)} | "
            f"{row.roic:.2%} | "
            f"{fmt_money(row.realized_npv_verifiable)} ({row.labeled_approved:,} labeled) |"
        )
    lines.extend(
        [
            "",
            "Comparison notes:",
            "",
            "- Use validation AUC/log loss/Brier when comparing held-out validation rows.",
            "- Use honest OOF as the broad training-population diagnostic; every OOF row is predicted by a fold model that did not train on that row.",
            "- Realized NPV is verifiable only on labeled validation rows; expected NPV covers validation + test approvals.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, test, feature_cols, categorical = prepare_features()

    val_pd = fit_predict_validation(train, validation, feature_cols, categorical)
    val_labeled = validation["default_flag"].notna().to_numpy()
    val_y = validation.loc[val_labeled, "default_flag"].astype(int).to_numpy()
    val_pred = val_pd[val_labeled]

    oof_y, oof_pred = out_of_fold_predictions(train, feature_cols, categorical)

    metrics = {
        "validation": metric_summary(val_y, val_pred),
        "honest_oof": metric_summary(oof_y, oof_pred),
    }
    val_cal = calibration_table(val_y, val_pred, n_bins=10)
    oof_cal = calibration_table(oof_y, oof_pred, n_bins=10)
    policies = policy_rows(validation, test)
    write_report(metrics, val_cal, oof_cal, policies)

    print(json.dumps({"metrics": metrics, "report": str(REPORT_DIR / "selected_model_report_card.md")}, indent=2))


if __name__ == "__main__":
    main()
