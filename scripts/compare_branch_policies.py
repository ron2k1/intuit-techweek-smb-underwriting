#!/usr/bin/env python3
"""Compare teammate branch A policies on the same local validation/test data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "csv-files"
WT = ROOT / "comparison" / "worktrees"

APR = 0.35
TERM_DAYS = 60
FEE = 0.03
TERM_INT = APR * TERM_DAYS / 365.0
PAID_MARGIN = FEE + TERM_INT
DAILY_DRAW_FACTOR = (1.0 + TERM_INT) / TERM_DAYS
BRIEF_LGD = 0.30
BRIEF_BREAK_EVEN = PAID_MARGIN / (PAID_MARGIN + BRIEF_LGD)


BRANCHES = [
    {
        "person": "Ronil",
        "branch": "ronil",
        "submission": WT / "ronil" / "submissions" / "submission_A_decisions.csv",
        "code_root": WT / "ronil",
        "model": "HistGradientBoosting + Logistic blend",
        "lgd_assumption": "0.30 amortizing / brief-standardized",
        "break_even_pd": BRIEF_BREAK_EVEN,
    },
    {
        "person": "Ayush",
        "branch": "ayush",
        "submission": WT / "ayush" / "submissions" / "submission_A_decisions.csv",
        "code_root": WT / "ayush",
        "model": "HistGradientBoosting bootstrap ensemble",
        "lgd_assumption": "0.30 amortizing",
        "break_even_pd": BRIEF_BREAK_EVEN,
    },
    {
        "person": "Abhi",
        "branch": "Abhimanyu",
        "submission": WT / "Abhimanyu" / "submissions" / "submission_A_decisions.csv",
        "code_root": WT / "Abhimanyu",
        "model": "CatBoost 10-fold + isotonic",
        "lgd_assumption": "0.9086 empirical recovery trap",
        "break_even_pd": 0.0879,
    },
    {
        "person": "Steven",
        "branch": "steven",
        "submission": WT / "steven" / "outputs" / "submission" / "submission_A_decisions.csv",
        "code_root": WT / "steven",
        "model": "LightGBM no-prior-score PD + hazard/recovery NPV policy",
        "lgd_assumption": "brief-faithful cash-flow formula",
        "break_even_pd": math.nan,
    },
]


def brief_realized_npv(df: pd.DataFrame) -> np.ndarray:
    amount = df["requested_amount"].to_numpy(float)
    default = df["default_flag"].fillna(0).to_numpy(float)
    t_star = df["days_to_default"].fillna(0).to_numpy(float)
    recovery = df["final_recovered_amount"].fillna(0).to_numpy(float)
    paid = amount * PAID_MARGIN
    defaulted = FEE * amount + DAILY_DRAW_FACTOR * amount * np.clip(t_star - 1, 0, None) + recovery - amount
    return np.where(default == 1, defaulted, paid)


def brief_expected_npv(
    amount: np.ndarray,
    p_default: np.ndarray,
    expected_t_star: np.ndarray,
    expected_recovery_rate: np.ndarray,
) -> np.ndarray:
    paid = amount * PAID_MARGIN
    recovery_amount = expected_recovery_rate * amount
    defaulted = (
        FEE * amount
        + DAILY_DRAW_FACTOR * amount * np.clip(expected_t_star - 1, 0, None)
        + recovery_amount
        - amount
    )
    return (1.0 - p_default) * paid + p_default * defaulted


def amortizing_profit(df: pd.DataFrame) -> np.ndarray:
    amount = df["requested_amount"].to_numpy(float)
    default = df["default_flag"].fillna(0).to_numpy(float)
    dtd = df["days_to_default"].fillna(0).to_numpy(float)
    recovery = df["final_recovered_amount"].fillna(0).to_numpy(float)
    frac = np.clip(np.minimum(dtd, TERM_DAYS) / TERM_DAYS, 0, 1)
    paid = amount * PAID_MARGIN
    defaulted = amount * (FEE + frac * (1.0 + TERM_INT) - 1.0) + recovery
    defaulted = np.minimum(defaulted, paid)
    return np.where(default == 1, defaulted, paid)


def fixed_lgd_profit(df: pd.DataFrame, lgd: float = BRIEF_LGD) -> np.ndarray:
    amount = df["requested_amount"].to_numpy(float)
    default = df["default_flag"].fillna(0).to_numpy(float)
    return np.where(default == 1, -lgd * amount, PAID_MARGIN * amount)


def normalized_text(path: Path) -> str:
    parts: list[str] = []
    for sub in ["src", "code", "scripts", "README.md", "markdown_files"]:
        p = path / sub
        if p.is_file():
            parts.append(p.read_text(errors="ignore"))
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".py", ".md", ".ipynb"}:
                    parts.append(f.read_text(errors="ignore"))
    return "\n".join(parts)


def infer_code_flags(branch: dict) -> dict[str, object]:
    text = normalized_text(branch["code_root"])
    lowered = text.lower()
    drops_prior_underwriter = bool(
        re.search(r"drop_for_a\s*=.*prior_underwriter_score", lowered, re.S)
        or re.search(r"drop_for_pd\s*=.*prior_underwriter_score", lowered, re.S)
    )
    keeps_prior_underwriter = ("prior_underwriter_score" in text) and not drops_prior_underwriter
    drops_prior_decision = bool(
        re.search(r"drop_for_a\s*=.*prior_decision", lowered, re.S)
        or re.search(r"exclude\s*=.*prior_decision", lowered, re.S)
        or re.search(r"drop_for_pd\s*=.*prior_decision", lowered, re.S)
    )
    return {
        "keeps_prior_underwriter_score_in_A": keeps_prior_underwriter,
        "drops_prior_decision_in_A": drops_prior_decision,
        "mentions_lgd_091_or_empirical_recovery": ("0.91" in text or "0.908" in text or "empirical recovery" in lowered),
        "mentions_lgd_030_or_amortizing": ("LGD = 0.30" in text or "lgd=0.30" in lowered or "amortiz" in lowered),
        "mentions_outcome_leakage_guard": ("leakage" in lowered and "default_flag" in lowered and "days_to_default" in lowered),
    }


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    mask = y.notna() & p.notna()
    if mask.sum() == 0 or y[mask].nunique() < 2:
        return math.nan
    return float(roc_auc_score(y[mask].astype(int), p[mask]))


def summarize_branch(
    branch: dict,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    curves: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    submission_path = branch["submission"]
    if not submission_path.exists():
        row = {k: branch[k] for k in ["person", "branch", "model", "lgd_assumption", "break_even_pd"]}
        row["status"] = "missing submission_A_decisions.csv"
        return row

    sub = pd.read_csv(submission_path)
    all_scored = pd.concat(
        [
            validation.assign(_split="validation"),
            test.assign(_split="test"),
        ],
        ignore_index=True,
    )
    merged = all_scored.merge(sub, on="applicant_id", how="left", validate="one_to_one")
    if merged["decision"].isna().any():
        raise ValueError(f"{branch['person']} submission missing applicant IDs")

    merged["decision"] = merged["decision"].astype(int)
    labeled_val = (merged["_split"] == "validation") & merged["default_flag"].notna()
    approved = merged["decision"] == 1
    prior_declined = merged["prior_decision"] == 0
    prior_approved = merged["prior_decision"] == 1
    no_bank_feed = merged["has_linked_bank_feed"] == 0

    val_labeled = merged.loc[labeled_val].copy()
    val_labeled_approved = val_labeled["decision"] == 1

    brief_npv = brief_realized_npv(val_labeled)
    amort_profit = amortizing_profit(val_labeled)
    fixed_profit = fixed_lgd_profit(val_labeled)

    pd_col = "predicted_pd" if "predicted_pd" in merged.columns else None
    auc = safe_auc(val_labeled["default_flag"], val_labeled[pd_col]) if pd_col else math.nan
    mean_pd_approved = float(merged.loc[approved, pd_col].mean()) if pd_col and approved.any() else math.nan
    model_ev_lgd030 = math.nan
    model_ev_brief_formula = math.nan
    if pd_col:
        pd_hat = merged[pd_col].astype(float).clip(0, 1)
        amount = merged["requested_amount"].astype(float)
        ev_per_dollar = (1.0 - pd_hat) * PAID_MARGIN - pd_hat * BRIEF_LGD
        model_ev_lgd030 = float((ev_per_dollar[approved] * amount[approved]).sum())
        expected_t = np.r_[curves["validation_t_star"], curves["test_t_star"]]
        expected_recovery = np.r_[curves["validation_recovery"], curves["test_recovery"]]
        model_ev_brief_formula = float(
            brief_expected_npv(
                amount.to_numpy(float),
                pd_hat.to_numpy(float),
                expected_t,
                expected_recovery,
            )[approved.to_numpy()].sum()
        )

    flags = infer_code_flags(branch)
    row = {
        "person": branch["person"],
        "branch": branch["branch"],
        "status": "ok",
        "model": branch["model"],
        "lgd_assumption": branch["lgd_assumption"],
        "break_even_pd": branch["break_even_pd"],
        "submission_rows": int(len(sub)),
        "approved_total": int(approved.sum()),
        "approval_rate_total": float(approved.mean()),
        "approval_rate_validation": float(merged.loc[merged["_split"] == "validation", "decision"].mean()),
        "approval_rate_test": float(merged.loc[merged["_split"] == "test", "decision"].mean()),
        "prior_declined_total": int(prior_declined.sum()),
        "prior_declined_approved": int((approved & prior_declined).sum()),
        "prior_declined_approval_rate": float((approved & prior_declined).sum() / max(prior_declined.sum(), 1)),
        "prior_approved_approval_rate": float((approved & prior_approved).sum() / max(prior_approved.sum(), 1)),
        "no_bank_feed_approval_rate": float((approved & no_bank_feed).sum() / max(no_bank_feed.sum(), 1)),
        "funds_prior_declined_region": bool((approved & prior_declined).sum() > 0),
        "labeled_validation_approved": int(val_labeled_approved.sum()),
        "labeled_validation_approval_rate": float(val_labeled_approved.mean()),
        "labeled_validation_default_rate_approved": (
            float(val_labeled.loc[val_labeled_approved, "default_flag"].mean())
            if val_labeled_approved.any()
            else math.nan
        ),
        "verifiable_val_npv_brief_formula": float(brief_npv[val_labeled_approved.to_numpy()].sum()),
        "verifiable_val_profit_amortizing": float(amort_profit[val_labeled_approved.to_numpy()].sum()),
        "verifiable_val_profit_fixed_lgd030": float(fixed_profit[val_labeled_approved.to_numpy()].sum()),
        "headline_npv_brief_formula_all_approved": model_ev_brief_formula,
        "model_optimistic_ev_lgd030_all_approved": model_ev_lgd030,
        "validation_pd_auc_labeled": auc,
        "mean_pd_approved_all": mean_pd_approved,
        **flags,
    }
    return row


def fmt_money(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"${float(x) / 1_000_000:.2f}M"


def fmt_pct(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * float(x):.1f}%"


def fmt_float(x: object, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{float(x):.{digits}f}"


def bool_word(x: object) -> str:
    return "Yes" if bool(x) else "No"


def recovery_trap_label(r: pd.Series) -> str:
    if "empirical recovery trap" in str(r["lgd_assumption"]):
        return "Fell in"
    if bool(r.get("mentions_lgd_030_or_amortizing")) or "brief-faithful cash-flow" in str(r["lgd_assumption"]):
        return "Avoided"
    return "Unclear"


def write_report(rows: pd.DataFrame) -> None:
    report = ROOT / "markdown_files" / "branch_comparison_analysis.md"
    screenshot_report = ROOT / "markdown_files" / "screenshot_criteria_branch_comparison.md"
    csv_path = ROOT / "outputs" / "reports" / "branch_comparison_analysis.csv"
    screenshot_csv_path = ROOT / "outputs" / "reports" / "screenshot_criteria_branch_comparison.csv"
    json_path = ROOT / "outputs" / "reports" / "branch_comparison_analysis.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows.to_dict(orient="records"), indent=2))

    screenshot_rows = []
    for _, r in rows.iterrows():
        screenshot_rows.append(
            {
                "person": r["person"],
                "model": r["model"],
                "lgd_assumption": r["lgd_assumption"],
                "break_even_pd": r["break_even_pd"],
                "keeps_prior_underwriter_score": bool(r["keeps_prior_underwriter_score_in_A"]),
                "funds_prior_declined_region": bool(r["funds_prior_declined_region"]),
                "headline_npv_brief_formula": r["headline_npv_brief_formula_all_approved"],
                "verifiable_npv_labeled_val_brief_formula": r["verifiable_val_npv_brief_formula"],
                "headline_npv_fixed_lgd030_reference": r["model_optimistic_ev_lgd030_all_approved"],
                "verifiable_npv_amortizing_reference": r["verifiable_val_profit_amortizing"],
                "recovery_trap": recovery_trap_label(r),
            }
        )
    screenshot_df = pd.DataFrame(screenshot_rows)
    screenshot_df.to_csv(screenshot_csv_path, index=False)

    lines = [
        "# Branch Comparison: Deliverable A Policy Traps",
        "",
        "This compares the four local branches from `https://github.com/ron2k1/intuit-techweek-smb-underwriting` against the same local validation/test data. Abhi's branch did not commit an A submission, so it was generated in the isolated comparison worktree from his pipeline.",
        "",
        "Valuation columns are included with the slide formula first and the teammate-screenshot convention as a reference:",
        "",
        "- `Brief NPV` follows the hackathon slide formula exactly: repaid `F + R*r*T/365`; default `F + D*(t*-1) + rec - R`.",
        "- `Brief headline` is expected NPV over approved validation+test rows using each branch's submitted PDs and common timing/recovery curves.",
        "- `Brief verifiable` is realized NPV on labeled-validation approved rows.",
        "- `Fixed-LGD EV` and `amortizing profit` are retained only to reconcile the teammate screenshot.",
        "",
        "| Person | Model | LGD / Break-even | Approves | Prior-declined funded | Brief headline | Val approved | Brief verifiable | Fixed-LGD EV ref | Amortizing ref | AUC | Key trap read |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in rows.iterrows():
        trap_bits = []
        if "empirical recovery trap" in str(r["lgd_assumption"]):
            trap_bits.append("fell into recovery trap")
        elif r.get("mentions_lgd_091_or_empirical_recovery"):
            trap_bits.append("identified recovery trap")
        if r.get("mentions_lgd_030_or_amortizing"):
            trap_bits.append("amortization-aware")
        if r.get("funds_prior_declined_region"):
            trap_bits.append("funds reject region")
        else:
            trap_bits.append("avoids reject region")
        if bool(r.get("keeps_prior_underwriter_score_in_A")):
            trap_bits.append("keeps prior score")
        else:
            trap_bits.append("drops prior score")
        lgd = f"{r['lgd_assumption']} / {fmt_float(r['break_even_pd'], 3)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["person"]),
                    str(r["model"]),
                    lgd,
                    f"{int(r['approved_total']):,} ({fmt_pct(r['approval_rate_total'])})",
                    f"{int(r['prior_declined_approved']):,} ({fmt_pct(r['prior_declined_approval_rate'])})",
                    fmt_money(r["headline_npv_brief_formula_all_approved"]),
                    f"{int(r['labeled_validation_approved']):,}",
                    fmt_money(r["verifiable_val_npv_brief_formula"]),
                    fmt_money(r["model_optimistic_ev_lgd030_all_approved"]),
                    fmt_money(r["verifiable_val_profit_amortizing"]),
                    fmt_float(r["validation_pd_auc_labeled"], 3),
                    "; ".join(trap_bits),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## What This Shows",
            "",
            "The biggest spread is not model AUC. It is policy/economics: whether the branch catches amortization/LGD, and whether it chooses to fund the prior-declined region where there are no observed outcomes.",
            "",
            "- Abhi has a reasonable CatBoost AUC, but uses empirical post-default recovery as LGD. That sets break-even PD near 8.8% and under-funds.",
            "- Ayush catches the amortization/LGD trap but uses a conservative policy that avoids the prior-declined region, so it gives up speculative upside but avoids unlabelled-region downside.",
            "- Ronil catches amortization and drops prior-underwriter outputs, but funds a large prior-declined region. That can create a much larger optimistic headline EV, while the verifiable labeled-validation number stays close to Ayush because the declined region has no labels.",
            "- Steven's current branch is the active project policy: brief-faithful NPV plus a direct-NPV blend. It approves more of the labeled validation book and is close to the best brief-formula validation NPV in this comparison, but it still needs a defensible story for reject-region uncertainty.",
            "",
            "## Scoring Implication",
            "",
            "For the hackathon, the safest defense is to separate verifiable performance from speculative reject-region extrapolation. Report the labeled-validation NPV as auditable, then present any prior-declined funding as a sensitivity analysis with pessimistic floors, not as guaranteed value.",
            "",
            f"CSV output: `{csv_path.relative_to(ROOT)}`",
            f"JSON output: `{json_path.relative_to(ROOT)}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n")

    shot = [
        "# Screenshot-Criteria Branch Comparison",
        "",
        "This mirrors the teammate screenshot categories, but uses the hackathon slide formula as the primary NPV convention. The screenshot's fixed-LGD/amortizing convention is listed separately because it explains why Ayush's screenshot shows `$5.28M` / `$2.75M`.",
        "",
        "| Criterion | Ronil | Ayush | Abhi | Steven |",
        "|---|---:|---:|---:|---:|",
    ]
    by_person = {r["person"]: r for _, r in rows.iterrows()}
    order = ["Ronil", "Ayush", "Abhi", "Steven"]

    def cells(fn):
        return " | ".join(fn(by_person[p]) for p in order)

    shot.extend(
        [
            f"| Model | {cells(lambda r: str(r['model']))} |",
            f"| LGD assumption | {cells(lambda r: str(r['lgd_assumption']))} |",
            f"| Break-even PD | {cells(lambda r: fmt_float(r['break_even_pd'], 3))} |",
            f"| Keeps `prior_underwriter_score`? | {cells(lambda r: bool_word(r['keeps_prior_underwriter_score_in_A']))} |",
            f"| Funds prior-declined region? | {cells(lambda r: bool_word(r['funds_prior_declined_region']))} |",
            f"| Headline NPV (slide formula) | {cells(lambda r: fmt_money(r['headline_npv_brief_formula_all_approved']))} |",
            f"| Verifiable NPV (slide formula) | {cells(lambda r: fmt_money(r['verifiable_val_npv_brief_formula']))} |",
            f"| Headline NPV (fixed-LGD reference) | {cells(lambda r: fmt_money(r['model_optimistic_ev_lgd030_all_approved']))} |",
            f"| Verifiable NPV (amortizing reference) | {cells(lambda r: fmt_money(r['verifiable_val_profit_amortizing']))} |",
            f"| Recovery trap | {cells(recovery_trap_label)} |",
            "",
            "## Notes",
            "",
            "- Abhi's branch had no committed A submission, so this table uses his own pipeline run in the isolated comparison worktree.",
            "- Ayush's screenshot numbers match the fixed-LGD/amortizing reference: `$5.28M` headline and `$2.75M` verifiable.",
            "- Ronil's current cloned branch does not match the screenshot's `~$15M` headline. The current committed branch produces about `$10.30M` under the slide formula and `$6.85M` under the fixed-LGD reference, while funding 943 prior-declined applicants.",
            "- The verifiable NPV lines intentionally ignore unlabelled prior-declined upside/downside because those outcomes are not observable in validation.",
            "",
            f"CSV output: `{screenshot_csv_path.relative_to(ROOT)}`",
        ]
    )
    screenshot_report.write_text("\n".join(shot) + "\n")


def main() -> None:
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    curves = np.load(ROOT / "outputs" / "deliverable_a_curves.npz")
    rows = pd.DataFrame([summarize_branch(b, validation, test, curves) for b in BRANCHES])
    write_report(rows)
    print(rows[
        [
            "person",
            "approved_total",
            "approval_rate_total",
            "prior_declined_approved",
            "labeled_validation_approved",
            "headline_npv_brief_formula_all_approved",
            "verifiable_val_npv_brief_formula",
            "model_optimistic_ev_lgd030_all_approved",
            "verifiable_val_profit_amortizing",
            "validation_pd_auc_labeled",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
