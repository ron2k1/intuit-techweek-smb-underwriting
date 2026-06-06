"""Deliverable C: causal do(feature=value) counterfactual PD via the g-formula.

Run AFTER build_a:  python -m src.build_c -> submissions/submission_C_counterfactuals.csv

TIED TO A. C reuses A's exact modeling spine (src.model) AND A's exact feature set
(src.data.feature_columns minus the prior-lender colliders). Every feature named in
dataset/intervention_queries.csv is therefore retained, so all 900 do() queries get
a genuine, data-driven counterfactual -- nothing is forced to 0 by feature exclusion.

  Dropped (same two as A): prior_underwriter_score, prior_decision
    -- prior_decision is constant==1 on the labelled book (zero variance) and
       prior_underwriter_score is a selection collider; neither is queried.

A do() query is answered by the g-formula for DETERMINISTIC descendants:
  1. take the applicant's raw row,
  2. set the intervened raw feature to its value,
  3. re-run add_engineered_features so downstream features recompute
     (e.g. do(observed_monthly_revenue) flows into lev_verified;
      do(stated_annual_revenue) flows into lev_stated + report_gap),
  4. predict with the model; everything else held fixed.

CAUSAL NOTE (defended in writeup D s3). The strict backdoor answer treats self-report
(stated_*) as a non-cause -> do(stated_*) ~ 0 (README trap #4). We instead RETAIN
stated_* (per direction: keep every queried column in) and let the model estimate the
effect from data. This is the "model-perturbation" counterfactual the template permits;
what we give up is the structural guarantee of an exact 0. We REPORT the measured
stated_* effect below so the magnitude is visible, not hidden.

90% bands: bootstrap-ensemble spread on the model, conformal-calibrated, widened in the
never-labelled (OOD) region -- the same recipe as A.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import model as M

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB = REPO_ROOT / "submissions"
DATA = REPO_ROOT / "dataset"
N_ENSEMBLE = 12
OOD_SCORE_CUT = 0.273
SEED = M.DEFAULT_SEED

# C uses A's feature set: drop only the prior-lender colliders (same as build_a).
# This keeps every queried column (incl. stated_*) and ties A <-> C to one spine.
DROP_FOR_C = ["prior_underwriter_score", "prior_decision"]


def feats_for_c(feats: list[str]) -> list[str]:
    return [c for c in feats if c not in DROP_FOR_C]


def to_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return D.to_model_matrix(df, feats).to_numpy()


def ood_flag(df: pd.DataFrame) -> np.ndarray:
    below = pd.to_numeric(df["prior_underwriter_score"], errors="coerce").to_numpy() < OOD_SCORE_CUT
    no_feed = df["no_bank_feed"].to_numpy().astype(bool)
    return (np.nan_to_num(below, nan=1.0).astype(bool) | no_feed).astype(float)


def main() -> None:
    SUB.mkdir(exist_ok=True)

    # --- train the shared spine on the labelled book, A's feature set ---
    tr = D.add_engineered_features(D.load_raw("train"))
    feats = feats_for_c(D.feature_columns(tr))
    ytr = D.target_vector(tr)
    lab = ytr.notna().to_numpy()
    cat_idx = D.categorical_indices(feats)
    Xtr = to_matrix(tr, feats)[lab]
    ytr_l = ytr[lab].astype(int).to_numpy()
    print(f"[C] features={len(feats)} (dropped {DROP_FOR_C})  labelled train={len(ytr_l)}")

    model = M.fit_calibrated(Xtr, ytr_l, cat_idx, seed=SEED, cv=5, scoring="roc_auc")

    # --- conformal lambda on the labelled validation book ---
    va = D.add_engineered_features(D.load_raw("val"))
    yva = D.target_vector(va)
    lva = yva.notna().to_numpy()
    Xva = to_matrix(va, feats)
    pd_va = model.predict_proba(Xva)[:, 1]
    std_va = M.bootstrap_pd(Xtr, ytr_l, [Xva[lva]], cat_idx, seed=SEED,
                            n_models=N_ENSEMBLE, scoring="roc_auc")[0].std(0)
    lam = M.conformal_lambda(pd_va[lva], std_va, yva[lva].astype(int).to_numpy())
    print(f"[C] conformal lambda={lam:.3f}")

    # --- raw rows for every applicant referenced by a query ---
    raw_all = pd.concat([D.load_raw("val"), D.load_raw("test")], ignore_index=True)
    raw_all["applicant_id"] = raw_all["applicant_id"].astype(str)
    raw_all = raw_all.drop_duplicates("applicant_id").set_index("applicant_id")

    q = pd.read_csv(DATA / "intervention_queries.csv")
    q["applicant_id"] = q["applicant_id"].astype(str)
    queried = sorted(q["feature_name"].unique())
    missing_from_model = [f for f in queried if f not in feats and f not in tr.columns]
    print(f"[C] queries={len(q)}  queried features={len(queried)}  "
          f"retained-in-model/raw -> not-representable={missing_from_model or 'none'}")

    # baseline rows (one per query, in query order) -- all 900 preserved
    base_raw = raw_all.reindex(q["applicant_id"]).reset_index()
    inter_raw = base_raw.copy()
    # apply each intervention. raw values are numeric codes for every feature here
    # (verified: 0% to-numeric coercion loss), incl. int-coded categoricals.
    for feat, grp in q.groupby("feature_name"):
        idx = grp.index.to_numpy()
        vals = pd.to_numeric(grp["intervention_value"], errors="coerce").to_numpy()
        if feat in inter_raw.columns:
            inter_raw.loc[idx, feat] = vals

    base_eng = D.add_engineered_features(base_raw)
    inter_eng = D.add_engineered_features(inter_raw)
    Xb = to_matrix(base_eng, feats)
    Xi = to_matrix(inter_eng, feats)

    pd_base = model.predict_proba(Xb)[:, 1]
    pd_cf = model.predict_proba(Xi)[:, 1]
    delta = pd_cf - pd_base

    # intervals on the counterfactual: bootstrap spread + OOD widening
    std_cf = M.bootstrap_pd(Xtr, ytr_l, [Xi], cat_idx, seed=SEED,
                            n_models=N_ENSEMBLE, scoring="roc_auc")[0].std(0)
    ood = ood_flag(inter_eng)
    lo, hi = M.make_interval(pd_cf, std_cf, lam, ood)

    out = pd.DataFrame({
        "query_id": q["query_id"],
        "predicted_pd_cf": np.clip(pd_cf, 0, 1),
        "pd_cf_lower_90": lo,
        "pd_cf_upper_90": hi,
    })
    assert len(out) == len(q), "lost query rows!"
    out.to_csv(SUB / "submission_C_counterfactuals.csv", index=False)

    # --- diagnostics: per-group interventional effect (esp. self-report) ---
    fn = q["feature_name"].to_numpy()
    self_report = np.isin(fn, ["stated_annual_revenue", "stated_time_in_business"])
    print(f"[C] wrote {len(out)} rows.  interventional |delta| by group:")
    print(f"     self-report (stated_*)  n={int(self_report.sum())}  "
          f"mean={np.abs(delta[self_report]).mean():.4f}  max={np.abs(delta[self_report]).max():.4f}  "
          f"(retained per directive; data-driven, not forced 0)")
    print(f"     all-other queries       n={int((~self_report).sum())}  "
          f"mean={np.abs(delta[~self_report]).mean():.4f}  max={np.abs(delta[~self_report]).max():.4f}")
    top = (pd.Series(np.abs(delta), index=fn).groupby(level=0).mean()
           .sort_values(ascending=False).head(6))
    print("     strongest causes (mean|delta|): "
          + ", ".join(f"{k}={v:.3f}" for k, v in top.items()))


if __name__ == "__main__":
    main()
