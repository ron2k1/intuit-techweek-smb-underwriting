"""Deliverable A: profit-maximizing approve/decline + calibrated PD + 90% interval.

Run:  python -m src.build_a   ->  submissions/submission_A_decisions.csv (13,306 rows)

Five steps, each tracing back to reports/audit_findings.md:
  1. Train PD on the LABELLED subset (approved & matured) over the shared features.
  2. Calibrate (isotonic, internal CV) so predicted_pd is a real probability.
  3. Quantify uncertainty: bootstrap-ensemble spread, then EXPLICITLY widen the
     90% interval on the never-labelled reject-inference region (declined / no-feed).
  4. Decide on PROFIT: pick the PD threshold that maximizes *realized* portfolio
     profit on the labelled validation book under the real loan economics -- not
     accuracy, not 0.5.
  5. Emit one row per val+test applicant, ordered to match expected_ids/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from . import data as D

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = REPO_ROOT / "submissions"
EXPECTED_IDS = REPO_ROOT / "expected_ids" / "applicant_ids.txt"

# --- loan economics (README / audit §8) -------------------------------------
GOOD_MARGIN = 0.35 * 60 / 365 + 0.03   # ~0.0875 net on a fully repaid loan
ORIG_FEE = 0.03                         # collected upfront, kept even on default
Z90 = 1.6448536269514722               # 90% two-sided normal half-width factor

# --- uncertainty / OOD knobs (the ML PM tunes these for S_cal) --------------
N_ENSEMBLE = 12
MIN_HALFWIDTH = 0.03                    # floor so intervals aren't absurdly tight
OOD_BOOST = 1.5                         # extra interval width in the unlabelled region
OOD_SCORE_CUT = 0.273                   # prior_underwriter_score below = never-labelled
RANDOM_SEED = 20260605


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def realized_profit(amount: np.ndarray, default_flag: np.ndarray,
                    recovered: np.ndarray) -> np.ndarray:
    """Per-loan realized profit under the challenge economics.

    Fully repaid -> amount * GOOD_MARGIN.
    Defaulted    -> keep the 3% origination fee + ACTUAL recovery, lose the rest
                    of principal. Uses real final_recovered_amount -- no LGD guess.
    Rows where default_flag is NaN (unlabelled) return NaN and must be masked out.
    """
    good = amount * GOOD_MARGIN
    bad = amount * ORIG_FEE + np.nan_to_num(recovered, nan=0.0) - amount
    out = np.where(default_flag == 1.0, bad, good)
    out[np.isnan(default_flag)] = np.nan
    return out


def ood_flag(df: pd.DataFrame) -> np.ndarray:
    """1 where the model extrapolates beyond labelled support (audit §1, §4)."""
    below_cut = df["prior_underwriter_score"].to_numpy() < OOD_SCORE_CUT
    no_feed = df["no_bank_feed"].to_numpy().astype(bool)
    return (below_cut | no_feed).astype(float)


def main() -> None:
    SUB_DIR.mkdir(exist_ok=True)
    tr, feats = D.load_features("train")
    va, _ = D.load_features("val")
    te, _ = D.load_features("test")

    ytr = D.target_vector(tr)
    lab = ytr.notna().to_numpy()                       # labelled = approved & matured
    Xtr = D.to_model_matrix(tr, feats).to_numpy()
    Xva = D.to_model_matrix(va, feats).to_numpy()
    Xte = D.to_model_matrix(te, feats).to_numpy()
    Xtr_lab, ytr_lab = Xtr[lab], ytr[lab].astype(int).to_numpy()
    print(f"[1] train labelled={len(ytr_lab)}  features={len(feats)}  "
          f"base default_rate={ytr_lab.mean():.4f}")

    # --- 2. calibrated PD model (isotonic, internal 5-fold) ---
    cal = CalibratedClassifierCV(_hgb(RANDOM_SEED), method="isotonic", cv=5)
    cal.fit(Xtr_lab, ytr_lab)
    pd_va = cal.predict_proba(Xva)[:, 1]
    pd_te = cal.predict_proba(Xte)[:, 1]
    print(f"[2] calibrated PD  mean(val)={pd_va.mean():.4f}  mean(test)={pd_te.mean():.4f}")

    # --- 3. bootstrap-ensemble spread for the 90% interval ---
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(Xtr_lab)
    ens_va = np.empty((N_ENSEMBLE, len(Xva)))
    ens_te = np.empty((N_ENSEMBLE, len(Xte)))
    for k in range(N_ENSEMBLE):
        idx = rng.integers(0, n, n)                    # bootstrap resample
        m = _hgb(RANDOM_SEED + 1 + k).fit(Xtr_lab[idx], ytr_lab[idx])
        ens_va[k] = m.predict_proba(Xva)[:, 1]
        ens_te[k] = m.predict_proba(Xte)[:, 1]
    std_va, std_te = ens_va.std(0), ens_te.std(0)

    # --- 4. profit-maximizing threshold on the labelled validation book ---
    yva = D.target_vector(va)
    lab_va = yva.notna().to_numpy()
    pr = realized_profit(va["requested_amount"].to_numpy(),
                         yva.to_numpy(),
                         va["final_recovered_amount"].to_numpy())
    approve_all = np.nansum(pr[lab_va])
    cands = np.unique(np.round(pd_va[lab_va], 4))
    best_tau, best_profit = cands[-1] + 1e-6, -np.inf
    for tau in cands:
        sel = lab_va & (pd_va < tau)
        tot = np.nansum(pr[sel])
        if tot > best_profit:
            best_profit, best_tau = tot, tau
    lift = best_profit - approve_all
    print(f"[4] tau*={best_tau:.4f}  realized val profit=${best_profit:,.0f}  "
          f"(approve-all=${approve_all:,.0f}, lift=${lift:,.0f})")

    # --- 5. assemble all 13,306 rows ---
    out = pd.DataFrame({
        "applicant_id": pd.concat([va["applicant_id"], te["applicant_id"]],
                                  ignore_index=True),
        "predicted_pd": np.clip(np.concatenate([pd_va, pd_te]), 0.0, 1.0),
    })
    std = np.concatenate([std_va, std_te])
    ood = np.concatenate([ood_flag(va), ood_flag(te)])
    halfwidth = np.maximum(Z90 * std, MIN_HALFWIDTH) * (1.0 + OOD_BOOST * ood)
    out["pd_lower_90"] = np.clip(out["predicted_pd"] - halfwidth, 0.0, 1.0)
    out["pd_upper_90"] = np.clip(out["predicted_pd"] + halfwidth, 0.0, 1.0)
    out["decision"] = (out["predicted_pd"] < best_tau).astype(int)

    # order exactly to expected_ids (validator checks the ID set; be exact)
    order = pd.read_csv(EXPECTED_IDS, header=None)[0].tolist()
    out = out.set_index("applicant_id").reindex(order).reset_index()
    out = out.rename(columns={"index": "applicant_id"})
    assert out["applicant_id"].notna().all() and not out.isna().any().any(), \
        "row/ID mismatch vs expected_ids"
    out = out[["applicant_id", "decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]]

    dest = SUB_DIR / "submission_A_decisions.csv"
    out.to_csv(dest, index=False)
    appr = out["decision"].mean()
    print(f"[5] wrote {dest.name}: {len(out)} rows  approve_rate={appr:.3f}  "
          f"PD[min/med/max]={out.predicted_pd.min():.3f}/"
          f"{out.predicted_pd.median():.3f}/{out.predicted_pd.max():.3f}")


if __name__ == "__main__":
    main()
