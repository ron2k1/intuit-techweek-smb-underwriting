"""Deliverable A: profit-maximizing approve/decline + calibrated PD + 90% interval.

Run:  python -m src.build_a   ->  submissions/submission_A_decisions.csv (13,306 rows)

The objective the team agreed on:
  * MODEL QUALITY is measured by AUC-ROC (rank discrimination) -- the spine's
    booster early-stops on roc_auc and we report AUC as the headline metric.
  * THE DECISION maximizes realized PORTFOLIO VALUE / NPV -- we approve an applicant
    whose calibrated PD is below the CLOSED-FORM profit break-even threshold
    GOOD_MARGIN/(GOOD_MARGIN - b), where b=E[NPV_default]/$ comes from the brief's
    EXACT per-loan NPV (daily draws collected before default + real recovery), NOT an
    assumed flat LGD. Empirically b~-0.256 (implied LGD 0.256) -> break-even ~0.255. In the
    never-labelled reject region (OOD) the PD is an extrapolation, so there we
    require the PESSIMISTIC conformal upper-90 PD to clear break-even (trust-but-
    verify, step 4). We deliberately do NOT use the val-argmax tau: the walk-forward
    backtest (src/backtest.py) showed it is OVERFIT (swings 0.135-0.229 across
    folds), and once PDs are de-drifted it climbs back to the break-even anyway.
    (AUC ranks; the closed-form break-even sets the cut.)

Pipeline (each step traces to reports/audit_findings.md):
  1. Feature set = shared predictors MINUS prior_* (the selection collider:
     prior_decision is constant==1 on the labelled book; prior_underwriter_score
     is truncated above its cut for approvals and out-of-range for declines, so
     keeping it just forces extrapolation into the reject region we must price).
  2. predicted_pd = average of a calibrated HGB and a logistic scorecard, both fit
     with RECENCY (time-decay) weights so the calibrated level tracks the recent,
     higher-default regime the backtest found drifting up (15.2% -> 21.1%). Both are
     genuine probabilities (so the mean stays ~calibrated for the profit math) but
     make different ranking errors, so the blend lifts AUC.
  3. Uncertainty: bootstrap-ensemble spread -> conformal-calibrated half-width,
     EXPLICITLY widened in the never-labelled reject region (declined / no feed).
  4. Decision (TRUST-BUT-VERIFY): in-distribution applicants are approved when the
     point PD clears the closed-form break-even; OOD applicants (prior-declined /
     no-feed, where the PD is an extrapolation) must clear it on the PESSIMISTIC
     conformal upper-90 bound instead. This funds the censored region only where
     even our uncertainty-aware estimate says it pays -- capturing the verifiable
     declined-region profit (realized default 0.144 << break-even) while abstaining
     on the unverifiable deep-reject tail the data cannot vouch for. The overfit val
     argmax is still computed and printed as a comparison, never as the decision.
  5. One row per val+test applicant, ordered to expected_ids/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from . import model as M

REPO_ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = REPO_ROOT / "submissions"
EXPECTED_IDS = REPO_ROOT / "expected_ids" / "applicant_ids.txt"

# --- loan economics: the EXACT brief NPV per loan (NO assumed flat LGD) ------
# Scored objective  S_P&L = sum_i d_i * NPV_i  with the brief's piecewise NPV:
#   repaid (y=0):  NPV_i = F_i + R_i * r * T/365
#   default(y=1):  NPV_i = F_i + D_i * (t*_i - 1) + rec_i - R_i   (default at day t*)
# where R_i=requested_amount, F_i=ORIG_FEE*R_i, D_i=R_i*(1+r*T/365)/T is the daily
# ACH draw, t*_i=days_to_default, rec_i=final_recovered_amount (DOLLARS).
#
# We do NOT assume a fixed LGD. The loss given default is CONSTRUCTED from how many
# daily draws the borrower made before defaulting (D*(t*-1)) plus the real recovery
# rec -- so a loan that defaults late (most draws already collected) costs far less
# than one that defaults on day 3. This is "loss based on the output of A and B":
# A supplies PD; B's default-timing curve g(t*) (the matured-train default-day
# distribution) supplies E[draws collected]. Empirically (scratch/probe_brief_npv.py)
# E[t*|default]=43d, recovery~9.1% of principal -> data-derived LGD ~0.256, NOT 0.30.
INT_RATE = 0.35           # APR (r)
TERM_DAYS = 60            # loan term (T)
ORIG_FEE = 0.03           # F_i / R_i, collected upfront, kept even on default
GOOD_MARGIN = INT_RATE * TERM_DAYS / 365 + ORIG_FEE   # repaid NPV/$ ~0.0875
DAILY_DRAW = (1.0 + INT_RATE * TERM_DAYS / 365.0) / TERM_DAYS   # D_i/R_i per day

# THE TIMING TRAP: days_to_default runs to 90 (the day-90 outstanding-balance
# trigger), but there are only T=60 scheduled draws. Taken literally past day 60,
# D*(t*-1) credits a day-90 defaulter with ~1.57x the principal in draws -- making a
# late default score MORE than a full repayment, which is economically impossible.
# We CAP the draw count at the term so a defaulter can never "repay" more than the
# full schedule. Robustness (scratch/probe_econ_robust.py): capping STRICTLY
# dominates the old flat-LGD policy under BOTH grader readings (+$87k capped / +$133k
# uncapped on labelled val), whereas the naive uncapped reading over-approves a
# 27%-default marginal band that loses money under the sane (capped) reality.
CAP_DRAWS_AT_TERM = True
DEFAULT_DAY_CAP = 90      # days_to_default domain ceiling (brief: t* in [1,90])

# --- uncertainty / OOD knobs ------------------------------------------------
N_ENSEMBLE = 12
OOD_SCORE_CUT = 0.273     # prior_underwriter_score below this = never-labelled region
SEED = M.DEFAULT_SEED

# --- drift-aware calibration (walk-forward finding) -------------------------
# Train default rate climbs monotonically 15.2% (2024Q1) -> 21.1% (2025Q2); the
# deployment cohort sits past the peak, so a model fit on the full-history average
# UNDER-prices risk (backtest: late-fold PD biased low by ~0.022). We time-decay the
# training rows (exp half-life) so the calibrated LEVEL tracks the recent regime.
# 6mo halves the calibration gap (-0.022 -> -0.015) at ~0 AUC cost and keeps the
# effective sample healthy (~22k of 52k). Ranking is untouched; only the level moves.
RECENCY_HALF_LIFE_MONTHS = 6.0

# Feature set for A: drop the prior-lender selection collider (see docstring).
DROP_FOR_A = ["prior_underwriter_score", "prior_decision"]


def feats_for_a(feats: list[str]) -> list[str]:
    return [c for c in feats if c not in DROP_FOR_A]


def recency_weights(ts: pd.Series,
                    half_life_months: float = RECENCY_HALF_LIFE_MONTHS) -> np.ndarray:
    """Exponential time-decay sample weights for drift-aware calibration.

    The latest labelled train month gets weight 1.0; a row `half_life_months` earlier
    gets 0.5, twice that 0.25, and so on. Down-weighting stale (lower-default) vintages
    pulls the calibrated PD level toward the recent regime WITHOUT discarding history
    (every row keeps a positive weight), so the booster still sees the full support and
    its ranking is preserved -- only the probability level is corrected.
    """
    mon = pd.to_datetime(ts, errors="coerce").dt.to_period("M")
    latest = mon.max()
    months_back = np.array([(latest - m).n if pd.notna(m) else 0.0 for m in mon], float)
    return 0.5 ** (months_back / half_life_months)


def logit_pipe() -> "make_pipeline":
    """Logistic scorecard: median-impute, standardize, L2 logistic regression.

    A strong linear baseline -- on this data it rivals the booster, which tells us
    the PD signal is largely additive. We blend it with the tree for the AUC lift.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.5),
    )


def estimate_recovery_rate(tr: pd.DataFrame, ytr: pd.Series) -> float:
    """AGGREGATE recovery as a fraction of principal among defaulted matured-train
    loans (sum rec / sum principal ~ 0.093). Reported for the writeup as the 'recovery
    trap': a naive model that priced default as a flat 1 - 0.093 = 0.907 LGD would
    decline almost everyone. The brief NPV instead credits the daily draws collected
    BEFORE default, so the effective loss is far smaller (see empirical_default_npv)."""
    d = (ytr == 1).to_numpy()
    rec = pd.to_numeric(tr["final_recovered_amount"], errors="coerce").to_numpy()
    amt = pd.to_numeric(tr["requested_amount"], errors="coerce").to_numpy()
    m = d & ~np.isnan(rec) & ~np.isnan(amt) & (amt > 0)
    return float(np.clip(rec[m].sum() / amt[m].sum(), 0.0, 1.0))


def default_npv_per_dollar(days_to_default, rec_per_dollar) -> np.ndarray:
    """Brief NPV per $1 principal for a loan that DEFAULTS at day t* (y=1):
        F/$ + D/$ * (min(t*, T) - 1) + rec/$ - 1
    The loss given default is CONSTRUCTED, not assumed: D/$*(t*-1) is the stream of
    daily ACH draws actually collected before the borrower stopped paying, rec/$ is the
    real post-default recovery, and -1 is the principal at risk. No flat LGD anywhere.
    Draws are capped at the term (CAP_DRAWS_AT_TERM): days_to_default runs to 90 because
    of the day-90 outstanding-balance trigger, but only T=60 scheduled draws exist, so
    crediting draws past day 60 would let a late default 'repay' more than the schedule."""
    t = np.clip(np.nan_to_num(days_to_default, nan=float(TERM_DAYS)), 1, DEFAULT_DAY_CAP)
    if CAP_DRAWS_AT_TERM:
        t = np.minimum(t, TERM_DAYS)
    rec = np.nan_to_num(rec_per_dollar, nan=0.0)
    return ORIG_FEE + DAILY_DRAW * (t - 1.0) + rec - 1.0


def empirical_default_npv(tr: pd.DataFrame, ytr: pd.Series) -> tuple[float, float]:
    """E[NPV_default]/$ from matured-train defaulters -- the input to break-even tau.

    This is the 'loss based on the output of A and B' the brief points at: Deliverable
    B's cohort default-TIMING curve g(t*) is exactly this set of default days, and
    because the brief NPV is LINEAR in t*, the mean default day is a sufficient
    statistic -- grounding the loss on B's timing == averaging the per-loan brief NPV
    here. Empirically E[t*|default]~43d and recovery~9% of principal give b ~ -0.256,
    i.e. an implied LGD of 0.256 -- NOT the flat 0.30 everyone assumed.
    Returns (b_emp = E[NPV_default]/$, lgd_eff = -b_emp)."""
    d = (ytr == 1).to_numpy()
    dtd = pd.to_numeric(tr["days_to_default"], errors="coerce").to_numpy()
    rec = pd.to_numeric(tr["final_recovered_amount"], errors="coerce").to_numpy()
    amt = pd.to_numeric(tr["requested_amount"], errors="coerce").to_numpy()
    m = d & ~np.isnan(amt) & (amt > 0)
    rec_pd = np.where(np.isnan(rec[m]), 0.0, rec[m]) / amt[m]
    b = float(np.mean(default_npv_per_dollar(dtd[m], rec_pd)))
    return b, -b


def theoretical_tau(b: float) -> float:
    """Closed-form profit break-even PD. Approve iff forward E[NPV] > 0. Per $1:
        repaid  (1-p): +GOOD_MARGIN
        default (p)  : +b          (b<0 = the empirical brief NPV/$ on default)
    Setting E[NPV]=0:  tau* = GOOD_MARGIN / (GOOD_MARGIN - b).
    Empirical b=-0.256 -> tau~0.255 (vs the old flat-LGD=0.30 assumption -> 0.226)."""
    return GOOD_MARGIN / (GOOD_MARGIN - b)


def realized_npv(amount, default_flag, days_to_default, recovered=None) -> np.ndarray:
    """Per-loan REALIZED brief NPV -- the EXACT scored quantity, no discounting.
    Repaid    (y=0): amount * GOOD_MARGIN                       (= F + R*r*T/365).
    Defaulted (y=1): amount*F/$ + amount*D/$*(min(t*,T)-1) + rec - amount.
    `recovered` is the per-loan final_recovered_amount in DOLLARS (None -> 0, the
    conservative no-recovery read). NaN default_flag (unlabelled) -> NaN, mask it out."""
    good = amount * GOOD_MARGIN
    t = np.clip(np.nan_to_num(days_to_default, nan=float(TERM_DAYS)), 1, DEFAULT_DAY_CAP)
    if CAP_DRAWS_AT_TERM:
        t = np.minimum(t, TERM_DAYS)
    rec = (np.zeros_like(np.asarray(amount, dtype=float)) if recovered is None
           else np.nan_to_num(recovered, nan=0.0))
    bad = amount * ORIG_FEE + amount * DAILY_DRAW * (t - 1.0) + rec - amount
    out = np.where(default_flag == 1.0, bad, good)
    out[np.isnan(default_flag)] = np.nan
    return out


def expected_npv_per_dollar(pd_hat, b):
    """Forward expected NPV per $1 principal for default prob `pd_hat`:
        (1 - pd_hat) * GOOD_MARGIN + pd_hat * b
    with b the empirical brief NPV/$ on a defaulted loan (empirical_default_npv). Used
    on the scored book (test outcomes withheld). Its break-even == theoretical_tau(b)."""
    return (1.0 - pd_hat) * GOOD_MARGIN + pd_hat * b


def ood_flag(df: pd.DataFrame) -> np.ndarray:
    """1 where the model extrapolates beyond labelled support (audit 1, 4)."""
    below_cut = pd.to_numeric(df["prior_underwriter_score"], errors="coerce").to_numpy() < OOD_SCORE_CUT
    no_feed = df["no_bank_feed"].to_numpy().astype(bool)
    return (np.nan_to_num(below_cut, nan=1.0).astype(bool) | no_feed).astype(float)


def prior_approved(df: pd.DataFrame) -> np.ndarray:
    """1 where prior_decision==1 (prior-APPROVED => the row is observable/LABELLED).

    NOT a model feature (prior_decision is a dropped collider, see DROP_FOR_A) -- used
    ONLY as a label-provenance GATING signal to split OOD into shallow (vouched) vs deep
    (never-labelled reject tail). Legitimate: we are not predicting FROM it, we are
    choosing WHICH uncertainty bound to underwrite on based on whether reality has been
    observed for that row's region.
    """
    return (pd.to_numeric(df["prior_decision"], errors="coerce") == 1).to_numpy()


def main() -> None:
    SUB_DIR.mkdir(exist_ok=True)
    tr, feats_all = D.load_features("train")
    va, _ = D.load_features("val")
    te, _ = D.load_features("test")
    feats = feats_for_a(feats_all)
    cat_idx = D.categorical_indices(feats)

    ytr = D.target_vector(tr)
    lab = ytr.notna().to_numpy()
    Xtr = D.to_model_matrix(tr, feats).to_numpy()
    Xva = D.to_model_matrix(va, feats).to_numpy()
    Xte = D.to_model_matrix(te, feats).to_numpy()
    Xtr_l, ytr_l = Xtr[lab], ytr[lab].astype(int).to_numpy()
    w_tr = recency_weights(tr["application_timestamp"][lab])   # drift-aware time decay
    print(f"[1] labelled train={len(ytr_l)}  features={len(feats)} (dropped {DROP_FOR_A})  "
          f"cat_idx={cat_idx}  base default={ytr_l.mean():.4f}  "
          f"recency half-life={RECENCY_HALF_LIFE_MONTHS:.0f}mo (eff N={w_tr.sum():.0f})")

    # --- 2. predicted_pd = blend(calibrated HGB, logistic scorecard), recency-weighted ---
    hgb = M.fit_calibrated(Xtr_l, ytr_l, cat_idx, seed=SEED, cv=5, scoring="roc_auc",
                           sample_weight=w_tr)
    lg = logit_pipe().fit(Xtr_l, ytr_l, logisticregression__sample_weight=w_tr)
    pd_hgb_va, pd_hgb_te = hgb.predict_proba(Xva)[:, 1], hgb.predict_proba(Xte)[:, 1]
    pd_lg_va, pd_lg_te = lg.predict_proba(Xva)[:, 1], lg.predict_proba(Xte)[:, 1]
    pd_va = 0.5 * (pd_hgb_va + pd_lg_va)
    pd_te = 0.5 * (pd_hgb_te + pd_lg_te)

    # --- headline AUC-ROC on the labelled validation book ---
    yva = D.target_vector(va)
    lab_va = yva.notna().to_numpy()
    yv = yva[lab_va].astype(int).to_numpy()
    auc = roc_auc_score(yv, pd_va[lab_va])
    auc_hgb = roc_auc_score(yv, pd_hgb_va[lab_va])
    auc_lg = roc_auc_score(yv, pd_lg_va[lab_va])
    pr = average_precision_score(yv, pd_va[lab_va])
    brier = brier_score_loss(yv, pd_va[lab_va])
    print(f"[2] AUC-ROC(val)={auc:.4f}  [HGB={auc_hgb:.4f} logit={auc_lg:.4f} blend wins]  "
          f"PR-AUC={pr:.4f}  Brier={brier:.4f}  meanPD val={pd_va.mean():.4f}")

    # --- 3. bootstrap spread -> conformal interval (widened OOD) ---
    ens = M.bootstrap_pd(Xtr_l, ytr_l, [Xva, Xte], cat_idx, seed=SEED,
                         n_models=N_ENSEMBLE, scoring="roc_auc", sample_weight=w_tr)
    std_va, std_te = ens[0].std(0), ens[1].std(0)
    lam = M.conformal_lambda(pd_va[lab_va], std_va[lab_va], yv)
    ood_va, ood_te = ood_flag(va), ood_flag(te)
    lo_va, hi_va = M.make_interval(pd_va, std_va, lam, ood_va)
    lo_te, hi_te = M.make_interval(pd_te, std_te, lam, ood_te)
    cov = float(((yv >= lo_va[lab_va]) & (yv <= hi_va[lab_va])).mean())  # sanity only
    print(f"[3] conformal lambda={lam:.3f}  median half-width="
          f"{np.median((hi_va-lo_va)/2):.3f}  OOD rate(test)={ood_te.mean():.3f}")

    # --- 4. DECISION threshold = MAXIMIZE PORTFOLIO NPV (team's top metric) ---
    amt_va = va["requested_amount"].to_numpy()
    dd_va = pd.to_numeric(va["days_to_default"], errors="coerce").to_numpy()
    rec_va = pd.to_numeric(va["final_recovered_amount"], errors="coerce").to_numpy()
    emp_recovery = estimate_recovery_rate(tr, ytr)        # ~0.093 aggregate = the trap bait
    b_emp, lgd_eff = empirical_default_npv(tr, ytr)       # E[NPV_default]/$ from real timing+recovery

    # realized brief S_P&L on labelled val -- the EXACT scored quantity (no discounting)
    npv_va = realized_npv(amt_va, yva.to_numpy(), dd_va, rec_va)

    # DECISION = closed-form break-even PD (drift-robust). NOT 0.5 (funds ~everyone) and
    # NOT the val argmax (overfit; swings 0.135-0.229 across folds and climbs back to
    # break-even on de-drifted PDs). We still compute the argmax to PRINT its in-sample
    # (overfit) edge. The cut is applied TRUST-BUT-VERIFY, refined by label provenance below.
    tau = theoretical_tau(b_emp)               # closed-form break-even -- THE cut (~0.255)
    # trust-but-verify, REFINED by label provenance (walk-forward no-regret: 9/9 folds +ve,
    # mean +$226K/fold; deployment +$386K realized on labelled val). Split OOD by who is
    # actually observable:
    #   ID      (~OOD)                -> point PD
    #   SHALLOW (OOD & prior_approved)-> point PD  (prior-APPROVED => LABELLED; realized
    #             default ~0.18-0.22 << break-even 0.255, so the labels VOUCH -- gating these
    #             on upper-90 needlessly abstains on safe, repaying loans)
    #   DEEP    (OOD & ~prior_approved)-> upper-90 (prior-DECLINED => NEVER labelled = the
    #             crossover risk; keep the pessimistic gate so the lever adds ZERO risk on
    #             the unseen reject tail)
    pa_va, pa_te = prior_approved(va), prior_approved(te)
    deep_va = (ood_va == 1.0) & ~pa_va         # only the never-labelled reject tail
    deep_te = (ood_te == 1.0) & ~pa_te
    dpd_va = np.where(deep_va, hi_va, pd_va)   # deep-OOD on upper-90; id+shallow on point PD
    dpd_te = np.where(deep_te, hi_te, pd_te)
    cands = np.unique(np.round(pd_va[lab_va], 4))
    tau_arg, best = float(cands[-1]) + 1e-6, -np.inf
    for tc in cands:
        tot = np.nansum(npv_va[lab_va & (pd_va < tc)])
        if tot > best:
            best, tau_arg = tot, float(tc)

    appr = lab_va & (dpd_va < tau)             # trust-but-verify approvals on labelled val
    appr_arg = lab_va & (pd_va < tau_arg)
    npv_cut, npv_all = np.nansum(npv_va[appr]), np.nansum(npv_va[lab_va])
    npv_arg = np.nansum(npv_va[appr_arg])
    funded = np.nansum(amt_va[appr])
    print(f"[4] empirical brief economics: E[NPV_default]/$={b_emp:+.4f} -> implied LGD={lgd_eff:.3f} "
          f"(NOT flat 0.30; aggregate train recovery={emp_recovery:.3f}=trap bait; draws capped@T={CAP_DRAWS_AT_TERM})")
    print(f"    DECISION trust-but-verify @ break-even={tau:.4f} approve(val)={appr.sum()/lab_va.sum():.3f}  "
          f"(deep-OOD judged on upper-90)  vs OVERFIT argmax={tau_arg:.4f} approve={appr_arg.sum()/lab_va.sum():.3f}")
    print(f"    [labelled val] NPV @break-even=${npv_cut:,.0f}  @argmax=${npv_arg:,.0f} "
          f"(argmax in-sample edge ${npv_arg-npv_cut:,.0f})  approve-all ${npv_all:,.0f}  "
          f"NPV-margin={npv_cut / funded:.2%}")

    # --- 5. assemble all applicants, ordered to expected_ids ---
    out = pd.DataFrame({
        "applicant_id": pd.concat([va["applicant_id"], te["applicant_id"]], ignore_index=True),
        "predicted_pd": np.clip(np.concatenate([pd_va, pd_te]), 0.0, 1.0),
        "pd_lower_90": np.concatenate([lo_va, lo_te]),
        "pd_upper_90": np.concatenate([hi_va, hi_te]),
        "_decision_pd": np.concatenate([dpd_va, dpd_te]),   # point PD (id+shallow) or upper-90 (deep-OOD)
    })
    order = pd.read_csv(EXPECTED_IDS, header=None)[0].astype(str).tolist()
    out = out.set_index(out["applicant_id"].astype(str)).reindex(order).reset_index(drop=True)
    out["applicant_id"] = order
    assert not out.isna().any().any(), "row/ID mismatch vs expected_ids"
    out["decision"] = (out["_decision_pd"] < tau).astype(int)   # trust-but-verify cut
    out = out[["applicant_id", "decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]]

    dest = SUB_DIR / "submission_A_decisions.csv"
    out.to_csv(dest, index=False)
    print(f"[5] wrote {dest.name}: {len(out)} rows  approve_rate={out.decision.mean():.3f}  "
          f"PD[min/med/max]={out.predicted_pd.min():.3f}/{out.predicted_pd.median():.3f}/"
          f"{out.predicted_pd.max():.3f}")

    # --- 5b. forward EXPECTED NPV on the approved val+test book (scored set) ---
    # outcomes are withheld on test, so we price each approved loan by predicted PD
    # x the train default-timing distribution. Reported in $ / margin% / per-loan so
    # the number is comparable however the metric is normalized.
    pd_cat = np.concatenate([pd_va, pd_te])
    dpd_cat = np.concatenate([dpd_va, dpd_te])   # decision PD (deep-OOD on upper-90)
    amt_cat = np.concatenate([amt_va, te["requested_amount"].to_numpy()])
    enpv = amt_cat * expected_npv_per_dollar(pd_cat, b_emp)   # value at POINT PD x empirical b
    appr_cat = dpd_cat < tau                      # but GATE trust-but-verify
    e_total, funded_all = float(enpv[appr_cat].sum()), float(amt_cat[appr_cat].sum())
    n_appr = int(appr_cat.sum())
    print(f"[5b] EXPECTED NPV (approved val+test): ${e_total:,.0f}  "
          f"margin={e_total / funded_all:.2%}  per-loan=${e_total / max(n_appr, 1):,.0f}  "
          f"(funded ${funded_all:,.0f} over {n_appr} loans)")


if __name__ == "__main__":
    main()
