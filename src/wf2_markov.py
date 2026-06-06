#!/usr/bin/env python3
"""wf2_markov — discrete-time absorbing-Markov / survival hazard model for PD.

Idea (organizer hint: "Markov chains are important"):
  Each loan-day is a step in an absorbing Markov chain over delinquency states.
  We don't observe per-day draws, only the absorption day (days_to_default) and
  whether absorption was to DEFAULT or PAID. That is exactly the data a
  DISCRETE-TIME SURVIVAL model consumes. We model a per-loan-day default hazard
    h(d | x_i) = P(absorb-to-default at day d | survived to d, x_i)
  and the implied PD_i = 1 - prod_{d=1..90} (1 - h(d|x_i)) is the marginal
  default-by-90 probability -- directly comparable to the GBT/blend PD.

  Two hazard specs are compared:
    (M1) Pooled-logistic discrete-time hazard: stack one row per loan-day,
         target = "defaulted on this day", features = x_i + a smooth function of
         day d (so the baseline hazard is data-driven). Fit ONE logistic on the
         person-day panel. PD = 1 - prod(1-h_d).
    (M2) Two-part / mixture: (a) a "term default" hazard over days 1..60 fit as
         above on the panel, plus (b) a day-90 balance-check absorption: among
         loans that survive the term, a logistic P(open-balance default | x).
         This matches the data-generating structure (no defaults in 61-89; a
         spike at 90).

  Both are compared to a plain GBT PD (same features, binary default_flag) and
  to the deployed blend recipe, all under the SAME 5-fold OOF split.

Metrics: pooled OOF AUC, Brier, LogLoss, ECE (10-bin). Also reports the OOF AUC
of the implied TIMING distribution as a bonus (Deliverable B relevance).

Run: .venv/Scripts/python.exe -m src.wf2_markov
Does NOT modify build_a.py or any pipeline file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_a import build_cat_dtypes, build_features  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SEED = 17
N_SPLITS = 5
HORIZON = 90
TERM = 60  # repayment term; balance check at day 90


def to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for c in X.columns:
        col = X[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            codes = col.cat.codes.astype("float64")
            codes[codes < 0] = np.nan
            out[c] = codes
        else:
            out[c] = pd.to_numeric(col, errors="coerce").astype("float64")
    return out


def ece(y, p, n_bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return e


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, categorical_features="from_dtype",
        random_state=seed,
    )


def make_logit(Xtr, ytr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr2, Xte2 = imp.transform(Xtr), imp.transform(Xte)
    sc = StandardScaler().fit(Xtr2)
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(Xtr2), ytr)
    return clf.predict_proba(sc.transform(Xte2))[:, 1]


# ------- M1: pooled-logistic discrete-time hazard ----------------------------
# Risk set & baseline: a small set of day-bucket dummies (data-driven baseline
# hazard). To keep the panel tractable we use coarse day buckets but evaluate
# hazard at the bucket level (hazard is empirically ~flat at 0.0023/day in 3-60).

DAY_BUCKETS = [(1, 2), (3, 6), (7, 14), (15, 30), (31, 45), (46, 60)]  # term-days
# Bucket midpoints used as a feature so hazard varies smoothly with age.


def build_panel(Xnum: np.ndarray, exit_day: np.ndarray, is_def: np.ndarray):
    """Person-day panel over the 60-day TERM only (no rows in 61-90).

    For each loan i and each day-bucket fully/partly before its exit, emit one
    row: features = [x_i, bucket_mid], target = 1 if loan defaulted in that
    bucket, else 0. Loans exit at min(exit_day, 60). A loan contributes rows for
    every bucket whose start <= its exit day; the bucket containing the exit gets
    the event label (1 iff it was a default), earlier buckets get 0.
    """
    rows_x, rows_age, rows_y = [], [], []
    n = len(exit_day)
    for bs, be in DAY_BUCKETS:
        mid = (bs + be) / 2.0
        # loans still at risk at bucket start (exit_day >= bs)
        at_risk = exit_day >= bs
        # event in this bucket: default with dtd in [bs,be]
        ev = is_def & (exit_day >= bs) & (exit_day <= be)
        ar_idx = np.where(at_risk)[0]
        rows_x.append(Xnum[ar_idx])
        rows_age.append(np.full(len(ar_idx), mid))
        rows_y.append(ev[ar_idx].astype(int))
    X = np.vstack(rows_x)
    age = np.concatenate(rows_age).reshape(-1, 1)
    y = np.concatenate(rows_y)
    return np.hstack([X, age]), y


def m1_pd(Xnum_tr, exit_tr, def_tr, Xnum_te):
    """Fit pooled-logistic hazard on the term panel; return per-loan PD_term
    (1 - prod over buckets of (1-h_bucket)). Day-90 balance handled separately."""
    Xp, yp = build_panel(Xnum_tr, exit_tr, def_tr)
    imp = SimpleImputer(strategy="median").fit(Xp)
    sc = StandardScaler().fit(imp.transform(Xp))
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(imp.transform(Xp)), yp)

    # Per-loan survival through the term: product over buckets of (1 - h_bucket).
    surv = np.ones(len(Xnum_te))
    cum_def = np.zeros(len(Xnum_te))
    haz_by_bucket = []
    for bs, be in DAY_BUCKETS:
        mid = (bs + be) / 2.0
        Xb = np.hstack([Xnum_te, np.full((len(Xnum_te), 1), mid)])
        h = clf.predict_proba(sc.transform(imp.transform(Xb)))[:, 1]
        haz_by_bucket.append(h)
        cum_def += surv * h          # prob default in this bucket
        surv *= (1 - h)
    pd_term = cum_def                  # P(default during 1..60)
    surv_term = surv                   # P(survive the term, i.e. open at day 60)
    return pd_term, surv_term, np.array(haz_by_bucket)


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)
    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    dtd = train.loc[obs, "days_to_default"].to_numpy()
    is_def = y.astype(bool)
    exit_day = np.where(is_def, np.nan_to_num(dtd), TERM).astype(int)
    # day-90 balance default indicator (among defaults, dtd==90)
    day90_def = is_def & (np.nan_to_num(dtd) >= 90)

    Xnum_df = to_numeric_frame(Xo)
    Xnum = Xnum_df.to_numpy()

    print(f"[data] {obs.sum():,} observed, default rate {y.mean():.4f}; "
          f"day90-balance defaults {day90_def.sum():,} "
          f"({day90_def.sum()/is_def.sum():.1%} of defaults)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in
           ["gbt", "logit", "blend", "m1_term", "m1_full", "m2_full"]}

    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        # --- baseline binary models (deployed recipe) ---
        m = make_hgb(SEED + k).fit(Xo.iloc[tr], y[tr])
        p_gbt = m.predict_proba(Xo.iloc[te])[:, 1]
        p_log = make_logit(Xnum[tr], y[tr], Xnum[te])
        p_blend = 0.4 * p_gbt + 0.6 * p_log
        oof["gbt"][te] = p_gbt
        oof["logit"][te] = p_log
        oof["blend"][te] = p_blend

        # --- M1: discrete-time hazard over the term ---
        pd_term, surv_term, _ = m1_pd(Xnum[tr], exit_day[tr], is_def[tr], Xnum[te])
        oof["m1_term"][te] = pd_term

        # --- M2: add day-90 balance-check absorption ---
        # Among TRAIN loans that survived the term (exit_day==60 & not term-default
        # i.e. paid OR day90-default), model P(day90 default | x).
        survived_term_tr = (exit_day[tr] >= TERM)  # reached day 60 open
        # Of these, day90_def==1 are the balance defaults; rest paid.
        st_idx = np.where(survived_term_tr)[0]
        y90 = day90_def[tr][st_idx].astype(int)
        if y90.sum() > 5 and (1 - y90).sum() > 5:
            p90_te = make_logit(Xnum[tr][st_idx], y90, Xnum[te])
        else:
            p90_te = np.full(len(te), y90.mean())
        # Full PD (M2) = term-default prob + survive-term * P(day90 default)
        pd_full_m2 = pd_term + surv_term * p90_te
        oof["m2_full"][te] = np.clip(pd_full_m2, 1e-6, 1 - 1e-6)

        # M1 "full" without separate day90 model: treat day90 hazard as the
        # marginal balance-default rate among term survivors (no covariates).
        base90 = y90.mean()
        oof["m1_full"][te] = np.clip(pd_term + surv_term * base90, 1e-6, 1 - 1e-6)

    print("\n" + "=" * 74)
    print(f"{'model':12s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s} {'ECE':>8s}")
    print("-" * 74)
    for name in ["gbt", "logit", "blend", "m1_term", "m1_full", "m2_full"]:
        p = np.clip(oof[name], 1e-6, 1 - 1e-6)
        print(f"{name:12s} {roc_auc_score(y,p):8.4f} {brier_score_loss(y,p):8.4f} "
              f"{log_loss(y,p):9.4f} {ece(y,p):8.4f}")

    # Correlation of M2 PD with blend PD (does the chain just reproduce the GBT?)
    print("\n[diagnostic] Pearson corr of PDs with deployed blend:")
    for name in ["gbt", "logit", "m1_full", "m2_full"]:
        r = np.corrcoef(oof[name], oof["blend"])[0, 1]
        rho = pd.Series(oof[name]).corr(pd.Series(oof["blend"]), method="spearman")
        print(f"  {name:10s} pearson={r:.4f}  spearman={rho:.4f}")

    # ---- TIMING evaluation: does the hazard recover the default-day shape? ----
    # Concordance of predicted vs actual default day among defaulters (proxy for B).
    print("\n[timing] mean predicted term-default share vs empirical (B relevance)")
    emp_term = (is_def & (np.nan_to_num(dtd) < 90)).sum() / len(y)
    print(f"  empirical P(term default 3-60) = {emp_term:.4f}")
    print(f"  empirical P(day90 balance def) = {day90_def.sum()/len(y):.4f}")
    print(f"  M2 mean pd_full = {oof['m2_full'].mean():.4f}  "
          f"(actual default rate {y.mean():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
