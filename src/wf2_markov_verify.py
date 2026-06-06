#!/usr/bin/env python3
"""wf2_markov_verify -- INDEPENDENT adversarial check: does the discrete-time
absorbing-Markov / survival hazard model BEAT the deployed 0.4*HGB+0.6*Logit
blend on Deliverable A's PD-by-90 (AUC / Brier / ECE), under identical 5-fold OOF?

I rebuild the hazard from scratch (NOT reusing the designer's bucket panel) two
ways to avoid being fooled by a single parameterization:
  (H-fine)  per-DAY logistic hazard on a true person-day risk set, days 1..60,
            day index as a numeric age feature (data-driven baseline). PD_term =
            1 - prod_d (1-h_d). Day-90 balance via a separate logistic among
            term-survivors. This is the textbook discrete-time survival object.
  (H-pwexp) piecewise-constant hazard via the designer's coarse buckets (parity
            check with their m2_full).

Compared head-to-head with:
  gbt    : HistGradientBoosting binary default_flag (deployed tree arm)
  logit  : L2 logistic binary default_flag (deployed linear arm)
  blend  : 0.4*gbt + 0.6*logit (DEPLOYED Deliverable A point PD)

All on the SAME StratifiedKFold(5, seed=17) splits the designer used. I report
pooled OOF AUC/Brier/LogLoss/ECE, plus paired per-fold AUC deltas (does the chain
win in ALL folds, or is the delta fold-noise?), plus a DeLong-style bootstrap on
the pooled AUC difference. I also recalibrate every score with isotonic on the
held-out fold's own OOF predictions of the OTHER folds is not possible cheaply, so
I additionally report metrics AFTER a global isotonic calibration fit on OOF (this
is the fair calibration comparison, since build_a calibrates on val).

Run: .venv/Scripts/python.exe -m src.wf2_markov_verify
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
from sklearn.isotonic import IsotonicRegression
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
TERM = 60


def to_numeric_frame(X):
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


def make_hgb(seed):
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, categorical_features="from_dtype",
        random_state=seed,
    )


def fit_logit(Xtr, ytr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(imp.transform(Xtr)), ytr)
    return clf.predict_proba(sc.transform(imp.transform(Xte)))[:, 1]


# ---------------- H-fine: TRUE per-day risk-set hazard ----------------------
# To keep the person-day panel tractable (51722 loans * up to 60 days ~ 3M rows
# over the full data, ~2.4M per train fold) we subsample the CENSORED (survived)
# day rows: every non-event person-day is kept with prob KEEP, and the fitted
# intercept is offset-corrected by log(KEEP) so hazards stay unbiased. Event rows
# are always kept. This is the standard case-base / sub-sampling trick for rare
# discrete-time hazards and makes the fit ~5x cheaper with negligible bias.

KEEP = 0.25  # keep 25% of non-event person-days


def build_daypanel(Xnum, exit_day, is_def, rng):
    """One row per at-risk loan-day d in 1..60. event=1 iff loan defaults on day d.
    Non-event rows subsampled at rate KEEP (event rows always kept)."""
    xs, ages, ys, w_off = [], [], [], []
    n = len(exit_day)
    for d in range(1, TERM + 1):
        at_risk = exit_day >= d            # still alive at start of day d
        event = is_def & (exit_day == d)   # defaults exactly on day d
        ar = np.where(at_risk)[0]
        ev = event[ar]
        # keep all events; subsample non-events
        keep_mask = ev | (rng.random(len(ar)) < KEEP)
        sel = ar[keep_mask]
        xs.append(Xnum[sel])
        ages.append(np.full(len(sel), d, dtype="float64"))
        ys.append(event[sel].astype(int))
    X = np.vstack(xs)
    age = np.concatenate(ages).reshape(-1, 1)
    y = np.concatenate(ys)
    return np.hstack([X, age]), y


def hfine_pd(Xnum_tr, exit_tr, def_tr, Xnum_te, rng):
    Xp, yp = build_daypanel(Xnum_tr, exit_tr, def_tr, rng)
    imp = SimpleImputer(strategy="median").fit(Xp)
    sc = StandardScaler().fit(imp.transform(Xp))
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(imp.transform(Xp)), yp)
    # intercept offset to undo subsampling of non-events: logit(h_true) =
    # logit(h_fit) + log(KEEP). Add log(KEEP) to the linear predictor for hazards.
    offset = np.log(KEEP)
    surv = np.ones(len(Xnum_te))
    pd_term = np.zeros(len(Xnum_te))
    haz_days = np.zeros((len(Xnum_te), TERM + 1))
    for d in range(1, TERM + 1):
        Xb = np.hstack([Xnum_te, np.full((len(Xnum_te), 1), float(d))])
        z = clf.decision_function(sc.transform(imp.transform(Xb))) + offset
        h = 1.0 / (1.0 + np.exp(-z))
        haz_days[:, d] = h
        pd_term += surv * h
        surv *= (1 - h)
    return pd_term, surv, haz_days


# ---------------- H-pwexp: designer's coarse buckets (parity) ---------------
BUCKETS = [(1, 2), (3, 6), (7, 14), (15, 30), (31, 45), (46, 60)]


def hpw_pd(Xnum_tr, exit_tr, def_tr, Xnum_te):
    xs, ages, ys = [], [], []
    for bs, be in BUCKETS:
        mid = (bs + be) / 2.0
        at_risk = exit_tr >= bs
        ev = def_tr & (exit_tr >= bs) & (exit_tr <= be)
        idx = np.where(at_risk)[0]
        xs.append(Xnum_tr[idx]); ages.append(np.full(len(idx), mid)); ys.append(ev[idx].astype(int))
    Xp = np.hstack([np.vstack(xs), np.concatenate(ages).reshape(-1, 1)])
    yp = np.concatenate(ys)
    imp = SimpleImputer(strategy="median").fit(Xp)
    sc = StandardScaler().fit(imp.transform(Xp))
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(imp.transform(Xp)), yp)
    surv = np.ones(len(Xnum_te))
    pd_term = np.zeros(len(Xnum_te))
    for bs, be in BUCKETS:
        mid = (bs + be) / 2.0
        Xb = np.hstack([Xnum_te, np.full((len(Xnum_te), 1), mid)])
        h = clf.predict_proba(sc.transform(imp.transform(Xb)))[:, 1]
        pd_term += surv * h
        surv *= (1 - h)
    return pd_term, surv


def paired_auc_bootstrap(y, pa, pb, n_boot=2000, seed=0):
    """Bootstrap the pooled AUC difference AUC(pa)-AUC(pb). Returns (mean_delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    base = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
    deltas = np.array(deltas)
    return base, np.quantile(deltas, 0.025), np.quantile(deltas, 0.975)


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)
    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    dtd = np.nan_to_num(train.loc[obs, "days_to_default"].to_numpy())
    is_def = y.astype(bool)
    exit_day = np.where(is_def, dtd, TERM).astype(int)
    day90 = is_def & (dtd >= 90)
    Xnum = to_numeric_frame(Xo).to_numpy()
    print(f"[data] {obs.sum():,} observed; default rate {y.mean():.4f}; "
          f"day90 balance defaults {day90.sum():,} ({day90.sum()/is_def.sum():.1%} of defaults)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    names = ["gbt", "logit", "blend", "hpw_full", "hfine_term", "hfine_full"]
    oof = {k: np.zeros(len(y)) for k in names}
    fold_auc = {k: [] for k in names}
    rng = np.random.default_rng(SEED)

    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        m = make_hgb(SEED + k).fit(Xo.iloc[tr], y[tr])
        p_gbt = m.predict_proba(Xo.iloc[te])[:, 1]
        p_log = fit_logit(Xnum[tr], y[tr], Xnum[te])
        p_blend = 0.4 * p_gbt + 0.6 * p_log
        oof["gbt"][te] = p_gbt; oof["logit"][te] = p_log; oof["blend"][te] = p_blend

        # day-90 balance logistic among term survivors (shared by hpw & hfine full)
        st = np.where(exit_day[tr] >= TERM)[0]
        y90 = day90[tr][st].astype(int)
        if y90.sum() > 5 and (1 - y90).sum() > 5:
            p90 = fit_logit(Xnum[tr][st], y90, Xnum[te])
        else:
            p90 = np.full(len(te), y90.mean())
        base90 = y90.mean()

        # H-pwexp
        pd_term_pw, surv_pw = hpw_pd(Xnum[tr], exit_day[tr], is_def[tr], Xnum[te])
        oof["hpw_full"][te] = np.clip(pd_term_pw + surv_pw * p90, 1e-6, 1 - 1e-6)

        # H-fine
        pd_term_f, surv_f, _ = hfine_pd(Xnum[tr], exit_day[tr], is_def[tr], Xnum[te], rng)
        oof["hfine_term"][te] = np.clip(pd_term_f, 1e-6, 1 - 1e-6)
        oof["hfine_full"][te] = np.clip(pd_term_f + surv_f * p90, 1e-6, 1 - 1e-6)

        for nm in names:
            fold_auc[nm].append(roc_auc_score(y[te], oof[nm][te]))

    print("\n" + "=" * 78)
    print("POOLED OOF (raw scores)")
    print(f"{'model':12s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s} {'ECE':>8s} {'mean_pd':>8s}")
    print("-" * 78)
    for nm in names:
        p = np.clip(oof[nm], 1e-6, 1 - 1e-6)
        print(f"{nm:12s} {roc_auc_score(y,p):8.4f} {brier_score_loss(y,p):8.4f} "
              f"{log_loss(y,p):9.4f} {ece(y,p):8.4f} {p.mean():8.4f}")

    # Fair calibration comparison: isotonic-recalibrate each OOF score on OOF
    # itself (in-sample optimistic but EQUAL for all models -> isolates ranking).
    print("\n" + "=" * 78)
    print("POOLED OOF (after global isotonic recalibration on OOF -- equalizes calibration)")
    print(f"{'model':12s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s} {'ECE':>8s}")
    print("-" * 78)
    for nm in names:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        pc = np.clip(iso.fit_transform(oof[nm], y), 1e-6, 1 - 1e-6)
        print(f"{nm:12s} {roc_auc_score(y,pc):8.4f} {brier_score_loss(y,pc):8.4f} "
              f"{log_loss(y,pc):9.4f} {ece(y,pc):8.4f}")

    print("\n" + "=" * 78)
    print("PAIRED per-fold AUC vs blend (does the chain win in ALL 5 folds?)")
    print(f"{'fold':>4s} {'blend':>8s} {'hpw':>8s} {'hfine':>8s} {'d_hpw':>8s} {'d_hfine':>8s}")
    for k in range(N_SPLITS):
        b = fold_auc["blend"][k]; pw = fold_auc["hpw_full"][k]; hf = fold_auc["hfine_full"][k]
        print(f"{k:4d} {b:8.4f} {pw:8.4f} {hf:8.4f} {pw-b:+8.4f} {hf-b:+8.4f}")
    d_pw = np.array(fold_auc["hpw_full"]) - np.array(fold_auc["blend"])
    d_hf = np.array(fold_auc["hfine_full"]) - np.array(fold_auc["blend"])
    print(f"  mean delta  hpw={d_pw.mean():+.4f} (wins {int((d_pw>0).sum())}/5)   "
          f"hfine={d_hf.mean():+.4f} (wins {int((d_hf>0).sum())}/5)")

    print("\n" + "=" * 78)
    print("Bootstrap 95% CI on pooled AUC difference (chain - blend)")
    for nm in ["hpw_full", "hfine_full"]:
        base, lo, hi = paired_auc_bootstrap(y, oof[nm], oof["blend"], seed=SEED)
        verdict = "BEATS blend" if lo > 0 else ("WORSE" if hi < 0 else "TIE (CI spans 0)")
        print(f"  {nm:12s} dAUC={base:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    # Spearman of chain PD vs blend PD (is it just reproducing the same ranking?)
    print("\n[diagnostic] Spearman corr of chain PD with deployed blend PD:")
    for nm in ["hpw_full", "hfine_full"]:
        rho = pd.Series(oof[nm]).corr(pd.Series(oof["blend"]), method="spearman")
        print(f"  {nm:12s} spearman={rho:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
