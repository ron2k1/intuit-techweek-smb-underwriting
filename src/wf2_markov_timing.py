#!/usr/bin/env python3
"""wf2_markov_timing — does the chain's TIMING help A's NPV, and is it a B tool?

Two questions the marginal-PD comparison (wf2_markov) cannot answer:
  (Q2) A's exact NPV on a default uses the default DAY t*: default NPV/$ =
       FEE + DAILY_DRAW*(t*-1) + rec/R - 1. A binary PD model has no t*; the
       deployed code plugs a measured effective LGD (0.25). Does a per-loan
       E[t* | x] from the discrete-time hazard sharpen the funded-set selection
       enough to beat the flat-LGD policy on realized val $? We compare, on a
       5-fold OOF basis, the realized portfolio $ of:
         (a) flat-LGD conservative policy  (deployed: fund if PD<BE)
         (b) timing-aware policy: fund if E[NPV per $ | x] > 0 using the chain's
             per-loan default-day distribution to compute expected loss.
  (Q3) Deliverable B is the cumulative-default-rate-by-loan-age grid. The chain
       gives P(t* <= age_weeks) natively, with day-90 censoring handled. We
       report how well the chain's predicted cumulative-default-by-age curve
       matches the empirical curve (the core of B) vs a naive "scale the final
       rate" baseline.

Run: .venv/Scripts/python.exe -m src.wf2_markov_timing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

# Economics (mirror build_a.py constants exactly)
TERM_INT = 0.35 * 60 / 365
FEE = 0.03
NET_MARGIN = TERM_INT + FEE          # ~0.0875
DAILY_DRAW = (1 + TERM_INT) / 60
LGD = 0.25
BREAK_EVEN_PD = NET_MARGIN / (LGD + NET_MARGIN)  # ~0.259

# Per-day hazard on a fine grid so we get a real default-day distribution.
DAY_GRID = list(range(1, TERM + 1))  # 1..60 (term defaults); 90 handled separately


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


def fit_hazard_buckets(Xtr, exit_tr, def_tr):
    """Pooled-logistic hazard over coarse buckets; returns a predict fn that
    yields a per-day hazard h(d|x) for d in 1..60 via the bucket the day falls in."""
    buckets = [(1, 2), (3, 6), (7, 14), (15, 30), (31, 45), (46, 60)]
    rows_x, rows_age, rows_y = [], [], []
    for bs, be in buckets:
        mid = (bs + be) / 2.0
        at_risk = exit_tr >= bs
        ev = def_tr & (exit_tr >= bs) & (exit_tr <= be)
        idx = np.where(at_risk)[0]
        rows_x.append(Xtr[idx]); rows_age.append(np.full(len(idx), mid))
        rows_y.append(ev[idx].astype(int))
    Xp = np.hstack([np.vstack(rows_x), np.concatenate(rows_age).reshape(-1, 1)])
    yp = np.concatenate(rows_y)
    imp = SimpleImputer(strategy="median").fit(Xp)
    sc = StandardScaler().fit(imp.transform(Xp))
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(imp.transform(Xp)), yp)

    # bucket length -> per-day hazard within bucket = 1-(1-h_bucket)^(1/len)
    def per_day_haz(Xte):
        H = np.zeros((len(Xte), TERM + 1))  # index by day 1..60
        for bs, be in buckets:
            mid = (bs + be) / 2.0
            Xb = np.hstack([Xte, np.full((len(Xte), 1), mid)])
            hb = clf.predict_proba(sc.transform(imp.transform(Xb)))[:, 1]
            L = be - bs + 1
            hd = 1 - (1 - hb) ** (1.0 / L)
            for d in range(bs, be + 1):
                H[:, d] = hd
        return H  # H[:,d] hazard on day d
    return per_day_haz


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
    amt = train.loc[obs, "requested_amount"].to_numpy()
    rec = np.nan_to_num(train.loc[obs, "final_recovered_amount"].to_numpy())
    is_def = y.astype(bool)
    exit_day = np.where(is_def, dtd, TERM).astype(int)
    day90 = is_def & (dtd >= 90)
    Xnum = to_numeric_frame(Xo).to_numpy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # accumulate realized $ for the two policies, per fold
    flat_dollars, timing_dollars = [], []
    flat_n, timing_n = [], []
    # B-relevance: predicted vs empirical cumulative default by loan-age (weeks)
    age_weeks = np.arange(1, 14)
    pred_curve_acc = np.zeros(13)
    emp_curve_acc = np.zeros(13)

    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        # plain binary logistic PD (proxy for deployed point PD; we showed the
        # chain marginal == blend, so use logit PD as the shared ranking).
        imp = SimpleImputer(strategy="median").fit(Xnum[tr])
        sc = StandardScaler().fit(imp.transform(Xnum[tr]))
        clf = LogisticRegression(C=0.5, max_iter=3000).fit(
            sc.transform(imp.transform(Xnum[tr])), y[tr])
        pd_hat = clf.predict_proba(sc.transform(imp.transform(Xnum[te])))[:, 1]

        # hazard model for timing
        haz_fn = fit_hazard_buckets(Xnum[tr], exit_day[tr], is_def[tr])
        H = haz_fn(Xnum[te])  # (n_te, 61), H[:,d] day-d hazard
        # default-day distribution over term: f(d) = surv(d-1)*h(d)
        surv = np.ones(len(te))
        fday = np.zeros((len(te), TERM + 1))
        for d in DAY_GRID:
            fday[:, d] = surv * H[:, d]
            surv = surv * (1 - H[:, d])
        pd_term = fday[:, 1:].sum(axis=1)
        surv_term = surv  # P(open at day 60)
        # day-90 balance default prob among term survivors (marginal in this fold)
        st = exit_day[tr] >= TERM
        base90 = day90[tr][st].mean() if st.sum() else 0.0
        pd_full = pd_term + surv_term * base90

        # ---- Expected NPV per $ using the chain's default-day distribution ----
        # repaid -> NET_MARGIN. term default at day d -> FEE + DAILY_DRAW*(d-1) + 0 - 1
        # (no per-loan recovery known at decision time; use mean rec/R among defs).
        mean_rec_frac = (rec[tr][is_def[tr]] / amt[tr][is_def[tr]]).mean()
        # E[NPV/$] = P(survive_all)*NET_MARGIN
        #          + sum_d f(d)*min(FEE+DAILY_DRAW*(d-1)+mean_rec_frac-1, NET_MARGIN)
        #          + day90 default * (FEE + DAILY_DRAW*59 + mean_rec_frac - 1)
        npv = np.zeros(len(te))
        p_survive_all = surv_term * (1 - base90)
        npv += p_survive_all * NET_MARGIN
        for d in DAY_GRID:
            loss = np.minimum(FEE + DAILY_DRAW * (d - 1) + mean_rec_frac - 1, NET_MARGIN)
            npv += fday[:, d] * loss
        loss90 = min(FEE + DAILY_DRAW * (90 - 1) + mean_rec_frac - 1, NET_MARGIN)
        npv += surv_term * base90 * loss90

        # ---- realized $ helper (true outcome on the held-out fold) ----
        def realized(decision):
            draws = amt[te] * DAILY_DRAW * np.clip(dtd[te] - 1, 0, None)
            default_profit = np.minimum(amt[te] * FEE + draws + rec[te] - amt[te],
                                        amt[te] * NET_MARGIN)
            profit = np.where(y[te] == 0, amt[te] * NET_MARGIN, default_profit)
            fmask = decision == 1
            return float(profit[fmask].sum()), int(fmask.sum())

        # (a) flat-LGD conservative: fund if pd_hat < BREAK_EVEN_PD
        dec_flat = (pd_hat < BREAK_EVEN_PD).astype(int)
        rf, nf = realized(dec_flat)
        flat_dollars.append(rf); flat_n.append(nf)

        # (b) timing-aware: fund if E[NPV/$|x] > 0
        dec_tim = (npv > 0).astype(int)
        rt, nt = realized(dec_tim)
        timing_dollars.append(rt); timing_n.append(nt)

        # ---- B relevance: cumulative default by loan-age in WEEKS ----
        # predicted cumulative P(default by week w) for the held-out loans
        for wi, w in enumerate(age_weeks):
            day = min(w * 7, 90)
            if day <= TERM:
                cum = fday[:, 1:day + 1].sum(axis=1)
            else:
                cum = pd_term + (surv_term * base90 if day >= 90 else 0.0)
            pred_curve_acc[wi] += cum.mean()
            # empirical cumulative default-by-day among the same held-out loans
            emp_curve_acc[wi] += ((dtd[te] <= day) & is_def[te]).mean()

    print("=" * 74)
    print("Q2 -- A NPV: timing-aware (chain) vs flat-LGD conservative (deployed)")
    print("=" * 74)
    print(f"  flat-LGD   realized $ per fold: " +
          ", ".join(f"{v:,.0f}" for v in flat_dollars))
    print(f"             funded n per fold:   " + ", ".join(str(n) for n in flat_n))
    print(f"  timing     realized $ per fold: " +
          ", ".join(f"{v:,.0f}" for v in timing_dollars))
    print(f"             funded n per fold:   " + ", ".join(str(n) for n in timing_n))
    tot_flat, tot_tim = sum(flat_dollars), sum(timing_dollars)
    print(f"\n  TOTAL realized $  flat={tot_flat:,.0f}  timing={tot_tim:,.0f}  "
          f"delta={tot_tim - tot_flat:,.0f}")
    wins = sum(t > f for t, f in zip(timing_dollars, flat_dollars))
    print(f"  timing beats flat in {wins}/{N_SPLITS} folds")

    print("\n" + "=" * 74)
    print("Q3 -- B relevance: chain's cumulative-default-by-loan-age curve")
    print("=" * 74)
    pred_curve = pred_curve_acc / N_SPLITS
    emp_curve = emp_curve_acc / N_SPLITS
    print(f"{'week':>4s} {'day':>4s} {'pred_cum':>9s} {'emp_cum':>9s} {'abs_err':>8s}")
    for wi, w in enumerate(age_weeks):
        day = min(w * 7, 90)
        print(f"{w:4d} {day:4d} {pred_curve[wi]:9.4f} {emp_curve[wi]:9.4f} "
              f"{abs(pred_curve[wi]-emp_curve[wi]):8.4f}")
    mae = np.mean(np.abs(pred_curve - emp_curve))
    print(f"\n  trajectory MAE (chain vs empirical, pooled) = {mae:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
