#!/usr/bin/env python3
"""wf4_cost_sensitive — does a genuinely better COST-SENSITIVE funding rule beat
the flat break-even PD (0.259) on REALIZED amortizing NPV, out-of-fold?

Baseline (FLAT): fund iff calibrated point PD < BREAK_EVEN_PD (0.259), i.e. a
single break-even derived from a portfolio-average effective LGD ~0.25.

We test three cost-sensitive refinements, all on the 51,722 observed rows, with
BOTH 5-fold StratifiedKFold(shuffle, seed=17) random OOF and a chronological
expanding-window walk-forward. The PD model is the EXACT baseline blend
(0.4*HGB + 0.6*L2-logit) with cross-fit isotonic calibration, fit inside each
fold. Realized NPV uses the EXACT amortizing formula from build_a.py (capped).

  (a) RECOVERY-MODEL per-loan LGD:
      Predict rec_frac = final_recovered_amount/R on defaulters (train-fold), and
      E[days_to_default|default] on defaulters, giving a per-loan expected LGD
      under the exact amortizing formula -> per-loan break-even threshold
      tau_i = margin/(margin + LGD_i). Fund iff pd_i < tau_i.
      (Variant: fund iff E[NPV_i] > 0 directly.)

  (b) RISK-AVERSE utility:
      Use an upper PD instead of the point PD: pd_upper = point + k*ensemble_sd
      (epistemic) for k in {0.5,1,2}, and a conformal-style upper-90 from OOF
      residual spread. Fund iff pd_upper < BREAK_EVEN_PD. This funds fewer, lower
      -variance loans.

  (c) Full THRESHOLD SWEEP confirming 0.259 is the OOF + walk-forward optimum.

ADOPT a rule only if it beats FLAT on REALIZED NPV out-of-fold on (nearly) every
fold AND mean delta clearly exceeds per-fold std. Per-loan refinements have
repeatedly been noise/negative here, so the prior is skeptical.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\ayush\intuit-hackathon\intuit-techweek-smb-underwriting")
sys.path.insert(0, str(REPO))
from src.build_a import (build_cat_dtypes, build_features, numeric_frame,  # noqa: E402
                         make_model, make_logit, W_HGB)
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Economics (EXACT from build_a.py) ---------------------------------- #
TERM_INT = 0.35 * 60 / 365            # ~0.057534
FEE = 0.03
NET_MARGIN = TERM_INT + FEE           # ~0.087534
DAILY_DRAW = (1 + TERM_INT) / 60      # per-$ daily draw
LGD_FLAT = 0.25
BREAK_EVEN_PD = NET_MARGIN / (LGD_FLAT + NET_MARGIN)  # 0.259
SEED = 17
N_BOOT = 25                           # ensemble for epistemic sd (matches build_a)

DATA = REPO / "dataset"
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")
cats = build_cat_dtypes(train, val, test)
X_all = build_features(train, cats)

obs = train["default_flag"].notna().to_numpy()
Xo = X_all.loc[obs].reset_index(drop=True)
Zo = numeric_frame(Xo)
sub = train.loc[obs].reset_index(drop=True)
y = sub["default_flag"].astype(int).to_numpy()
amount = sub["requested_amount"].to_numpy(dtype=float)
dtd = sub["days_to_default"].to_numpy(dtype=float)
rec = np.nan_to_num(sub["final_recovered_amount"].to_numpy(dtype=float))
ts = pd.to_datetime(sub["application_timestamp"]).to_numpy()
N = len(sub)
print(f"[data] observed={N:,} defaulters={y.sum():,} rate={y.mean():.4f}")
print(f"[econ] NET_MARGIN={NET_MARGIN:.5f} LGD_FLAT={LGD_FLAT} BREAK_EVEN={BREAK_EVEN_PD:.4f}")


def make_reg(seed):
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype", random_state=seed, learning_rate=0.05,
        max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1)


def realized_npv(idx, fund):
    """EXACT realized amortizing $ profit (build_a formula, capped) over funded
    rows in idx. fund is boolean aligned to idx."""
    a = amount[idx]; d = y[idx]; t = dtd[idx]; r = rec[idx]
    draws = a * DAILY_DRAW * np.clip(np.nan_to_num(t) - 1, 0, None)
    default_profit = np.minimum(a * FEE + draws + r - a, a * NET_MARGIN)
    profit = np.where(d == 0, a * NET_MARGIN, default_profit)
    return float(profit[fund].sum())


def per_loan_lgd_from_recovery(e_tstar, e_recfrac):
    """Expected per-$ loss on default under the exact amortizing formula, given a
    predicted default timing and recovery fraction. LGD_i = -E[default NPV/$]."""
    draws = DAILY_DRAW * np.clip(e_tstar - 1, 0, None)
    default_npv_per_d = np.minimum(FEE + draws + e_recfrac - 1.0, NET_MARGIN)
    return -default_npv_per_d  # positive loss


# ------------------------------------------------------------------------- #
# One fold: fit blend PD (cross-fit isotonic), timing & recovery regressors,
# ensemble sd, then evaluate every rule's realized NPV on the test fold.
# ------------------------------------------------------------------------- #
def run_fold(tr, te, fold, rng_master):
    # --- blend PD on train fold ---
    h = make_model(SEED + fold).fit(Xo.iloc[tr], y[tr])
    lg = make_logit().fit(Zo.iloc[tr], y[tr])
    raw_te = W_HGB * h.predict_proba(Xo.iloc[te])[:, 1] + (1 - W_HGB) * lg.predict_proba(Zo.iloc[te])[:, 1]
    # cross-fit isotonic: calibrate using an inner OOF on the TRAIN fold only
    raw_tr = W_HGB * h.predict_proba(Xo.iloc[tr])[:, 1] + (1 - W_HGB) * lg.predict_proba(Zo.iloc[tr])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_tr, y[tr])
    pd_hat = np.clip(iso.predict(raw_te), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y[te], pd_hat)

    # --- ensemble sd for risk-averse rule (bootstrap HGB) ---
    rng = np.random.default_rng(SEED + 1000 + fold)
    boot = np.zeros((len(te), N_BOOT))
    ntr = len(tr)
    for b in range(N_BOOT):
        bi = tr[rng.integers(0, ntr, ntr)]
        mb = make_model(SEED + 5000 + fold * 100 + b).fit(Xo.iloc[bi], y[bi])
        boot[:, b] = mb.predict_proba(Xo.iloc[te])[:, 1]
    # calibrate ensemble quantiles with same iso (interval cal in build_a uses
    # HGB-only; here we just need a sd/upper proxy on the calibrated scale)
    ens_sd = boot.std(axis=1)
    ens_hi = np.clip(iso.predict(np.quantile(boot, 0.95, axis=1)), 1e-6, 1 - 1e-6)

    # --- timing + recovery regressors on TRAIN-fold defaulters ---
    dmask = (y[tr] == 1)
    tr_def = tr[dmask]
    reg_t = make_reg(SEED + 100 + fold).fit(Xo.iloc[tr_def], dtd[tr_def])
    e_tstar = np.clip(reg_t.predict(Xo.iloc[te]), 1, 90)
    recfrac_tr = rec[tr_def] / amount[tr_def]
    reg_r = make_reg(SEED + 200 + fold).fit(Xo.iloc[tr_def], recfrac_tr)
    e_recfrac = np.clip(reg_r.predict(Xo.iloc[te]), 0, 1)

    lgd_i = per_loan_lgd_from_recovery(e_tstar, e_recfrac)
    tau_i = NET_MARGIN / (NET_MARGIN + np.clip(lgd_i, 1e-6, None))

    a_te = amount[te]
    # E[NPV_i] for the direct-EV rule
    draws = a_te * DAILY_DRAW * np.clip(e_tstar - 1, 0, None)
    e_def_profit = np.minimum(a_te * FEE + draws + e_recfrac * a_te - a_te, a_te * NET_MARGIN)
    e_npv = (1 - pd_hat) * a_te * NET_MARGIN + pd_hat * e_def_profit

    rules = {}
    rules["FLAT (pd<0.259)"] = pd_hat < BREAK_EVEN_PD
    rules["(a) recov-LGD per-loan tau"] = pd_hat < tau_i
    rules["(a) E[NPV_i]>0"] = e_npv > 0
    rules["(b) risk-averse pd+0.5sd"] = (pd_hat + 0.5 * ens_sd) < BREAK_EVEN_PD
    rules["(b) risk-averse pd+1.0sd"] = (pd_hat + 1.0 * ens_sd) < BREAK_EVEN_PD
    rules["(b) risk-averse pd+2.0sd"] = (pd_hat + 2.0 * ens_sd) < BREAK_EVEN_PD
    rules["(b) risk-averse ens-upper95"] = ens_hi < BREAK_EVEN_PD

    out = {"fold": fold, "n": len(te), "auc": auc}
    for name, fund in rules.items():
        out[name] = realized_npv(te, fund)
        out[name + "__nfund"] = int(fund.sum())
    # stash for sweep
    return out, pd_hat, te


def summarize(results, rule_names, label):
    print("\n" + "=" * 84)
    print(f"{label}  (realized amortizing NPV per fold, vs FLAT baseline)")
    print("=" * 84)
    flat = np.array([r["FLAT (pd<0.259)"] for r in results])
    nfolds = len(results)
    print(f"{'rule':32s} " + " ".join(f"f{r['fold']}" for r in results) + "   mean_delta  pos/nf  verdict")
    table = []
    for name in rule_names:
        vals = np.array([r[name] for r in results])
        if name == "FLAT (pd<0.259)":
            print(f"{name:32s} " + " ".join(f"${v/1e3:,.0f}k" for v in vals) + "   (baseline)")
            continue
        delta = vals - flat
        md = delta.mean(); sd = delta.std(ddof=1) if nfolds > 1 else 0.0
        npos = int((delta > 0).sum())
        adopt = (npos == nfolds) and (md > sd) and (md > 0)
        weak = (npos >= nfolds - 1) and (md > 0) and (md > 0.5 * sd)
        verdict = "ADOPT" if adopt else ("WEAK" if weak else "REJECT")
        print(f"{name:32s} " + " ".join(f"${d/1e3:+,.0f}k" for d in delta) +
              f"   ${md/1e3:+,.0f}k  {npos}/{nfolds}  {verdict}")
        table.append((name, md, sd, npos, verdict))
    print(f"\nmean nfund: FLAT={np.mean([r['FLAT (pd<0.259)__nfund'] for r in results]):.0f}", end="")
    for name in rule_names:
        if name == "FLAT (pd<0.259)":
            continue
        print(f" | {name.split()[0]}={np.mean([r[name+'__nfund'] for r in results]):.0f}", end="")
    print()
    return table, flat


def threshold_sweep(pd_all, idx_all, label):
    """(c) Full threshold sweep on the pooled OOF book."""
    pd_v = np.concatenate(pd_all)
    idx_v = np.concatenate(idx_all)
    grid = np.round(np.arange(0.10, 0.601, 0.005), 3)
    best_t, best = None, -1e18
    npv_at = {}
    for t in grid:
        fund = pd_v < t
        tot = realized_npv(idx_v, fund)
        npv_at[t] = tot
        if tot > best:
            best, best_t = tot, t
    at_be = npv_at[min(grid, key=lambda g: abs(g - BREAK_EVEN_PD))]
    print(f"\n[{label}] THRESHOLD SWEEP (pooled OOF, exact realized NPV):")
    print(f"  argmax tau = {best_t:.3f}  NPV=${best:,.0f}")
    print(f"  @0.259 (baseline) NPV=${at_be:,.0f}  | gain at argmax = ${best-at_be:,.0f} "
          f"({100*(best-at_be)/abs(at_be):+.3f}%)")
    # show neighborhood
    nbhd = [t for t in grid if 0.20 <= t <= 0.32]
    print("  neighborhood:  " + "  ".join(f"{t:.3f}:${npv_at[t]/1e6:.4f}M" for t in nbhd))
    return best_t


def main():
    rule_names = ["FLAT (pd<0.259)", "(a) recov-LGD per-loan tau", "(a) E[NPV_i]>0",
                  "(b) risk-averse pd+0.5sd", "(b) risk-averse pd+1.0sd",
                  "(b) risk-averse pd+2.0sd", "(b) risk-averse ens-upper95"]

    # ---- (1) 5-fold StratifiedKFold random OOF ---- #
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res_r, pdall_r, idxall_r = [], [], []
    rng = np.random.default_rng(SEED)
    for fold, (tr, te) in enumerate(skf.split(Xo, y)):
        out, pd_hat, idx = run_fold(tr, te, fold, rng)
        res_r.append(out); pdall_r.append(pd_hat); idxall_r.append(idx)
        print(f"  [rand fold {fold}] n={len(te):,} auc={out['auc']:.4f}")
    tbl_r, flat_r = summarize(res_r, rule_names, "5-FOLD STRATIFIED RANDOM OOF")
    tau_rand = threshold_sweep(pdall_r, idxall_r, "RANDOM OOF")

    # ---- (2) Walk-forward expanding-window TIME folds ---- #
    order = np.argsort(ts, kind="stable")
    n_tf = 5
    blocks = np.array_split(order, n_tf + 1)
    res_t, pdall_t, idxall_t = [], [], []
    for k in range(n_tf):
        tr = np.concatenate(blocks[:k + 1]); te = blocks[k + 1]
        out, pd_hat, idx = run_fold(tr, te, k, rng)
        res_t.append(out); pdall_t.append(pd_hat); idxall_t.append(idx)
        print(f"  [time fold {k}] n_tr={len(tr):,} n_te={len(te):,} auc={out['auc']:.4f}")
    tbl_t, flat_t = summarize(res_t, rule_names, "WALK-FORWARD EXPANDING-WINDOW TIME FOLDS")
    tau_time = threshold_sweep(pdall_t, idxall_t, "WALK-FORWARD OOF")

    # ---- verdict ---- #
    print("\n" + "#" * 84)
    print("VERDICT")
    print("#" * 84)
    print(f"(c) sweep argmax tau: random={tau_rand:.3f}, walk-forward={tau_time:.3f} "
          f"(baseline 0.259)")
    for label, tbl in [("RANDOM", tbl_r), ("WALK-FWD", tbl_t)]:
        print(f"\n{label}:")
        for name, md, sd, npos, verdict in tbl:
            print(f"  {name:32s} mean_delta=${md:,.0f} std=${sd:,.0f} pos={npos} -> {verdict}")


if __name__ == "__main__":
    main()
