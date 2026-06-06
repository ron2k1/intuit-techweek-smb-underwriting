#!/usr/bin/env python3
"""LEVER: how to treat the prior-DECLINED (unlabeled) region.

Compare three funding policies on EXPECTED NPV (val+test, amortizing per-dollar
EV with LGD=0.30) and stress-test under a pessimistic floor on the declined
region's true PD.

Policies (break-even PD = NET_MARGIN/(NET_MARGIN+LGD) ~= 0.226):
  conservative : fund only prior-APPROVED rows with point pd < 0.226.
  gated        : conservative PLUS prior-DECLINED rows whose conformal
                 UPPER-90 pd < 0.226 (trust-but-verify).
  trust        : fund ANY row (approved or declined) with point pd < 0.226.

Stress test (pessimistic): floor the DECLINED region's true PD at the empirical
cutoff-edge rate 0.245 when computing EV for declined-funded loans (approved EV
unchanged). This penalizes policies that lean on the unlabeled region.

Realized validation profit is ONLY measurable on prior-APPROVED matured rows
(declined are unlabeled), so we report realized $ for the approved-funded subset
separately and explicitly. We do NOT tune any threshold on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
PREDS = REPO / "reports" / "a_predictions.csv"
DATA = REPO / "dataset"

TERM_INT = 0.35 * 60 / 365          # ~0.057534
FEE = 0.03
NET_MARGIN = TERM_INT + FEE         # ~0.087534
LGD = 0.30
BREAK_EVEN = NET_MARGIN / (NET_MARGIN + LGD)   # ~0.2259
EDGE_PD = 0.245                     # empirical cutoff-edge rate (pessimistic floor)


def ev_per_dollar(pd_arr, lgd=LGD):
    """Amortizing expected profit per dollar of principal."""
    return (1 - pd_arr) * NET_MARGIN - pd_arr * lgd


def realized_profit(amt, dflag, dtd, rec):
    """Realized $ profit per loan (only valid where outcomes are observed)."""
    rec = np.nan_to_num(rec)
    frac = np.clip(np.minimum(np.nan_to_num(dtd), 60) / 60.0, 0, 1)
    default_profit = amt * (FEE + frac * (1 + TERM_INT) - 1) + rec
    default_profit = np.minimum(default_profit, amt * NET_MARGIN)
    return np.where(dflag == 0, amt * NET_MARGIN, default_profit)


def main() -> int:
    df = pd.read_csv(PREDS)
    amt = df["requested_amount"].to_numpy(float)
    pd_p = df["predicted_pd"].to_numpy(float)
    pd_u = df["pd_upper_90"].to_numpy(float)
    declined = (df["prior_decision"] == 0).to_numpy()
    approved = ~declined

    print("=" * 74)
    print("SETUP")
    print("=" * 74)
    print(f"  total rows (val+test) ........ {len(df):,}")
    print(f"  prior-APPROVED ............... {approved.sum():,}")
    print(f"  prior-DECLINED (unlabeled) ... {declined.sum():,}")
    print(f"  labeled rows (approved+matured): {df['default_flag'].notna().sum():,}")
    print(f"  break-even PD ................ {BREAK_EVEN:.4f}")
    print(f"  pessimistic declined floor ... {EDGE_PD:.4f}")

    # ---- funded masks per policy ----
    funded = {
        "conservative": approved & (pd_p < BREAK_EVEN),
        "gated":        (approved & (pd_p < BREAK_EVEN)) | (declined & (pd_u < BREAK_EVEN)),
        "trust":        pd_p < BREAK_EVEN,
    }

    # ---- EXPECTED NPV (model PD) and PESSIMISTIC NPV (declined PD floored) ----
    ev_model = ev_per_dollar(pd_p)
    pd_pess = pd_p.copy()
    pd_pess[declined] = np.maximum(pd_pess[declined], EDGE_PD)
    ev_pess = ev_per_dollar(pd_pess)

    rows = []
    for name in ["conservative", "gated", "trust"]:
        m = funded[name]
        n = int(m.sum())
        n_app = int((m & approved).sum())
        n_dec = int((m & declined).sum())
        principal = amt[m].sum()
        npv_exp = (ev_model[m] * amt[m]).sum()
        npv_pess = (ev_pess[m] * amt[m]).sum()
        rows.append((name, n, n_app, n_dec, principal, npv_exp, npv_pess))

    print("\n" + "=" * 74)
    print("EXPECTED NPV vs PESSIMISTIC-FLOOR NPV (val+test, per-dollar amortizing EV)")
    print("=" * 74)
    print(f"  {'policy':<13}{'funded':>7}{'#app':>7}{'#dec':>7}"
          f"{'principal':>14}{'EXP NPV':>14}{'PESS NPV':>14}")
    cons_exp = cons_pess = None
    for name, n, na, nd, pr, ne, npe in rows:
        if name == "conservative":
            cons_exp, cons_pess = ne, npe
        print(f"  {name:<13}{n:>7,}{na:>7,}{nd:>7,}{pr:>14,.0f}{ne:>14,.0f}{npe:>14,.0f}")

    print("\n  -- deltas vs conservative --")
    print(f"  {'policy':<13}{'dEXP NPV':>14}{'dPESS NPV':>14}{'robust?':>10}")
    for name, n, na, nd, pr, ne, npe in rows:
        d_exp = ne - cons_exp
        d_pess = npe - cons_pess
        robust = "yes" if npe >= cons_pess - 1e-6 else "NO"
        print(f"  {name:<13}{d_exp:>14,.0f}{d_pess:>14,.0f}{robust:>10}")

    # ---- REALIZED profit on the APPROVED-matured funded subset (measurable only) ----
    print("\n" + "=" * 74)
    print("REALIZED $ on APPROVED-MATURED funded rows ONLY (declined are UNLABELED)")
    print("=" * 74)
    lab = df["default_flag"].notna().to_numpy()
    rp = realized_profit(
        amt, df["default_flag"].to_numpy(),
        df["days_to_default"].to_numpy() if "days_to_default" in df else np.zeros(len(df)),
        df["final_recovered_amount"].to_numpy(),
    )
    print(f"  {'policy':<13}{'app-mat funded':>16}{'realized $':>16}{'$/loan':>10}")
    for name in ["conservative", "gated", "trust"]:
        m = funded[name] & approved & lab   # approved funded set is identical across policies
        n = int(m.sum())
        tot = rp[m].sum()
        per = tot / max(n, 1)
        print(f"  {name:<13}{n:>16,}{tot:>16,.0f}{per:>10,.0f}")
    print("  NOTE: the approved-funded set is IDENTICAL across all three policies,")
    print("        so realized profit cannot distinguish them. The only difference")
    print("        is the declined-region funding, which is unobservable.")

    # ---- declined-region diagnostic: what point/upper PD do we admit? ----
    print("\n" + "=" * 74)
    print("DECLINED-REGION ADMISSION DIAGNOSTIC")
    print("=" * 74)
    g_dec = declined & (pd_u < BREAK_EVEN)
    t_dec = declined & (pd_p < BREAK_EVEN)
    for tag, mk in [("gated declined-admitted", g_dec), ("trust declined-admitted", t_dec)]:
        if mk.sum():
            print(f"  {tag:<26} n={mk.sum():>5,}  "
                  f"mean point pd={pd_p[mk].mean():.3f}  mean upper90={pd_u[mk].mean():.3f}  "
                  f"principal=${amt[mk].sum():,.0f}")
        else:
            print(f"  {tag:<26} n=0")

    # break-even check: at EDGE_PD the true EV per dollar
    print(f"\n  EV/$ at point break-even ({BREAK_EVEN:.3f}): "
          f"{ev_per_dollar(BREAK_EVEN):+.5f} (=0 by construction)")
    print(f"  EV/$ at pessimistic floor ({EDGE_PD:.3f}): "
          f"{ev_per_dollar(EDGE_PD):+.5f}  <- declined-funded loans realize THIS if floor is true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
