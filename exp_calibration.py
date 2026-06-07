"""Calibration and sensitivity pass over the parameters added in Phases 1-2.

Each new parameter is given a literature-anchored central value and a plausible
range (see the calibration table in the report). This script runs a one-at-a-time
sensitivity sweep around the calibrated centre, on a fully-instrumented economy
(the toolkit plus reinstatement and unemployment all on), and reports how far the
key outcomes move as each parameter spans its range. A short bar means the result
is robust to that parameter; a long bar means the magnitude leans on it.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

OUT = "figures_v3"
TEAL, SIENNA, SAND, BLUE, PURP = "#1d4e44", "#7c2d12", "#c2956b", "#3b6ea5", "#6b3fa0"
W = slice(250, 600)

# central calibration and plausible ranges (anchors documented in the report)
CENTRAL = dict(mu_frac=0.25, trade_leak=0.5, ai_supply_elasticity=0.5,
               reinstate_frac=0.5, unemployment_pass_through=0.3,
               unemployment_benefit=0.5)
RANGES = dict(mu_frac=(0.15, 0.35), trade_leak=(0.3, 0.7),
              ai_supply_elasticity=(0.25, 1.0), reinstate_frac=(0.3, 0.7),
              unemployment_pass_through=(0.1, 0.5), unemployment_benefit=(0.37, 0.58))
LABEL = dict(mu_frac="AI rent size (mu_frac)", trade_leak="repatriation share (phi)",
             ai_supply_elasticity="AI-supply elasticity (eta)",
             reinstate_frac="reinstatement (rho)",
             unemployment_pass_through="unemployment pass-through",
             unemployment_benefit="benefit replacement rate")
# fixed instruments so every channel is active
FIXED = dict(dst_ai=0.10, tax_repat=0.30)


def _run(**over):
    p = S.two_channel_base(); p.seed = 0; p.periods = 600
    cfg = {**CENTRAL, **FIXED, **over}
    for k, v in cfg.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(600):
        m.step()
    return m


def _out(m):
    et = m.h_eq.sum() + m.eq_state + m.eq_row
    return dict(
        own=100 * m.eq_row / et,
        rev=100 * (np.array(m.hist.corp_tax_rev)[W].sum()
                   + np.array(m.hist.repat_revenue)[W].sum()) / np.array(m.hist.Y)[W].sum(),
        lshare=float((np.array(m.hist.w_Lr)[W] + np.array(m.hist.w_Lc)[W]).sum()
                     / np.array(m.hist.Y)[W].sum()),
        unemp=100 * float(np.array(m.hist.unemployment)[W].mean()),
    )


def run():
    base = _out(_run())
    params = list(RANGES)
    swings = {k: {} for k in ("own", "rev", "lshare", "unemp")}
    for pr in params:
        lo, hi = RANGES[pr]
        olo, ohi = _out(_run(**{pr: lo})), _out(_run(**{pr: hi}))
        for o in swings:
            swings[o][pr] = (olo[o], ohi[o])
    titles = dict(own="Foreign ownership of capital (%)", rev="Capital-sector revenue (% of output)",
                  lshare="Labour share", unemp="Unemployment rate (%)")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    cols = {"own": SIENNA, "rev": TEAL, "lshare": PURP, "unemp": SAND}
    for ax, o in zip(axes.flat, ("own", "rev", "lshare", "unemp")):
        order = sorted(params, key=lambda k: abs(swings[o][k][1] - swings[o][k][0]))
        y = np.arange(len(order)); b = base[o]
        for i, pr in enumerate(order):
            lo, hi = swings[o][pr]
            ax.barh(i, hi - lo, left=min(lo, hi), height=0.6, color=cols[o], alpha=0.85)
        ax.axvline(b, color="black", lw=1.0, ls="--", label="central")
        ax.set_yticks(y); ax.set_yticklabels([LABEL[k] for k in order], fontsize=8)
        ax.set_title(titles[o], fontsize=10); ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Sensitivity of the headline outcomes to the calibrated parameter ranges",
                 fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/calib_tornado.png", dpi=110); plt.close(fig)
    print("saved calib_tornado.png")
    print(f"central: own={base['own']:.1f}%  rev={base['rev']:.2f}%  lshare={base['lshare']:.3f}  unemp={base['unemp']:.1f}%")
    for o in ("own", "rev", "lshare", "unemp"):
        widths = {k: abs(swings[o][k][1] - swings[o][k][0]) for k in params}
        top = max(widths, key=widths.get)
        print(f"  {o}: most sensitive to {top} (swing {widths[top]:.2f})")


if __name__ == "__main__":
    run()
