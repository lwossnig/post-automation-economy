"""Regenerate the figures affected by the hardening (frontier + composition).

The composition figure is new: it shows the domestic vs true (offshore-inclusive)
Gini across wealth-tax rates, the corrected and calibrated version of the finding.
"""
import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as sv3

OUT = "figures_v3"
os.makedirs(OUT, exist_ok=True)
NSEED = 10        # canonical seed count, matching run_v3_experiments.py and the captions


def mseed(p, keys):
    acc = {k: [] for k in keys}
    for s in range(NSEED):
        h = ModelV3(replace(p, seed=s)).run()
        for k in keys:
            series = getattr(h, k)
            acc[k].append(np.mean(series[-30:]) if k not in ("gov_nw",) else series[-1])
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---- expA: behavioural policy frontier (regenerated) ----
def frontier():
    order = ["laissez_faire", "income_tax_ubi", "wealth_tax_ubi", "wealth_tax_frictionless",
             "progressive_wealth", "state_ownership", "citizens_fund", "foreign_ownership"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for nm in order:
        p = sv3.REGISTRY[nm]()
        gv = []; gi = []
        for s in range(NSEED):
            h = ModelV3(replace(p, seed=s)).run()
            yf = h.Y[-1] / p.n_agents
            gv.append(h.gov_nw[-1] / p.n_agents / yf)
            gi.append(np.mean(h.gini[-30:]))
        mk = "D" if nm == "wealth_tax_frictionless" else "o"
        ax.scatter(np.mean(gi), np.mean(gv), s=90, marker=mk, zorder=3,
                   label=nm.replace("_", " "))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("domestic wealth Gini (lower = more equal)")
    ax.set_ylabel("government net worth / output (higher = more solvent)")
    ax.set_title("Behavioural policy frontier: equality versus solvency")
    ax.legend(loc="center", fontsize=9, framealpha=0.92, title="policy regime",
              ncol=2, borderpad=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/expA_frontier.png", dpi=110); plt.close(fig)
    print("expA_frontier.png")


# ---- expB: corrected composition effect (domestic vs true Gini across tax rates) ----
def composition():
    rates = [0.0, 0.01, 0.02, 0.03, 0.05]
    dom = []; true = []; off = []
    for tw in rates:
        p = replace(sv3.wealth_tax_ubi(), tax_wealth=tw)
        dd = []; tt = []; oo = []
        for s in range(NSEED):
            h = ModelV3(replace(p, seed=s)).run()
            dd.append(np.mean(h.gini[-30:]))
            tt.append(np.mean(h.gini_true[-30:]))
            oo.append(np.mean(h.offshore_share[-30:]) * 100)
        dom.append(np.mean(dd)); true.append(np.mean(tt)); off.append(np.mean(oo))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.array(rates) * 100
    ax[0].plot(x, dom, "o-", color="#1d4e44", label="domestic Gini (measured)")
    ax[0].plot(x, true, "s--", color="#7c2d12", label="true Gini (incl. offshore)")
    ax[0].fill_between(x, dom, true, color="#e8c9b8", alpha=0.5, label="composition wedge")
    ax[0].set_xlabel("wealth-tax rate (%)"); ax[0].set_ylabel("wealth Gini")
    ax[0].set_title("Domestic vs true inequality"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(x, off, "o-", color="#7c2d12")
    ax[1].set_xlabel("wealth-tax rate (%)"); ax[1].set_ylabel("offshore share of household wealth (%)")
    ax[1].set_title("Capital flight scales with the rate (~2% per pp)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Composition effect: the wedge between measured and true inequality (calibrated)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/expB_composition.png", dpi=110); plt.close(fig)
    print("expB_composition.png   domestic:", np.round(dom, 3), " true:", np.round(true, 3),
          " offshore%:", np.round(off, 1))


if __name__ == "__main__":
    frontier()
    composition()
    print("done")
