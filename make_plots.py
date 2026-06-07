"""Generate per-scenario diagnostic panels and a cross-scenario comparison.

    python make_plots.py

Writes PNGs to ./figures/. Each quantity that is a level is scaled by output
(mean per-capita output = 1), so values read as multiples of one period's
output.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm_sfc import Model
from abm_sfc import scenarios

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

TITLES = {
    "laissez_faire": "Laissez-faire (private ownership, no redistribution)",
    "income_tax_ubi": "Income + corporate tax funding UBI",
    "wealth_tax_ubi": "Wealth (stock) tax funding UBI",
    "state_ownership": "State majority ownership (60%)",
    "foreign_ownership": "Foreign majority ownership (60%)",
}


def run_all():
    results = {}
    for name, fn in scenarios.REGISTRY.items():
        p = fn()
        results[name] = (p, Model(p).run())
    return results


def panel(name, p, h):
    n = p.n_agents
    t = np.arange(len(h.gini))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(TITLES[name], fontsize=14, fontweight="bold")

    ax[0, 0].plot(t, h.alpha, color="tab:purple")
    ax[0, 0].set_title("Automation index $\\alpha_t$")
    ax[0, 0].set_ylim(-0.02, 1.02)

    ax[0, 1].plot(t, h.labour_share, color="tab:blue")
    ax[0, 1].set_title("Labour share of output")
    ax[0, 1].set_ylim(-0.02, 0.75)

    ax[0, 2].plot(t, np.array(h.top1_share) * 100, color="tab:red")
    ax[0, 2].set_title("Top 1% wealth share (%)")
    ax[0, 2].set_ylim(0, max(25, max(h.top1_share) * 100 * 1.1))

    ax[1, 0].plot(t, h.gini, color="tab:orange")
    ax[1, 0].set_title("Wealth Gini")
    ax[1, 0].set_ylim(0, 0.7)

    ax[1, 1].plot(t, np.array(h.house_nw) / n, label="households", color="tab:blue")
    ax[1, 1].plot(t, np.array(h.gov_nw) / n, label="government", color="tab:green")
    ax[1, 1].plot(t, np.array(h.row_nw) / n, label="rest of world", color="tab:red")
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set_title("Sector net worth (multiples of output)")
    ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(t, np.array(h.gov_balance) / n, color="tab:green")
    ax[1, 2].axhline(0, color="k", lw=0.6)
    ax[1, 2].set_title("Government primary balance / output")

    for a in ax.flat:
        a.set_xlabel("period")
        a.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def comparison(results):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Cross-scenario comparison", fontsize=14, fontweight="bold")
    for name, (p, h) in results.items():
        n = p.n_agents
        t = np.arange(len(h.gini))
        ax[0, 0].plot(t, h.gini, label=name)
        ax[0, 1].plot(t, np.array(h.top1_share) * 100, label=name)
        ax[1, 0].plot(t, np.array(h.gov_nw) / n, label=name)
        ax[1, 1].plot(t, np.array(h.row_nw) / n, label=name)
    ax[0, 0].set_title("Wealth Gini")
    ax[0, 1].set_title("Top 1% wealth share (%)")
    ax[1, 0].set_title("Government net worth / output")
    ax[1, 1].set_title("Rest-of-world net worth / output (wealth leakage)")
    for a in ax.flat:
        a.set_xlabel("period")
        a.grid(alpha=0.3)
        a.axhline(0, color="k", lw=0.5)
        a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, "comparison.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    res = run_all()
    paths = [panel(name, p, h) for name, (p, h) in res.items()]
    paths.append(comparison(res))
    print("\n".join(paths))
