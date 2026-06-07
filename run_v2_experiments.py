"""Run the full v2 experiment battery and write figures + a results table.

    python run_v2_experiments.py

Each experiment maps to a hypothesis in the plan (H1..H6) plus robustness.
Levels are scaled by per-capita output where noted; ratios/shares are scale-free.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm_sfc.model_v2 import ModelV2, ParamsV2
from abm_sfc import scenarios_v2 as sv2
from abm_sfc.mc import replicate, summarise

OUT = "figures_v2"
os.makedirs(OUT, exist_ok=True)
NSEED = 12
RESULTS = []


def log(msg):
    print(msg)
    RESULTS.append(msg)


# ---------------------------------------------------------------------------
# Experiment 1 (H1, H2, H3): scenario comparison with Monte Carlo bands
# ---------------------------------------------------------------------------
def exp_scenarios():
    log("\n" + "=" * 74)
    log("EXPERIMENT 1  Scenario comparison (H1/H2/H3), %d seeds" % NSEED)
    log("=" * 74)
    log(f"{'scenario':20s}{'Gini':>14s}{'top1%':>14s}{'gov_nw/Yfin':>14s}")
    traj = {}
    for name, fn in sv2.REGISTRY.items():
        p = fn()
        m = replicate(p, NSEED)
        s = summarise(m)
        # one representative trajectory for plotting
        h = ModelV2(replace(p, seed=0)).run()
        traj[name] = h
        yf = h.Y[-1] / p.n_agents
        log(f"{name:20s}{s['gini'][0]:7.3f}[{s['gini'][1]:.2f},{s['gini'][2]:.2f}]"
            f"{s['top1'][0]*100:7.1f}[{s['top1'][1]*100:4.1f},{s['top1'][2]*100:4.1f}]"
            f"{s['gov_nw'][0]/p.n_agents/yf:14.2f}")
    # comparison plot
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("v2 scenario comparison", fontweight="bold")
    for name, h in traj.items():
        n = sv2.REGISTRY[name]().n_agents
        ax[0, 0].plot(h.gini, label=name)
        ax[0, 1].plot(np.array(h.top1_share) * 100, label=name)
        ax[1, 0].plot(np.array(h.gov_nw) / n / (np.array(h.Y) / n), label=name)
        ax[1, 1].plot(np.array(h.labour_share), label=name)
    ax[0, 0].set_title("Wealth Gini")
    ax[0, 1].set_title("Top 1% share (%)")
    ax[1, 0].set_title("Government net worth / output")
    ax[1, 1].set_title("Labour share")
    for a in ax.flat:
        a.set_xlabel("period"); a.grid(alpha=0.3); a.legend(fontsize=7); a.axhline(0, color="k", lw=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/exp1_scenarios.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2 (H2): concentration speed vs automation level, coupled model
# ---------------------------------------------------------------------------
def exp_speed():
    log("\n" + "=" * 74)
    log("EXPERIMENT 2  Concentration speed vs automation level (H2), kappa=1")
    log("=" * 74)
    log(f"{'auto_max':>9}{'final Gini':>12}{'t_half':>9}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for am in [0.0, 0.15, 0.30, 0.45]:
        gs = []
        for s in range(NSEED):
            p = replace(sv2.coupled(), auto_max=am, periods=300, seed=s)
            gs.append(ModelV2(p).run().gini)
        g = np.array(gs).mean(0)
        final = g[-40:].mean(); g0 = g[:20].mean()
        half = g0 + 0.5 * (final - g0)
        c = int(np.argmax(g >= half)) if final > g0 else -1
        log(f"{am:9.2f}{final:12.3f}{c:9d}")
        ax.plot(g, label=f"auto_max={am} (I->{0.5+am:.2f})")
    ax.axvline(80, ls="--", color="grey", lw=0.8, label="automation onset")
    ax.set_title("Coupled model: concentration tracks automation level")
    ax.set_xlabel("period"); ax.set_ylabel("Gini"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2_speed.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3 (H1): r vs g and fiscal sustainability frontier
# ---------------------------------------------------------------------------
def exp_rg():
    log("\n" + "=" * 74)
    log("EXPERIMENT 3  r-vs-g and fiscal sustainability (H1)")
    log("=" * 74)
    # sweep required return (shifts steady-state r) x tax regime; report terminal gov nw
    regimes = {"income_tax": dict(tax_corp=0.25, tax_income=0.30, ubi=sv2.UBI),
               "wealth_tax": dict(tax_corp=0.25, tax_income=0.15, tax_wealth=0.02, ubi=sv2.UBI)}
    log(f"{'r_required':>11}{'regime':>13}{'mean r_net':>12}{'mean g':>9}{'gov_nw/Y':>11}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for rr in [0.02, 0.04, 0.06]:
        for rname, cfg in regimes.items():
            gn = []
            rnet = []; grate = []
            for s in range(NSEED):
                p = replace(sv2.BASE, r_required=rr, periods=300, seed=s, **cfg)
                h = ModelV2(p).run()
                yf = h.Y[-1] / p.n_agents
                gn.append(h.gov_nw[-1] / p.n_agents / yf)
                rnet.append(np.mean(h.r_net[-40:])); grate.append(np.mean(h.g_rate[-40:]))
            log(f"{rr:11.2f}{rname:>13}{np.mean(rnet):12.3f}{np.mean(grate):9.3f}{np.mean(gn):11.1f}")
    # illustrate r and g trajectories for the base case
    h = ModelV2(replace(sv2.income_tax_ubi(), seed=0)).run()
    ax.plot(h.r_net, label="r (net return on capital)")
    ax.plot(h.g_rate, label="g (capital growth rate)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Endogenous r and g over the automation transition")
    ax.set_xlabel("period"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp3_rg.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 4 (H1/H2): elasticity of substitution sweep
# ---------------------------------------------------------------------------
def exp_eps():
    log("\n" + "=" * 74)
    log("EXPERIMENT 4  Elasticity of substitution sweep (H1/H2)")
    log("=" * 74)
    log(f"{'eps':>6}{'final labour_share':>20}{'wage rel t0':>14}{'final Gini':>12}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for eps in [0.5, 0.8, 1.5, 2.0]:
        ls = []; wg = []; gi = []
        rep_h = None
        for s in range(NSEED):
            p = replace(sv2.laissez_faire(), eps=eps, periods=300, seed=s)
            h = ModelV2(p).run()
            if rep_h is None:
                rep_h = h
            ls.append(h.labour_share[-1]); wg.append(h.wage[-1] / h.wage[0]); gi.append(np.mean(h.gini[-40:]))
        log(f"{eps:6.1f}{np.mean(ls):20.3f}{np.mean(wg):14.3f}{np.mean(gi):12.3f}")
        ax[0].plot(rep_h.labour_share, label=f"eps={eps}")
        ax[1].plot(np.array(rep_h.wage) / rep_h.wage[0], label=f"eps={eps}")
    ax[0].set_title("Labour share vs elasticity"); ax[1].set_title("Wage (relative to t0) vs elasticity")
    for a in ax: a.set_xlabel("period"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp4_eps.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 5 (H3): foreign-ownership leakage sweep
# ---------------------------------------------------------------------------
def exp_foreign():
    log("\n" + "=" * 74)
    log("EXPERIMENT 5  Foreign-ownership leakage (H3)")
    log("=" * 74)
    log(f"{'foreign share':>14}{'RoW_nw/Y':>11}{'gov_nw/Y':>11}")
    fig, ax = plt.subplots(figsize=(8, 5))
    shares = [0.0, 0.2, 0.4, 0.6, 0.8]
    leak = []; gov = []
    for fs in shares:
        rn = []; gn = []
        for s in range(NSEED):
            p = replace(sv2.BASE, own_households=1.0 - fs, own_row=fs,
                        tax_corp=0.25, tax_income=0.30, ubi=sv2.UBI, periods=300, seed=s)
            h = ModelV2(p).run()
            yf = h.Y[-1] / p.n_agents
            rn.append(h.row_nw[-1] / p.n_agents / yf); gn.append(h.gov_nw[-1] / p.n_agents / yf)
        leak.append(np.mean(rn)); gov.append(np.mean(gn))
        log(f"{fs:14.2f}{np.mean(rn):11.1f}{np.mean(gn):11.1f}")
    ax.plot(shares, leak, "o-", label="RoW net worth / output (leakage)")
    ax.plot(shares, gov, "s-", label="government net worth / output")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Foreign ownership: wealth leakage and fiscal erosion")
    ax.set_xlabel("foreign ownership share"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp5_foreign.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 6 (H4): buyer collapse with endogenous portfolio choice
# ---------------------------------------------------------------------------
def exp_buyer():
    log("\n" + "=" * 74)
    log("EXPERIMENT 6  Equity-purchase dynamics under automation (H4, revised)")
    log("=" * 74)
    h = ModelV2(replace(sv2.income_tax_ubi(), seed=0)).run()
    ep = np.array(h.eq_purchase_house)
    wage = np.array(h.wage)
    g = np.array(h.g_rate)
    log(f"peak household equity purchase = {ep.max():.1f} at period {int(np.argmax(ep))}")
    log(f"late-period (last 40) mean purchase = {ep[-40:].mean():.3f}")
    log("NOTE: with endogenous capital deepening the wage per unit RISES, so the")
    log("v1 affordability-driven buyer collapse does NOT bind; equity purchases")
    log("instead track net investment demand (they end when the boom ends).")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, color="tab:blue", label="household equity purchases")
    ax1.set_xlabel("period"); ax1.set_ylabel("equity purchases", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(g, color="tab:green", label="capital growth rate g")
    ax2.plot(wage / wage[0] / 100.0, color="tab:red", lw=0.8, label="wage rel t0 (/100)")
    ax2.set_ylabel("g  and  scaled wage", color="tab:green")
    ax1.set_title("Equity purchases track net investment, not a vanishing wage")
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp6_buyer.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 7 (H5): policy frontier (inequality vs solvency)
# ---------------------------------------------------------------------------
def exp_frontier():
    log("\n" + "=" * 74)
    log("EXPERIMENT 7  Policy frontier: inequality vs solvency (H5)")
    log("=" * 74)
    log(f"{'policy':20s}{'Gini':>9}{'gov_nw/Y':>11}")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["laissez_faire", "income_tax_ubi", "wealth_tax_ubi",
                 "progressive_wealth", "state_ownership", "citizens_fund", "foreign_ownership"]:
        p = sv2.REGISTRY[name]()
        gi = []; gn = []
        for s in range(NSEED):
            h = ModelV2(replace(p, seed=s)).run()
            yf = h.Y[-1] / p.n_agents
            gi.append(np.mean(h.gini[-40:])); gn.append(h.gov_nw[-1] / p.n_agents / yf)
        log(f"{name:20s}{np.mean(gi):9.3f}{np.mean(gn):11.1f}")
        ax.scatter(np.mean(gi), np.mean(gn), s=80)
        ax.annotate(name, (np.mean(gi), np.mean(gn)), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("wealth Gini (lower = more equal)")
    ax.set_ylabel("government net worth / output (higher = more solvent)")
    ax.set_title("Policy frontier: inequality vs fiscal position")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp7_frontier.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 8 (H6): intervention timing / hysteresis
# ---------------------------------------------------------------------------
def exp_timing():
    log("\n" + "=" * 74)
    log("EXPERIMENT 8  Intervention timing / hysteresis (H6)")
    log("=" * 74)
    log(f"{'wealth-tax onset':>17}{'final Gini':>12}")
    fig, ax = plt.subplots(figsize=(8, 5))
    base = sv2.coupled(); base = replace(base, kappa=1.0, periods=320)
    for onset in [60, 110, 160, 210]:
        gs = []
        for s in range(NSEED):
            # wealth tax switches on at `onset` by raising tax_wealth via a wrapper
            p = replace(base, seed=s, tax_wealth=0.0)
            m = ModelV2(p)
            for t in range(p.periods):
                m.p.tax_wealth = 0.03 if t >= onset else 0.0
                m.step()
            gs.append(m.hist.gini)
        g = np.array(gs).mean(0)
        log(f"{onset:17d}{g[-40:].mean():12.3f}")
        ax.plot(g, label=f"intervene at t={onset}")
    ax.axvline(80, ls="--", color="grey", lw=0.8, label="automation onset")
    ax.set_title("Intervention timing: earlier wealth tax, lower terminal inequality")
    ax.set_xlabel("period"); ax.set_ylabel("Gini"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp8_timing.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 9 (robustness): finite-size sweep
# ---------------------------------------------------------------------------
def exp_size():
    log("\n" + "=" * 74)
    log("EXPERIMENT 9  Finite-size robustness")
    log("=" * 74)
    log(f"{'N':>6}{'Gini mean':>11}{'Gini sd':>9}{'top1% mean':>12}{'top1% sd':>10}")
    for n in [250, 500, 1000, 2000, 4000]:
        gm = []; tm = []
        for s in range(NSEED):
            p = replace(sv2.laissez_faire(), n_agents=n, periods=250, seed=s)
            h = ModelV2(p).run()
            gm.append(np.mean(h.gini[-40:])); tm.append(np.mean(h.top1_share[-40:]))
        log(f"{n:6d}{np.mean(gm):11.3f}{np.std(gm):9.4f}{np.mean(tm)*100:12.2f}{np.std(tm)*100:10.3f}")


if __name__ == "__main__":
    exp_scenarios()
    exp_speed()
    exp_rg()
    exp_eps()
    exp_foreign()
    exp_buyer()
    exp_frontier()
    exp_timing()
    exp_size()
    with open(f"{OUT}/results.txt", "w") as f:
        f.write("\n".join(RESULTS))
    print("\nAll experiments complete. Figures in", OUT)
