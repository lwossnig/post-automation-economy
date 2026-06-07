"""Phase 0 and Phase 1 experiments.

Phase 0  - harden the existing two-channel results:
  (a) mu_frac sweep: is the policy ranking robust to the (calibrated) rent size?
  (b) uncertainty quantification: seed dispersion on the headline metrics.
Phase 1  - the two structural fixes:
  (a) open-economy trade leak (phi): foreign ownership as a range, not a point.
  (b) AI-supply elasticity: the efficiency cost of taxing the rent.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

OUT = "figures_v3"
STEADY = slice(250, 600)
TEAL, SIENNA, SAND, BLUE, PURP = "#1d4e44", "#7c2d12", "#c2956b", "#3b6ea5", "#6b3fa0"


def _run(make, seed=0, **over):
    p = make(); p.seed = seed; p.periods = 600
    for k, v in over.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    return m


def _ss(arr, win=STEADY):
    return float(np.array(arr)[win].mean())


def _foreign_share(m):
    et = m.h_eq.sum() + m.eq_state + m.eq_row
    return m.eq_row / et if et > 0 else 0.0


def _cap_rev_over_Y(m, win=STEADY):
    Y = np.array(m.hist.Y)[win].sum()
    return (np.array(m.hist.corp_tax_rev)[win].sum()
            + np.array(m.hist.repat_revenue)[win].sum()) / Y


# ---------------------------------------------------------------- Phase 0a
def mu_sweep():
    """Foreign ownership and capital-sector revenue vs the rent size, by policy.
    Shows the ranking is robust even though the rent level is calibrated."""
    pols = [("two_channel_base", "untaxed", SIENNA),
            ("robot_tax_only", "robot tax", SAND),
            ("ai_dst", "AI digital levy", TEAL),
            ("sovereign_compute", "full toolkit", PURP)]
    mus = [0.10, 0.20, 0.30, 0.40]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, lab, col in pols:
        own, rev = [], []
        for mu in mus:
            m = _run(S.REGISTRY[name], mu_frac=mu)
            own.append(100 * _foreign_share(m))
            rev.append(100 * _cap_rev_over_Y(m))
        ax[0].plot(mus, own, "o-", color=col, label=lab, lw=1.6)
        ax[1].plot(mus, rev, "o-", color=col, label=lab, lw=1.6)
    ax[0].set_xlabel("AI rent size (mu_frac)"); ax[0].set_ylabel("foreign ownership of capital (%)")
    ax[0].set_title("Ownership rises with the rent, ranking holds")
    ax[1].set_xlabel("AI rent size (mu_frac)"); ax[1].set_ylabel("capital-sector revenue (% of output)")
    ax[1].set_title("Revenue rises with the rent, ranking holds")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/p0_mu_sweep.png", dpi=110); plt.close(fig)
    print("saved p0_mu_sweep.png")


# ---------------------------------------------------------------- Phase 0b
def uq(nseeds=16):
    """Seed dispersion (mean and 95% interval) on the headline metrics."""
    pols = [("two_channel_base", "untaxed"), ("robot_tax_only", "robot tax"),
            ("ai_dst", "AI digital levy"), ("ai_withholding", "rent withholding"),
            ("sovereign_compute", "full toolkit")]
    labels, rev_m, rev_e, own_m, own_e = [], [], [], [], []
    for name, lab in pols:
        revs, owns = [], []
        for s in range(nseeds):
            m = _run(S.REGISTRY[name], seed=s)
            revs.append(100 * _cap_rev_over_Y(m))
            owns.append(100 * _foreign_share(m))
        labels.append(lab)
        rev_m.append(np.mean(revs)); rev_e.append(1.96 * np.std(revs) / np.sqrt(nseeds))
        own_m.append(np.mean(owns)); own_e.append(1.96 * np.std(owns) / np.sqrt(nseeds))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].barh(y, rev_m, xerr=rev_e, color=TEAL, alpha=0.85, capsize=3)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels); ax[0].invert_yaxis()
    ax[0].set_xlabel("capital-sector revenue (% of output)")
    ax[0].set_title(f"Revenue (mean, 95% CI, {nseeds} seeds)")
    ax[1].barh(y, own_m, xerr=own_e, color=SIENNA, alpha=0.85, capsize=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel("foreign ownership of capital (%)")
    ax[1].set_title("Foreign ownership (mean, 95% CI)")
    for a in ax:
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/p0_uq.png", dpi=110); plt.close(fig)
    print(f"saved p0_uq.png  (rev CIs: {[round(e,2) for e in rev_e]}, own CIs: {[round(e,2) for e in own_e]})")


# ---------------------------------------------------------------- Phase 1a
def trade_sweep():
    """Foreign ownership and the capital stock vs the repatriation share phi:
    the ownership result as a range between the closed and fully-open poles."""
    phis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    own, krel = [], []
    Kref = None
    for phi in phis:
        m = _run(S.two_channel_base, trade_leak=phi)
        own.append(100 * _foreign_share(m))
        if Kref is None:
            Kref = _ss(m.hist.Y)
        krel.append(100 * _ss(m.hist.Y) / Kref)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(phis, own, "o-", color=SIENNA, lw=1.8, label="foreign ownership of capital (%)")
    ax.set_xlabel("share of the rent repatriated as goods (trade_leak, phi)")
    ax.set_ylabel("foreign ownership of capital (%)", color=SIENNA)
    ax.tick_params(axis="y", labelcolor=SIENNA)
    ax.set_ylim(0, 75); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(phis, krel, "s--", color=TEAL, lw=1.5, label="output, % of closed economy")
    ax2.set_ylabel("steady-state output (% of closed economy)", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    ax.set_title("Open economy: ownership drift is a range, not a point")
    ax.annotate("closed model\n(fully reinvested)", (0.0, own[0]), textcoords="offset points",
                xytext=(8, -28), fontsize=8, color=SIENNA)
    ax.annotate("fully repatriated\n(no equity bought)", (1.0, own[-1]), textcoords="offset points",
                xytext=(-70, 18), fontsize=8, color=SIENNA)
    fig.tight_layout(); fig.savefig(f"{OUT}/p1_trade_sweep.png", dpi=110); plt.close(fig)
    print(f"saved p1_trade_sweep.png  (ownership {own[0]:.0f}% -> {own[-1]:.0f}%)")


# ---------------------------------------------------------------- Phase 1b
def ai_supply_sweep():
    """Output level and revenue vs the AI-supply elasticity, under the DST +
    withholding: taxing the rent is no longer free."""
    etas = [0.0, 0.1, 0.2, 0.3, 0.5]
    Yb = _ss(_run(S.two_channel_base).hist.Y)
    ylev, rev = [], []
    for eta in etas:
        m = _run(S.two_channel_base, dst_ai=0.10, tax_repat=0.30, ai_supply_elasticity=eta)
        ylev.append(100 * _ss(m.hist.Y) / Yb)
        rev.append(100 * _cap_rev_over_Y(m))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(etas, ylev, "o-", color=TEAL, lw=1.8)
    ax.set_xlabel("AI-supply response to the net-of-tax rent (ai_supply_elasticity)")
    ax.set_ylabel("steady-state output (% of untaxed)", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL); ax.grid(alpha=0.3); ax.set_ylim(70, 102)
    ax2 = ax.twinx()
    ax2.plot(etas, rev, "s--", color=SIENNA, lw=1.5)
    ax2.set_ylabel("capital-sector revenue (% of output)", color=SIENNA)
    ax2.tick_params(axis="y", labelcolor=SIENNA)
    ax.set_title("Taxing the rent is not free once AI deployment responds")
    fig.tight_layout(); fig.savefig(f"{OUT}/p1_ai_supply_sweep.png", dpi=110); plt.close(fig)
    print(f"saved p1_ai_supply_sweep.png  (output {ylev[0]:.0f}% -> {ylev[-1]:.0f}% of untaxed)")


if __name__ == "__main__":
    mu_sweep()
    uq(16)
    trade_sweep()
    ai_supply_sweep()
