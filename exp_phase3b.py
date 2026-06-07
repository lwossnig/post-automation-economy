"""Phase 3: endogenous markup via market contestability.

The AI rent is durable only if the frontier stays monopolistic. As competition
(open-weight catch-up, more providers) erodes the markup, the rent shrinks. This
makes competition policy an alternative lever to taxation, but a different one:
competition destroys the rent (and with it the tax base), while taxation captures
it. The two are substitutes for the ownership goal and opposites for the fiscal one.
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


def _run(periods=600, **o):
    p = S.two_channel_base(); p.seed = 0; p.periods = periods
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(periods):
        m.step()
    return m


def _own(m):
    et = m.h_eq.sum() + m.eq_state + m.eq_row
    return 100 * m.eq_row / et


def _revY(m):
    return 100 * (np.array(m.hist.corp_tax_rev)[W].sum()
                  + np.array(m.hist.repat_revenue)[W].sum()) / np.array(m.hist.Y)[W].sum()


def fig():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel 1: competition sweep (untaxed) - rent, ownership
    kappas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    own, rentshare = [], []
    for k in kappas:
        m = _run(competition=k)
        own.append(_own(m))
        rentshare.append(100 * 0.25 * (1 - k))     # mu_eff as % of cognitive value added
    x = [100 * k for k in kappas]
    ax[0].plot(x, own, "o-", color=SIENNA, lw=1.9, label="foreign ownership of capital")
    ax[0].plot(x, rentshare, "s--", color=PURP, lw=1.6, label="effective rent share (mu_eff)")
    ax[0].set_xlabel("market contestability (competition, %)")
    ax[0].set_ylabel("per cent")
    ax[0].set_title("Competition erodes the rent, and with it foreign ownership")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="upper right")
    # panel 2: the two levers in (ownership, revenue) space
    tax_own, tax_rev = [], []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        m = _run(dst_ai=w / 2, tax_repat=w / 2)
        tax_own.append(_own(m)); tax_rev.append(_revY(m))
    comp_own, comp_rev = [], []
    for k in kappas:
        m = _run(competition=k, dst_ai=0.05, tax_repat=0.05)   # a light fixed levy, so revenue is visible
        comp_own.append(_own(m)); comp_rev.append(_revY(m))
    ax[1].plot(tax_own, tax_rev, "o-", color=TEAL, lw=1.9, label="taxation (raise the wedge)")
    ax[1].plot(comp_own, comp_rev, "s-", color=SIENNA, lw=1.9, label="competition (raise contestability)")
    ax[1].annotate("captures the rent\n(revenue up)", (tax_own[-1], tax_rev[-1]),
                   textcoords="offset points", xytext=(8, -6), fontsize=8, color=TEAL)
    ax[1].annotate("no rent left to capture\n(revenue stays low)", (comp_own[-1], comp_rev[-1]),
                   textcoords="offset points", xytext=(10, 10), fontsize=8, color=SIENNA)
    ax[1].set_xlabel("foreign ownership of capital (%)")
    ax[1].set_ylabel("capital-sector revenue (% of output)")
    ax[1].set_title("Two ways to cut foreign ownership, opposite for revenue")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(f"{OUT}/p3_competition.png", dpi=110); plt.close(fig)
    print(f"saved p3_competition.png  (own {own[0]:.0f}->{own[-1]:.0f}% as competition rises; "
          f"tax raises rev to {tax_rev[-1]:.1f}%, competition cuts it to {comp_rev[-1]:.1f}%)")


if __name__ == "__main__":
    fig()
