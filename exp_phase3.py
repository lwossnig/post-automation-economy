"""Phase 3: the optimising foreign owner (profit-shifting of the rent).

The territoriality test (section 6K) showed the host's take is invariant to where
the compute SITS. This asks the sharper question: what if the owner relocates
where the rent is BOOKED in response to the tax? A share of the rent recognition
shifts offshore, rising with the host's wedge, so the host's take is bounded and
has a revenue-maximising rate. Reach survives server relocation but is limited by
profit-shifting of the recognition itself.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

OUT = "figures_v3"
TEAL, SIENNA, SAND, BLUE = "#1d4e44", "#7c2d12", "#c2956b", "#3b6ea5"
W = slice(250, 600)


def _run(periods=600, **o):
    p = S.two_channel_base(); p.seed = 0; p.periods = periods
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(periods):
        m.step()
    return m


def _revY(m):
    return 100 * (np.array(m.hist.corp_tax_rev)[W].sum()
                  + np.array(m.hist.repat_revenue)[W].sum()) / np.array(m.hist.Y)[W].sum()


def _own(m):
    et = m.h_eq.sum() + m.eq_state + m.eq_row
    return 100 * m.eq_row / et


def fig():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel 1: host take vs wedge, non-strategic vs optimising owner
    wedges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rev0, rev1 = [], []
    for w in wedges:
        d, r = w / 2, w / 2
        rev0.append(_revY(_run(dst_ai=d, tax_repat=r, owner_shift_elasticity=0.0)))
        rev1.append(_revY(_run(dst_ai=d, tax_repat=r, owner_shift_elasticity=1.0)))
    x = [100 * w for w in wedges]
    ax[0].plot(x, rev0, "o-", color=SAND, lw=1.7, label="non-strategic owner")
    ax[0].plot(x, rev1, "o-", color=SIENNA, lw=1.9, label="optimising owner (shifts the rent)")
    ax[0].fill_between(x, rev1, rev0, color=SIENNA, alpha=0.08)
    ax[0].set_xlabel("host tax wedge on the rent, DST + withholding (%)")
    ax[0].set_ylabel("capital-sector revenue (% of output)")
    ax[0].set_title("Profit-shifting bends the take below the non-strategic line")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].annotate("shifted offshore\n(widens with the rate)", (x[-2], (rev0[-2] + rev1[-2]) / 2),
                   textcoords="offset points", xytext=(-96, 6), fontsize=8, color=SIENNA)
    # panel 2: erosion vs the owner's mobility, at the full-toolkit wedge
    elasts = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    revE, ownE = [], []
    for e in elasts:
        m = _run(dst_ai=0.10, tax_repat=0.30, owner_shift_elasticity=e)
        revE.append(_revY(m)); ownE.append(_own(m))
    ax2 = ax[1]
    ax2.plot(elasts, revE, "o-", color=TEAL, lw=1.9)
    ax2.set_xlabel("owner mobility (profit-shift elasticity)")
    ax2.set_ylabel("capital-sector revenue (% of output)", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL); ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(elasts, ownE, "s--", color=SIENNA, lw=1.5)
    ax3.set_ylabel("foreign ownership of capital (%)", color=SIENNA)
    ax3.tick_params(axis="y", labelcolor=SIENNA)
    ax2.set_title("A more mobile owner pays less and owns more")
    fig.tight_layout(); fig.savefig(f"{OUT}/p3_optimising_owner.png", dpi=110); plt.close(fig)
    print(f"saved p3_optimising_owner.png  (erosion gap at 60%% wedge {rev0[-1]-rev1[-1]:.2f}pp; "
          f"rev {revE[0]:.2f}->{revE[-1]:.2f}, own {ownE[0]:.0f}->{ownE[-1]:.0f}% as mobility rises)")


if __name__ == "__main__":
    fig()
