"""Phase 2 (piece 2): unemployment from automation, and funding the safety net.

Left: the unemployment rate over time depends on the balance between displacement
(pass-through) and reinstatement (new tasks), spanning the optimistic and the
"bloodbath" range in the debate. Right: a safety net for the displaced is
affordable when the state reaches the AI rent, and strains the public balance
sheet when it leans on the eroding income-tax base alone.
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


def _run(periods=600, **o):
    p = S.two_channel_base(); p.seed = 0; p.periods = periods
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(periods):
        m.step()
    return m


def fig():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a_start = S.two_channel_base().a_ai_start
    # ---- left: unemployment paths
    ax[0].axvspan(a_start, a_start + 120, color=BLUE, alpha=0.06)
    ax[0].annotate("AI ramp", (a_start + 60, 0.40), color=BLUE, fontsize=8, ha="center")
    combos = [(0.0, 0.4, SIENNA, "no reinstatement, high pass-through"),
              (0.5, 0.4, SAND, "balanced reinstatement, high pass-through"),
              (0.5, 0.2, TEAL, "balanced reinstatement, moderate pass-through")]
    for rf, lam, col, lab in combos:
        m = _run(reinstate_frac=rf, unemployment_pass_through=lam, unemployment_benefit=0.5)
        ax[0].plot(100 * np.array(m.hist.unemployment), color=col, lw=1.7, label=lab)
    ax[0].axhspan(10, 20, color="grey", alpha=0.10)
    ax[0].annotate("10-20% (Amodei range)", (300, 15), fontsize=7.5, color="dimgrey")
    ax[0].set_xlabel("period"); ax[0].set_ylabel("unemployment rate (%)")
    ax[0].set_title("Unemployment depends on displacement vs new tasks")
    ax[0].set_ylim(0, 50); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5, loc="upper left")
    # ---- right: funding the safety net
    base = dict(reinstate_frac=0.5, unemployment_pass_through=0.3, unemployment_benefit=0.5)
    unf = _run(**base)
    fund = _run(**base, dst_ai=0.10, tax_repat=0.30)
    yU = np.array(unf.hist.Y); yF = np.array(fund.hist.Y)
    ax[1].plot(np.array(unf.hist.gov_nw) / yU, color=SIENNA, lw=1.8,
               label="income tax only (safety net unfunded)")
    ax[1].plot(np.array(fund.hist.gov_nw) / yF, color=TEAL, lw=1.8,
               label="plus DST + withholding (reaches the rent)")
    ax[1].axhline(0, color="black", lw=0.7, ls=":")
    ax[1].set_xlabel("period"); ax[1].set_ylabel("government net worth (multiple of output)")
    ax[1].set_title("A safety net is affordable if it reaches the rent")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/p2b_unemployment.png", dpi=110); plt.close(fig)
    print(f"saved p2b_unemployment.png  (unfunded gov_nw/Y={unf.hist.gov_nw[-1]/yU[-1]:.2f}, "
          f"funded={fund.hist.gov_nw[-1]/yF[-1]:.2f})")


if __name__ == "__main__":
    fig()
