"""Phase 2 experiment: the reinstatement margin and the labour share.

Automation alone drives the labour share down; new tasks (reinstatement) pull it
back up. The figure shows the labour-share path is not mechanical but depends on
the balance between the two, and the steady-state trade-off between the labour
share and output (a more labour-heavy economy saves and accumulates less)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

OUT = "figures_v3"
TEAL, SIENNA, SAND, BLUE = "#1d4e44", "#7c2d12", "#c2956b", "#3b6ea5"


def _run(rho, periods=600, **o):
    p = S.two_channel_base(); p.seed = 0; p.periods = periods; p.reinstate_frac = rho
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(periods):
        m.step()
    return m


def _lshare_series(m):
    Y = np.array(m.hist.Y)
    wl = np.array(m.hist.w_Lr) + np.array(m.hist.w_Lc)
    return wl / np.clip(Y, 1e-9, None)


def reinstatement_fig():
    rhos = [(0.0, "no reinstatement", SIENNA),
            (0.3, "partial (rho = 0.3)", SAND),
            (0.5, "balanced (rho = 0.5)", TEAL)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: labour-share path over time
    a_ai_start = S.two_channel_base().a_ai_start
    ax[0].axvspan(a_ai_start, a_ai_start + 120, color=BLUE, alpha=0.06)
    ax[0].annotate("AI ramp", (a_ai_start + 60, 0.66), color=BLUE, fontsize=8, ha="center")
    for rho, lab, col in rhos:
        ls = _lshare_series(_run(rho))
        ax[0].plot(ls, color=col, lw=1.7, label=lab)
    ax[0].set_xlabel("period"); ax[0].set_ylabel("labour share of output")
    ax[0].set_title("The labour-share fall is conditional, not mechanical")
    ax[0].set_ylim(0.30, 0.70); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="lower left")
    # right: steady-state labour share and output vs rho
    rr = [0.0, 0.2, 0.4, 0.5, 0.6]
    Yb = float(np.array(_run(0.0).hist.Y)[250:].mean())
    lsh, yrel = [], []
    for rho in rr:
        m = _run(rho)
        lsh.append(float(_lshare_series(m)[250:].mean()))
        yrel.append(100 * float(np.array(m.hist.Y)[250:].mean()) / Yb)
    ax2 = ax[1]
    ax2.plot(rr, lsh, "o-", color=TEAL, lw=1.8)
    ax2.set_xlabel("reinstatement strength (reinstate_frac, rho)")
    ax2.set_ylabel("steady-state labour share", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL); ax2.grid(alpha=0.3); ax2.set_ylim(0.35, 0.75)
    ax3 = ax2.twinx()
    ax3.plot(rr, yrel, "s--", color=SIENNA, lw=1.5)
    ax3.set_ylabel("steady-state output (% of no-reinstatement)", color=SIENNA)
    ax3.tick_params(axis="y", labelcolor=SIENNA)
    ax2.set_title("Labour share rises, output eases as saving falls")
    fig.tight_layout(); fig.savefig(f"{OUT}/p2_reinstatement.png", dpi=110); plt.close(fig)
    print(f"saved p2_reinstatement.png  (labour share {lsh[0]:.2f} -> {lsh[-1]:.2f}; "
          f"output {yrel[-1]:.0f}% at rho={rr[-1]})")


if __name__ == "__main__":
    reinstatement_fig()
