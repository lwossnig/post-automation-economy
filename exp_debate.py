"""Testing specific claims from the AI-tax debate against the model.

Figure A (Laffer/erosion): Friedman's "the base shrinks the moment you tax it"
and the IMF's "could stifle productivity". Sweep the digital levy with and
without the AI-supply response and track revenue level and output.

Figure B (territoriality): Friedman's "subsidy for foreign inference" and
Luckey's "makes foreign models more attractive". Move the compute offshore and
see whether a recognised-value levy (DST, withholding) loses its grip.
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


def laffer_fig():
    Yb = float(np.array(_run().hist.Y)[W].mean())
    rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    rev0, rev1, out1 = [], [], []
    for d in rates:
        m0 = _run(dst_ai=d, ai_supply_elasticity=0.0)
        m1 = _run(dst_ai=d, ai_supply_elasticity=0.5)
        y1 = float(np.array(m1.hist.Y)[W].mean())
        rev0.append(_revY(m0) * float(np.array(m0.hist.Y)[W].mean()) / Yb)
        rev1.append(_revY(m1) * y1 / Yb)
        out1.append(100 * y1 / Yb)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = [100 * r for r in rates]
    ax.plot(x, rev0, "o-", color=SAND, lw=1.7, label="revenue, no supply response")
    ax.plot(x, rev1, "o-", color=SIENNA, lw=1.9, label="revenue, with supply response")
    ax.set_xlabel("digital-levy rate on AI (%)")
    ax.set_ylabel("capital-sector revenue (% of untaxed output)", color=SIENNA)
    ax.tick_params(axis="y", labelcolor=SIENNA); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, out1, "s--", color=TEAL, lw=1.5, label="output (with response)")
    ax2.set_ylabel("steady-state output (% of untaxed)", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL); ax2.set_ylim(70, 104)
    ax.set_title("Erosion and an output cost, but no Laffer peak in range")
    fig.tight_layout(); fig.savefig(f"{OUT}/debate_laffer.png", dpi=110); plt.close(fig)
    print(f"saved debate_laffer.png  (rev w/resp {rev1[1]:.1f}->{rev1[-1]:.1f}; output ->{out1[-1]:.0f}%)")


def territory_fig():
    shs = [1.0, 0.8, 0.6, 0.4, 0.2]
    corp, dst, wh = [], [], []
    for sh in shs:
        corp.append(_revY(_run(s_home=sh)))
        dst.append(_revY(_run(s_home=sh, dst_ai=0.10)))
        wh.append(_revY(_run(s_home=sh, tax_repat=0.30)))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(shs, dst, "o-", color=TEAL, lw=1.9, label="digital levy (recognised AI value)")
    ax.plot(shs, wh, "s-", color=SIENNA, lw=1.9, label="withholding (repatriated rent)")
    ax.plot(shs, corp, "^--", color=SAND, lw=1.6, label="corporate tax only")
    ax.set_xlabel("share of AI compute located in the host (s_home)  <- more offshore")
    ax.set_ylabel("capital-sector revenue (% of output)")
    ax.set_title("A recognised-value base is invariant to where the compute sits")
    ax.invert_xaxis(); ax.set_ylim(-0.3, 5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    ax.annotate("base recognised in the host,\nso moving servers offshore\nchanges nothing",
                (0.4, dst[3]), textcoords="offset points", xytext=(10, -40), fontsize=8, color=TEAL)
    fig.tight_layout(); fig.savefig(f"{OUT}/debate_territory.png", dpi=110); plt.close(fig)
    print(f"saved debate_territory.png  (DST {dst[0]:.2f}->{dst[-1]:.2f}, withholding {wh[0]:.2f}->{wh[-1]:.2f})")


if __name__ == "__main__":
    laffer_fig()
    territory_fig()
