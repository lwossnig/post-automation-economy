"""Experiment H: AI vs robotic automation, taxed with different instruments.

Robots are physical and domestic, so their income is source-taxable. AI carries
a mobile IP rent that leaks abroad as a deductible licence fee. We compare what
each instrument actually collects, in the mature steady state (where the rent is
the only durable surplus) and in the transition profit window (where ordinary
capital profit is briefly positive). Headline metrics are seed-averaged.
"""
from __future__ import annotations

import numpy as np

from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

SEEDS = 5
STEADY = slice(250, 600)     # mature economy: rent is the only durable surplus
TRANS = slice(110, 200)      # AI ramp: ordinary capital profit briefly positive


def _run(make, seed):
    p = make(); p.seed = seed; p.periods = 600
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    return m


def _metrics(m, win):
    Y = np.array(m.hist.Y)
    yo = Y[win].sum()
    rent = np.array(m.hist.rent_ai)[win].sum() / yo
    # gov revenue from the capital-sector instruments (corp + robot + DST + rent
    # withholding), all booked through corp_tax_rev and repat_revenue
    caprev = (np.array(m.hist.corp_tax_rev)[win].sum()
              + np.array(m.hist.repat_revenue)[win].sum()) / yo
    return rent, caprev


def main():
    names = ["two_channel_base", "robot_tax_only", "ai_dst",
             "ai_withholding", "offshore_compute", "sovereign_compute"]
    rows = []
    for name in names:
        make = S.REGISTRY[name]
        rent_s = caprev_s = caprev_t = gov = gini = kaik = 0.0
        for sd in range(SEEDS):
            m = _run(make, sd)
            r_s, c_s = _metrics(m, STEADY)
            _, c_t = _metrics(m, TRANS)
            Y = np.array(m.hist.Y)
            rent_s += r_s; caprev_s += c_s; caprev_t += c_t
            gov += m.hist.gov_nw[-1] / m.p.n_agents / (Y[-1] / m.p.n_agents)
            gini += np.mean(m.hist.gini[-30:])
            et = m.h_eq.sum() + m.eq_state + m.eq_row
            kaik += m.eq_row / et            # foreign ownership share of capital
        k = SEEDS
        rows.append((name, rent_s/k, caprev_s/k, caprev_t/k, gov/k, gini/k, kaik/k))

    hdr = ("scenario", "rent/Y_ss", "capRev/Y_ss", "capRev/Y_tr", "gov_nw/Y", "Gini", "f_row")
    print(f"{hdr[0]:20s} {hdr[1]:>10s} {hdr[2]:>12s} {hdr[3]:>12s} {hdr[4]:>9s} {hdr[5]:>6s} {hdr[6]:>7s}")
    for r in rows:
        print(f"{r[0]:20s} {r[1]*100:9.2f}% {r[2]*100:11.2f}% {r[3]*100:11.2f}% "
              f"{r[4]:9.2f} {r[5]:6.3f} {r[6]:7.3f}")

    # figure: revenue reached by each instrument (steady state) vs the rent that leaks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["base\n(untaxed)", "robot\ntax", "AI\nDST", "AI\nwithhold",
              "compute\nabroad", "full\ntoolkit"]
    rentv = [r[1] * 100 for r in rows]
    revv = [r[2] * 100 for r in rows]
    frow = [r[6] * 100 for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, rentv, 0.4, label="AI rent leaving / output", color="#c44")
    ax.bar(x + 0.2, revv, 0.4, label="capital-sector revenue / output", color="#48a")
    ax.plot(x, frow, "ko-", label="foreign ownership of capital (%)", lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% (steady state)")
    ax.set_title("AI vs robotic automation: rent, revenue, and who ends up owning the capital")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures_v3/expH_two_channel.png", dpi=110)
    print("\nsaved figures_v3/expH_two_channel.png")


def extra_figures():
    """Two time-series figures: the foreign-ownership drift under each policy,
    and the income decomposition (where each pound of output goes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- foreign ownership over time, by policy ---
    pol = [("two_channel_base", "untaxed", "#7c2d12"),
           ("robot_tax_only", "robot tax", "#c2956b"),
           ("ai_dst", "AI digital levy", "#1d4e44"),
           ("ai_withholding", "rent withholding (rebated)", "#3b6ea5"),
           ("sovereign_compute", "full toolkit", "#6b3fa0")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, lab, col in pol:
        fr = _foreign_share_series(S.REGISTRY[name], 0)
        ax.plot(range(len(fr)), 100 * np.array(fr), label=lab, color=col, lw=1.6)
    ax.axvspan(110, 200, color="#000", alpha=0.04)
    ax.set_xlabel("period"); ax.set_ylabel("foreign ownership of capital (%)")
    ax.set_title("Who ends up owning the capital: foreign ownership over time")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    ax.text(155, 4, "AI ramp", fontsize=8, ha="center", color="#555")
    fig.tight_layout(); fig.savefig("figures_v3/expH_foreign_ownership.png", dpi=110)
    plt.close(fig)
    print("saved figures_v3/expH_foreign_ownership.png")

    # --- income decomposition over time (base scenario) ---
    m = _run(S.two_channel_base, 0)
    Y = np.array(m.hist.Y)
    comp = {
        "routine wages": np.array(m.hist.w_Lr) / Y,
        "cognitive wages": np.array(m.hist.w_Lc) / Y,
        "robotic capital income": np.array(m.hist.ci_Kr) / Y,
        "AI compute income": np.array(m.hist.ci_Kai) / Y,
        "AI IP rent (leaves the UK)": np.array(m.hist.rent_ai) / Y,
    }
    cols = ["#2e7d32", "#66bb6a", "#8d6e63", "#bcaaa4", "#c44"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.stackplot(range(len(Y)), *comp.values(), labels=list(comp.keys()), colors=cols)
    ax.set_xlim(0, len(Y) - 1); ax.set_ylim(0, 1)
    ax.set_xlabel("period"); ax.set_ylabel("share of output")
    ax.set_title("Where each pound of output goes (untaxed foreign AI rent)")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout(); fig.savefig("figures_v3/expH_income_decomposition.png", dpi=110)
    plt.close(fig)
    print("saved figures_v3/expH_income_decomposition.png")


def _foreign_share_series(make, seed):
    m = make(); m.seed = seed; m.periods = 600
    md = ModelV3(m)
    out = []
    for _ in range(md.p.periods):
        md.step()
        et = md.h_eq.sum() + md.eq_state + md.eq_row
        out.append(md.eq_row / et if et > 0 else 0.0)
    return out


if __name__ == "__main__":
    main()
    extra_figures()
