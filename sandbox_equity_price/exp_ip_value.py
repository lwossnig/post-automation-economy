"""Figure for the capitalised-rent section: value the rent stream the model produces,
show the book-vs-market ownership gap, the valuation channel of a host levy, and the
negative equity yield that makes a market-clearing price degenerate."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

OUT = "/home/claude/figures_v3"
TEAL, SIENNA, SAND, PURP = "#1d4e44", "#7c2d12", "#c2956b", "#6b3fa0"


def _run(**o):
    p = S.two_channel_base(); p.seed = 0; p.periods = 600
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(600):
        m.step()
    return m


def fig():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel 1: across the rent, the IP value and the book-vs-market ownership gap
    mus = [0.10, 0.20, 0.25, 0.30, 0.40]
    vip, bk, mk = [], [], []
    for mu in mus:
        m = _run(mu_frac=mu)
        vip.append(m.hist.v_ip[-1] / m.hist.Y[-1])
        bk.append(100 * m.hist.own_row[-1]); mk.append(100 * m.hist.own_row_mkt[-1])
    ax0 = ax[0]; x = [100 * mu for mu in mus]
    ax0.plot(x, bk, "o-", color=SAND, lw=1.7, label="foreign ownership, book")
    ax0.plot(x, mk, "o-", color=SIENNA, lw=1.9, label="foreign ownership, market value")
    ax0.fill_between(x, bk, mk, color=SIENNA, alpha=0.08)
    ax0.set_xlabel("rent share of cognitive value added (%)")
    ax0.set_ylabel("foreign ownership (%)")
    ax0.set_title("The capitalised rent lifts ownership above book")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=8, loc="upper left")
    ax0b = ax0.twinx()
    ax0b.plot(x, vip, "s--", color=PURP, lw=1.5)
    ax0b.set_ylabel("IP value (years of output)", color=PURP)
    ax0b.tick_params(axis="y", labelcolor=PURP)
    # panel 2: the levy erodes the owner's capitalised wealth (valuation channel)
    wedges = [0.0, 0.1, 0.2, 0.3, 0.4]
    vipw = []
    for w in wedges:
        m = _run(dst_ai=w / 2, tax_repat=w / 2)
        vipw.append(m.hist.v_ip[-1] / m.hist.Y[-1])
    ax1 = ax[1]
    ax1.plot([100 * w for w in wedges], vipw, "o-", color=TEAL, lw=1.9)
    ax1.set_xlabel("host tax wedge on the rent (%)")
    ax1.set_ylabel("IP value (years of output)")
    ax1.set_title("A source levy erodes the owner's capitalised wealth")
    ax1.grid(alpha=0.3)
    ax1.annotate("a market price on domestic equity is\ndegenerate: its yield is ~0 to negative,\nso the value sits in this foreign-held IP",
                 (0.5, 0.12), xycoords="axes fraction", fontsize=7.5, color="#555",
                 ha="center", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/p3_ip_value.png", dpi=110); plt.close(fig)
    print(f"saved p3_ip_value.png  (IP {vip[2]:.1f}y of output at baseline; "
          f"own book {bk[2]:.0f}->market {mk[2]:.0f}%; levy cuts IP {vipw[0]:.1f}->{vipw[-1]:.1f}y)")


fig()
