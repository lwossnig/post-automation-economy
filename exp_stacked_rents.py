"""Experiment Q: two stacked foreign rents (compute vs model).

The durable foreign surplus is split into two rentiers. The model/IP rent
(mu_frac) is an ongoing licence flow on the cognitive cluster, reached by the DST
and the withholding. The compute/chip rent (mu_compute) is a foreign chip-maker's
margin EMBEDDED IN THE PRICE of AI capital, peeled from the AI capital income; it
leaves abroad as a goods price, so the DST and the withholding miss it and only a
border tariff, a usage levy, or a domestic chip-maker reach it. Onshoring the
servers (s_home) does not capture it.

We show three things, seed-averaged over the mature steady state:
  (1) a second foreign rent accelerates the drift of capital ownership abroad;
  (2) a scorecard of what each instrument actually collects from the compute rent
      (tariff / usage reach it; DST / withholding do not);
  (3) moving the chip-maker's domicile home (compute_foreign = 0) keeps the rent.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

SEEDS = 5
STEADY = slice(250, 600)     # mature economy: rent is the only durable surplus


def _run(make, seed):
    p = make(); p.seed = seed; p.periods = 600
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    return m


def _metrics(m):
    Y = np.array(m.hist.Y)
    yo = Y[STEADY].sum()
    rent_ai = np.array(m.hist.rent_ai)[STEADY].sum() / yo
    rent_comp = np.array(m.hist.rent_compute)[STEADY].sum() / yo
    comp_rev = np.array(m.hist.compute_rev)[STEADY].sum() / yo
    # all capital-sector revenue (corp + robot + DST + withholding), booked through
    # corp_tax_rev and repat_revenue; the licence-flow instruments live here
    licence_rev = (np.array(m.hist.corp_tax_rev)[STEADY].sum()
                   + np.array(m.hist.repat_revenue)[STEADY].sum()) / yo
    own_row_end = float(m.hist.own_row[-1])
    own_row_start = float(np.mean(m.hist.own_row[80:110]))   # pre-AI-ramp baseline
    return dict(rent_ai=rent_ai, rent_comp=rent_comp, comp_rev=comp_rev,
                licence_rev=licence_rev, own_row=own_row_end, drift=own_row_end - own_row_start)


# Scenarios: the importer baseline, the two licence-flow instruments applied to a
# stacked-rent economy (to show they miss the compute rent), the two new
# instruments that reach it, the offshore-compute case, and the two domestic-owner
# domicile cases. The licence-flow instruments are built inline from the importer.
def _imp(**kw):
    return replace(S.stacked_rents_importer(), **kw)


SCEN = {
    "no compute rent (IP only)":   S.two_channel_base,
    "+ compute rent (importer)":   S.stacked_rents_importer,
    "  ... + DST (misses it)":     lambda: _imp(dst_ai=0.10),
    "  ... + withholding (misses)": lambda: _imp(tax_repat=0.30, repat_rebate=True),
    "  ... + tariff (reaches it)": S.stacked_rents_tariff,
    "  ... + usage levy (partial)": S.stacked_rents_usage,
    "  ... compute offshored":     S.stacked_rents_offshore,
    "domestic chips, foreign IP":  S.stacked_rents_us_chips,
    "full owner (both at home)":   S.stacked_rents_full_owner,
}


def main():
    rows = {}
    for name, make in SCEN.items():
        agg = None
        for sd in range(SEEDS):
            mt = _metrics(_run(make, sd))
            agg = mt if agg is None else {k: agg[k] + mt[k] for k in agg}
        rows[name] = {k: v / SEEDS for k, v in agg.items()}

    hdr = f"{'scenario':30s} {'rentAI/Y':>9s} {'rentC/Y':>8s} {'compRev/Y':>10s} {'licRev/Y':>9s} {'f_row':>7s} {'drift':>7s}"
    print(hdr); print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:30s} {r['rent_ai']:9.4f} {r['rent_comp']:8.4f} "
              f"{r['comp_rev']:10.4f} {r['licence_rev']:9.4f} {r['own_row']:7.3f} {r['drift']:+7.3f}")

    # --- Figure: foreign ownership (left) and what each instrument reaches (right)
    names = list(rows.keys())
    f_row = [rows[n]["own_row"] for n in names]
    comp_rev = [rows[n]["comp_rev"] for n in names]
    lic_rev = [rows[n]["licence_rev"] for n in names]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    y = np.arange(len(names))
    axL.barh(y, f_row, color="#b5651d")
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=8)
    axL.invert_yaxis()
    axL.set_xlabel("foreign ownership share of capital (steady state)")
    axL.set_title("Ownership drift: a second foreign rent pushes it up;\n"
                  "only a tariff, a usage levy, or home chips pull it back")

    w = 0.4
    axR.barh(y - w/2, comp_rev, height=w, color="#2a9d8f", label="reaches compute rent (tariff + usage)")
    axR.barh(y + w/2, lic_rev, height=w, color="#888888", label="licence-flow take (corp + DST + withholding)")
    axR.set_yticks(y); axR.set_yticklabels([""] * len(names))
    axR.invert_yaxis()
    axR.set_xlabel("government take / steady-state output")
    axR.set_title("What each instrument collects:\nDST/withholding leave the compute rent untouched")
    axR.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures_v3/expQ_stacked_rents.png", dpi=110)
    print("saved figures_v3/expQ_stacked_rents.png")
    plt.close(fig)

    extra_figure()


def extra_figure():
    """Foreign-ownership-drift time series for the key contrast scenarios."""
    keys = {
        "IP rent only": S.two_channel_base,
        "+ compute rent (importer)": S.stacked_rents_importer,
        "+ tariff": S.stacked_rents_tariff,
        "domestic chips": S.stacked_rents_us_chips,
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, make in keys.items():
        paths = []
        for sd in range(SEEDS):
            m = _run(make, sd)
            paths.append(np.array(m.hist.own_row))
        mean = np.mean(np.vstack(paths), axis=0)
        ax.plot(mean, label=label, lw=1.8)
    ax.set_xlabel("period"); ax.set_ylabel("foreign ownership share of capital")
    ax.set_title("Experiment Q: a stacked compute rent accelerates ownership drift abroad")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("figures_v3/expQ_ownership_drift.png", dpi=110)
    print("saved figures_v3/expQ_ownership_drift.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
