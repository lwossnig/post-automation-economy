"""Experiment R: elastic labour supply and the bottleneck wage.

The baseline pays each labour type its marginal product with FIXED supply. The task
AI cannot do (the physical/routine bottleneck) is then a scarce complement to the
accumulating robotic capital, so its per-unit wage explodes: this is the model's
"automation raises wages" result. The reviewer's objection is that a task hard for
AI but easy for any human need not command a high wage, because abundant labour
competes it down.

We test that by letting the bottleneck labour supply slope up in its own wage (with
a reservation floor), solved as a short fixed point against the CES. The clean
result is that the scarcity premium is competed away: the bottleneck per-unit wage
is dampened and the routine/cognitive wage ratio collapses. Within the bottleneck
cluster the surplus shifts to capital (its labour share falls). At the AGGREGATE the
labour share need not fall, because the labour-intensive bottleneck cluster expands
as labour floods in; we report this honestly. So the result is "the bottleneck-wage
spike is an artefact of inelastic supply," not "automation lowers the labour share."
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

SEEDS = 5
STEADY = slice(250, 600)

SCEN = ["labour_inelastic", "labour_elastic_routine", "labour_elastic_strong"]


def _run(make, seed):
    p = make(); p.seed = seed; p.periods = 600
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    return m


def _metrics(m):
    W = STEADY
    wLr = np.array(m.hist.w_Lr)[W].sum(); ciKr = np.array(m.hist.ci_Kr)[W].sum()
    wLc = np.array(m.hist.w_Lc)[W].sum(); ciKai = np.array(m.hist.ci_Kai)[W].sum()
    tot = wLr + wLc + ciKr + ciKai
    return dict(
        routine_labshare=wLr / (wLr + ciKr),
        agg_labshare=(wLr + wLc) / tot,
        agg_capshare=(ciKr + ciKai) / tot,
        wu_r=float(np.mean(m.hist.wage_unit_r[-50:])),
        wu_c=float(np.mean(m.hist.wage_unit_c[-50:])),
        ratio=float(np.mean(m.hist.wage_unit_r[-50:]) / max(np.mean(m.hist.wage_unit_c[-50:]), 1e-9)),
        gini_labour=float(np.mean(m.hist.gini_labour[-50:])),
        sf_r=float(m._sf_r),
    )


def main():
    rows = {}
    for name in SCEN:
        make = S.REGISTRY[name]
        agg = None
        for sd in range(SEEDS):
            mt = _metrics(_run(make, sd))
            agg = mt if agg is None else {k: agg[k] + mt[k] for k in agg}
        rows[name] = {k: v / SEEDS for k, v in agg.items()}

    hdr = (f"{'scenario':24s} {'sf_r':>5s} {'routLabShr':>10s} {'aggLabShr':>9s} "
           f"{'aggCapShr':>9s} {'wuR':>8s} {'wuC':>7s} {'wuR/wuC':>8s} {'giniLab':>7s}")
    print(hdr); print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:24s} {r['sf_r']:5.2f} {r['routine_labshare']:10.3f} "
              f"{r['agg_labshare']:9.3f} {r['agg_capshare']:9.3f} {r['wu_r']:8.1f} "
              f"{r['wu_c']:7.1f} {r['ratio']:8.1f} {r['gini_labour']:7.3f}")

    # --- Figure: bottleneck-wage path and the routine/cognitive premium ---------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in SCEN:
        paths = [np.array(_run(S.REGISTRY[name], sd).hist.wage_unit_r) for sd in range(SEEDS)]
        ax1.plot(np.mean(np.vstack(paths), axis=0), lw=1.8, label=name)
    ax1.set_xlabel("period"); ax1.set_ylabel("routine (bottleneck) per-unit wage")
    ax1.set_title("Elastic supply competes away the bottleneck-wage spike")
    ax1.legend(fontsize=8)

    names = list(rows.keys())
    x = np.arange(len(names)); w = 0.35
    ratio = [rows[n]["ratio"] for n in names]
    rls = [rows[n]["routine_labshare"] for n in names]
    ax2.bar(x - w/2, ratio, width=w, color="#c44e52", label="routine/cognitive wage ratio")
    ax2b = ax2.twinx()
    ax2b.bar(x + w/2, rls, width=w, color="#4c72b0", label="routine-cluster labour share")
    ax2.set_xticks(x); ax2.set_xticklabels([n.replace("labour_", "") for n in names], fontsize=8)
    ax2.set_ylabel("wuR / wuC (scarcity premium)", color="#c44e52")
    ax2b.set_ylabel("routine labour share", color="#4c72b0")
    ax2.set_title("Premium collapses; bottleneck surplus shifts to capital")
    fig.tight_layout()
    fig.savefig("figures_v3/expR_labour_supply.png", dpi=110)
    print("saved figures_v3/expR_labour_supply.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
