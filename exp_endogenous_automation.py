"""Experiment S: endogenous, cost-driven automation.

Automation is an exogenous logistic ramp in the baseline. Here that ramp is only
the TECHNICAL frontier: a task is actually automated when the machine is cheaper
than the human wage for it (MIT "Beyond AI Exposure"). Machine cost starts above
the early wage and falls at a calibrated rate, so realised automation lags the
frontier and its pace is set by how fast cost declines; rising bottleneck wages
open the gate further (the feedback). We compare the realised path against the
exogenous frontier and trace how the cost-decline speed moves the timing of the
distributional shift. The transition still arrives, so the result is "how fast and
how far," not "whether."
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
SCEN = ["endog_auto_slow", "endog_auto_mit", "endog_auto_fast", "endog_auto_scale"]


def _run(make, seed):
    p = make(); p.seed = seed; p.periods = 600
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    return m


def main():
    # exogenous frontier (the baseline two-channel ramp), seed-averaged
    exo = np.mean(np.vstack([np.array(_run(S.two_channel_base, sd).hist.a_ai)
                             for sd in range(SEEDS)]), axis=0)

    rows = {}
    paths = {}
    for name in SCEN:
        a_paths, owns, ginis = [], [], []
        for sd in range(SEEDS):
            m = _run(S.REGISTRY[name], sd)
            a_paths.append(np.array(m.hist.a_ai))
            owns.append(m.hist.own_row[-1])
            ginis.append(np.mean(m.hist.gini[-30:]))
        a_mean = np.mean(np.vstack(a_paths), axis=0)
        paths[name] = a_mean
        arr = np.where(a_mean >= 0.9)[0]
        rows[name] = dict(a150=a_mean[150], a250=a_mean[250], a400=a_mean[400],
                          arrival=int(arr[0]) if len(arr) else -1,
                          own_row=float(np.mean(owns)), gini=float(np.mean(ginis)))

    arr_exo = np.where(exo >= 0.9)[0]
    print(f"{'scenario':18s} {'a_ai@150':>9s} {'a_ai@250':>9s} {'a_ai@400':>9s} "
          f"{'arrival':>8s} {'f_row':>7s} {'gini':>6s}")
    print("-" * 70)
    print(f"{'(exogenous)':18s} {exo[150]:9.3f} {exo[250]:9.3f} {exo[400]:9.3f} "
          f"{(int(arr_exo[0]) if len(arr_exo) else -1):8d} {'-':>7s} {'-':>6s}")
    for name, r in rows.items():
        print(f"{name:18s} {r['a150']:9.3f} {r['a250']:9.3f} {r['a400']:9.3f} "
              f"{r['arrival']:8d} {r['own_row']:7.3f} {r['gini']:6.3f}")

    # --- Figure: realised automation paths vs the exogenous frontier -----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(exo, "k--", lw=2, label="exogenous frontier")
    for name in SCEN:
        ax1.plot(paths[name], lw=1.7, label=name.replace("endog_auto_", ""))
    ax1.set_xlabel("period"); ax1.set_ylabel("realised AI task share (a_ai)")
    ax1.set_title("Cost-driven automation lags the frontier;\nthe pace is set by how fast cost falls")
    ax1.legend(fontsize=8)

    names = list(rows.keys())
    x = np.arange(len(names)); w = 0.38
    gini = [rows[n]["gini"] for n in names]
    frow = [rows[n]["own_row"] for n in names]
    ax2.bar(x - w/2, gini, width=w, color="#dd8452", label="mature wealth Gini")
    ax2.bar(x + w/2, frow, width=w, color="#55a868", label="foreign ownership share")
    ax2.set_xticks(x); ax2.set_xticklabels([n.replace("endog_auto_", "") for n in names], fontsize=8)
    ax2.set_title("The distributional shift moves with the cost-decline speed")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures_v3/expS_endogenous_automation.png", dpi=110)
    print("saved figures_v3/expS_endogenous_automation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
