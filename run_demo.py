"""Demo: run the five scenarios, then a small Monte Carlo sweep.

    python run_demo.py

Prints a scenario comparison and a sensitivity sweep over the wealth-tax rate
and the foreign-ownership share, reporting the end-of-horizon top-1% wealth
share, Gini, government net worth and rest-of-world (repatriated) net worth.
All quantities are scaled by output (mean per-capita output = 1).
"""
from __future__ import annotations

import numpy as np

from abm_sfc import Model
from abm_sfc import scenarios
from abm_sfc.model import Params
from dataclasses import replace


def run_scenarios() -> None:
    print("=" * 78)
    print("SCENARIO COMPARISON  (end of horizon; NW scaled by output)")
    print("=" * 78)
    print(f"{'scenario':17s}{'top1%':>8s}{'gini':>8s}{'labour_sh':>11s}"
          f"{'gov_NW/Y':>11s}{'RoW_NW/Y':>11s}")
    for name, fn in scenarios.REGISTRY.items():
        p = fn()
        h = Model(p).run()
        print(f"{name:17s}{h.top1_share[-1]*100:7.1f}%{h.gini[-1]:8.3f}"
              f"{h.labour_share[-1]:11.2f}{h.gov_nw[-1]/p.n_agents:11.1f}"
              f"{h.row_nw[-1]/p.n_agents:11.1f}")


def monte_carlo(n_draws: int = 200, periods: int = 250, seed: int = 0) -> None:
    print("\n" + "=" * 78)
    print(f"MONTE CARLO SWEEP  ({n_draws} draws)")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_draws):
        tax_wealth = float(rng.uniform(0.0, 0.04))
        own_row = float(rng.uniform(0.0, 0.7))
        own_state = float(rng.uniform(0.0, 1.0 - own_row))
        p = replace(
            scenarios.BASE,
            periods=periods,
            n_agents=800,
            own_row=own_row,
            own_state=own_state,
            own_households=1.0 - own_row - own_state,
            tax_corp=0.2,
            tax_income=0.2,
            tax_wealth=tax_wealth,
            ubi=scenarios.UBI,
            seed=k,
        )
        h = Model(p).run()
        rows.append((tax_wealth, own_row, own_state,
                     h.top1_share[-1], h.gini[-1],
                     h.gov_nw[-1] / p.n_agents, h.row_nw[-1] / p.n_agents))
    arr = np.array(rows)
    # correlations of levers with outcomes
    def corr(i, j):
        return float(np.corrcoef(arr[:, i], arr[:, j])[0, 1])

    print("Correlations (Spearman-like Pearson on sweep):")
    print(f"  wealth-tax rate  vs final Gini        : {corr(0, 4):+.2f}")
    print(f"  wealth-tax rate  vs government NW      : {corr(0, 5):+.2f}")
    print(f"  foreign-own share vs government NW     : {corr(1, 5):+.2f}")
    print(f"  foreign-own share vs RoW NW (leakage)  : {corr(1, 6):+.2f}")
    print(f"  state-own share  vs government NW      : {corr(2, 5):+.2f}")


if __name__ == "__main__":
    run_scenarios()
    monte_carlo()
