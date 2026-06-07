"""Item 4: global (Sobol) sensitivity analysis.

Addresses the over-parameterisation criticism head-on: rather than asserting the
results are robust, we quantify how much each free parameter drives the two
headline outcomes (terminal wealth Gini and terminal government net worth /
output). Sobol first-order (S1) and total (ST) indices decompose the output
variance across parameters. A result that hinges on one or two parameters is
fragile; one whose variance is spread is more robust, and either way the reader
sees which knobs matter.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model_v3 import ModelV3
from . import scenarios_v3 as sv3


# the free parameters we vary, with plausible ranges. Note c_profit (the
# propensity to consume out of capital income) replaces the old inv_speed knob:
# under the residual-investment closure investment is no longer a behavioural
# speed, whereas c_profit sets the aggregate saving rate and is a genuine driver.
PROBLEM = {
    "num_vars": 8,
    "names": ["eps", "ret_sigma", "ret_persist", "c_profit",
              "avoidance_elasticity", "migration_semi_elast", "c_wealth", "demographic_reset"],
    "bounds": [[0.4, 0.9], [0.03, 0.08], [0.85, 0.97], [0.20, 0.50],
               [0.3, 1.2], [0.005, 0.04], [0.01, 0.05], [0.01, 0.04]],
}


def _evaluate(values, base_factory, periods=200, n_agents=500, seed=0):
    p = base_factory()
    p = replace(p, periods=periods, n_agents=n_agents, seed=seed,
                eps=values[0], ret_sigma=values[1], ret_persist=values[2],
                c_profit=values[3], avoidance_elasticity=values[4],
                migration_semi_elast=values[5], c_wealth=values[6],
                demographic_reset=values[7])
    h = ModelV3(p).run()
    gini = float(np.mean(h.gini[-30:]))
    gov = float(h.gov_nw[-1] / p.n_agents / max(h.Y[-1] / p.n_agents, 1e-9))
    return gini, gov


def run_sobol(base_name="wealth_tax_ubi", N=64):
    """Return Sobol S1/ST for Gini and gov-nw outcomes.

    N is the base sample size; total model runs = N*(2D+2). D=8 -> 18*N.
    Use a modest N here for tractability; raise for a publication run.
    """
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    base_factory = sv3.REGISTRY[base_name]
    X = sobol_sample.sample(PROBLEM, N, calc_second_order=False)
    Yg = np.empty(X.shape[0]); Yv = np.empty(X.shape[0])
    for i, row in enumerate(X):
        g, v = _evaluate(row, base_factory, seed=i % 4)
        Yg[i] = g; Yv[i] = v
    Sg = sobol_analyze.analyze(PROBLEM, Yg, calc_second_order=False, print_to_console=False)
    Sv = sobol_analyze.analyze(PROBLEM, Yv, calc_second_order=False, print_to_console=False)
    return PROBLEM["names"], Sg, Sv


if __name__ == "__main__":
    names, Sg, Sv = run_sobol(N=64)
    print("Sobol indices on the wealth-tax scenario (N=64; ~1150 runs)")
    print(f"{'parameter':22s}{'Gini S1':>10}{'Gini ST':>10}{'govNW S1':>10}{'govNW ST':>10}")
    for i, nm in enumerate(names):
        print(f"{nm:22s}{Sg['S1'][i]:10.3f}{Sg['ST'][i]:10.3f}{Sv['S1'][i]:10.3f}{Sv['ST'][i]:10.3f}")
