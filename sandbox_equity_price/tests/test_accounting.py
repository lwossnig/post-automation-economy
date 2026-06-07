"""Watertight-accounting tests.

These are the tests that justify calling the model stock-flow *consistent*.
Both invariants must hold to machine precision at every period, for every
parameterisation, including with taxes, UBI, foreign ownership and the
stochastic kinetic kernel switched on.
"""
import numpy as np
import pytest

from abm_sfc import Model, Params


def deposits_sum(m: Model) -> float:
    return float(m.h_dep.sum() + m.gov_dep + m.row_dep + m.firm_dep)


PARAM_SETS = [
    Params(n_agents=300, periods=60, seed=1),                       # bare baseline
    Params(n_agents=300, periods=60, auto_max=0.9, auto_start=20,
           tax_corp=0.25, tax_income=0.3, tax_wealth=0.02, ubi=1.0,
           gov_cost=0.1, seed=2),                                   # full policy
    Params(n_agents=300, periods=60, own_households=0.3, own_state=0.4,
           own_row=0.3, auto_max=0.8, tax_corp=0.2, ubi=0.5, seed=3),
]


@pytest.mark.parametrize("p", PARAM_SETS)
def test_deposits_net_to_zero_every_period(p):
    m = Model(p)
    for _ in range(p.periods):
        m.step()
        assert abs(deposits_sum(m)) < 1e-6, "deposits must net to zero"


@pytest.mark.parametrize("p", PARAM_SETS)
def test_total_net_worth_equals_capital(p):
    m = Model(p)
    for _ in range(p.periods):
        m.step()
        assert abs(m.total_nw() - m.K) < 1e-6, "sum of net worths must equal K"


@pytest.mark.parametrize("p", PARAM_SETS)
def test_change_in_total_nw_equals_dK(p):
    """d(sum of net worths) must equal the change in the capital stock."""
    m = Model(p)
    prev_nw = m.total_nw()
    prev_K = m.K
    for _ in range(p.periods):
        m.step()
        d_nw = m.total_nw() - prev_nw
        d_K = m.K - prev_K
        assert abs(d_nw - d_K) < 1e-6
        prev_nw, prev_K = m.total_nw(), m.K
