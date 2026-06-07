"""v2 tests: accounting invariants, production economics, and behaviour.

The invariants are the same two as v1, generalised: with behavioural investment
the firm holds inventories of unsold output, so the real-asset total is
K + inventories, and the sum of sector net worths must equal that.
"""
import numpy as np
import pytest

from abm_sfc.model_v2 import ModelV2, ParamsV2
from abm_sfc.production import Production
from abm_sfc import scenarios_v2 as sv2


CONFIGS = [
    dict(auto_max=0.0),
    dict(tax_corp=0.25, tax_income=0.2, tax_wealth=0.02, ubi=0.3, kappa=1.0),
    dict(own_households=0.4, own_state=0.3, own_row=0.3, tax_corp=0.2),
    dict(own_households=0.4, own_state=0.6, citizens_fund=True),
    dict(wealth_brackets=((2.0, 0.01), (10.0, 0.03)), ubi=0.3),
    dict(skill_dispersion=0.6, displacement=True, ubi=0.3),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_deposits_net_to_zero(cfg):
    p = ParamsV2(n_agents=300, periods=120, seed=1, **cfg)
    m = ModelV2(p)
    for _ in range(p.periods):
        m.step()
        assert abs(m.deposits_sum()) < 1e-4


@pytest.mark.parametrize("cfg", CONFIGS)
def test_net_worth_equals_real_assets(cfg):
    p = ParamsV2(n_agents=300, periods=120, seed=1, **cfg)
    m = ModelV2(p)
    for _ in range(p.periods):
        m.step()
        assert abs(m.total_nw() - m.real_assets()) < 1e-4


# ---- production economics ----

def test_factor_payments_exhaust_output():
    pr = Production(eps=0.6)
    for I in [0.1, 0.3, 0.6, 0.9]:
        wb, ci, pi, rg = pr.factor_prices(K=12000.0, L=2000.0, I=I)
        Y = pr.output(12000.0, 2000.0, I)
        assert abs((wb + ci) - Y) < 1e-6 * Y          # Euler exhaustion
        assert abs(pi - ci / Y) < 1e-9                # capital share consistency


def test_capital_share_rises_with_automation():
    pr = Production(eps=0.6)
    shares = [pr.factor_prices(12000.0, 2000.0, I)[2] for I in [0.2, 0.5, 0.8, 0.95]]
    assert all(b > a for a, b in zip(shares, shares[1:]))


def test_no_automation_is_stationary():
    p = ParamsV2(n_agents=400, periods=200, seed=0, auto_max=0.0)
    h = ModelV2(p).run()
    assert abs(h.Y[-1] / h.Y[80] - 1.0) < 0.05        # output roughly constant
    assert abs(h.g_rate[-1]) < 0.01                   # growth ~ 0


# ---- behaviour ----

def test_full_automation_collapses_labour_share():
    p = sv2.laissez_faire()
    h = ModelV2(p).run()
    assert h.labour_share[0] > 0.6
    assert h.labour_share[-1] < 0.55                  # capital share has risen


def test_closed_economy_when_no_foreign():
    p = sv2.income_tax_ubi()
    p.periods = 100
    h = ModelV2(p).run()
    assert max(abs(x) for x in h.row_nw) < 1e-6


def test_foreign_ownership_leaks_abroad():
    p = sv2.foreign_ownership()
    p.periods = 150
    h = ModelV2(p).run()
    assert h.row_nw[-1] > h.row_nw[10]


def test_coupling_raises_concentration_with_automation():
    """With kappa>0, automation should raise the stationary Gini relative to the
    same run with the coupling off."""
    off = sv2.coupled(); off.kappa = 0.0
    on = sv2.coupled(); on.kappa = 1.0
    g_off = np.mean(ModelV2(off).run().gini[-40:])
    g_on = np.mean(ModelV2(on).run().gini[-40:])
    assert g_on > g_off


def test_citizens_fund_reduces_inequality_vs_state_keep():
    """Rebating state dividends per capita should leave lower inequality than the
    state keeping them, holding ownership fixed."""
    keep = ModelV2(sv2.state_ownership()).run()
    fund = ModelV2(sv2.citizens_fund()).run()
    assert np.mean(fund.gini[-40:]) < np.mean(keep.gini[-40:])


def test_equity_purchases_collapse_after_boom():
    """Household equity purchases (out of saving) collapse once the
    automation-driven capital-deepening boom ends and net investment returns to
    zero. (In v2, with endogenous capital deepening, wages rise in absolute
    terms, so this collapse is driven by investment demand ending, not by the
    v1 affordability channel.) Use income_tax_ubi where households do buy."""
    p = sv2.income_tax_ubi()
    h = ModelV2(p).run()
    peak = np.max(h.eq_purchase_house)
    late = np.mean(h.eq_purchase_house[-40:])
    assert peak > 0.0
    assert late < 0.1 * peak
