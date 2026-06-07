"""Hand-checkable cases and qualitative behaviour.

These pin down the economics with cases simple enough to verify by hand, then
confirm the headline qualitative results the model exists to demonstrate.
"""
import numpy as np

from abm_sfc import Model, Params
from abm_sfc import scenarios


def test_no_automation_labour_share_constant():
    """With auto_max=0 the capital share never moves, so labour share == 1-pi0
    in every period. Hand-checkable."""
    p = Params(n_agents=100, periods=40, pi0=0.3, auto_max=0.0, seed=0)
    h = Model(p).run()
    assert np.allclose(h.labour_share, 0.70, atol=1e-12)


def test_full_automation_drives_labour_share_to_zero():
    p = Params(n_agents=200, periods=300, pi0=0.3,
               auto_start=80, auto_speed=0.06, auto_max=0.999, seed=0)
    h = Model(p).run()
    assert h.labour_share[0] > 0.69                 # starts near 1 - pi0
    assert h.labour_share[-1] < 0.05                 # ends near 0


def test_government_balance_identity():
    """Government balance reported each period must equal revenue minus outlays,
    reconstructed independently from the deposit change net of dividends/interest
    is awkward; instead verify the *flow* definition directly via a one-period
    hand computation with no automation and known taxes."""
    p = Params(n_agents=10, periods=1, pi0=0.3, auto_max=0.0,
               tax_corp=0.5, div_payout=0.0, tax_income=0.0, tax_wealth=0.0,
               ubi=0.0, gov_cost=0.0, own_state=0.0, own_households=1.0, seed=0)
    m = Model(p)
    Y = p.Y0
    gross_profit = p.pi0 * Y
    expected_corp_tax = p.tax_corp * gross_profit    # only revenue this period
    m.step()
    assert abs(m.hist.gov_balance[0] - expected_corp_tax) < 1e-9


def test_closed_economy_recovered_when_row_share_zero():
    """own_row=0 must reproduce the closed economy bit-for-bit: RoW net worth
    stays exactly zero throughout."""
    p = scenarios.income_tax_ubi()
    p.periods = 80
    h = Model(p).run()
    assert max(abs(x) for x in h.row_nw) < 1e-9


def test_foreign_ownership_leaks_wealth_abroad():
    """With foreign ownership, RoW net worth must grow strictly positive as
    dividends and retained earnings are repatriated."""
    p = scenarios.foreign_ownership()
    p.periods = 150
    h = Model(p).run()
    assert h.row_nw[-1] > 1.0
    assert h.row_nw[-1] > h.row_nw[10]               # monotone-ish accumulation


def test_income_tax_alone_does_not_prevent_condensation():
    """Reproduce the kinetic-theory result (Bouchaud-Mezard; Barros & Martins):
    a flow tax (income/corporate) funding a UBI barely dents wealth condensation,
    whereas a stock tax (wealth) does. Concentration is driven by the equity
    stock, which flow taxes never touch."""
    base = Model(scenarios.laissez_faire()).run()
    inc = Model(scenarios.income_tax_ubi()).run()
    weal = Model(scenarios.wealth_tax_ubi()).run()
    # income tax leaves the top share essentially where laissez-faire leaves it
    assert abs(inc.top1_share[-1] - base.top1_share[-1]) < 0.02
    # the stock (wealth) tax cuts the top share by a large margin
    assert weal.top1_share[-1] < 0.5 * inc.top1_share[-1]
    assert weal.gini[-1] < inc.gini[-1]


def test_wealth_tax_improves_fiscal_position():
    """Socialising ownership through a wealth tax leaves the state with higher
    net worth than funding the same UBI from income tax alone."""
    inc = Model(scenarios.income_tax_ubi()).run()
    weal = Model(scenarios.wealth_tax_ubi()).run()
    assert weal.gov_nw[-1] > inc.gov_nw[-1]
