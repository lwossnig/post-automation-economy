"""Scenario presets: the regimes you actually want to compare.

Each returns a Params. The shared baseline is an economy that automates over the
horizon (capital share rises towards `auto_max`). The scenarios differ only in
*who owns the automated capital* and *how it is taxed*, which is the whole point.
"""
from __future__ import annotations

from dataclasses import replace

from .model import Params

BASE = Params(
    n_agents=2000,
    periods=300,
    Y0=0.0,                 # -> set to n_agents (mean per-capita output = 1)
    pi0=0.30,
    auto_start=80,
    auto_speed=0.06,
    auto_max=0.95,          # near-full automation by the end of the horizon
    div_payout=0.6,
    depreciation=0.05,
    c_income=0.80,
    c_wealth=0.03,
    r_debt=0.02,
    bm_J=0.10,              # tail exponent alpha = 1 + 2J/sigma^2 ~ 1.66 (heavy, realistic)
    bm_sigma=0.55,
    bm_dt=0.05,
    bm_substeps=4,
    gov_cost=0.10,          # state running cost = 10% of output
    seed=0,
)

# per-capita UBI (mean per-capita output is 1.0), sized to be roughly fundable
UBI = 0.25


def _ubi(frac: float = UBI) -> float:
    return frac


def laissez_faire() -> Params:
    """Automation, private household ownership, no redistribution. The control."""
    return replace(BASE, own_households=1.0, own_state=0.0, own_row=0.0,
                   tax_corp=0.0, tax_income=0.0, tax_wealth=0.0, ubi=0.0)


def income_tax_ubi() -> Params:
    """Fund a UBI from a flat income + corporate tax. Tests the doc-1 result that
    income tax alone cannot stop condensation."""
    return replace(BASE, own_households=1.0,
                   tax_corp=0.25, tax_income=0.30, tax_wealth=0.0,
                   ubi=_ubi())   # per-capita UBI ~ a slice of mean output


def wealth_tax_ubi() -> Params:
    """Same UBI but funded partly by a wealth tax on the stock, which the
    kinetic literature says is what actually shifts the condensation phase."""
    return replace(BASE, own_households=1.0,
                   tax_corp=0.25, tax_income=0.15, tax_wealth=0.02,
                   wealth_exempt=0.0, ubi=_ubi())


def state_ownership() -> Params:
    """The state is majority shareholder; profit returns to the fisc as
    dividends rather than as a distortionary tax (sovereign-wealth logic)."""
    return replace(BASE, own_households=0.4, own_state=0.6, own_row=0.0,
                   tax_corp=0.25, tax_income=0.15, tax_wealth=0.0,
                   ubi=_ubi())


def foreign_ownership() -> Params:
    """A large share of the automated capital is owned abroad: dividends and
    retained earnings leak out, eroding the personal-tax base."""
    return replace(BASE, own_households=0.4, own_state=0.0, own_row=0.6,
                   tax_corp=0.25, tax_income=0.30, tax_wealth=0.0,
                   ubi=_ubi())


REGISTRY = {
    "laissez_faire": laissez_faire,
    "income_tax_ubi": income_tax_ubi,
    "wealth_tax_ubi": wealth_tax_ubi,
    "state_ownership": state_ownership,
    "foreign_ownership": foreign_ownership,
}
