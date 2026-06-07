"""v2 scenario presets.

Shared automating baseline; scenarios differ in ownership and policy. All use
the endogenous-production, portfolio-choice model. Mechanisms can be toggled per
hypothesis (kappa for the concentration coupling, citizens_fund, brackets, etc).
"""
from __future__ import annotations

from dataclasses import replace

from .model_v2 import ParamsV2

BASE = ParamsV2(
    n_agents=2000,
    periods=300,
    eps=0.6,
    I_base=0.50,
    auto_start=80,
    auto_speed=0.06,
    auto_max=0.45,            # I_t rises 0.50 -> ~0.95
    div_payout=0.6,
    r_required=0.04,
    inv_speed=0.5,
    c_income=0.80,
    c_wealth=0.03,
    phi_equity=0.5,
    gov_cost=0.10,
    r_debt=0.02,
    bm_J=0.10,
    bm_sigma=0.55,
    kappa=0.0,
    seed=0,
)

UBI = 0.30


def laissez_faire():
    return replace(BASE, own_households=1.0, tax_corp=0.0, tax_income=0.0,
                   tax_wealth=0.0, ubi=0.0)


def income_tax_ubi():
    return replace(BASE, own_households=1.0, tax_corp=0.25, tax_income=0.30,
                   tax_wealth=0.0, ubi=UBI)


def wealth_tax_ubi():
    return replace(BASE, own_households=1.0, tax_corp=0.25, tax_income=0.15,
                   tax_wealth=0.02, ubi=UBI)


def progressive_wealth():
    return replace(BASE, own_households=1.0, tax_corp=0.25, tax_income=0.15,
                   wealth_brackets=((3.0, 0.01), (10.0, 0.03)), ubi=UBI)


def state_ownership():
    return replace(BASE, own_households=0.4, own_state=0.6, tax_corp=0.25,
                   tax_income=0.15, ubi=UBI)


def citizens_fund():
    return replace(BASE, own_households=0.4, own_state=0.6, citizens_fund=True,
                   tax_corp=0.25, tax_income=0.15, ubi=UBI)


def foreign_ownership():
    return replace(BASE, own_households=0.4, own_row=0.6, tax_corp=0.25,
                   tax_income=0.30, ubi=UBI)


def coupled():
    """Concentration-speed coupling switched on (kappa>0)."""
    return replace(BASE, own_households=1.0, kappa=1.0, tax_corp=0.0,
                   tax_income=0.0, ubi=0.0)


REGISTRY = {
    "laissez_faire": laissez_faire,
    "income_tax_ubi": income_tax_ubi,
    "wealth_tax_ubi": wealth_tax_ubi,
    "progressive_wealth": progressive_wealth,
    "state_ownership": state_ownership,
    "citizens_fund": citizens_fund,
    "foreign_ownership": foreign_ownership,
}
