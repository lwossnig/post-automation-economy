"""Agent-based, stock-flow-consistent model of an automating economy.

Design contract
---------------
Four sectors carry balance sheets: households (N agents), the firm /
capital-owning sector, government, and rest-of-world (RoW). Two asset classes:

  * ``dep``    : a liquid claim (think government money / deposits). By
                 construction every transaction is a transfer between two
                 deposit accounts, so the deposits of the four sectors sum to
                 zero at all times.
  * ``eq``     : equity claims on the firm's productive capital stock ``K``.

Two invariants hold every period by construction of the double-entry transfers
(and are asserted in the tests):

    (1) the four sectors' deposit balances sum to zero;
    (2) the sum of all sector net worths equals the real capital stock K
        (all financial claims net out).

Hence d(sum of net worths) = dK = I - depreciation*K each period. The firm
sector holds any residual operating cash as part of its own net worth.

Mechanisms
----------
* Automation: capital's share of value added rises with the automation index
  alpha_t,   pi_t = pi0 + (1 - pi0) * alpha_t   (reduced form of the
  Acemoglu-Restrepo result that automation raises the capital share). At
  alpha=1 labour earns nothing.
* Distribution: equity income (dividends + booked retained earnings) accrues
  pro-rata to equity held, so wealth begets wealth; on top, the conservative
  Bouchaud-Mezard kernel reshuffles household equity (asset trading) and drives
  Pareto-tailed condensation.
* Ownership: equity is split between households, the state, and RoW. State
  equity turns corporate profit into a non-distortionary fiscal revenue stream;
  RoW equity sends dividends + retained earnings abroad (profit repatriation),
  eroding the personal-tax base while leaving corporate tax intact.
* Fiscal loop: government taxes (corporate, personal income, wealth), receives
  its own dividends, pays UBI + running cost + debt interest, and runs the
  residual into its deposit balance (negative balance = public debt).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinetic import kinetic_step


@dataclass
class Params:
    # population / horizon
    n_agents: int = 2000
    periods: int = 200

    # production
    Y0: float = 0.0             # initial real output; if 0, set to n_agents (per-capita ~1)
    g_Y: float = 0.0            # exogenous output growth per period (baseline 0)
    pi0: float = 0.30           # capital share of value added with no automation

    # automation index alpha_t in [0, 1]: logistic ramp
    auto_start: int = 50        # period at which the ramp is centred
    auto_speed: float = 0.10    # logistic steepness
    auto_max: float = 0.0       # ceiling of the ramp (0 = no automation baseline)

    # firm profit / investment policy
    div_payout: float = 0.6     # fraction of after-tax profit paid as dividends
    depreciation: float = 0.05

    # ownership split of equity (must sum to 1)
    own_households: float = 1.0
    own_state: float = 0.0
    own_row: float = 0.0

    # behaviour (Godley-Lavoie consumption out of income and wealth)
    c_income: float = 0.80
    c_wealth: float = 0.04

    # policy
    tax_corp: float = 0.0
    tax_income: float = 0.0
    tax_wealth: float = 0.0     # flat wealth tax on household net worth above exemption
    wealth_exempt: float = 0.0
    ubi: float = 0.0            # per-capita UBI per period (real)
    gov_cost: float = 0.0       # government running cost as a fraction of output
    r_debt: float = 0.02        # interest on deposit balances (paid by debtors)

    # distributional engine (Bouchaud-Mezard)
    bm_J: float = 0.5
    bm_sigma: float = 0.3
    bm_dt: float = 0.05
    bm_substeps: int = 4

    seed: int = 0

    def validate(self) -> None:
        s = self.own_households + self.own_state + self.own_row
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"ownership shares must sum to 1, got {s}")


@dataclass
class History:
    gini: list[float] = field(default_factory=list)
    top1_share: list[float] = field(default_factory=list)
    labour_share: list[float] = field(default_factory=list)
    gov_nw: list[float] = field(default_factory=list)
    row_nw: list[float] = field(default_factory=list)
    house_nw: list[float] = field(default_factory=list)
    K: list[float] = field(default_factory=list)
    gov_balance: list[float] = field(default_factory=list)   # surplus(+)/deficit(-)
    alpha: list[float] = field(default_factory=list)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.clip(x - x.min(), 0, None)) if x.min() < 0 else np.sort(x)
    n = x.size
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


class Model:
    """A single deterministic+stochastic run of the AB-SFC economy."""

    def __init__(self, p: Params):
        p.validate()
        if p.Y0 <= 0:
            p.Y0 = float(p.n_agents)      # per-capita output ~ 1
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        n = p.n_agents

        # --- initial stocks ---
        self.K = p.Y0 * 2.0                       # capital stock (arbitrary scale)
        # equity split across owners
        self.eq_state = self.K * p.own_state
        self.eq_row = self.K * p.own_row
        eq_house_total = self.K * p.own_households
        # spread household equity unevenly to seed a realistic starting point
        seed_w = self.rng.lognormal(0.0, 0.5, n)
        self.h_eq = eq_house_total * seed_w / seed_w.sum()

        # deposits start at zero for every sector (they net to zero forever)
        self.h_dep = np.zeros(n)
        self.gov_dep = 0.0
        self.row_dep = 0.0
        self.firm_dep = 0.0

        self.t = 0
        self.hist = History()

    # ---- helpers -------------------------------------------------------
    def alpha_t(self) -> float:
        p = self.p
        if p.auto_max <= 0:
            return 0.0
        z = p.auto_speed * (self.t - p.auto_start)
        return p.auto_max / (1.0 + np.exp(-z))

    def house_nw(self) -> np.ndarray:
        return self.h_eq + self.h_dep

    def total_nw(self) -> float:
        return (self.house_nw().sum()
                + (self.eq_state + self.gov_dep)
                + (self.eq_row + self.row_dep)
                + (self.K + self.firm_dep - (self.h_eq.sum() + self.eq_state + self.eq_row)))

    # ---- one period ----------------------------------------------------
    def step(self) -> None:
        p = self.p
        n = p.n_agents
        Y = p.Y0 * (1.0 + p.g_Y) ** self.t
        alpha = self.alpha_t()
        pi = p.pi0 + (1.0 - p.pi0) * alpha            # capital share rises w/ automation

        # ---------- factor income ----------
        wage_bill = (1.0 - pi) * Y
        gross_profit = pi * Y
        wage_i = np.full(n, wage_bill / n)            # uniform labour endowment (v1)

        # ---------- firm: tax, dividends, retained earnings ----------
        corp_tax = p.tax_corp * gross_profit
        after_tax = gross_profit - corp_tax
        dividends = p.div_payout * after_tax
        retained = after_tax - dividends              # booked as new equity (pro-rata)

        # equity ownership fractions BEFORE booking retained earnings
        eq_total = self.h_eq.sum() + self.eq_state + self.eq_row
        f_house = self.h_eq / eq_total
        f_state = self.eq_state / eq_total
        f_row = self.eq_row / eq_total

        # dividends paid in cash (deposits) pro-rata to equity
        div_house = dividends * f_house
        div_state = dividends * f_state
        div_row = dividends * f_row

        # deposit interest (debtors pay, creditors receive; sums to zero)
        int_house = p.r_debt * self.h_dep
        int_gov = p.r_debt * self.gov_dep
        int_row = p.r_debt * self.row_dep
        int_firm = p.r_debt * self.firm_dep   # closes the interest transfer to zero sum

        # ---------- household income, taxes, consumption ----------
        gross_income = wage_i + div_house + int_house
        income_tax = p.tax_income * np.clip(gross_income, 0, None)
        nw = self.house_nw()
        ubi_i = np.full(n, p.ubi)

        # Wealth tax falls on the wealth STOCK and is collected from equity and
        # cash in proportion to how each household holds its (positive) wealth.
        # The equity leg transfers ownership to the state, which is what actually
        # de-concentrates holdings; the cash leg is ordinary revenue.
        wealth_tax = p.tax_wealth * np.clip(nw - p.wealth_exempt, 0, None)
        wealth_tax = np.minimum(wealth_tax, np.clip(nw, 0, None))   # cannot exceed wealth
        pos_eq = np.clip(self.h_eq, 0, None)
        pos_dep = np.clip(self.h_dep, 0, None)
        pos_nw = pos_eq + pos_dep + 1e-12
        wt_eq = wealth_tax * pos_eq / pos_nw      # paid by surrendering equity to the state
        wt_cash = wealth_tax - wt_eq              # paid in cash

        # cash-flow disposable income (equity leg does not consume liquidity)
        disposable = gross_income - income_tax - wt_cash + ubi_i
        desired_c = np.clip(p.c_income * disposable + p.c_wealth * np.clip(nw, 0, None), 0, None)
        # no unlimited borrowing: consumption cannot push liquid deposits below zero
        max_c = np.clip(self.h_dep + disposable, 0, None)
        consumption = np.minimum(desired_c, max_c)
        C = consumption.sum()

        # ---------- government ----------
        G = p.gov_cost * Y
        ubi_total = ubi_i.sum()
        gov_revenue = corp_tax + income_tax.sum() + wt_cash.sum() + div_state + int_gov
        gov_outlay = ubi_total + G
        gov_balance = gov_revenue - gov_outlay        # +surplus / -deficit (cash basis)

        # ---------- investment / capital ----------
        # goods-market closure: output not consumed by households or government
        # is investment (so no output is lost); capital then depreciates.
        I = Y - C - G
        dK = I - p.depreciation * self.K

        # ================= apply double-entry deposit transfers =================
        # households (cash legs only)
        self.h_dep += (wage_i + div_house + int_house + ubi_i
                       - consumption - income_tax - wt_cash)
        # government (residual into its deposit -> negative balance is public debt)
        self.gov_dep += (corp_tax + income_tax.sum() + wt_cash.sum() + div_state
                         + int_gov - ubi_total - G)
        # rest of world
        self.row_dep += (div_row + int_row)
        # firm cash flow (interest leg included so all deposit flows net to zero)
        self.firm_dep += (C + G) - wage_bill - corp_tax - dividends + int_firm

        # ---------- equity transfers ----------
        # wealth-tax equity leg: ownership moves from households to the state
        self.h_eq -= wt_eq
        self.eq_state += wt_eq.sum()
        # book retained earnings pro-rata to (pre-transfer) ownership
        self.h_eq += retained * f_house
        self.eq_state += retained * f_state
        self.eq_row += retained * f_row

        # ---------- capital stock ----------
        self.K += dK
        # Invariants (asserted in tests): the four sectors' deposits sum to zero,
        # and the sum of all sector net worths equals K. The firm sector holds the
        # residual cash (firm_dep) as part of its own net worth.

        # ---------- distributional engine: reshuffle household equity ----------
        for _ in range(p.bm_substeps):
            self.h_eq = kinetic_step(self.h_eq, p.bm_J, p.bm_sigma, p.bm_dt, self.rng)

        # ---------- record ----------
        nw_house = self.house_nw()
        self.hist.alpha.append(alpha)
        self.hist.labour_share.append(1.0 - pi)
        self.hist.gini.append(gini(nw_house))
        srt = np.sort(nw_house)
        self.hist.top1_share.append(float(srt[int(0.99 * n):].sum() / max(srt.sum(), 1e-9)))
        self.hist.gov_nw.append(self.eq_state + self.gov_dep)
        self.hist.row_nw.append(self.eq_row + self.row_dep)
        self.hist.house_nw.append(float(nw_house.sum()))
        self.hist.K.append(self.K)
        self.hist.gov_balance.append(gov_balance)

        self.t += 1

    def run(self) -> History:
        for _ in range(self.p.periods):
            self.step()
        return self.hist
