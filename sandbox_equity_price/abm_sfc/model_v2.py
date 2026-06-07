"""AB-SFC model v2: endogenous production, portfolio choice, richer policy.

Built on the v1 accounting backbone (same two invariants, still asserted in
tests) with mechanisms added as switches so v1 is recoverable as a special
case. New blocks, by plan phase:

  P1 endogenous production  : output, wages and the return on capital come from
                              the task-based CES block (production.py); K
                              accumulates; r and the growth rate g are endogenous.
  P2 concentration coupling : the kinetic noise sigma scales with the capital
                              share, so automation drives the *speed* of
                              wealth condensation (kappa).
  P3 portfolio choice       : households split saving between deposits and newly
                              issued equity; equity is valued at book (NAV). When
                              wage income collapses, saving and hence household
                              equity purchases collapse, so ownership can only
                              broaden through purchases that have dried up
                              (buyer-collapse channel).
  P4 policy                 : progressive wealth tax (bracket schedule) and a
                              citizens'-fund mode in which the state holds equity
                              as custodian and rebates its dividends per capita.
  P5 heterogeneous labour   : a skill distribution plus an automation-driven
                              displacement process, so technological unemployment
                              is explicit rather than a uniformly shrinking wage.

The two invariants (sector deposits net to zero; sum of sector net worths equals
the capital stock K) hold every period by double-entry construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinetic import kinetic_step
from .production import Production
from .model import gini


@dataclass
class ParamsV2:
    # population / horizon
    n_agents: int = 2000
    periods: int = 300

    # production (CES task block)
    A: float = 0.45
    A_L: float = 1.0
    A_K: float = 1.0
    eps: float = 0.6             # elasticity of substitution between task bundles
    K0_per_capita: float = 6.0   # initial capital per worker (sets initial K/L)
    depreciation: float = 0.05

    # automation index I_t in [0,1]: logistic ramp (interpreted as task share)
    I_base: float = 0.50         # capital-task share before the AI ramp (gives pi~0.23)
    auto_start: int = 80
    auto_speed: float = 0.06
    auto_max: float = 0.45       # ramp adds up to this, so I_t rises 0.50 -> ~0.95

    # firm policy
    div_payout: float = 0.6
    r_required: float = 0.04     # required net return; investment closes the gap
    inv_speed: float = 0.5       # speed of investment response to the return gap

    # ownership split of equity at t=0 (must sum to 1)
    own_households: float = 1.0
    own_state: float = 0.0
    own_row: float = 0.0

    # household behaviour
    c_income: float = 0.80       # MPC out of disposable income
    c_wealth: float = 0.03       # MPC out of net worth
    phi_equity: float = 0.5      # fraction of saving households try to put in equity

    # heterogeneous labour (P5)
    skill_dispersion: float = 0.0    # sd of log skill; 0 => uniform labour (v1-like)
    displacement: bool = False       # if True, automation unemploys a share of workers

    # policy
    tax_corp: float = 0.0
    tax_income: float = 0.0
    tax_wealth: float = 0.0          # flat-rate leg (used if no brackets given)
    wealth_exempt: float = 0.0
    wealth_brackets: tuple = ()      # ((threshold, rate), ...) progressive schedule
    ubi: float = 0.0
    gov_cost: float = 0.10
    r_debt: float = 0.02
    citizens_fund: bool = False      # state rebates dividends on its equity per capita

    # distributional engine
    bm_J: float = 0.10
    bm_sigma: float = 0.55
    bm_dt: float = 0.05
    bm_substeps: int = 4
    kappa: float = 0.0               # coupling: sigma_eff = sigma*(1+kappa*(pi/pi_ref-1))

    seed: int = 0

    def validate(self) -> None:
        s = self.own_households + self.own_state + self.own_row
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"ownership shares must sum to 1, got {s}")


@dataclass
class HistoryV2:
    gini: list = field(default_factory=list)
    top1_share: list = field(default_factory=list)
    labour_share: list = field(default_factory=list)
    gov_nw: list = field(default_factory=list)
    row_nw: list = field(default_factory=list)
    house_nw: list = field(default_factory=list)
    K: list = field(default_factory=list)
    Y: list = field(default_factory=list)
    gov_balance: list = field(default_factory=list)
    I_auto: list = field(default_factory=list)
    r_net: list = field(default_factory=list)
    g_rate: list = field(default_factory=list)
    wage: list = field(default_factory=list)
    unemployment: list = field(default_factory=list)
    eq_purchase_house: list = field(default_factory=list)
    house_eq_frac: list = field(default_factory=list)


class ModelV2:
    def __init__(self, p: ParamsV2):
        p.validate()
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        n = p.n_agents
        self.prod = Production(A=p.A, A_L=p.A_L, A_K=p.A_K, eps=p.eps)

        # labour endowment / skills
        if p.skill_dispersion > 0:
            s = self.rng.lognormal(0.0, p.skill_dispersion, n)
            self.skill = s / s.mean()
        else:
            self.skill = np.ones(n)
        self.employed = np.ones(n, dtype=float)

        # initial capital and equity
        self.L0 = float(self.skill.sum())
        self.K = p.K0_per_capita * n
        self.eq_state = self.K * p.own_state
        self.eq_row = self.K * p.own_row
        seed_w = self.rng.lognormal(0.0, 0.5, n)
        self.h_eq = (self.K * p.own_households) * seed_w / seed_w.sum()

        self.h_dep = np.zeros(n)
        self.gov_dep = 0.0
        self.row_dep = 0.0
        self.firm_dep = 0.0
        self.inventories = 0.0

        # reference capital share at baseline automation (for the kappa coupling)
        _, _, pi_ref, _ = self.prod.factor_prices(self.K, self.L0, p.I_base)
        self.pi_ref = max(pi_ref, 1e-3)

        self.t = 0
        self.hist = HistoryV2()

    # ---- helpers ----
    def I_t(self) -> float:
        p = self.p
        z = p.auto_speed * (self.t - p.auto_start)
        ramp = p.auto_max / (1.0 + np.exp(-z))
        return min(max(p.I_base + ramp, p.I_base), 0.999)

    def house_nw(self) -> np.ndarray:
        return self.h_eq + self.h_dep

    def total_nw(self) -> float:
        firm_nw = (self.K + self.inventories + self.firm_dep
                   - (self.h_eq.sum() + self.eq_state + self.eq_row))
        return (self.house_nw().sum() + self.eq_state + self.gov_dep
                + self.eq_row + self.row_dep + firm_nw)

    def real_assets(self) -> float:
        return self.K + self.inventories

    def deposits_sum(self) -> float:
        return float(self.h_dep.sum() + self.gov_dep + self.row_dep + self.firm_dep)

    def _wealth_tax(self, nw: np.ndarray) -> np.ndarray:
        p = self.p
        if p.wealth_brackets:
            tax = np.zeros_like(nw)
            for thr, rate in p.wealth_brackets:
                tax += rate * np.clip(nw - thr, 0, None)
            return np.minimum(tax, np.clip(nw, 0, None))
        tax = p.tax_wealth * np.clip(nw - p.wealth_exempt, 0, None)
        return np.minimum(tax, np.clip(nw, 0, None))

    # ---- one period ----
    def step(self) -> None:
        p = self.p
        n = p.n_agents
        I = self.I_t()

        # ----- labour supply: displacement (P5) -----
        if p.displacement:
            # employment probability falls as automation rises (reduced form)
            p_emp = np.clip(1.0 - I, 0.02, 1.0)
            self.employed = (self.rng.random(n) < p_emp).astype(float)
        L = float((self.skill * self.employed).sum())
        L = max(L, 1e-6)

        # ----- production -----
        Y = self.prod.output(self.K, L, I)
        wage_bill, capital_income, pi, r_gross = self.prod.factor_prices(self.K, L, I)
        r_net = r_gross - p.depreciation

        # wage paid per efficiency unit, distributed to employed workers by skill
        wage_per_unit = wage_bill / L
        wage_i = wage_per_unit * self.skill * self.employed

        # ----- firm: tax on net profit, dividends, retained -----
        net_profit = capital_income - p.depreciation * self.K
        corp_tax = p.tax_corp * np.clip(net_profit, 0, None)
        after_tax = net_profit - corp_tax
        dividends = np.clip(p.div_payout * after_tax, 0, None)
        retained = after_tax - dividends

        eq_total = self.h_eq.sum() + self.eq_state + self.eq_row
        f_house = self.h_eq / eq_total
        f_state = self.eq_state / eq_total
        f_row = self.eq_row / eq_total

        div_house = dividends * f_house
        div_state = dividends * f_state
        div_row = dividends * f_row

        # interest on deposit balances (nets to zero across sectors)
        int_house = p.r_debt * self.h_dep
        int_gov = p.r_debt * self.gov_dep
        int_row = p.r_debt * self.row_dep
        int_firm = p.r_debt * self.firm_dep

        # ----- citizens' fund: rebate state dividends per capita (P4) -----
        # The state always RECEIVES its dividend (div_state); under the fund it
        # then pays an equal per-capita rebate out again. Net effect on the state
        # balance is zero, but the cash genuinely flows firm -> state -> households.
        if p.citizens_fund:
            fund_rebate = np.full(n, div_state / n)
        else:
            fund_rebate = np.zeros(n)

        # ----- household income, taxes, consumption -----
        gross_income = wage_i + div_house + int_house
        income_tax = p.tax_income * np.clip(gross_income, 0, None)
        nw = self.house_nw()
        ubi_i = np.full(n, p.ubi)

        wealth_tax = self._wealth_tax(nw)
        pos_eq = np.clip(self.h_eq, 0, None)
        pos_dep = np.clip(self.h_dep, 0, None)
        pos_nw = pos_eq + pos_dep + 1e-12
        wt_eq = wealth_tax * pos_eq / pos_nw      # surrendered as equity to the state
        wt_cash = wealth_tax - wt_eq

        disposable = gross_income - income_tax - wt_cash + ubi_i + fund_rebate
        desired_c = np.clip(p.c_income * disposable + p.c_wealth * np.clip(nw, 0, None), 0, None)
        max_c = np.clip(self.h_dep + disposable, 0, None)
        consumption = np.minimum(desired_c, max_c)
        C = consumption.sum()

        # ----- government -----
        G = p.gov_cost * Y
        ubi_total = ubi_i.sum()
        gov_balance = (corp_tax + income_tax.sum() + wt_cash.sum() + div_state
                       + int_gov - ubi_total - G - fund_rebate.sum())

        # ----- investment / capital -----
        # Behavioural investment (not a pure residual): firms invest to close the
        # gap between the marginal return on capital and a required return, plus
        # replacement of depreciation. This disciplines accumulation so capital
        # does not run away when the marginal product is temporarily high.
        required_return = p.r_required
        inv_response = p.inv_speed * (r_net - required_return) * self.K
        I_gross = p.depreciation * self.K + np.clip(inv_response, -0.1 * self.K, 0.1 * self.K)
        I_gross = max(I_gross, 0.0)
        dK = I_gross - p.depreciation * self.K
        # any output not consumed or invested is absorbed by firm inventories
        # (firm net worth), so the goods market still closes with no output lost.
        inventory_change = Y - C - G - I_gross

        # ================= deposit transfers (pre-equity-purchase) =================
        self.h_dep += (wage_i + div_house + int_house + ubi_i + fund_rebate
                       - consumption - income_tax - wt_cash)
        self.gov_dep += (corp_tax + income_tax.sum() + wt_cash.sum() + div_state
                         + int_gov - ubi_total - G - fund_rebate.sum())
        self.row_dep += (div_row + int_row)
        self.firm_dep += (C + G) - wage_bill - corp_tax - dividends + int_firm

        # firm real assets: capital and inventories of unsold output
        self.inventories += inventory_change

        # ----- equity transfers -----
        # wealth-tax equity leg: households -> state
        self.h_eq -= wt_eq
        self.eq_state += wt_eq.sum()
        # retained earnings booked pro-rata to existing owners (concentrating force)
        self.h_eq += retained * f_house
        self.eq_state += retained * f_state
        self.eq_row += retained * f_row

        # ----- portfolio choice: new equity purchases out of saving (P3) -----
        # external equity financing need this period
        PP = max(dK - retained, 0.0)
        saving = np.clip(self.h_dep, 0, None)             # liquid available to invest
        desired_house = p.phi_equity * saving
        tot_des = desired_house.sum()
        if PP > 0 and tot_des > 0:
            scale = min(1.0, PP / tot_des)
            buy = desired_house * scale
            self.h_dep -= buy
            self.h_eq += buy
            self.firm_dep += float(buy.sum())             # cash to firm for new shares
            eq_purchase_house = float(buy.sum())
        else:
            eq_purchase_house = 0.0

        # ----- capital stock -----
        self.K += dK

        # ----- distributional engine with capital-share coupling (P2) -----
        sigma_eff = p.bm_sigma * (1.0 + p.kappa * (pi / self.pi_ref - 1.0))
        sigma_eff = float(np.clip(sigma_eff, 1e-3, 5.0))
        for _ in range(p.bm_substeps):
            self.h_eq = kinetic_step(self.h_eq, p.bm_J, sigma_eff, p.bm_dt, self.rng)

        # ----- record -----
        nwh = self.house_nw()
        self.hist.gini.append(gini(nwh))
        srt = np.sort(nwh)
        self.hist.top1_share.append(float(srt[int(0.99 * n):].sum() / max(srt.sum(), 1e-9)))
        self.hist.labour_share.append(1.0 - pi)
        self.hist.gov_nw.append(self.eq_state + self.gov_dep)
        self.hist.row_nw.append(self.eq_row + self.row_dep)
        self.hist.house_nw.append(float(nwh.sum()))
        self.hist.K.append(self.K)
        self.hist.Y.append(Y)
        self.hist.gov_balance.append(gov_balance)
        self.hist.I_auto.append(I)
        self.hist.r_net.append(r_net)
        self.hist.g_rate.append(dK / max(self.K - dK, 1e-9))
        self.hist.wage.append(wage_per_unit)
        self.hist.unemployment.append(1.0 - self.employed.mean())
        self.hist.eq_purchase_house.append(eq_purchase_house)
        self.hist.house_eq_frac.append(float(self.h_eq.sum() / (self.h_eq.sum() + self.eq_state + self.eq_row)))

        self.t += 1

    def run(self) -> HistoryV2:
        for _ in range(self.p.periods):
            self.step()
        return self.hist
