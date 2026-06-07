"""AB-SFC model v3 (corrected closure): microfounded concentration, behavioural
taxes, endogenous capital mobility, and a CONSISTENT goods market.

This builds on the v2 accounting backbone and the CES production block. Relative
to the first v3 draft it CORRECTS the goods-market closure, which is what makes
the level and fiscal results trustworthy. Four design pillars:

  (1) BEHAVIOURAL WEALTH TAX + ENDOGENOUS CAPITAL MOBILITY.
      The wealth-tax base erodes with the rate (avoidance), calibrated to the
      bunching elasticity of taxable wealth w.r.t. the net-of-tax rate
      (Brulhart et al. 2022, ~0.7-0.8). Domestically held equity also relocates
      abroad in response to the tax (capital flight), calibrated to a ~2%-per-pp
      semi-elasticity (Jakobsen-Kleven-Zucman). The foreign ownership share is an
      ENDOGENOUS state variable: the tax itself moves capital.

  (2) MICROFOUNDED CONCENTRATION (no imposed kinetic kernel).
      Households earn heterogeneous, persistent returns on equity (Fagereng et
      al. 2020). The Pareto tail emerges from this random-growth process
      (Benhabib-Bisin-Zhu) plus a small demographic reset, not from a
      mean-reverting kernel.

  (3) CONSISTENT GOODS MARKET (the correction, neoclassical closure).
      Output is supply-determined at full utilisation, Y = Y_potential from the
      CES, so factor incomes satisfy Euler exactly. The goods market is closed by
      letting INVESTMENT be the clearing residual, I = Y - C - G: whatever output
      is not consumed or bought by government is invested, so nothing ever piles
      up as unsold inventory and there is no demand-deficiency spiral. Under gross
      complements (eps < 1) potential output is bounded in K while depreciation is
      linear in K, so the capital stock converges to a finite steady state rather
      than collapsing or exploding; the consumption wealth effect sets where that
      steady state sits (richer households consume more, raising C and lowering
      residual investment as capital deepens). This is the standard neoclassical
      saving=investment closure used in the automation-and-distribution literature
      (e.g. Moll-Rachel-Restrepo), chosen over a demand-determined closure because
      the latter produces a paradox-of-thrift collapse once the labour share falls.
      Equity grows by EXACTLY the change in the capital stock dK, financed by
      retained earnings (booked pro-rata) and, when dK exceeds internal funds, by
      new equity issued to savers. Consequently total equity claims equal the
      capital stock K at all times (eq/K = 1), the firm carries no drifting cash
      buffer (firm_dep ~ 0), and government / foreign net-worth levels are
      economically meaningful. Both diagnostics are recorded every period.

  (4) CALIBRATION DISCIPLINE.
      The CES elasticity is held in the empirically defensible gross-complements
      range; consumption propensities are calibrated so the no-automation
      baseline is stationary with a realistic capital-output ratio; the initial
      wealth distribution is seeded to a plausible Gini; and the return-dispersion
      primitive is taken from the returns-to-wealth literature.

Two accounting invariants hold every period by double-entry construction and are
asserted in the tests:
  (i)  the four sectors' deposit balances sum to zero;
  (ii) the sum of sector net worths equals the capital stock K
       (there are no inventories under the corrected closure, so real assets = K).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .production import Production
from .model import gini


@dataclass
class ParamsV3:
    # ---- population and horizon ----
    n_agents: int = 2000
    periods: int = 300

    # ---- production (task-based CES block) ----
    A: float = 0.45                # total factor productivity (sets output scale)
    A_L: float = 1.0               # labour-augmenting productivity
    A_K: float = 1.0               # capital-augmenting productivity
    eps: float = 0.6               # elasticity of substitution (gross complements)
    K0_per_capita: float = 5.0     # initial capital per agent (baseline K/Y ~ 2.8)
    depreciation: float = 0.05     # capital depreciation per period

    # ---- automation index I_t in [0,1]: logistic ramp in the capital-task share ----
    I_base: float = 0.50           # capital-task share before the AI ramp (pi ~ 0.23)
    auto_start: int = 80           # midpoint period of the ramp
    auto_speed: float = 0.06       # steepness of the ramp
    auto_max: float = 0.45         # ramp adds up to this, so I rises 0.50 -> ~0.95

    # ---- firm investment and dividend policy ----
    # Output is supply-determined (Y = potential) and investment is the goods-
    # market-clearing residual I = Y - C - G (see step, section 6). The capital
    # stock is disciplined by the consumption wealth effect plus gross-complements
    # diminishing returns, NOT by the behavioural-investment knobs below, which
    # are retained for backward compatibility and as reporting diagnostics only
    # (they no longer drive accumulation).
    inv_speed: float = 0.30        # [unused under residual closure] kept for compat
    inv_adjust: float = 8.0        # [unused under residual closure] kept for compat
    r_required: float = 0.04       # [diagnostic] reference net return for reporting
    max_g: float = 0.06            # [unused under residual closure] kept for compat
    div_payout: float = 1.0        # max share of free cash flow paid as dividends

    # ---- equity ownership at t=0 (own_row then evolves endogenously) ----
    own_households: float = 1.0
    own_state: float = 0.0
    own_row: float = 0.0

    # ---- household behaviour (differential-saving consumption) ----
    # Consumption = c_income * labour disposable income
    #             + c_profit * capital disposable income
    #             + c_wealth * productive equity.
    # The split propensities are the classical Kaldor (1957) / Pasinetti (1962)
    # device: workers consume most of their wages, capital owners save most of
    # their profits. It is also what makes the steady state STABLE under the
    # residual-investment closure. When capital becomes scarce its return spikes
    # (gross complements); with a single high propensity that windfall would be
    # consumed, starving investment and decumulating K without limit. A low
    # propensity to consume profits means the scarcity rent is reinvested instead,
    # so the capital stock self-corrects and K/Y settles at a finite value. The
    # wealth term is the secondary stabiliser (deeper capital -> more consumption
    # -> less residual investment). Calibrated so the no-automation baseline is
    # stationary at K/Y ~ 2.9, converging from any initial capital level.
    c_income: float = 0.85         # propensity to consume out of labour income
    c_profit: float = 0.35         # propensity to consume out of capital income (rest saved)
    c_wealth: float = 0.03         # propensity to consume out of equity wealth
    phi_equity: float = 0.5        # [unused under saving-financed issuance] kept for compat

    # ---- heterogeneous labour (optional) ----
    skill_dispersion: float = 0.0  # sd of log skill; 0 => homogeneous labour
    displacement: bool = False     # if True, automation unemploys a share of workers

    # ---- policy instruments ----
    tax_corp: float = 0.0
    tax_income: float = 0.0
    tax_wealth: float = 0.0
    wealth_exempt: float = 0.0
    wealth_brackets: tuple = ()    # ((threshold, rate), ...) progressive schedule
    ubi: float = 0.0
    gov_cost: float = 0.10         # government consumption as a share of output
    r_debt: float = 0.02           # interest on deposit balances (the r in r>g)
    citizens_fund: bool = False    # state rebates its dividends per capita
    # Source-based withholding tax on capital income leaving the country: a levy
    # on dividends repatriated to foreign owners, collected at source (where the
    # automated output is produced) before the income leaves. tax_repat is the
    # rate; if repat_rebate is True the proceeds are paid straight back to
    # residents as an equal per-capita transfer (tax-the-robot-owner, rebate-to-
    # citizens), otherwise they are ordinary government revenue. With profit
    # shifting (below) the same rate also acts as a royalty withholding tax.
    tax_repat: float = 0.0
    repat_rebate: bool = False
    # Profit shifting by foreign-operated automation (the US-company-in-host-state
    # case). foreign_operated is the share of the capital stock run by a foreign
    # multinational that owns the IP; profit_shift is the fraction of that
    # operation's economic profit relocated abroad each period as a deductible
    # royalty / licence / management charge, recognised in the parent's
    # jurisdiction rather than the host's. The royalty is deducted BEFORE the
    # host's corporate tax (so it erodes the domestic taxable base) and leaves
    # immediately as cash to the rest of the world (it is not reinvested as
    # domestic equity). tax_repat acts as the host's withholding tax on that
    # royalty. foreign_operated is exogenous and fixed, so the extraction is a
    # structural feature rather than something that dilutes away with the equity
    # stake; it is deliberately distinct from own_row (legal equity ownership),
    # because a multinational can strip profit via IP charges regardless of who
    # holds the subsidiary's shares.
    profit_shift: float = 0.0
    foreign_operated: float = 0.0

    # ---- TWO-CHANNEL AUTOMATION (v4): AI vs robotic, separately taxed ----
    # When two_channel is True the single capital-task CES is replaced by the
    # nested three-cluster structure (production_v4): a routine cluster pairing
    # routine labour with robotic capital K_r, a cognitive cluster pairing
    # cognitive labour with AI compute capital K_ai, and a top CES between the
    # two clusters. Robots displace routine labour, AI displaces cognitive
    # labour, and e_top above the within-cluster elasticities makes the two
    # capitals more substitutable with each other than with their own labour.
    two_channel: bool = False
    e_top: float = 1.20            # elasticity between routine and cognitive clusters
    e_routine: float = 0.60        # within-routine elasticity (gross complements)
    e_cog: float = 0.60            # within-cognitive elasticity
    theta_cog: float = 0.50        # top-level weight on the cognitive cluster
    cognitive_share: float = 0.50  # fraction of workers (top by skill) doing cognitive work
    # two independent automation ramps (capital-task share inside each cluster)
    a_r_base: float = 0.50         # robotic task share before the ramp (matches v3 I_base)
    a_r_max: float = 0.45          # robotic ramp addition (-> 0.95)
    a_r_start: int = 60            # robots ramp earlier...
    a_r_speed: float = 0.05        # ...and slower
    a_ai_base: float = 0.50        # AI task share before the ramp (matches v3 I_base)
    a_ai_max: float = 0.49         # AI ramp addition (-> 0.99, larger reach)
    a_ai_start: int = 110          # AI ramps later...
    a_ai_speed: float = 0.10       # ...and faster
    # ---- Phase 2: reinstatement margin (Acemoglu-Restrepo new tasks) ----
    # Automation raises the capital-task share, but new labour-intensive tasks are
    # created in its wake and pull that share back down. reinstate_frac (rho) is the
    # long-run fraction of the automation increment that is offset by new tasks;
    # reinstate_lag is how far this offset trails automation (new tasks take time to
    # emerge). With rho=0 the ramp is unoffset (the prior model exactly); with rho>0
    # the capital-task share settles below its automation ceiling, so labour keeps a
    # standing share of each cluster's tasks rather than being driven to near-zero.
    reinstate_frac: float = 0.0
    reinstate_lag: int = 40
    # ---- Phase 2 (piece 2): unemployment from automation ----
    # The labour-share fall (automation net of reinstatement) is realised on the
    # EXTENSIVE margin: a fraction of the fall shows up as workers with no job
    # rather than as uniformly lower wages. This is output-neutral by construction
    # (the displaced tasks are done by capital, which is why output holds while the
    # job goes), so it adds an unemployment rate, within-labour inequality, and a
    # safety-net cost without disturbing the production or accumulation path.
    # unemployment_pass_through (lambda_u) is the fraction of the labour-share fall
    # turned into unemployment; unemployment_benefit (b_u) is the benefit paid to
    # each unemployed worker as a fraction of the economy's average wage, financed
    # by the state; ubi_labour_supply adds voluntary non-participation rising with
    # the UBI replacement rate. All zero by default (full employment, prior model).
    unemployment_pass_through: float = 0.0
    unemployment_benefit: float = 0.0
    ubi_labour_supply: float = 0.0
    # two capital stocks: split of the initial stock and separate depreciation
    robot_capital_share0: float = 0.5   # share of initial K that is robotic (rest AI compute)
    depreciation_r: float = 0.05        # robotic capital depreciation
    depreciation_ai: float = 0.18       # AI compute depreciates faster (obsolescence)
    invest_damping: float = 0.25        # speed of return-equalising reallocation (0..1)
    # AI rent (IP monopoly markup) and where the compute sits
    mu_frac: float = 0.0           # IP rent as a share of cognitive-cluster value added
    # ---- Phase 3: endogenous markup via market contestability ----
    # The AI rent is not a fixed share if the frontier is contestable. competition
    # (kappa) erodes the markup toward zero: the effective rent is mu_frac*(1-kappa),
    # so kappa = 0 is the monopolistic frontier (the prior model) and kappa = 1 is a
    # fully commoditised one in which open-weight competition prices the rent away.
    # This makes competition policy an alternative lever to taxation: competition
    # shrinks the rent (no fiscal handle left), taxation captures it (rent persists).
    competition: float = 0.0
    # ---- rising-rent extensions (both default 0 = the flat-markup baseline) ----
    # markup_power: the markup itself RISES with AI automation, representing greater
    # pricing power as the model becomes more essential. The effective markup is
    # mu_frac*(1-competition)*(1 + markup_power*auto), with auto in [0,1] the degree
    # of AI automation (a_ai from its base toward its ceiling). 1.0 roughly doubles
    # the markup at full automation. cognitive_capture: the AI/cognitive cluster
    # captures a GROWING share of total output as it automates (the top-level weight
    # theta rises with auto), representing AI-native production displacing the
    # human/routine cluster rather than only substituting within the cognitive one.
    markup_power: float = 0.0
    cognitive_capture: float = 0.0
    # robot_ip: an embodied-IP rent on the robotic cluster (default 0). Robots are
    # hardware (competitive) plus IP (the control/foundation model, scarce). This
    # peels a foreign-held markup from the robotic cluster's value added, exactly as
    # the AI markup is peeled from the cognitive cluster; it is reached by the same
    # source levy and missed by the hardware robot tax. 0 = robots earn no rent.
    robot_ip: float = 0.0
    s_home: float = 1.0            # fraction of AI compute located on home servers
    # ai_ip_foreign is the share of the AI IP (and so of its rent) owned abroad. At 1
    # the rent leaves as a cross-border licence fee and is reachable only by source
    # levies: the rent-importer case (the UK with a US/Chinese owner), which is the
    # paper's main setting. At 0 the rent stays home as ordinary capital income to
    # resident owners, reachable by the domestic income and wealth taxes: the
    # domestic-owner case (the US with a home-grown owner). Default 1 reproduces the
    # prior model exactly.
    ai_ip_foreign: float = 1.0
    # differential instruments
    robot_tax: float = 0.0         # source tax on robotic capital income (fully collected)
    dst_ai: float = 0.0            # digital-services levy on AI revenue (rent + AI capital income)
    # ---- Phase 1a: open-economy goods trade (current account) ----
    # trade_leak (phi) is the fraction of the foreign owner's AFTER-HOST-TAX rent
    # that is repatriated as REAL RESOURCES: the owner consumes domestic output
    # (an export) instead of reinvesting the cash in domestic equity. Exports are
    # final demand and so come out of investment (lower capital formation), and the
    # repatriated rent buys goods rather than ownership. phi=0 reproduces the
    # closed, fully-reinvested model exactly; phi=1 takes the entire net rent out
    # as goods, so the flow drains but no equity is acquired. The truth lies
    # between, so phi indexes the ownership result from a bound to a range.
    trade_leak: float = 0.0
    # ---- Phase 1b: AI-supply response to the net-of-tax rent ----
    # The AI owner deploys less capability when its rent is taxed: the effective AI
    # task share is scaled by the net-of-tax retention rate raised to this
    # elasticity, keep_ai = (1 - dst_ai - tax_repat) ** ai_supply_elasticity. This
    # is the AI-side analogue of the wealth-tax avoidance elasticity and gives the
    # levy an efficiency cost (less AI, lower output, a smaller rent base), so
    # taxing the rent is no longer free. ai_supply_elasticity=0 reproduces the
    # no-response model exactly.
    ai_supply_elasticity: float = 0.0
    # ---- Phase 3: optimising foreign owner (profit-shifting of the rent) ----
    # The owner relocates where the IP rent is BOOKED in response to the host's tax
    # wedge on it. Moving the servers does not help (the levy is on recognised value,
    # not location, see section 6K), but shifting the recognition of the licence fee
    # to a low-tax entity does. A share of the rent, rising with the wedge and capped
    # because some rent is tied to recognised domestic use, escapes the host's DST and
    # withholding. This is the AI-side analogue of profit-shifting and turns the
    # territoriality result into a bounded one. owner_shift_elasticity = 0 is the
    # non-strategic owner (the prior model exactly).
    owner_shift_elasticity: float = 0.0

    # ---- item 1: behavioural responses to the wealth tax ----
    avoidance_elasticity: float = 0.75   # d ln(taxable base) / d ln(1 - tau_w)
    migration_semi_elast: float = 0.02   # target extra foreign share per pp of tau_w
    mobility_on: bool = True             # if True, capital location responds to the tax

    # ---- item 2: heterogeneous persistent returns ----
    ret_sigma: float = 0.05        # cross-sectional SD of idiosyncratic wealth returns
    ret_persist: float = 0.92      # AR(1) persistence of an agent's return type
    ret_scale_dep: float = 0.0     # optional: returns rising in wealth rank (0 = off)
    demographic_reset: float = 0.02  # per-period replacement prob (pins the Pareto tail)

    # ---- initial wealth distribution ----
    init_wealth_sigma: float = 1.15  # log-sd of seed equity (initial Gini ~ 0.6)

    seed: int = 0

    def validate(self) -> None:
        s = self.own_households + self.own_state + self.own_row
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"ownership shares must sum to 1, got {s}")


@dataclass
class HistoryV3:
    """Per-period recorded series. Distribution metrics come in two flavours:
    'domestic' (on net worth held onshore) and 'true' (including each household's
    offshore wealth attributed back to it), so the composition effect of capital
    flight is visible."""
    gini: list = field(default_factory=list)              # domestic wealth Gini
    gini_true: list = field(default_factory=list)         # incl. offshore wealth
    top1_share: list = field(default_factory=list)        # domestic top 1% share
    top1_share_true: list = field(default_factory=list)   # incl. offshore
    top10_share: list = field(default_factory=list)
    labour_share: list = field(default_factory=list)
    gov_nw: list = field(default_factory=list)            # eq_state + gov_dep
    row_nw: list = field(default_factory=list)            # eq_row + row_dep
    house_nw: list = field(default_factory=list)
    K: list = field(default_factory=list)
    Y: list = field(default_factory=list)
    gov_balance: list = field(default_factory=list)       # government primary+interest balance
    I_auto: list = field(default_factory=list)            # automation index
    r_net: list = field(default_factory=list)             # net return on capital
    g_rate: list = field(default_factory=list)            # capital growth rate
    own_row: list = field(default_factory=list)           # endogenous foreign equity share
    offshore_share: list = field(default_factory=list)    # offshore / true household wealth
    wealth_tax_base_frac: list = field(default_factory=list)  # taxable base after avoidance
    repat_revenue: list = field(default_factory=list)     # withholding on repatriated dividends
    royalty: list = field(default_factory=list)           # profit shifted abroad (royalty outflow)
    corp_tax_rev: list = field(default_factory=list)       # corporate tax actually collected
    K_r: list = field(default_factory=list)                # robotic capital stock
    K_ai: list = field(default_factory=list)               # AI compute capital stock
    rent_ai: list = field(default_factory=list)            # AI IP rent (markup) per period
    rent_robot: list = field(default_factory=list)         # embodied-IP rent on robots (default 0)
    a_r: list = field(default_factory=list)                # robotic task share
    a_ai: list = field(default_factory=list)               # AI task share
    w_Lr: list = field(default_factory=list)               # routine wage bill
    w_Lc: list = field(default_factory=list)               # cognitive wage bill
    ci_Kr: list = field(default_factory=list)              # robotic capital income
    ci_Kai: list = field(default_factory=list)             # AI capital income
    exports: list = field(default_factory=list)            # rent repatriated as real goods (Phase 1a)
    unemployment: list = field(default_factory=list)       # unemployment rate (Phase 2 piece 2)
    capital_flight: list = field(default_factory=list)    # equity relocated abroad this period
    eq_to_K: list = field(default_factory=list)           # equity claims / K (diagnostic; should be ~1)
    firm_dep: list = field(default_factory=list)          # firm cash buffer (diagnostic; should be ~0)


class ModelV3:
    def __init__(self, p: ParamsV3):
        p.validate()
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        n = p.n_agents
        self.prod = Production(A=p.A, A_L=p.A_L, A_K=p.A_K, eps=p.eps)
        if p.two_channel:
            from .production_v4 import NestedProduction
            self.prod4 = NestedProduction(
                A=p.A, e_top=p.e_top, e_routine=p.e_routine, e_cog=p.e_cog,
                theta=p.theta_cog)

        # --- labour endowment: per-agent skill (efficiency units) and employment ---
        if p.skill_dispersion > 0:
            s = self.rng.lognormal(0.0, p.skill_dispersion, n)
            self.skill = s / s.mean()           # normalise mean skill to 1
        else:
            self.skill = np.ones(n)
        self.employed = np.ones(n, dtype=float)

        # --- capital and its ownership ---
        # The capital stock K is owned via equity claims split between households
        # (a per-agent vector h_eq), the state (eq_state), and the rest of the
        # world (eq_row). By construction the three sum to K at all times.
        self.K = p.K0_per_capita * n
        if p.two_channel:
            # two capital stocks summing to K; cognitive workers are the top
            # cognitive_share of agents by skill (AI displaces higher-skill work)
            self.K_r = self.K * p.robot_capital_share0
            self.K_ai = self.K - self.K_r
            k = int(round(p.cognitive_share * n))
            order = np.argsort(self.skill)            # ascending skill
            self.cognitive = np.zeros(n, dtype=bool)
            if k > 0:
                self.cognitive[order[n - k:]] = True   # top-skill agents are cognitive
        self.eq_state = self.K * p.own_state
        self.eq_row = self.K * p.own_row
        seed_w = self.rng.lognormal(0.0, p.init_wealth_sigma, n)   # heavy-tailed seed
        self.h_eq = (self.K * p.own_households) * seed_w / seed_w.sum()

        # --- deposits (liquid claims). By construction these net to zero across
        # the four sectors every period. Households start with none. ---
        self.h_dep = np.zeros(n)
        self.gov_dep = 0.0
        self.row_dep = 0.0
        self.firm_dep = 0.0          # firm cash buffer; the corrected closure keeps this ~0

        # --- persistent idiosyncratic return type (AR(1)), mean zero ---
        self.ret_z = self.rng.standard_normal(n)

        # --- per-agent offshore wealth: equity each household has relocated abroad,
        # kept attributed to its original owner so we can measure TRUE inequality.
        # It is grown each period at the same reinvestment rate as onshore equity,
        # so it reflects not just the fled principal but its subsequent growth. ---
        self.offshore = np.zeros(n)

        # --- lagged distributed income, used in the consumption decision to break
        # the consumption<->output simultaneity. Output is demand-determined, so
        # consumption must be predetermined (decided on income already received).
        # We carry each agent's disposable income from last period. ---
        self.div_house_lag = np.zeros(n)
        self.fund_rebate_lag = np.zeros(n)
        self.disp_lag = np.zeros(n)        # per-agent disposable income, previous period
        # lagged net return on capital, used to predetermine the investment
        # decision (so output can be demand-determined without a simultaneity).
        self.r_lag = 0.06
        # lagged SUSTAINABLE capital-income yield per unit of equity = last period's
        # after-tax profit / equity. This is what the consumption rule treats as
        # capital income, deliberately EXCLUDING any capital returned through
        # share buybacks when the firm disinvests (dK < 0). Consuming buyback /
        # liquidation proceeds as if they were permanent income is what produced a
        # runaway disinvestment spiral; anchoring on the sustainable yield removes
        # it while leaving the steady state (where buybacks are zero) unchanged.
        self.eq_yield_lag = 0.0

        self.t = 0
        self.hist = HistoryV3()

    # ----------------------------------------------------------------- helpers
    def I_t(self) -> float:
        """Automation index at the current period: a logistic ramp from I_base to
        I_base + auto_max, centred on auto_start."""
        p = self.p
        z = p.auto_speed * (self.t - p.auto_start)
        ramp = p.auto_max / (1.0 + np.exp(-z))
        return min(max(p.I_base + ramp, p.I_base), 0.999)

    def _logistic_ramp(self, base, mx, start, speed):
        z = speed * (self.t - start)
        return min(max(base + mx / (1.0 + np.exp(-z)), base), 0.999)

    def a_r_t(self) -> float:
        """Robotic capital-task share inside the routine cluster."""
        p = self.p
        return self._logistic_ramp(p.a_r_base, p.a_r_max, p.a_r_start, p.a_r_speed)

    def a_ai_t(self) -> float:
        """AI capital-task share inside the cognitive cluster."""
        p = self.p
        return self._logistic_ramp(p.a_ai_base, p.a_ai_max, p.a_ai_start, p.a_ai_speed)

    def _reinstate_share(self) -> float:
        """Phase 2: the fraction of competitive capital income reinstated to labour
        by new tasks. Ramps from zero to reinstate_frac, trailing the AI automation
        ramp by reinstate_lag periods (new tasks emerge in the wake of automation)."""
        p = self.p
        if p.reinstate_frac <= 0.0:
            return 0.0
        z = p.a_ai_speed * (self.t - p.a_ai_start - p.reinstate_lag)
        return float(p.reinstate_frac / (1.0 + np.exp(-z)))

    def _employment(self, a_r, a_ai):
        """Phase 2 (piece 2): turn part of the labour-share fall into unemployment.

        Production is untouched (the full effective labour still enters the CES, so
        output and accumulation stay on their path). This is purely an extensive-
        margin overlay: a fraction of the labour DISPLACED by automation (net of
        reinstatement) is recorded as out of work rather than as a lower wage for
        all. The unchanged cluster wage bill is then concentrated on those still
        employed (section 7), and the out-of-work receive a transfer (section 2).
        Displacement falls on the lowest-skill workers within each cluster first,
        which matches the entry-level pattern the policy debate describes. With
        unemployment_pass_through and ubi_labour_supply both zero this returns the
        all-employed mask, so the prior model is reproduced exactly."""
        p = self.p
        emp = self.employed.copy()
        if not p.two_channel or (p.unemployment_pass_through <= 0.0 and p.ubi_labour_supply <= 0.0):
            return emp
        rs = self._reinstate_share()
        def disp(a, base):  # fractional fall in the cluster's labour-task share, net of reinstatement
            return float(np.clip((a - base) / max(1.0 - base, 1e-6), 0.0, 1.0)) * (1.0 - rs)
        u_vol = float(np.clip(p.ubi_labour_supply * p.ubi, 0.0, 0.5)) if p.ubi_labour_supply > 0 else 0.0
        u_by_cluster = {True: float(np.clip(p.unemployment_pass_through * disp(a_ai, p.a_ai_base) + u_vol, 0.0, 0.95)),
                        False: float(np.clip(p.unemployment_pass_through * disp(a_r, p.a_r_base) + u_vol, 0.0, 0.95))}
        for is_cog, u in u_by_cluster.items():
            idx = np.where(self.cognitive == is_cog)[0]
            if len(idx) == 0 or u <= 0.0:
                continue
            order = idx[np.argsort(self.skill[idx])]        # lowest skill displaced first
            target = u * float(self.skill[idx].sum())        # effective labour to displace
            k = int(np.searchsorted(np.cumsum(self.skill[order]), target))
            if k > 0:
                emp[order[:k]] = 0.0
        return emp

    def _decompose(self, K_r, K_ai, L_r, L_c, a_r, a_ai):
        """Technological decomposition with the reinstatement margin applied as an
        OUTPUT-NEUTRAL income shift. Automation sets output and the capital stock
        (so the scale does not collapse); reinstatement then moves a share of each
        cluster's competitive capital income to its labour, capturing that new tasks
        are as productive as the automated ones but are performed by workers. With
        reinstate_frac = 0 this is exactly prod4.decompose."""
        if self.p.cognitive_capture > 0.0:
            auto_int = float(np.clip((a_ai - self.p.a_ai_base) / max(self.p.a_ai_max, 1e-9), 0.0, 1.0))
            theta = self.p.theta_cog + self.p.cognitive_capture * auto_int * (1.0 - self.p.theta_cog)
            self.prod4.theta = float(np.clip(theta, 0.0, 0.98))
        d = self.prod4.decompose(K_r, K_ai, L_r, L_c, a_r, a_ai)
        rs = self._reinstate_share()
        if rs > 0.0:
            for ci, w in (("ci_Kr", "w_Lr"), ("ci_Kai", "w_Lc")):
                shift = rs * d[ci]
                d[w] += shift
                d[ci] -= shift
            d["r_Kr"] = d["ci_Kr"] / K_r if K_r > 0 else 0.0
            d["r_Kai"] = d["ci_Kai"] / K_ai if K_ai > 0 else 0.0
        return d

    def house_nw(self) -> np.ndarray:
        """Per-agent domestic net worth = equity held onshore + deposits."""
        return self.h_eq + self.h_dep

    def total_nw(self) -> float:
        """Sum of all sector net worths. Firm net worth = its assets (capital plus
        any cash buffer) minus the equity claims outstanding against it."""
        firm_nw = (self.K + self.firm_dep
                   - (self.h_eq.sum() + self.eq_state + self.eq_row))
        return (self.house_nw().sum() + self.eq_state + self.gov_dep
                + self.eq_row + self.row_dep + firm_nw)

    def real_assets(self) -> float:
        """The economy's real assets. With the corrected closure there are no
        inventories, so this is just the capital stock."""
        return self.K

    def deposits_sum(self) -> float:
        """Should be 0 every period (double-entry check)."""
        return float(self.h_dep.sum() + self.gov_dep + self.row_dep + self.firm_dep)

    def _enforce_household_floor(self):
        """No-overdraft constraint on households, a net-zero transfer WITHIN the
        household sector. Deposits are holdings of the economy's single
        fixed-income asset; a household cannot issue it, so its balance cannot go
        negative. Any momentarily-negative household is raised to zero and the
        (small) shortfall is recouped pro-rata from households holding positive
        balances. This touches no other sector, so the four deposit balances still
        net to zero and total net worth still equals K. It is the residual cleanup
        for the expectation error between section-5 (expected) and section-8
        (realised) income; equity issuance is funded from the saving FLOW
        (section 10) so it does not by itself overdraw anyone. Settling every
        period also stops a negative balance being carried and charged overdraft
        interest, which previously corrupted the Gini. Returns the shortfall."""
        neg = np.clip(self.h_dep, None, 0.0)     # negative parts (<= 0), per agent
        shortfall = float(neg.sum())             # total household overdraft (<= 0)
        if shortfall < 0.0:
            self.h_dep -= neg                    # raise every negative balance to 0
            pos = np.clip(self.h_dep, 0.0, None)
            ps = pos.sum()
            if ps > 1e-12:
                self.h_dep -= pos * (-shortfall) / ps   # savers fund it, pro-rata
        return shortfall

    def _gross_wealth_tax(self, nw):
        """Statutory wealth-tax liability per agent, before behavioural avoidance.
        Supports either a flat rate above an exemption or a progressive schedule of
        TRUE MARGINAL brackets. For brackets ((t0, r0), (t1, r1), ...) sorted by
        threshold, the marginal rate r_i applies only to wealth in the band
        [t_i, t_{i+1}); the top rate applies to all wealth above the last
        threshold. (The earlier implementation summed rate_i * max(nw - t_i, 0)
        across brackets, which stacked the rates into a cumulative surtax: wealth
        above the top threshold was taxed at the SUM of all bracket rates. This
        version taxes each band once at its own marginal rate, the conventional
        meaning of a progressive schedule.)"""
        p = self.p
        if p.wealth_brackets:
            brk = sorted(p.wealth_brackets, key=lambda b: b[0])
            tax = np.zeros_like(nw)
            for i, (thr, rate) in enumerate(brk):
                upper = brk[i + 1][0] if i + 1 < len(brk) else np.inf
                band = np.clip(np.minimum(nw, upper) - thr, 0.0, None)
                tax += rate * band
            return tax
        return p.tax_wealth * np.clip(nw - p.wealth_exempt, 0, None)

    def _effective_wealth_rate(self):
        """Top statutory wealth rate, used for the behavioural-response strength."""
        p = self.p
        if p.wealth_brackets:
            return max(r for _, r in p.wealth_brackets)
        return p.tax_wealth

    # -------------------------------------------------------------- one period
    def step(self) -> None:
        p = self.p
        n = p.n_agents
        I = self.I_t()

        # === 1. LABOUR SUPPLY ===============================================
        # Optional displacement: as automation rises, a share of workers lose
        # employment (probability of being employed falls with I).
        if p.displacement:
            p_emp = np.clip(1.0 - I, 0.02, 1.0)
            self.employed = (self.rng.random(n) < p_emp).astype(float)
        L = max(float((self.skill * self.employed).sum()), 1e-6)  # effective labour
        if p.two_channel:
            eff = self.skill * self.employed
            L_r = max(float(eff[~self.cognitive].sum()), 1e-6)   # routine labour pool
            L_c = max(float(eff[self.cognitive].sum()), 1e-6)    # cognitive labour pool
            a_r, a_ai = self.a_r_t(), self.a_ai_t()
            # Phase 2 (piece 2): employment overlay. Production keeps the full labour
            # pools L_r, L_c above; emp_status only governs who EARNS the wage bill.
            self.emp_status = self._employment(a_r, a_ai)
            eff_e = self.skill * self.emp_status
            self._Lr_emp = max(float(eff_e[~self.cognitive].sum()), 1e-6)
            self._Lc_emp = max(float(eff_e[self.cognitive].sum()), 1e-6)

        # === 2. POTENTIAL OUTPUT AND THE INCOME SPLIT (CES technology) =======
        # Potential output is what the installed capital and employed labour could
        # produce at full utilisation. The capital share pi is technology-given and
        # rises endogenously with automation I. Actual output is demand-determined
        # below; factor incomes are paid out of ACTUAL output, so Euler holds at
        # the realised level and idle capital simply earns less.
        if p.two_channel:
            d = self._decompose(self.K_r, self.K_ai, L_r, L_c, a_r, a_ai)
            Y_pot = d["Y"]
            # Phase 3: competition erodes the markup. mu_eff is the rent share net
            # of contestability and replaces mu_frac everywhere in the rent split.
            self._mu_eff = p.mu_frac * (1.0 - p.competition)
            if p.markup_power > 0.0:
                auto_int = float(np.clip((a_ai - p.a_ai_base) / max(p.a_ai_max, 1e-9), 0.0, 1.0))
                self._mu_eff = self._mu_eff * (1.0 + p.markup_power * auto_int)
            self._mu_eff = float(np.clip(self._mu_eff, 0.0, 0.9))   # keep the cognitive net share positive
            # rent peels mu_eff of cognitive value added before the factor split
            rent_pot = self._mu_eff * d["s_c"] * Y_pot
            # embodied-IP rent on the robotic cluster (default 0): a foreign-held markup
            # peeled from the routine cluster's value added, mirroring the AI rent.
            self._mu_robot = float(np.clip(p.robot_ip, 0.0, 0.9))
            # competitive capital incomes net of the rent wedge on each side
            ci_Kr_pot = d["ci_Kr"] * (1.0 - self._mu_robot)
            ci_Kai_pot = d["ci_Kai"] * (1.0 - self._mu_eff)
            wlc_pot = d["w_Lc"] * (1.0 - self._mu_eff)
            wlr_pot = d["w_Lr"] * (1.0 - self._mu_robot)
            cap_pot = ci_Kr_pot + ci_Kai_pot
            pi = cap_pot / Y_pot                       # aggregate capital share (owners')
            # per-agent expected wage rate by cluster (for the consumption rule),
            # concentrated on the employed so the wage bill is unchanged
            self._wrate = np.where(self.cognitive, wlc_pot / self._Lc_emp, wlr_pot / self._Lr_emp)
            self._ret4 = (d["r_Kr"], d["r_Kai"])       # gross returns for the split
        else:
            Y_pot = self.prod.output(self.K, L, I)
            _wb, _ci, pi, _rg = self.prod.factor_prices(self.K, L, I)

        # Ownership fractions of the equity stock (distribute income, book retained).
        eq_total = self.h_eq.sum() + self.eq_state + self.eq_row
        f_house = self.h_eq / eq_total
        f_state = self.eq_state / eq_total
        f_row = self.eq_row / eq_total

        # Interest on deposit balances (the r in the r>g public-debt dynamic).
        int_house = p.r_debt * self.h_dep
        int_gov = p.r_debt * self.gov_dep
        int_row = p.r_debt * self.row_dep
        int_firm = p.r_debt * self.firm_dep
        ubi_i = np.full(n, p.ubi)

        # === 3. (INVESTMENT IS THE GOODS-MARKET RESIDUAL) ===================
        # Under the neoclassical closure there is no separate investment decision
        # here: output is produced at potential (section 6) and investment is
        # whatever output is left after consumption and government, I = Y - C - G
        # (computed in section 6, once consumption is known). The capital stock is
        # disciplined by the consumption wealth effect and gross-complements
        # diminishing returns, not by a behavioural-investment rule. We still
        # report the net return r_net (section 7) for diagnostics.

        # === 4. WEALTH TAX WITH BEHAVIOURAL AVOIDANCE (item 1a) =============
        # Base erodes with the rate (bunching). The liability splits into an equity
        # leg (paid in kind, ownership transferred to the state) and a cash leg
        # (ordinary revenue). Assessed on beginning-of-period net worth.
        nw = self.house_nw()
        gross_wt = self._gross_wealth_tax(nw)
        tau_w = self._effective_wealth_rate()
        base_frac = (1.0 - min(tau_w, 0.99)) ** p.avoidance_elasticity if tau_w > 0 else 1.0
        wealth_tax = np.minimum(gross_wt * base_frac, np.clip(nw, 0, None))
        pos_eq = np.clip(self.h_eq, 0, None)
        pos_dep = np.clip(self.h_dep, 0, None)
        pos_nw = pos_eq + pos_dep + 1e-12
        wt_eq = wealth_tax * pos_eq / pos_nw      # equity leg (in kind, to the state)
        wt_cash = wealth_tax - wt_eq              # cash leg (revenue)

        # === 5. CONSUMPTION (predetermined, behavioural) ====================
        # Households consume out of EXPECTED income plus a wealth effect, capped at
        # cash on hand. Expected income uses the wage the economy would pay at full
        # utilisation (predetermined from potential output) plus last period's
        # distributions; this anchors demand near capacity and avoids a degenerate
        # low-output equilibrium, while still letting demand fall short of capacity
        # when propensities or wealth are low. Consumption is genuinely behavioural
        # (not forced to clear), so it does not distort the wealth distribution.
        #
        # The stock WEALTH EFFECT operates on PRODUCTIVE EQUITY only (clip(h_eq,0)),
        # not on total net worth. This is deliberate and matters for consistency:
        #   (i) Deposit balances are government debt held by the private sector. In
        #       a scenario with a persistent primary deficit (e.g. pure laissez
        #       faire, where the state buys G but levies no tax) that debt grows
        #       without bound as the mirror of the government's negative net worth.
        #       Letting it drive consumption would make demand explode purely from a
        #       financial position, ration real output, and spuriously collapse the
        #       capital stock. Anchoring the wealth effect to equity removes that
        #       artefact: the real economy converges to its fundamentals-driven K/Y
        #       while the government's insolvency shows up (correctly) in its own net
        #       worth rather than by crashing production.
        #   (ii) Interest earned on deposits already enters the income term
        #       (int_house in exp_gross), so excluding deposits from the stock term
        #       avoids double-counting bond holdings as both income and stimulus.
        #   (iii) Empirically the marginal propensity to consume out of illiquid /
        #       equity wealth is what the macro wealth effect mostly captures.
        eq_wealth = np.clip(self.h_eq, 0.0, None)
        # Phase 2: the safety-net transfer to displaced workers. The unchanged wage
        # bill is concentrated on the employed (emp_status), and each out-of-work
        # worker receives unemployment_benefit times the economy's average wage,
        # financed by the state. It enters income like UBI (an untaxed transfer).
        benefit_i = np.zeros(n)
        if p.two_channel and p.unemployment_benefit > 0.0:
            avg_wage = (wlr_pot + wlc_pot) / max(n, 1)
            benefit_i = (1.0 - self.emp_status) * p.unemployment_benefit * avg_wage
        if p.two_channel:
            exp_wage_i = self._wrate * self.skill * self.emp_status
        else:
            exp_wage_i = ((1.0 - pi) * Y_pot / L) * self.skill * self.employed
        # Expected capital income is the SUSTAINABLE after-tax yield on the equity
        # the household holds (eq_yield_lag * equity), NOT last period's dividend.
        # The two coincide in steady state, but the dividend balloons with returned
        # capital whenever the firm disinvests; using the sustainable yield stops
        # that transient from being consumed and decumulating the capital stock.
        exp_cap_inc = self.eq_yield_lag * eq_wealth
        # Split disposable income into a LABOUR stream and a CAPITAL stream and
        # apply the differential propensities. Income tax falls on both; UBI and the
        # citizens-fund rebate are labour-side transfers; the wealth-tax cash leg is
        # netted off the capital stream (it is a levy on capital).
        #
        # NOTE the capital stream is the PRODUCTIVE equity yield only; interest on
        # deposits (int_house) is deliberately excluded from the consumption
        # decision. Deposits are government debt, and under a persistent primary
        # deficit that debt is on an explosive r>g path. Letting its interest drive
        # consumption would feed the explosion back into real demand and crash
        # production. Excluding it (like the equity-only wealth effect above) cleanly
        # separates the REAL equilibrium, which then converges to its fundamentals,
        # from FISCAL sustainability, which shows up as the trajectory of government
        # net worth (the "who goes bankrupt" question) rather than by contaminating
        # output. Households still RECEIVE the interest as cash (section 8); they
        # simply do not treat volatile public-debt service as consumable income.
        labour_disp = (exp_wage_i - p.tax_income * np.clip(exp_wage_i, 0, None)
                       + ubi_i + self.fund_rebate_lag + benefit_i)
        capital_gross = exp_cap_inc
        capital_disp = (capital_gross - p.tax_income * np.clip(capital_gross, 0, None)
                        - wt_cash)
        desired_c = np.clip(p.c_income * np.clip(labour_disp, 0, None)
                            + p.c_profit * np.clip(capital_disp, 0, None)
                            + p.c_wealth * eq_wealth, 0, None)
        # cash-on-hand cap: deposits plus this period's net disposable inflow
        max_c = np.clip(self.h_dep + np.clip(labour_disp + capital_disp, 0, None), 0, None)
        consumption = np.minimum(desired_c, max_c)
        C = consumption.sum()

        # === 6. SUPPLY-DETERMINED OUTPUT, RESIDUAL INVESTMENT ===============
        # Output is produced at full utilisation, Y = Y_pot, so factor incomes
        # satisfy Euler exactly. The goods market clears through INVESTMENT as the
        # residual: I = Y - C - G. Whatever is not consumed or bought by
        # government is invested, so there is no unsold inventory and no demand-
        # deficiency spiral. If consumption plus government would exceed output
        # (over-consumption), consumption is rationed down so investment floors at
        # zero and the capital stock simply depreciates; under gross complements
        # this keeps K in a bounded, stable range.
        Y = Y_pot
        if p.two_channel and p.ai_supply_elasticity > 0.0:
            # Phase 1b: taxing the AI rent has an efficiency cost. The AI owner
            # restricts deployment in response to the net-of-tax wedge, a standard
            # deadweight loss that scales with the AI tax wedge (dst_ai + tax_repat)
            # and the AI cluster's share of output, s_c. This is a current-period
            # output loss (it does not feed the accumulation technology, so it does
            # not compound implausibly), and because the rent base shrinks with the
            # rate it produces a Laffer-type trade-off rather than free revenue.
            dwl_ai = float(np.clip(p.ai_supply_elasticity * (p.dst_ai + p.tax_repat) * d["s_c"],
                                   0.0, 0.08))
            Y = Y_pot * (1.0 - dwl_ai)
        G = p.gov_cost * Y
        resource_for_c = Y - G                       # output available for consumption
        if C > resource_for_c and C > 0:             # ration over-consumption pro-rata
            consumption *= resource_for_c / C
            C = consumption.sum()
        I_gross = max(Y - C - G, 0.0)                # residual investment (clears goods market)
        # Phase 1a: open-economy trade. A fraction trade_leak of the foreign owner's
        # after-host-tax rent is repatriated as real goods (an export), which is
        # final demand and so comes out of investment. Treated exactly like the
        # government-consumption term G: the firm sells the goods for cash and the
        # buyer (RoW) pays, so the deposit and net-worth invariants are preserved by
        # the same algebra. Capped so investment cannot go negative.
        X_row = 0.0
        if p.two_channel and p.trade_leak > 0.0:
            rent_now = self._mu_eff * d["s_c"] * Y + self._mu_robot * d["s_r"] * Y   # total IP rent at realised output
            keep_rent = max(1.0 - p.dst_ai - p.tax_repat, 0.0)        # owner's share after host tax
            X_row = min(p.trade_leak * rent_now * keep_rent, I_gross) # cannot exceed investable residual
            I_gross = I_gross - X_row
        self._X_row = X_row
        if p.two_channel:
            # allocate total gross investment between the two stocks by a damped
            # move toward equalising NET returns, then net off each stock's own
            # depreciation. Total net capital formation is still I_gross - total
            # depreciation, so the aggregate goods-market closure is unchanged.
            rn_r = self._ret4[0] - p.depreciation_r
            rn_ai = self._ret4[1] - p.depreciation_ai
            frac_ai = self.K_ai / max(self.K_r + self.K_ai, 1e-9)
            gap = (rn_ai - rn_r) / (abs(rn_ai) + abs(rn_r) + 1e-6)   # in [-1, 1]
            w_ai = float(np.clip(frac_ai + p.invest_damping * gap, 0.02, 0.98))
            dep_flow = p.depreciation_r * self.K_r + p.depreciation_ai * self.K_ai
            self._dKr = I_gross * (1.0 - w_ai) - p.depreciation_r * self.K_r
            self._dKai = I_gross * w_ai - p.depreciation_ai * self.K_ai
            dK = self._dKr + self._dKai
        else:
            dep_flow = p.depreciation * self.K
            dK = I_gross - dep_flow                   # net capital formation
        ubi_total = ubi_i.sum()

        # === 7. FACTOR INCOMES, PROFIT, TAXES (on realised output) ==========
        if p.two_channel:
            sc = (Y / Y_pot) if Y_pot > 0 else 1.0    # scale potential to realised (DWL haircut)
            d2 = self._decompose(self.K_r, self.K_ai, L_r, L_c, a_r, a_ai)
            rent_ai = self._mu_eff * d2["s_c"] * Y
            rent_robot = self._mu_robot * d2["s_r"] * Y
            ci_Kr = d2["ci_Kr"] * (1.0 - self._mu_robot) * sc
            ci_Kai = d2["ci_Kai"] * (1.0 - self._mu_eff) * sc
            w_Lr = d2["w_Lr"] * (1.0 - self._mu_robot) * sc
            w_Lc = d2["w_Lc"] * (1.0 - self._mu_eff) * sc
            wage_bill = w_Lr + w_Lc
            capital_income = ci_Kr + ci_Kai           # owners' competitive return
            # wage paid per agent, by cluster. The unchanged cluster wage bill is
            # concentrated on those still employed (emp_status); the displaced earn
            # no wage and instead receive the safety-net transfer below.
            wage_i = np.where(self.cognitive, (w_Lc / self._Lc_emp), (w_Lr / self._Lr_emp)) \
                * self.skill * self.emp_status
            r_net = (capital_income / self.K if self.K > 0 else 0.0) - dep_flow / max(self.K, 1e-9)
            net_profit = capital_income - dep_flow
        else:
            wage_bill = (1.0 - pi) * Y
            capital_income = pi * Y
            r_net = (capital_income / self.K if self.K > 0 else 0.0) - p.depreciation
            wage_i = (wage_bill / L) * self.skill * self.employed
            rent_ai = 0.0
            rent_robot = 0.0
            ci_Kr = ci_Kai = 0.0

            net_profit = capital_income - dep_flow
        if p.two_channel:
            # The AI rent is the IP owner's licence fee. It is already peeled from
            # output (capital_income excludes it), so it is NOT deducted from net
            # profit again; it leaves to the rest of the world (section 8) net of
            # withholding and DST. The host reaches the rent only via DST and
            # withholding, never corporate tax.
            rent_total = rent_ai + rent_robot       # total foreign-held IP rent (AI + embodied robot IP)
            royalty = rent_total
            # Owner domicile: the foreign-owned share leaves as a cross-border royalty
            # (what the DST and withholding reach); the domestic share stays home as
            # capital income to resident owners (reached instead by domestic taxes).
            self._royalty_foreign = p.ai_ip_foreign * rent_total
            self._rent_dom = rent_total - self._royalty_foreign
            # Phase 3: the optimising owner shifts a share of the rent recognition
            # offshore when the host taxes it, escaping the DST and the withholding.
            self._rent_shift = 0.0
            if p.owner_shift_elasticity > 0.0:
                wedge = p.dst_ai + p.tax_repat                            # host's tax rate on the rent
                self._rent_shift = float(np.clip(p.owner_shift_elasticity * wedge, 0.0, 0.6))
            robot_tax_amt = p.robot_tax * np.clip(ci_Kr, 0, None)          # source levy on robots
            foreign_ai = (1.0 - p.s_home) * np.clip(ci_Kai, 0, None)      # AI income earned abroad
            base = np.clip(net_profit, 0, None) - foreign_ai              # home-recognised profit
            ord_corp = p.tax_corp * np.clip(base, 0, None)
            dst_cap = p.dst_ai * np.clip(ci_Kai, 0, None)                 # DST on AI capital income
            self._dst_rent = p.dst_ai * np.clip(self._royalty_foreign, 0, None) * (1.0 - self._rent_shift)  # on recognised foreign rent
            corp_tax = ord_corp + robot_tax_amt + dst_cap                 # taxes from firm surplus
            after_tax = net_profit - corp_tax
        else:
            self._dst_rent = 0.0
            self._rent_shift = 0.0
            # Profit shifting by foreign-operated automation: a fraction profit_shift
            # of the economic profit on the foreign-operated share of the capital is
            # relocated abroad as a deductible royalty/IP charge, recognised in the
            # parent's jurisdiction. It is deducted BEFORE the host's corporate tax,
            # so the host taxes only the residual base, and it leaves immediately as
            # cash (section 8) rather than being reinvested as domestic equity.
            royalty = p.profit_shift * p.foreign_operated * np.clip(net_profit, 0, None)
            self._royalty_foreign = royalty   # single-channel royalty is foreign by construction
            self._rent_dom = 0.0
            taxable_profit = net_profit - royalty
            corp_tax = p.tax_corp * np.clip(taxable_profit, 0, None)
            after_tax = net_profit - royalty - corp_tax

        # Sustainable capital-income yield carried to next period's consumption rule
        # (section 5). This is after-tax economic profit per unit of equity; it never
        # includes returned capital, so households do not consume buyback proceeds.
        self.eq_yield_lag = after_tax / max(eq_total, 1e-9)

        gross_income = wage_i + int_house + self.div_house_lag + self.fund_rebate_lag
        income_tax = p.tax_income * np.clip(gross_income, 0, None)

        # === 7b. FIRM FINANCING: equity grows by EXACTLY dK ==================
        # The capital stock grows by dK, so the equity claims on it must grow by
        # dK too (that is what keeps total claims == K). The firm finances dK
        # first from after-tax profit (retained, booked pro-rata to existing
        # owners) and, only if investment exceeds internal funds, by issuing new
        # equity to household savers. Whatever profit is left after financing
        # investment is paid out as dividends.
        issuance = max(dK - after_tax, 0.0)            # external finance needed (usually 0)
        pro_rata_growth = dK - issuance                 # = min(dK, after_tax), retained
        dividends_tot = max(p.div_payout * (after_tax - dK), 0.0)

        div_house = dividends_tot * f_house
        div_state = dividends_tot * f_state
        div_row = dividends_tot * f_row
        fund_rebate = np.full(n, div_state / n) if p.citizens_fund else np.zeros(n)

        # Source-based tax on the capital income foreign owners earn domestically,
        # collected where the automated output is produced. It taxes their full
        # attributed share of after-corporate-tax profit, f_row * after_tax, at
        # rate tax_repat, in two matched legs. The CASH leg falls on the dividend
        # they would repatriate (div_row) and is collected as revenue here; the
        # in-kind EQUITY leg falls on the earnings reinvested in their name (their
        # share of retained profit) and is applied in section 9 by diverting that
        # slice of new equity to the state. Taxing only the cash dividend would
        # miss most of the income, because in a growth phase profit is largely
        # retained and distributed dividends are near zero. If repat_rebate is set,
        # the cash leg is paid straight back to residents per capita; the equity
        # leg always accrues to the state. Both legs are matched, so deposits net
        # to zero and equity still equals K.
        repat_tax = p.tax_repat * div_row                       # cash leg on dividends
        repat_tax_roy = p.tax_repat * self._royalty_foreign * (1.0 - self._rent_shift)   # withholding on recognised royalty
        self._repat_eq = p.tax_repat * np.clip(pro_rata_growth, 0, None) * f_row  # equity leg (s.9)
        repat_cash = repat_tax + repat_tax_roy                  # total cash withholding
        repat_rebate_i = (np.full(n, repat_cash / n) if (p.repat_rebate and repat_cash > 0.0)
                          else np.zeros(n))

        # === 8. DEPOSIT (CASH) TRANSFERS ====================================
        # Every cash flow is a matched transfer between sectors, so the four
        # balances continue to net to zero. The firm receives sales (C+G), pays
        # wages, corporate tax and dividends, and receives any equity-issuance
        # proceeds (added in step 10). With the corrected closure its balance
        # stays at ~0 instead of drifting. We record each non-firm sector's NET
        # SAVING FLOW this period (its deposit inflow), which is what funds the new
        # equity issuance in section 10: by the S = I identity the non-firm
        # sectors' net saving sums to exactly the external finance the firm needs,
        # so issuance is fully funded out of current income without overdrawing
        # anyone (the buyer pays from the income just credited, not from a depleted
        # stock).
        # Domestic share of the AI rent (owner at home): paid out by the firm as
        # capital income to resident owners, allocated pro-rata to equity holdings so
        # it concentrates among capital owners and drives DOMESTIC inequality, rather
        # than leaving the country. Zero when ai_ip_foreign = 1 (the rent-importer).
        eqw = np.clip(self.h_eq, 0.0, None)
        rent_dom_i = ((self._rent_dom * eqw / eqw.sum()) if (self._rent_dom > 0.0 and eqw.sum() > 0.0)
                      else np.zeros(n))
        house_flow = (wage_i + int_house + ubi_i + fund_rebate + div_house + rent_dom_i
                      + repat_rebate_i + benefit_i - consumption - income_tax - wt_cash)
        gov_flow = (corp_tax + income_tax.sum() + wt_cash.sum() + div_state
                    + int_gov + repat_cash + self._dst_rent - repat_rebate_i.sum()
                    - ubi_total - G - fund_rebate.sum() - benefit_i.sum())
        row_flow = div_row + int_row + self._royalty_foreign - repat_cash - self._dst_rent - self._X_row
        self.h_dep += house_flow
        self.gov_dep += gov_flow
        self.row_dep += row_flow
        self.firm_dep += (C + G + self._X_row) - wage_bill - corp_tax - dividends_tot - royalty + int_firm

        # government balance (diagnostic): revenue + interest - outlays
        gov_balance = (corp_tax + income_tax.sum() + wt_cash.sum() + div_state
                       + int_gov - ubi_total - G - fund_rebate.sum() - benefit_i.sum())

        # No-overdraft cleanup of the expectation error between section-5 expected
        # income and section-8 realised income (net-zero within households).
        self._enforce_household_floor()

        # === 9. EQUITY TRANSFERS ============================================
        # Ownership-timing convention (made explicit per review item 6): all of
        # THIS period's equity returns accrue to BEGINNING-of-period owners, on the
        # shares f_house/f_state/f_row computed in section 2. That is the same base
        # used for dividends in section 7b, so dividends and retained earnings are
        # allocated consistently. The wealth-tax equity leg is a policy transfer
        # that settles at period END, so it is applied AFTER the retained-earnings
        # allocation. (The two operations are independent additions to the holding
        # array and so commute numerically; the order is fixed here only to make
        # the economic sequence unambiguous.)
        #
        # (a) retained earnings finance the pro-rata part of capital growth: every
        #     beginning-of-period owner's claim grows in proportion to its stake.
        self.h_eq += pro_rata_growth * f_house
        self.eq_state += pro_rata_growth * f_state
        self.eq_row += pro_rata_growth * f_row
        # (a') source-tax equity leg: the slice of the foreign owners' reinvested
        #      earnings taxed at source is diverted to the state (in kind), so the
        #      tax reaches the automation income that is retained rather than
        #      repatriated. Matched within equity, so total equity still grows by
        #      exactly dK and eq == K.
        self.eq_row -= self._repat_eq
        self.eq_state += self._repat_eq
        # (b) wealth-tax equity leg: ownership then moves from households to state.
        self.h_eq -= wt_eq
        self.eq_state += wt_eq.sum()

        # === 10. NEW EQUITY ISSUANCE FINANCED BY SAVERS =====================
        # When net investment exceeds the firm's internal funds (dK > after_tax) the
        # firm must raise external finance equal to `issuance`. This is the S = I
        # identity made operational: by national-accounting, net investment dK is
        # identically equal to aggregate net saving, so the cash to buy the new
        # equity always exists somewhere in the system. We allocate the WHOLE
        # issuance across every sector in proportion to the saving it just did
        # (its positive deposit balance after the section-8 cash transfers), and
        # the buyers pay the firm in full. Two things follow:
        #   * firm_dep stays identically ~0. With full subscription the firm's cash
        #     flow each period is exactly int_firm (algebra: after_tax - dK -
        #     dividends + issuance = int_firm in both the dividend and the issuance
        #     regime), so starting from zero it never drifts. No silent leak.
        #   * the saver acquires the claim. Crucially this includes the STATE and
        #     the REST OF WORLD, not just households: in a capital-tax regime the
        #     big saver is the government, so its surplus flows into productive
        #     equity (eq_state) instead of piling up as ever-growing public claims
        #     on a firm that never issued to it. Ownership therefore evolves
        #     endogenously with saving (a persistent saver gradually owns more of
        #     the capital stock), which is an economic result, not an artefact.
        # Total equity still grows by exactly dK (pro_rata_growth in section 9 plus
        # issuance here), so equity == K is preserved to machine precision.
        if issuance > 1e-12:
            # saving = positive deposit balance each sector is holding right now
            # (the accumulated surplus, not just this period's flow). Using the
            # stock is what routes a persistent saver's ACCUMULATED surplus into
            # equity: in a capital-tax regime the big saver is the government, and
            # draining its surplus into eq_state each period is what keeps gov_dep
            # from piling up as an ever-growing public deposit with no valid
            # negative counterpart (households cannot be its mirror under the
            # no-overdraft rule). The buyers pay the firm in full, so with full
            # subscription firm_dep stays identically ~0 (after_tax - dK -
            # dividends + issuance == int_firm in both regimes). Total equity grows
            # by exactly dK (pro_rata in section 9 plus issuance here) so eq == K.
            h_sav = np.clip(self.h_dep, 0.0, None)
            g_sav = max(self.gov_dep, 0.0)
            r_sav = max(self.row_dep, 0.0)
            total_sav = h_sav.sum() + g_sav + r_sav
            if total_sav > 1e-12:
                w = issuance / total_sav                 # uniform fraction of saving used
                buy_h = h_sav * w
                buy_g = g_sav * w
                buy_r = r_sav * w
            else:
                # degenerate fallback (no positive balances anywhere): split per
                # existing ownership so the books still close exactly.
                buy_h = issuance * f_house
                buy_g = issuance * f_state
                buy_r = issuance * f_row
            # buyers pay cash (deposits down) and receive equity (claims up)
            self.h_dep -= buy_h
            self.h_eq += buy_h
            self.gov_dep -= buy_g
            self.eq_state += buy_g
            self.row_dep -= buy_r
            self.eq_row += buy_r
            # firm receives the full proceeds -> firm_dep returns to ~0
            self.firm_dep += float(buy_h.sum() + buy_g + buy_r)

        # === 10b. NO-OVERDRAFT VIA A SECONDARY EQUITY MARKET (review item 1) ==
        # A household cannot hold a negative deposit: deposits are claims on the
        # economy's single safe asset (government debt) and a household cannot
        # ISSUE that asset. Any household left with a negative balance (it consumed
        # or subscribed to more equity than its realised cash) must therefore raise
        # cash by SELLING equity, at book value, to the sectors that are holding
        # surplus cash. This is a portfolio swap, matched on both legs: the seller's
        # deposit rises to zero and its equity falls by the same amount, while the
        # buyers' deposits fall and their equity rises by that amount. Aggregate
        # deposits still net to zero and total equity still equals K.
        #
        # Economically this is what closes the model under the no-overdraft rule.
        # When the government runs a persistent surplus its deposit would otherwise
        # accumulate with no valid negative counterpart (no sector can be its
        # debtor once households are floored); here that surplus is spent buying
        # households' equity, so it is converted into a sovereign equity stake
        # (eq_state) rather than a phantom deposit. The buyers are whichever
        # sectors actually hold surplus cash, pro-rata, so the same mechanism lets
        # surplus households or the rest of the world absorb the equity when they,
        # not the state, are the savers. The swap always clears: by the deposit
        # identity the total surplus held by the positive sectors equals the total
        # overdraft of the negative households exactly.
        neg = np.clip(self.h_dep, None, 0.0)         # per-household overdraft (<= 0)
        od = float(-neg.sum())                       # total overdraft (>= 0)
        if od > 1e-12:
            self.h_dep -= neg                        # sellers raise cash to zero (+od)
            self.h_eq += neg                         # sellers give up equity      (-od)
            surplus_h = np.clip(self.h_dep, 0.0, None)
            surplus_g = max(self.gov_dep, 0.0)
            surplus_r = max(self.row_dep, 0.0)
            tot = surplus_h.sum() + surplus_g + surplus_r
            if tot > 1e-12:                          # buyers absorb cash & equity pro-rata
                self.h_dep -= surplus_h / tot * od
                self.h_eq += surplus_h / tot * od
                self.gov_dep -= surplus_g / tot * od
                self.eq_state += surplus_g / tot * od
                self.row_dep -= surplus_r / tot * od
                self.eq_row += surplus_r / tot * od

        # === 11. CAPITAL STOCK UPDATE =======================================
        if p.two_channel:
            self.K_r = max(self.K_r + self._dKr, 1e-6)
            self.K_ai = max(self.K_ai + self._dKai, 1e-6)
            self.K = self.K_r + self.K_ai
        else:
            self.K += dK

        # === 12. CAPITAL FLIGHT (endogenous foreign mobility, item 1b) ======
        # A wealth tax pushes a share of household equity abroad. We move the
        # offshore stock partially towards a target share of household wealth,
        # target_extra = migration_semi_elast * (tau in pp), so the foreign share
        # settles at the empirically-calibrated level rather than draining the
        # sector. Flight is drawn disproportionately from the top of the
        # distribution (migration is a top-tail response).
        flight = 0.0
        if p.mobility_on and tau_w > 0 and self.h_eq.sum() > 0:
            target_extra = min(p.migration_semi_elast * (tau_w * 100.0), 0.6)
            onshore = self.h_eq.sum()
            target_offshore = target_extra * (onshore + self.offshore.sum())
            gap = target_offshore - self.offshore.sum()
            if gap > 0:
                move = 0.10 * gap                       # partial adjustment toward target
                w = np.clip(self.h_eq, 0, None)
                if w.sum() > 0:
                    share = w / w.sum()
                    tilt = np.clip(0.25 + 0.75 * share * n, 0, 4.0)   # top holders more mobile
                    flight_vec = share * tilt
                    flight_vec *= move / max(flight_vec.sum(), 1e-12)  # scale so total moved = move
                    flight_vec = np.minimum(flight_vec, 0.5 * w)       # at most half a holding/period
                    self.h_eq -= flight_vec
                    self.eq_row += flight_vec.sum()
                    self.offshore += flight_vec                        # attribute to original owner
                    flight = float(flight_vec.sum())

        # === 13. HETEROGENEOUS PERSISTENT RETURNS (item 2) ==================
        # Each household's equity earns an idiosyncratic return drawn around the
        # aggregate. The return TYPE is persistent (AR(1)), which is what produces
        # a heavy (Pareto) tail via random growth. The multiplier is mean-1, and
        # the holdings are then rescaled to preserve the household equity total
        # (the balance-sheet identity that household claims sum to the household
        # share of K). This is redistribution among households, not new wealth.
        innov = self.rng.standard_normal(n)
        self.ret_z = p.ret_persist * self.ret_z + np.sqrt(1.0 - p.ret_persist ** 2) * innov
        if p.ret_scale_dep != 0.0:
            rank = np.argsort(np.argsort(self.h_eq)) / max(n - 1, 1) - 0.5
            prem = p.ret_sigma * self.ret_z + p.ret_scale_dep * rank
        else:
            prem = p.ret_sigma * self.ret_z
        mult = np.exp(prem - 0.5 * p.ret_sigma ** 2)     # mean-1 lognormal multiplier
        pos = np.clip(self.h_eq, 0, None)
        target = self.h_eq.sum()
        if target > 0 and pos.sum() > 0:
            redistributed = pos * mult
            redistributed *= target / redistributed.sum()    # enforce sum == household claims
            self.h_eq = redistributed

        # Offshore wealth grows at the SAME aggregate reinvestment rate as onshore
        # equity (its share of dK), so the attribution reflects not just the fled
        # principal but its subsequent growth abroad. This avoids the earlier
        # principal-only undercount while staying bounded (it grows at the
        # disciplined equity growth rate, not an idiosyncratic compounding rate).
        if self.offshore.sum() > 0 and eq_total > 0:
            reinvest_rate = pro_rata_growth / eq_total
            self.offshore *= (1.0 + reinvest_rate)

        # === 14. DEMOGRAPHIC TURNOVER (pins the stationary tail) ============
        # A small share of agents are replaced each period. The estate of the
        # departed is split equally among the replacements (conserving the pool),
        # which caps runaway condensation and gives the random-growth process a
        # stationary Pareto tail (Benhabib-Bisin-Zhu). Offshore estates are spread
        # across survivors so the attribution is conserved.
        if p.demographic_reset > 0 and self.h_eq.sum() > 0:
            die = self.rng.random(n) < p.demographic_reset
            if die.any():
                k = int(die.sum())
                self.h_eq[die] = self.h_eq[die].sum() / k    # equal split, conserves the pool
                self.ret_z[die] = self.rng.standard_normal(k)
                off_pool = self.offshore[die].sum()
                self.offshore[die] = 0.0
                alive = ~die
                if alive.any() and off_pool > 0:
                    self.offshore[alive] += off_pool / alive.sum()

        # === 15. CARRY DISTRIBUTED INCOME FORWARD (for next period's demand) =
        # Consumption next period is decided on the disposable income received this
        # period (predetermined), which is what keeps output demand-determined
        # without a within-period simultaneity.
        self.div_house_lag = div_house
        self.fund_rebate_lag = fund_rebate
        self.disp_lag = gross_income - income_tax - wt_cash + ubi_i
        self.r_lag = r_net

        # === 16. RECORD METRICS =============================================
        # Household net worth is non-negative by construction at this point:
        # equity is clipped at zero throughout sections 12-14, and deposits were
        # floored at zero by _enforce_household_floor in sections 8 and 10 and are
        # not touched afterwards. The Gini below is therefore the ordinary Gini on
        # non-negative wealth, with no negative-value shifting convention invoked
        # (review item 2).
        nwh = self.house_nw()
        srt = np.sort(nwh); tot = max(srt.sum(), 1e-9)
        self.hist.gini.append(gini(nwh))
        self.hist.top1_share.append(float(srt[int(0.99 * n):].sum() / tot))
        self.hist.top10_share.append(float(srt[int(0.90 * n):].sum() / tot))
        # true distribution: domestic net worth plus offshore attributed to owners
        nw_true = nwh + self.offshore
        srt_t = np.sort(nw_true); tot_t = max(srt_t.sum(), 1e-9)
        self.hist.gini_true.append(gini(nw_true))
        self.hist.top1_share_true.append(float(srt_t[int(0.99 * n):].sum() / tot_t))
        self.hist.offshore_share.append(float(self.offshore.sum() / max(nw_true.sum(), 1e-9)))
        self.hist.labour_share.append(1.0 - pi)
        self.hist.gov_nw.append(self.eq_state + self.gov_dep)
        self.hist.row_nw.append(self.eq_row + self.row_dep)
        self.hist.repat_revenue.append(float(repat_tax + repat_tax_roy + self._repat_eq))
        self.hist.royalty.append(float(royalty))
        self.hist.corp_tax_rev.append(float(corp_tax))
        self.hist.house_nw.append(float(nwh.sum()))
        self.hist.K.append(self.K)
        self.hist.Y.append(Y)
        self.hist.gov_balance.append(gov_balance)
        self.hist.I_auto.append(I)
        self.hist.r_net.append(r_net)
        if p.two_channel:
            self.hist.K_r.append(float(self.K_r))
            self.hist.K_ai.append(float(self.K_ai))
            self.hist.rent_ai.append(float(rent_ai))
            self.hist.rent_robot.append(float(rent_robot))
            self.hist.a_r.append(float(a_r))
            self.hist.a_ai.append(float(a_ai))
            self.hist.w_Lr.append(float(w_Lr))
            self.hist.w_Lc.append(float(w_Lc))
            self.hist.ci_Kr.append(float(ci_Kr))
            self.hist.ci_Kai.append(float(ci_Kai))
            self.hist.exports.append(float(self._X_row))
            self.hist.unemployment.append(float(1.0 - self.emp_status.mean()))
        self.hist.g_rate.append(dK / max(self.K - dK, 1e-9))
        eq_claims = self.h_eq.sum() + self.eq_state + self.eq_row
        self.hist.own_row.append(self.eq_row / max(eq_claims, 1e-9))
        self.hist.wealth_tax_base_frac.append(base_frac)
        self.hist.capital_flight.append(flight)
        self.hist.eq_to_K.append(eq_claims / max(self.K, 1e-9))   # diagnostic: ~1
        self.hist.firm_dep.append(self.firm_dep)                   # diagnostic: ~0

        self.t += 1

    def run(self) -> HistoryV3:
        for _ in range(self.p.periods):
            self.step()
        return self.hist
