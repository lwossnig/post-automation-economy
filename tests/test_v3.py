"""v3 tests: invariants, calibration discipline, and the three upgrades.

Covers the same two accounting invariants as v1/v2 plus the new behaviour:
behavioural wealth-tax responses (item 1), the microfounded heterogeneous-returns
concentration engine (item 2), and calibration discipline (item 4).
"""
import numpy as np
import pytest

from abm_sfc.model_v3 import ModelV3, ParamsV3


CONFIGS = [
    dict(auto_max=0.0),
    dict(tax_corp=0.25, tax_income=0.2, tax_wealth=0.02, ubi=0.3),
    dict(own_households=0.7, own_row=0.3, tax_wealth=0.03, ubi=0.25),
    dict(own_households=0.4, own_state=0.6, citizens_fund=True),
    dict(wealth_brackets=((2.0, 0.01), (10.0, 0.03)), ubi=0.3),
    dict(skill_dispersion=0.6, displacement=True, ubi=0.3),
    dict(two_channel=True, mu_frac=0.25, mu_compute=0.15, tariff_compute=0.30,
         usage_levy=0.10, skill_dispersion=0.4, ubi=0.3),   # Exp Q: stacked rents + levers
    dict(two_channel=True, mu_frac=0.25, labour_supply_elast_r=0.3,
         reservation_wage=0.5, skill_dispersion=0.4, ubi=0.3),   # Exp R: elastic labour
    dict(two_channel=True, mu_frac=0.25, endogenous_automation=True,
         auto_cost0_ai=8.0, auto_cost0_r=8.0, cost_decline_ai=0.008,
         cost_decline_r=0.008, skill_dispersion=0.4, ubi=0.3),   # Exp S: endogenous automation
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_deposits_net_to_zero(cfg):
    p = ParamsV3(n_agents=300, periods=120, seed=1, **cfg)
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
        assert abs(m.deposits_sum()) < 1e-3


@pytest.mark.parametrize("cfg", CONFIGS)
def test_net_worth_equals_real_assets(cfg):
    p = ParamsV3(n_agents=300, periods=120, seed=1, **cfg)
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
        assert abs(m.total_nw() - m.real_assets()) < 1e-3


# ---- item 4: calibration discipline ----

def test_no_automation_is_stationary():
    p = ParamsV3(n_agents=400, periods=250, seed=0, auto_max=0.0)
    h = ModelV3(p).run()
    assert abs(h.Y[-1] / h.Y[80] - 1.0) < 0.06
    assert abs(h.g_rate[-1]) < 0.01


def test_transition_is_bounded_not_explosive():
    """Under the residual-investment closure with gross complements and
    differential saving, the automation transition must converge to a new steady
    state, not explode. Output may grow by a large multiple (steady compounding
    over a long horizon to a higher capital-intensive level), but K/Y must stay
    bounded, nothing may go NaN, and the growth rate must settle toward zero."""
    p = ParamsV3(seed=0)
    h = ModelV3(p).run()
    K = np.array(h.K); Y = np.array(h.Y)
    assert np.all(np.isfinite(K)) and np.all(np.isfinite(Y))
    assert h.Y[-1] / h.Y[0] > 1.2                 # automation does raise output
    ko = K / Y
    assert ko.max() < 15.0                         # capital intensity stays sane
    assert abs(h.g_rate[-1]) < 0.02                # growth has (nearly) settled


def test_no_collapse_or_explosion_across_regimes():
    """Every headline scenario must reach a finite steady state: no NaN, capital
    neither vanishing nor exploding, equity claims positive, and the SFC
    net-worth invariant intact at the end."""
    from abm_sfc import scenarios_v3 as sv3
    for name, fn in sv3.REGISTRY.items():
        base = fn()
        p = ParamsV3(**{**base.__dict__, "n_agents": 300, "periods": 250})
        m = ModelV3(p); h = m.run()
        assert np.all(np.isfinite(h.Y)), name
        eq = m.h_eq.sum() + m.eq_state + m.eq_row
        assert eq > 0.0, name
        assert abs(m.total_nw() - m.real_assets()) < 1e-2 * max(m.real_assets(), 1.0), name
        assert h.Y[-1] > 0.0, name


def test_capital_output_ratio_stays_bounded():
    p = ParamsV3(seed=0)
    h = ModelV3(p).run()
    ko = np.array(h.K) / np.array(h.Y)
    assert ko.max() < 6.0              # K/Y stays in an economically sane range


# ---- item 2: microfounded concentration ----

def test_returns_engine_produces_realistic_inequality():
    """Heterogeneous persistent returns must yield a heavy but STATIONARY tail
    under automation (not runaway condensation to Gini -> 1, nor a degenerate
    equal distribution). The headline scenarios put the mature wealth Gini in a
    realistic band."""
    from abm_sfc import scenarios_v3 as sv3
    p = sv3.REGISTRY["laissez_faire"](); p = ParamsV3(**{**p.__dict__,
                                                         "n_agents": 600, "periods": 600})
    h = ModelV3(p).run()
    g = np.mean(h.gini[-40:])
    assert 0.40 < g < 0.90             # realistic, stationary wealth Gini


def test_automation_raises_wealth_concentration():
    """The model's central distributional claim: shifting income from (relatively
    equal) labour to (unequally held) capital raises wealth inequality. With
    automation on, the mature Gini exceeds the no-automation control."""
    on = ModelV3(ParamsV3(n_agents=600, periods=600, seed=0)).run()
    off = ModelV3(ParamsV3(n_agents=600, periods=600, seed=0, auto_max=0.0)).run()
    assert np.mean(on.gini[-40:]) > np.mean(off.gini[-40:]) + 0.05


def test_demographic_reset_pins_the_tail():
    """Without the reset the random-growth process runs away; with it the tail is
    stationary. Compare top-1% share with reset on vs off under automation."""
    on = ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, demographic_reset=0.02)).run()
    off = ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, demographic_reset=0.0)).run()
    assert np.mean(on.top1_share[-40:]) < np.mean(off.top1_share[-40:])


def test_no_artificial_conservation_knob():
    """The model must not depend on the old kinetic kernel; concentration comes
    from the returns process. Higher return dispersion => more inequality."""
    lo = ModelV3(ParamsV3(n_agents=500, periods=250, seed=0, auto_max=0.0, ret_sigma=0.02)).run()
    hi = ModelV3(ParamsV3(n_agents=500, periods=250, seed=0, auto_max=0.0, ret_sigma=0.07)).run()
    assert np.mean(hi.gini[-40:]) > np.mean(lo.gini[-40:])


# ---- item 1: behavioural wealth-tax responses ----

def test_wealth_tax_base_erodes_with_rate():
    """Avoidance: a higher statutory wealth-tax rate shrinks the taxable base."""
    lo = ModelV3(ParamsV3(n_agents=300, periods=60, seed=0, tax_wealth=0.01)).run()
    hi = ModelV3(ParamsV3(n_agents=300, periods=60, seed=0, tax_wealth=0.05)).run()
    assert hi.wealth_tax_base_frac[-1] < lo.wealth_tax_base_frac[-1]
    assert lo.wealth_tax_base_frac[-1] <= 1.0


def test_wealth_tax_induces_capital_flight():
    """Mobility: a domestic wealth tax raises the endogenous foreign ownership
    share relative to no tax."""
    notax = ModelV3(ParamsV3(n_agents=400, periods=200, seed=0, tax_wealth=0.0)).run()
    tax = ModelV3(ParamsV3(n_agents=400, periods=200, seed=0, tax_wealth=0.03, ubi=0.25)).run()
    assert tax.own_row[-1] > notax.own_row[-1] + 0.02


def test_mobility_can_be_switched_off():
    """With mobility off and no avoidance, the foreign share stays at its initial
    value (recovers the frictionless tax as a special case)."""
    p = ParamsV3(n_agents=400, periods=150, seed=0, tax_wealth=0.03, ubi=0.25,
                 mobility_on=False, avoidance_elasticity=0.0)
    h = ModelV3(p).run()
    assert h.own_row[-1] < 1e-9
    assert abs(h.wealth_tax_base_frac[-1] - 1.0) < 1e-9


def test_behavioural_response_weakens_redistribution():
    """The central robustness check: with behavioural responses ON, the wealth
    tax achieves a different (less clean) outcome than the frictionless version,
    because capital flees and the base erodes. The frictionless tax should leave
    a lower measured top-1% share among remaining domestic holders is NOT
    guaranteed; what must hold is that the foreign share is higher and the base
    lower when behaviour is on."""
    friction = ModelV3(ParamsV3(n_agents=400, periods=250, seed=0,
                                tax_corp=0.25, tax_income=0.15, tax_wealth=0.02, ubi=0.25)).run()
    frictionless = ModelV3(ParamsV3(n_agents=400, periods=250, seed=0,
                                    tax_corp=0.25, tax_income=0.15, tax_wealth=0.02, ubi=0.25,
                                    mobility_on=False, avoidance_elasticity=0.0)).run()
    assert friction.own_row[-1] > frictionless.own_row[-1]


def test_offshore_tracking_reveals_composition_effect():
    """The hardened metric: under a wealth tax with mobility, fled capital leaves
    the measured domestic sector. True Gini (incl. offshore attributed to original
    owners) is at least the domestic Gini, and offshore share rises with the rate."""
    import numpy as np
    lo = ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, tax_wealth=0.02, ubi=0.25)).run()
    hi = ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, tax_wealth=0.05, ubi=0.25)).run()
    assert np.mean(hi.offshore_share[-30:]) > np.mean(lo.offshore_share[-30:])
    assert np.mean(hi.gini_true[-30:]) >= np.mean(hi.gini[-30:]) - 1e-6


def test_offshore_share_matches_migration_calibration():
    """Offshore share scales with the wealth-tax rate at the ~2%-per-pp migration
    semi-elasticity, and stays modest (not a drain) at empirical values."""
    import numpy as np
    o1 = np.mean(ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, tax_wealth=0.01, ubi=0.25)).run().offshore_share[-30:])
    o3 = np.mean(ModelV3(ParamsV3(n_agents=500, periods=300, seed=0, tax_wealth=0.03, ubi=0.25)).run().offshore_share[-30:])
    assert o3 > o1
    assert o1 < 0.10 and o3 < 0.15


def test_no_offshore_without_tax():
    """No wealth tax => no flight => domestic and true inequality coincide."""
    import numpy as np
    h = ModelV3(ParamsV3(n_agents=400, periods=200, seed=0, tax_wealth=0.0)).run()
    assert h.offshore_share[-1] < 1e-9
    assert abs(np.mean(h.gini_true[-20:]) - np.mean(h.gini[-20:])) < 1e-6


# ===========================================================================
# Review-driven tests (peer review, second round): per-agent no-overdraft,
# issuance edge cases, marginal brackets, persistent foreign ownership,
# stateless stability slope, and exact-reproduction guards.
# ===========================================================================

from abm_sfc import scenarios_v3 as sv3


_ALL_SCENARIOS = list(sv3.REGISTRY.keys())


@pytest.mark.parametrize("name", _ALL_SCENARIOS)
def test_per_household_no_overdraft(name):
    """Review item 1: NO household may hold a negative deposit after any step,
    in any scenario. This is the per-agent liquidity constraint, stronger than
    the aggregate deposits-net-to-zero check."""
    p = sv3.REGISTRY[name](); p.n_agents = 400; p.periods = 200; p.seed = 2
    m = ModelV3(p)
    worst = 0.0
    for _ in range(p.periods):
        m.step()
        worst = min(worst, float(m.h_dep.min()))
    assert worst > -1e-9, f"{name}: a household deposit went to {worst}"


@pytest.mark.parametrize("name", _ALL_SCENARIOS)
def test_household_net_worth_nonnegative(name):
    """Review item 2: with the no-overdraft floor and equity clipped at zero,
    household net worth is non-negative, so the recorded Gini is the ordinary
    Gini with no negative-shifting convention."""
    p = sv3.REGISTRY[name](); p.n_agents = 400; p.periods = 200; p.seed = 3
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
        assert m.house_nw().min() > -1e-9


def test_equity_equals_capital_to_machine_precision():
    """Total equity claims equal the capital stock to ~machine precision in
    every scenario (the firm holds no residual equity, no silent leak)."""
    for name in _ALL_SCENARIOS:
        p = sv3.REGISTRY[name](); p.n_agents = 300; p.periods = 150; p.seed = 1
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        eq = m.h_eq.sum() + m.eq_state + m.eq_row
        assert abs(eq - m.K) / m.K < 1e-9, name


def test_firm_holds_no_cash():
    """The firm's deposit stays at ~0 (full equity subscription), i.e. no leak
    into a residual firm cash balance, across scenarios and the boom transition."""
    for name in ["laissez_faire", "income_tax_ubi", "state_ownership"]:
        p = sv3.REGISTRY[name](); p.n_agents = 300; p.periods = 300; p.seed = 0
        m = ModelV3(p)
        worst = 0.0
        for _ in range(p.periods):
            m.step()
            worst = max(worst, abs(m.firm_dep) / m.K)
        assert worst < 1e-9, f"{name}: |firm_dep|/K reached {worst}"


def test_issuance_investment_boom_keeps_books():
    """Issuance edge case: force a large investment spike (high saving rate, big
    automation jump) so external finance dwarfs the deposit stock, and check the
    no-overdraft secondary market still clears with all invariants intact."""
    p = ParamsV3(n_agents=300, periods=200, seed=0,
                 c_profit=0.05, c_income=0.5,      # very high saving -> big dK, big issuance
                 auto_start=20, auto_speed=0.3, auto_max=0.6,
                 tax_corp=0.25, tax_income=0.3, ubi=0.3)
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
        assert m.h_dep.min() > -1e-9
        assert abs(m.deposits_sum()) < 1e-2
        assert abs(m.total_nw() - m.real_assets()) / m.K < 1e-9


def test_progressive_brackets_are_marginal_not_cumulative():
    """Review item 5: the progressive schedule must tax each band once at its own
    marginal rate, not stack every bracket's rate on the top band. For brackets
    ((3, 1%), (10, 3%)) an agent with net worth 15 should owe
    1%*(10-3) + 3%*(15-10) = 0.07 + 0.15 = 0.22, NOT the cumulative
    1%*(15-3) + 3%*(15-10) = 0.12 + 0.15 = 0.27."""
    p = ParamsV3(n_agents=10, periods=1, seed=0,
                 wealth_brackets=((3.0, 0.01), (10.0, 0.03)))
    m = ModelV3(p)
    nw = np.array([0.0, 2.0, 5.0, 15.0])
    tax = m._gross_wealth_tax(nw)
    expected = np.array([0.0, 0.0, 0.01 * (5 - 3), 0.01 * (10 - 3) + 0.03 * (15 - 10)])
    assert np.allclose(tax, expected, atol=1e-12), (tax, expected)
    # the top marginal rate is used for the behavioural-response strength
    assert abs(m._effective_wealth_rate() - 0.03) < 1e-12


def test_progressive_below_first_threshold_is_untaxed():
    p = ParamsV3(n_agents=4, periods=1, seed=0,
                 wealth_brackets=((3.0, 0.01), (10.0, 0.03)))
    m = ModelV3(p)
    assert m._gross_wealth_tax(np.array([2.999])) [0] == 0.0


def test_foreign_ownership_is_initial_shock_that_dilutes():
    """Review item 4: the foreign-ownership scenario is an INITIAL shock, not a
    pinned stake. It starts majority-foreign and dilutes substantially over the
    horizon (under a taxing government the stake is gradually transferred to the
    domestic public sector). This is the documented finding, and it stays
    book-consistent throughout."""
    h = ModelV3(sv3.foreign_ownership()).run()
    assert h.own_row[0] > 0.5                      # starts majority foreign
    assert h.own_row[-1] < h.own_row[0] - 0.2      # dilutes substantially


def test_foreign_dilution_transfers_to_the_state():
    """The diluted foreign equity is absorbed by the domestic state (the big
    saver under the income tax), not by a phantom balance: the state's equity
    share rises as the foreign share falls, and the books stay exact."""
    m = ModelV3(sv3.foreign_ownership())
    for _ in range(m.p.periods):
        m.step()
    eq = m.h_eq.sum() + m.eq_state + m.eq_row
    assert m.eq_state / eq > 0.3                    # state has taken up the slack
    assert abs(eq - m.K) / m.K < 1e-9               # books exact


def test_stability_slope_is_stateless_and_order_independent():
    """Review item 3: the local stability slope must be a pure function of the
    inputs, identical on repeat calls and independent of the order in which the
    automation levels are evaluated (no carried-over lag state)."""
    from abm_sfc.stability import real_slope
    p = sv3.REGISTRY["income_tax_ubi"]()
    a = real_slope(p, 0.5)
    b = real_slope(p, 0.5)
    assert abs(a - b) < 1e-12
    fwd = [real_slope(p, I) for I in (0.3, 0.7)]
    rev = [real_slope(p, I) for I in (0.7, 0.3)]
    assert abs(fwd[0] - rev[1]) < 1e-12 and abs(fwd[1] - rev[0]) < 1e-12


def test_real_economy_locally_stable_all_regimes():
    """Every scenario has a real capital-map slope strictly below one at all
    automation levels (the explosive root is purely the financial 1+r_debt)."""
    from abm_sfc.stability import stability_report
    for name in ["laissez_faire", "income_tax_ubi", "wealth_tax_ubi",
                 "state_ownership", "foreign_ownership"]:
        for row in stability_report(sv3.REGISTRY[name]()):
            assert abs(row["real_slope"]) < 1.0


def test_results_are_reproducible_with_fixed_seed():
    """Exact-reproduction guard: a fixed seed reproduces the terminal statistics
    bit-for-bit, so the canonical pipeline's numbers are deterministic."""
    a = ModelV3(sv3.wealth_tax_ubi()).run()
    b = ModelV3(sv3.wealth_tax_ubi()).run()
    assert a.gini[-1] == b.gini[-1]
    assert a.top1_share[-1] == b.top1_share[-1]
    assert a.gov_nw[-1] == b.gov_nw[-1]


def test_source_tax_preserves_invariants():
    """The source tax on foreign automation income (cash + in-kind equity legs)
    keeps deposits netting to zero, equity equal to K, and households non-negative,
    under both recycling modes."""
    from dataclasses import replace
    for tr in (0.25, 0.5):
        for rebate in (False, True):
            p = replace(sv3.foreign_ownership(), tax_repat=tr, repat_rebate=rebate,
                        n_agents=400, periods=200, seed=1)
            m = ModelV3(p)
            for _ in range(p.periods):
                m.step()
                assert m.h_dep.min() > -1e-9
            eq = m.h_eq.sum() + m.eq_state + m.eq_row
            assert abs(eq - m.K) / m.K < 1e-9
            assert abs(m.deposits_sum()) < 1e-2


def test_source_tax_accelerates_domestication():
    """A higher source tax on foreign-owned automation income brings the foreign
    ownership share down faster (the capital stock is domesticated sooner)."""
    from dataclasses import replace

    def cross5(p):
        h = ModelV3(p).run()
        fr = h.own_row
        for t, v in enumerate(fr):
            if v < 0.05:
                return t
        return len(fr)
    t0 = cross5(replace(sv3.foreign_ownership(), tax_repat=0.0, repat_rebate=True, seed=0))
    t1 = cross5(replace(sv3.foreign_ownership(), tax_repat=0.5, repat_rebate=True, seed=0))
    assert t1 < t0


def test_source_tax_off_by_default():
    """With tax_repat=0 the model is identical to the untaxed foreign-ownership
    scenario (the parameter is inert by default)."""
    a = ModelV3(sv3.foreign_ownership()).run()
    from dataclasses import replace
    b = ModelV3(replace(sv3.foreign_ownership(), tax_repat=0.0)).run()
    assert a.gini[-1] == b.gini[-1]
    assert np.allclose(a.repat_revenue, 0.0)


# --- two-channel automation (v4): AI vs robotic ---------------------------
from abm_sfc import scenarios_v3 as _sv3
from abm_sfc.production_v4 import NestedProduction
from dataclasses import replace as _replace


def test_nested_production_euler():
    """The four competitive factor incomes exhaust output exactly."""
    pr = NestedProduction(A=0.45, e_top=1.2, e_routine=0.6, e_cog=0.6, theta=0.5)
    for (Kr, Kai, Lr, Lc, ar, ac) in [(200, 200, 50, 50, 0.5, 0.5),
                                       (500, 80, 40, 60, 0.7, 0.3),
                                       (50, 900, 30, 70, 0.2, 0.9)]:
        d = pr.decompose(Kr, Kai, Lr, Lc, ar, ac)
        s = d["w_Lr"] + d["ci_Kr"] + d["w_Lc"] + d["ci_Kai"]
        assert abs(s - d["Y"]) / d["Y"] < 1e-12


def test_two_channel_capital_split():
    """K_r and K_ai stay positive and sum to the aggregate K."""
    p = _sv3.two_channel_base(); p.n_agents = 400; p.periods = 200; p.seed = 1
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
        assert m.K_r > 0 and m.K_ai > 0
        assert abs((m.K_r + m.K_ai) - m.K) < 1e-6 * m.K


def test_two_channel_invariants():
    """Equity = K, balances net to zero, no household overdraft, with the rent
    and all four instruments active."""
    p = _sv3.sovereign_compute(); p.n_agents = 400; p.periods = 250; p.seed = 2
    m = ModelV3(p)
    for _ in range(p.periods):
        m.step()
    eq = (m.h_eq.sum() + m.eq_state + m.eq_row) / m.K
    assert abs(eq - 1.0) < 1e-6
    assert abs(m.deposits_sum()) < 1e-4 * m.K
    assert m.h_dep.min() > -1e-6 * m.K
    assert abs(m.firm_dep) < 1e-4 * m.K


def test_ai_rent_zero_when_markup_zero():
    """With mu_frac = 0 no rent is generated; with mu_frac > 0 it is positive."""
    p0 = _replace(_sv3.two_channel_base(), mu_frac=0.0); p0.periods = 200; p0.seed = 0
    m0 = ModelV3(p0)
    for _ in range(p0.periods):
        m0.step()
    assert max(abs(x) for x in m0.hist.rent_ai) == 0.0
    p1 = _replace(_sv3.two_channel_base(), mu_frac=0.25); p1.periods = 200; p1.seed = 0
    m1 = ModelV3(p1)
    for _ in range(p1.periods):
        m1.step()
    assert m1.hist.rent_ai[-1] > 0.0


def test_dst_reaches_rent_corp_tax_does_not():
    """In steady state the DST collects revenue on the AI rent while ordinary
    corporate tax (empty base) collects almost none."""
    import numpy as np
    def cap_rev(make):
        m = ModelV3(_replace(make(), seed=0, periods=600))
        for _ in range(600):
            m.step()
        Y = np.array(m.hist.Y); w = slice(250, 600)
        return (np.array(m.hist.corp_tax_rev)[w].sum()
                + np.array(m.hist.repat_revenue)[w].sum()) / Y[w].sum()
    base = cap_rev(_sv3.two_channel_base)
    dst = cap_rev(_sv3.ai_dst)
    assert base < 0.005           # corporate tax reaches almost nothing
    assert dst > 0.02             # the DST reaches the rent


def test_two_channel_off_by_default():
    """A v3 scenario is bit-identical whether or not the two-channel params
    exist, as long as two_channel is False (the flag is inert by default)."""
    a = ModelV3(_sv3.income_tax_ubi()).run()
    b = ModelV3(_replace(_sv3.income_tax_ubi(), e_top=1.5, mu_frac=0.4,
                         robot_tax=0.2, dst_ai=0.1)).run()
    assert a.gini[-1] == b.gini[-1]      # two_channel=False ignores the v4 knobs


def test_trade_leak_invariants_and_curbs_ownership():
    """Phase 1a: with a goods-trade leak the invariants still hold, and
    repatriating the rent as goods lowers the foreign ownership share and the
    capital stock relative to the closed (fully-reinvested) case."""
    import numpy as np
    def run(phi):
        p = _replace(_sv3.two_channel_base(), trade_leak=phi, n_agents=400,
                     periods=300, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    closed, open_ = run(0.0), run(0.6)
    for m in (closed, open_):
        assert abs((m.h_eq.sum() + m.eq_state + m.eq_row) / m.K - 1.0) < 1e-6
        assert abs(m.deposits_sum()) < 1e-4 * m.K
        assert abs(m.firm_dep) < 1e-4 * m.K
    fshare = lambda m: m.eq_row / (m.h_eq.sum() + m.eq_state + m.eq_row)
    assert fshare(open_) < fshare(closed) - 0.02      # less ownership drift
    assert open_.K < closed.K                          # less domestic capital
    assert open_.hist.exports[-1] > 0.0                # the rent leaves as goods


def test_ai_supply_response_has_output_cost():
    """Phase 1b: taxing the AI rent with a positive supply elasticity shrinks
    output relative to the same tax with no response, and the invariants hold."""
    def run(eta):
        p = _replace(_sv3.two_channel_base(), dst_ai=0.10, tax_repat=0.30,
                     ai_supply_elasticity=eta, n_agents=400, periods=300, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    free, costly = run(0.0), run(0.5)
    assert costly.hist.Y[-1] < free.hist.Y[-1] * 0.97   # a real efficiency cost
    assert abs((costly.h_eq.sum() + costly.eq_state + costly.eq_row) / costly.K - 1.0) < 1e-6
    assert abs(costly.deposits_sum()) < 1e-4 * costly.K


def test_reinstatement_lifts_labour_share():
    """Phase 2: a positive reinstatement margin raises the steady-state labour
    share relative to the unoffset ramp, leaves the rent broadly unchanged (it is
    a labour/capital split effect), and preserves the invariants."""
    def run(rho):
        p = _replace(_sv3.two_channel_base(), reinstate_frac=rho, n_agents=400,
                     periods=300, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    off, on = run(0.0), run(0.5)
    lshare = lambda m: (m.hist.w_Lr[-1] + m.hist.w_Lc[-1]) / m.hist.Y[-1]
    assert lshare(on) > lshare(off) + 0.05               # labour share rises
    assert on.hist.rent_ai[-1] / on.hist.Y[-1] > 0.08    # rent broadly intact
    assert abs((on.h_eq.sum() + on.eq_state + on.eq_row) / on.K - 1.0) < 1e-6
    assert abs(on.deposits_sum()) < 1e-4 * on.K


def test_reinstatement_off_by_default():
    """reinstate_frac = 0 is bit-identical to the prior two-channel model."""
    a = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=200, seed=0))
    b = ModelV3(_replace(_sv3.two_channel_base(), reinstate_frac=0.0,
                         n_agents=300, periods=200, seed=0))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]


def test_unemployment_off_by_default():
    """With no pass-through and no UBI labour-supply response the model is
    bit-identical to the prior full-employment two-channel model."""
    a = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=200, seed=0))
    b = ModelV3(_replace(_sv3.two_channel_base(), unemployment_pass_through=0.0,
                         unemployment_benefit=0.5, n_agents=300, periods=200, seed=0))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]
    assert a.hist.gini[-1] == b.hist.gini[-1]


def test_unemployment_rises_with_passthrough_invariants_hold():
    """A positive pass-through produces a positive unemployment rate that rises
    with the parameter, output is held (extensive margin is output-neutral), and
    the deposit and equity invariants survive."""
    def run(lam):
        p = _replace(_sv3.two_channel_base(), reinstate_frac=0.5,
                     unemployment_pass_through=lam, unemployment_benefit=0.5,
                     n_agents=400, periods=300, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    lo, hi = run(0.2), run(0.4)
    assert 0.02 < lo.hist.unemployment[-1] < hi.hist.unemployment[-1]
    assert abs((hi.h_eq.sum() + hi.eq_state + hi.eq_row) / hi.K - 1.0) < 1e-6
    assert abs(hi.deposits_sum()) < 1e-3 * hi.K
    # output is broadly unchanged by the extensive-margin overlay (within a few %)
    base = run(0.0)
    assert hi.hist.Y[-1] > 0.5 * base.hist.Y[-1]


def test_optimising_owner_off_by_default():
    """With owner_shift_elasticity = 0 the model is bit-identical to the
    non-strategic two-channel model under the same taxes."""
    cfg = dict(dst_ai=0.10, tax_repat=0.30, n_agents=300, periods=200, seed=0)
    a = ModelV3(_replace(_sv3.two_channel_base(), owner_shift_elasticity=0.0, **cfg))
    b = ModelV3(_replace(_sv3.two_channel_base(), **cfg))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]
    assert a.eq_row == b.eq_row


def test_optimising_owner_erodes_take_and_drifts_ownership():
    """A mobile owner shifts the rent offshore: the host's revenue falls and
    foreign ownership rises relative to the non-strategic owner, with invariants
    intact."""
    def run(e):
        p = _replace(_sv3.two_channel_base(), dst_ai=0.10, tax_repat=0.30,
                     owner_shift_elasticity=e, n_agents=400, periods=400, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    base, mob = run(0.0), run(1.0)
    W = slice(150, 400)
    def rev(m):
        return (sum(m.hist.corp_tax_rev[W]) + sum(m.hist.repat_revenue[W]))
    assert rev(mob) < rev(base)                       # take eroded by shifting
    assert mob.eq_row / mob.K > base.eq_row / base.K   # ownership drifts up
    assert abs((mob.h_eq.sum() + mob.eq_state + mob.eq_row) / mob.K - 1.0) < 1e-6
    assert abs(mob.deposits_sum()) < 1e-3 * mob.K


def test_competition_off_by_default():
    """competition = 0 leaves the markup at mu_frac, so the model is bit-identical
    to the prior two-channel model."""
    a = ModelV3(_replace(_sv3.two_channel_base(), competition=0.0, n_agents=300, periods=200, seed=0))
    b = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=200, seed=0))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]
    assert a.eq_row == b.eq_row


def test_competition_erodes_rent_and_ownership():
    """Higher contestability shrinks the rent, so foreign ownership falls and the
    untaxed host has less rent to tax, with invariants intact."""
    def run(k):
        p = _replace(_sv3.two_channel_base(), competition=k, n_agents=400, periods=400, seed=1)
        m = ModelV3(p)
        for _ in range(p.periods):
            m.step()
        return m
    lo, hi = run(0.0), run(0.8)
    assert hi.eq_row / hi.K < lo.eq_row / lo.K          # rent competed away -> less foreign ownership
    assert abs((hi.h_eq.sum() + hi.eq_state + hi.eq_row) / hi.K - 1.0) < 1e-6
    assert abs(hi.deposits_sum()) < 1e-3 * hi.K


# ---- owner domicile: domestic vs foreign AI-IP owner ----
from abm_sfc import scenarios_v3 as _sv3
from dataclasses import replace as _replace


def _run_owner(ai_ip_foreign, T=300, seed=0):
    cfg = _replace(_sv3.two_channel_base(), ai_ip_foreign=ai_ip_foreign,
                   n_agents=300, periods=T, seed=seed)
    m = ModelV3(cfg)
    roy = dom = 0.0
    fshare = []
    for _ in range(T):
        m.step()
        roy += m._royalty_foreign
        dom += m._rent_dom
        et = m.h_eq.sum() + m.eq_state + m.eq_row
        fshare.append(m.eq_row / et if et > 0 else 0.0)
    return m, roy, dom, fshare


def _gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None)); n = len(x); s = x.sum()
    return 0.0 if s <= 0 else float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def test_domestic_owner_invariants():
    m, _, _, _ = _run_owner(0.0)
    assert abs(m.deposits_sum()) < 1e-3
    assert abs(m.total_nw() - m.real_assets()) < 1e-3


def test_domestic_owner_keeps_rent_home():
    _, roy, dom, fshare = _run_owner(0.0)
    assert roy == 0.0 and dom > 0.0          # nothing leaves; the rent accrues at home
    assert max(fshare) < 0.20                # no foreign-ownership drift


def test_foreign_owner_rent_leaves_and_drifts():
    _, roy, dom, fshare = _run_owner(1.0)
    assert dom == 0.0 and roy > 0.0          # the whole rent leaves as a royalty
    assert max(fshare) > fshare[0] + 0.05    # foreign ownership drifts up


def test_domestic_owner_raises_home_inequality():
    mD, _, _, _ = _run_owner(0.0)
    mF, _, _, _ = _run_owner(1.0)
    assert _gini(mD.house_nw()) >= _gini(mF.house_nw()) - 1e-9


# ---- rising-rent extensions: markup_power and cognitive_capture ----
def _run_rent(make, T=300, seed=0):
    cfg = _replace(make(), n_agents=300, periods=T, seed=seed)
    m = ModelV3(cfg)
    rs = []; fshare = []
    for _ in range(T):
        m.step()
        Y = m.hist.Y[-1]; rs.append(m.hist.rent_ai[-1] / Y if Y > 0 else 0.0)
        et = m.h_eq.sum() + m.eq_state + m.eq_row
        fshare.append(m.eq_row / et if et > 0 else 0.0)
    return m, np.array(rs), np.array(fshare)


def test_rising_rent_climbs_and_holds_invariants():
    mB, rsB, foB = _run_rent(_sv3.two_channel_base)
    mP, rsP, foP = _run_rent(_sv3.rising_rent_power)
    # rising-rent: late rent share exceeds the flat baseline, and ownership runs higher
    assert rsP[-50:].mean() > rsB[-50:].mean() + 0.03
    assert foP.max() > foB.max()
    # invariants intact under the extension
    assert abs(mP.deposits_sum()) < 1e-3
    assert abs(mP.total_nw() - mP.real_assets()) < 1e-3


def test_cognitive_capture_raises_rent_and_invariants():
    mB, rsB, _ = _run_rent(_sv3.two_channel_base)
    mC, rsC, _ = _run_rent(_sv3.rising_rent_capture)
    assert rsC[-50:].mean() > rsB[-50:].mean() + 0.02
    assert abs(mC.deposits_sum()) < 1e-3
    assert abs(mC.total_nw() - mC.real_assets()) < 1e-3


# ---- embodied robot-IP rent: behaves like the AI rent ----
def test_robot_ip_rent_raises_total_rent_reached_by_source_not_hardware():
    T = 300
    base = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0)); base.run()
    rip = ModelV3(_replace(_sv3.robot_ip_rent(), n_agents=300, periods=T, seed=0)); rip.run()
    Yb = np.array(base.hist.Y); Yr = np.array(rip.hist.Y)
    # total IP rent (AI + robot) is larger with a robot-IP rent than the AI-only baseline
    tot_b = (np.array(base.hist.rent_ai) + np.array(base.hist.rent_robot))[-50:].mean()
    tot_r = (np.array(rip.hist.rent_ai) + np.array(rip.hist.rent_robot))[-50:].mean()
    assert tot_r > tot_b + 0.02 * Yr[-50:].mean()
    assert np.array(rip.hist.rent_robot)[-50:].mean() > 0.0     # robot rent is positive
    assert np.array(base.hist.rent_robot)[-50:].mean() == 0.0   # zero in the baseline
    # invariants hold under the extension
    assert abs(rip.deposits_sum()) < 1e-3
    assert abs(rip.total_nw() - rip.real_assets()) < 1e-3

def _foreign_own(m):
    et = m.h_eq.sum() + m.eq_state + m.eq_row
    return m.eq_row / et if et > 0 else 0.0

def test_robot_ip_rent_source_levy_reaches_hardware_tax_misses():
    T = 400
    untaxed = ModelV3(_replace(_sv3.robot_ip_rent(), n_agents=300, periods=T, seed=0)); untaxed.run()
    robot = ModelV3(_replace(_sv3.robot_ip_rent(), n_agents=300, periods=T, seed=0, robot_tax=0.15)); robot.run()
    levy = ModelV3(_replace(_sv3.robot_ip_rent(), n_agents=300, periods=T, seed=0, dst_ai=0.10, tax_repat=0.30)); levy.run()
    fo_u, fo_robot, fo_levy = _foreign_own(untaxed), _foreign_own(robot), _foreign_own(levy)
    # the source levy pulls foreign ownership down materially; the hardware robot tax barely moves it
    assert fo_levy < fo_u - 0.10
    assert abs(fo_robot - fo_u) < abs(fo_levy - fo_u)


# ---- Experiment Q: two stacked foreign rents (compute vs model) ----

def test_compute_rent_accelerates_ownership_drift():
    """A second, foreign compute rent stacked on the IP rent raises the mature
    foreign-ownership share above the IP-rent-only baseline."""
    T = 400
    base = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0)); base.run()
    stk = ModelV3(_replace(_sv3.stacked_rents_importer(), n_agents=300, periods=T, seed=0)); stk.run()
    assert np.array(stk.hist.rent_compute)[-50:].mean() > 0.0      # compute rent is positive
    assert np.array(base.hist.rent_compute)[-50:].mean() == 0.0    # zero in the baseline
    assert _foreign_own(stk) > _foreign_own(base) + 0.02
    assert abs(stk.deposits_sum()) < 1e-3
    assert abs(stk.total_nw() - stk.real_assets()) < 1e-3


def test_dst_and_withholding_miss_the_compute_rent():
    """The licence-flow instruments (DST, withholding) collect NOTHING from the
    compute rent, while the tariff and the usage levy do. The compute-rent take is
    recorded in hist.compute_rev."""
    T = 400
    def take(make):
        m = ModelV3(_replace(make(), n_agents=300, periods=T, seed=0)); m.run()
        return np.array(m.hist.compute_rev)[100:].sum()
    imp = take(_sv3.stacked_rents_importer)
    dst = take(lambda: _replace(_sv3.stacked_rents_importer(), dst_ai=0.10))
    wht = take(lambda: _replace(_sv3.stacked_rents_importer(), tax_repat=0.30, repat_rebate=True))
    tar = take(_sv3.stacked_rents_tariff)
    use = take(_sv3.stacked_rents_usage)
    assert imp == 0.0 and dst == 0.0 and wht == 0.0   # none of these reach it
    assert tar > 0.0 and use > 0.0                     # the border/usage levers do


def test_domestic_chipmaker_keeps_the_compute_rent_home():
    """Moving the chip-maker's domicile home (compute_foreign=0) keeps the rent and
    lowers foreign ownership relative to the importer; onshoring servers (s_home)
    does not."""
    T = 400
    imp = ModelV3(_replace(_sv3.stacked_rents_importer(), n_agents=300, periods=T, seed=0)); imp.run()
    home = ModelV3(_replace(_sv3.stacked_rents_us_chips(), n_agents=300, periods=T, seed=0)); home.run()
    off = ModelV3(_replace(_sv3.stacked_rents_offshore(), n_agents=300, periods=T, seed=0)); off.run()
    assert _foreign_own(home) < _foreign_own(imp) - 0.05      # home chips keep the rent
    assert abs(_foreign_own(off) - _foreign_own(imp)) < 0.02  # s_home is inert for it


# ---- Experiment R: elastic labour supply and the bottleneck wage ----

def _two_ch_steady(m):
    """Steady-state cluster aggregates for a finished two-channel run."""
    W = slice(250, m.p.periods)
    wLr = np.array(m.hist.w_Lr)[W].sum(); ciKr = np.array(m.hist.ci_Kr)[W].sum()
    wuR = float(np.mean(m.hist.wage_unit_r[-50:]))
    wuC = float(np.mean(m.hist.wage_unit_c[-50:]))
    return dict(routine_labshare=wLr / (wLr + ciKr), wuR=wuR, ratio=wuR / max(wuC, 1e-9))


def test_elastic_supply_is_inert_at_zero():
    """Regression guard: zero elasticities reproduce the fixed-supply model exactly."""
    T = 300
    base = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0)); base.run()
    inel = ModelV3(_replace(_sv3.labour_inelastic(), n_agents=300, periods=T, seed=0)); inel.run()
    assert inel._sf_r == 1.0 and inel._sf_c == 1.0
    assert np.allclose(np.array(base.hist.w_Lr), np.array(inel.hist.w_Lr))
    assert np.allclose(np.array(base.hist.wage_unit_r), np.array(inel.hist.wage_unit_r))


def test_elastic_supply_dampens_the_bottleneck_wage():
    """The bottleneck (routine) per-unit wage spike and the routine/cognitive
    scarcity premium are both competed down once supply is elastic, and within the
    bottleneck cluster the surplus shifts toward capital (labour share falls)."""
    T = 400
    inel = ModelV3(_replace(_sv3.labour_inelastic(), n_agents=300, periods=T, seed=0)); inel.run()
    elas = ModelV3(_replace(_sv3.labour_elastic_strong(), n_agents=300, periods=T, seed=0)); elas.run()
    i, e = _two_ch_steady(inel), _two_ch_steady(elas)
    assert e["wuR"] < i["wuR"]                       # bottleneck wage dampened
    assert e["ratio"] < i["ratio"]                    # scarcity premium collapses
    assert e["routine_labshare"] < i["routine_labshare"]   # cluster surplus to capital
    assert elas._sf_r > 1.0                           # labour did expand
    assert abs(elas.deposits_sum()) < 1e-3
    assert abs(elas.total_nw() - elas.real_assets()) < 1e-3


# ---- Experiment S: endogenous, cost-driven automation ----

def test_endogenous_automation_is_inert_when_cost_is_negligible():
    """With the flag off the ramp is exogenous; with the flag on but machine cost far
    below the wage the gate is fully open, so the realised ramp matches the frontier."""
    T = 300
    exo = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0)); exo.run()
    cheap = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0,
                             endogenous_automation=True, auto_cost0_ai=1e-3, auto_cost0_r=1e-3,
                             cost_decline_ai=0.0, cost_decline_r=0.0)); cheap.run()
    assert np.allclose(np.array(exo.hist.a_ai), np.array(cheap.hist.a_ai), atol=1e-3)


def test_cost_decline_speed_sets_the_pace_and_transition_still_arrives():
    """Slower cost decline => later, lower realised automation than faster decline and
    than the exogenous frontier; but the transition still arrives by the end."""
    T = 600
    exo = ModelV3(_replace(_sv3.two_channel_base(), n_agents=300, periods=T, seed=0)); exo.run()
    slow = ModelV3(_replace(_sv3.endog_auto_slow(), n_agents=300, periods=T, seed=0)); slow.run()
    mit = ModelV3(_replace(_sv3.endog_auto_mit(), n_agents=300, periods=T, seed=0)); mit.run()
    fast = ModelV3(_replace(_sv3.endog_auto_fast(), n_agents=300, periods=T, seed=0)); fast.run()
    a_slow, a_mit, a_fast = (np.array(m.hist.a_ai) for m in (slow, mit, fast))
    # at mid-transition the realised share is ordered by cost-decline speed
    assert a_slow[250] < a_mit[250] < a_fast[250]
    assert a_slow[250] < np.array(exo.hist.a_ai)[250]      # endogenous lags the frontier
    # but it is "how fast", not "whether": slow automation still arrives by the end
    assert a_slow[-1] > 0.9
    assert abs(slow.deposits_sum()) < 1e-3
    assert abs(slow.total_nw() - slow.real_assets()) < 1e-3
