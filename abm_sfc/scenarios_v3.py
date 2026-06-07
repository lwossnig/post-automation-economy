"""v3 scenario presets. Same regimes as v2 but on the v3 model (microfounded
returns, behavioural wealth tax, endogenous mobility). Behavioural responses are
ON by default; each policy has a *_frictionless twin for the robustness contrast.
"""
from __future__ import annotations

from dataclasses import replace

from .model_v3 import ParamsV3

BASE = ParamsV3(
    n_agents=2000,
    periods=600,
    eps=0.6,
    K0_per_capita=5.0,
    I_base=0.50,
    auto_start=80,
    auto_speed=0.06,
    auto_max=0.45,
    div_payout=1.0,
    c_income=0.85,
    c_profit=0.35,
    c_wealth=0.03,
    gov_cost=0.10,
    r_debt=0.02,
    ret_sigma=0.05,
    ret_persist=0.92,
    demographic_reset=0.02,
    avoidance_elasticity=0.75,
    migration_semi_elast=0.02,
    mobility_on=True,
    init_wealth_sigma=1.15,
    seed=0,
)

UBI = 0.30


def laissez_faire():
    # Pure free-market benchmark: no taxes, no transfers AND no government
    # consumption (gov_cost=0). A government that bought goods without taxing
    # would run a perpetual unfunded primary deficit; the resulting public debt
    # is on an explosive r>g path and would contaminate the real economy through
    # interest income. The honest no-policy counterfactual is simply no state.
    return replace(BASE, own_households=1.0, tax_corp=0.0, tax_income=0.0,
                   tax_wealth=0.0, ubi=0.0, gov_cost=0.0)


def income_tax_ubi():
    return replace(BASE, tax_corp=0.25, tax_income=0.30, ubi=UBI)


def wealth_tax_ubi():
    return replace(BASE, tax_corp=0.25, tax_income=0.15, tax_wealth=0.02, ubi=UBI)


def wealth_tax_frictionless():
    """Same wealth tax but with behavioural responses switched off: the v2-style
    frictionless instrument, for the robustness contrast."""
    return replace(BASE, tax_corp=0.25, tax_income=0.15, tax_wealth=0.02, ubi=UBI,
                   mobility_on=False, avoidance_elasticity=0.0)


def progressive_wealth():
    return replace(BASE, tax_corp=0.25, tax_income=0.15,
                   wealth_brackets=((3.0, 0.01), (10.0, 0.03)), ubi=UBI)


def state_ownership():
    return replace(BASE, own_households=0.4, own_state=0.6, tax_corp=0.25, tax_income=0.15, ubi=UBI)


def citizens_fund():
    return replace(BASE, own_households=0.4, own_state=0.6, citizens_fund=True,
                   tax_corp=0.25, tax_income=0.15, ubi=UBI)


def foreign_ownership():
    # Initial foreign-ownership SHOCK (review item 4): the rest of the world
    # starts holding 60% of equity, but the stake is NOT pinned. Because new
    # equity issuance is subscribed by whoever is saving, and under a taxing
    # government the big domestic saver is the state, the foreign stake dilutes
    # over the horizon and is, in effect, gradually transferred to the domestic
    # public sector. That dilution-to-the-state is the finding this scenario
    # reports, not an artefact. Sustaining a constant foreign share would require
    # modelling persistent FDI inflows, whose domestic counterpart is an
    # ever-growing gross creditor position under r > g; that is left as future
    # work rather than imposed here.
    return replace(BASE, own_households=0.4, own_row=0.6, tax_corp=0.25,
                   tax_income=0.30, ubi=UBI)


REGISTRY = {
    "laissez_faire": laissez_faire,
    "income_tax_ubi": income_tax_ubi,
    "wealth_tax_ubi": wealth_tax_ubi,
    "wealth_tax_frictionless": wealth_tax_frictionless,
    "progressive_wealth": progressive_wealth,
    "state_ownership": state_ownership,
    "citizens_fund": citizens_fund,
    "foreign_ownership": foreign_ownership,
}

# --- two-channel automation (v4): AI vs robotic, taxed differently -----------
# The base case is a standard income-tax-and-UBI state in which a foreign-owned
# AI IP layer charges a markup (mu_frac) on the cognitive cluster, so a durable
# rent leaks abroad as a deductible licence fee. The compute starts home-located
# (s_home = 1). Each policy variant then adds the one instrument that tries to
# reach the rent or the robots, holding everything else fixed, so the experiment
# isolates what each lever actually catches.
TWO_CH = replace(income_tax_ubi(), two_channel=True, skill_dispersion=0.4,
                 mu_frac=0.25, s_home=1.0)


def two_channel_base():
    """Foreign-owned AI rent, untaxed beyond ordinary corporate/income tax."""
    return replace(TWO_CH)


def robot_tax_only():
    """Source levy on robotic capital income (the physical 'robot tax')."""
    return replace(TWO_CH, robot_tax=0.15)


def ai_dst():
    """Digital-services levy on AI revenue, the handle that does not leak."""
    return replace(TWO_CH, dst_ai=0.10)


def ai_withholding():
    """Withholding on the rent as it is repatriated, rebated to citizens."""
    return replace(TWO_CH, tax_repat=0.30, repat_rebate=True)


def offshore_compute():
    """Same rent, but most AI compute sits on foreign servers (low s_home)."""
    return replace(TWO_CH, s_home=0.20)


def sovereign_compute():
    """Onshored compute plus a robot tax and a DST: the full domestic toolkit."""
    return replace(TWO_CH, s_home=1.0, robot_tax=0.15, dst_ai=0.10)


def open_economy():
    """Phase 1a: half the foreign owner's net rent is repatriated as real goods
    (a current-account leak) rather than reinvested in domestic equity."""
    return replace(TWO_CH, trade_leak=0.5)


def ai_dst_with_response():
    """Phase 1b: the digital levy on AI, with the AI owner deploying less in
    response to the net-of-tax wedge (the levy carries an efficiency cost)."""
    return replace(TWO_CH, dst_ai=0.10, ai_supply_elasticity=0.5)


def reinstatement():
    """Phase 2: automation is partly offset by new labour tasks (reinstatement),
    so the capital share of each cluster is reinstated toward labour over time."""
    return replace(TWO_CH, reinstate_frac=0.5)


def automation_unemployment():
    """Phase 2 (piece 2): displacement realised on the extensive margin. Balanced
    reinstatement plus a moderate pass-through give a double-digit unemployment
    rate, with a state-funded safety net for the out-of-work."""
    return replace(TWO_CH, reinstate_frac=0.5, unemployment_pass_through=0.3,
                   unemployment_benefit=0.5)


def optimising_owner():
    """Phase 3: the foreign owner shifts a share of the rent recognition offshore
    in response to the host's tax wedge, eroding the take and drifting ownership
    up. Run with the full toolkit so there is a wedge to respond to."""
    return replace(TWO_CH, dst_ai=0.10, tax_repat=0.30, owner_shift_elasticity=1.0)


def contestable_frontier():
    """Phase 3: a contestable AI frontier in which open-weight competition prices
    half the rent away (the endogenous-markup channel)."""
    return replace(TWO_CH, competition=0.5)


REGISTRY.update({
    "two_channel_base": two_channel_base,
    "robot_tax_only": robot_tax_only,
    "ai_dst": ai_dst,
    "ai_withholding": ai_withholding,
    "offshore_compute": offshore_compute,
    "sovereign_compute": sovereign_compute,
    "open_economy": open_economy,
    "ai_dst_with_response": ai_dst_with_response,
    "reinstatement": reinstatement,
    "automation_unemployment": automation_unemployment,
    "optimising_owner": optimising_owner,
    "contestable_frontier": contestable_frontier,
})


# --- domestic-owner baseline (the simpler case: the US owns the AI) ----------
# Same two-channel economy, but the AI IP is owned at home (ai_ip_foreign = 0),
# so the rent does NOT leave: it accrues to resident owners as capital income and
# concentrates among them. The policy problem is domestic inequality rather than
# lost national ownership, and the instrument that bites is a domestic tax on the
# owners (a wealth tax), not a source levy on a cross-border fee.
def domestic_owner():
    """AI IP owned at home: the rent stays, concentrating among capital owners."""
    return replace(TWO_CH, ai_ip_foreign=0.0)


def domestic_owner_wealthtax():
    """Domestic owner, with a wealth tax reaching the home-held rent."""
    return replace(TWO_CH, ai_ip_foreign=0.0, tax_wealth=0.05, wealth_exempt=1.0)


# --- rising-rent extensions (the markup need not stay a fixed share) ----------
def rising_rent_power():
    """Pricing power grows with automation: the markup roughly doubles by full AI."""
    return replace(TWO_CH, markup_power=1.0)


def rising_rent_capture():
    """The AI cluster captures a growing share of total output as it automates."""
    return replace(TWO_CH, cognitive_capture=0.6)


def rising_rent_both():
    """Both channels together: the extreme but not implausible case."""
    return replace(TWO_CH, markup_power=1.5, cognitive_capture=0.6)


# --- embodied robot-IP rent (robots carry a foreign-held IP markup too) ------
def robot_ip_rent():
    """Robots carry an IP rent of the same order as the AI markup (0.25)."""
    return replace(TWO_CH, robot_ip=0.25)


# --- Experiment Q: two stacked foreign rents (compute vs model) --------------
# A second durable rent sits on the COMPUTE layer (chip design + ecosystem lock-in),
# embedded in the price of AI capital rather than flowing as a licence fee. The base
# case adds it on top of the foreign IP rent (the pure importer); variants move its
# domicile home or reach for it with the two new instruments. mu_compute = 0.15 is a
# compute markup of the same order as a fraction of the AI markup.
def stacked_rents_importer():
    """Both rents foreign: a compute/chip rent stacked on the foreign IP rent."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=1.0)


def stacked_rents_us_chips():
    """Compute rent domestic (a home chip-maker), IP rent still foreign."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=0.0)


def stacked_rents_full_owner():
    """Both rents domestic: the full owner (compute and model at home)."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=0.0, ai_ip_foreign=0.0)


def stacked_rents_tariff():
    """Pure importer plus a border tariff on the imported compute rent."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=1.0, tariff_compute=0.30)


def stacked_rents_usage():
    """Pure importer plus a usage levy on the compute bill (partial reach)."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=1.0, usage_levy=0.10)


def stacked_rents_offshore():
    """Pure importer with compute offshored: onshoring (s_home) does NOT reach it."""
    return replace(TWO_CH, mu_compute=0.15, compute_foreign=1.0, s_home=0.20)


REGISTRY.update({
    "stacked_rents_importer": stacked_rents_importer,
    "stacked_rents_us_chips": stacked_rents_us_chips,
    "stacked_rents_full_owner": stacked_rents_full_owner,
    "stacked_rents_tariff": stacked_rents_tariff,
    "stacked_rents_usage": stacked_rents_usage,
    "stacked_rents_offshore": stacked_rents_offshore,
})


# --- Experiment R: elastic labour supply and the bottleneck wage -------------
# The baseline pays each labour type its marginal product with FIXED supply, so the
# task AI cannot do (the physical/routine bottleneck) becomes a scarce complement to
# robotic capital and its per-unit wage explodes: the model's "automation raises
# wages" result. labour_inelastic is that baseline. The elastic scenarios make the
# bottleneck labour abundant (an upward-sloping supply with a reservation floor), so
# the scarcity premium is competed away: the bottleneck wage is dampened, the
# wuR/wuC premium collapses, and within the bottleneck cluster the surplus shifts to
# capital. Elasticities are kept in the range where the fixed point converges below
# the supply safety band. (At the aggregate the labour share need not fall, because
# the labour-intensive bottleneck cluster expands; the clean result is the bottleneck
# WAGE, not the economy-wide share.)
def labour_inelastic():
    """Fixed labour supply (the current model); the sanity anchor."""
    return replace(TWO_CH)


def labour_elastic_routine():
    """Abundant, elastic routine (bottleneck) labour with a reservation floor."""
    return replace(TWO_CH, labour_supply_elast_r=0.2, reservation_wage=0.5)


def labour_elastic_strong():
    """A stronger supply response: the bottleneck premium is competed away further."""
    return replace(TWO_CH, labour_supply_elast_r=0.3, reservation_wage=0.5)


REGISTRY.update({
    "labour_inelastic": labour_inelastic,
    "labour_elastic_routine": labour_elastic_routine,
    "labour_elastic_strong": labour_elastic_strong,
})
