import numpy as np
from abm_sfc import Model, scenarios
from dataclasses import replace

# Citizens' wealth fund: the state acquires equity via the wealth-tax equity leg
# (and/or initial state ownership) but pays ALL dividends on state-held equity
# straight back out as an equal per-capita dividend. Ownership is socialised;
# the income is distributed; the state is a custodian, not a hoarder.
def run_fund(p, seeds=6):
    ginis=[]; govs=[]; extra=[]
    for s in range(seeds):
        pp=replace(p,seed=s); m=Model(pp)
        for _ in range(pp.periods):
            # state dividend this period = div_payout share of state's equity return,
            # approximated as payout * (after-tax profit) * state equity fraction.
            eq_total = m.h_eq.sum()+m.eq_state+m.eq_row
            f_state = m.eq_state/eq_total if eq_total>0 else 0
            # rebate: add equal per-capita dividend to every household's deposit
            # (estimate from last realised dividends via the model's own bookkeeping)
            m.step()
            # after step, state received div_state into gov_dep; rebate it equally
            # reconstruct div_state from f_state and realised profit proxy:
            # simpler: move a fraction of the period's gov dividend back out.
            # We approximate the state dividend as f_state * div_payout * pi*Y*(1-tax_corp)
            pi = pp.pi0+(1-pp.pi0)*m.alpha_t()
            Y = pp.Y0
            div_state = f_state*pp.div_payout*pi*Y*(1-pp.tax_corp)
            m.gov_dep -= div_state
            m.h_dep += div_state/pp.n_agents
        ginis.append(np.mean(m.hist.gini[-50:]))
        govs.append(m.hist.gov_nw[-1]/pp.n_agents)
    return np.mean(ginis), np.mean(govs)

# combine: modest wealth tax building a fund, dividends rebated
p = replace(scenarios.wealth_tax_ubi(), periods=300, tax_wealth=0.02, wealth_exempt=0.0)
g,v = run_fund(p)
print(f"Citizens' fund (wealth tax 2%, dividends rebated as UBI):")
print(f"  final Gini = {g:.3f}   gov NW / Y = {v:.1f}")

# also a state-ownership-from-the-start citizens fund
p2 = replace(scenarios.state_ownership(), periods=300)
g2,v2 = run_fund(p2)
print(f"State ownership 60% + dividend rebate:")
print(f"  final Gini = {g2:.3f}   gov NW / Y = {v2:.1f}")
