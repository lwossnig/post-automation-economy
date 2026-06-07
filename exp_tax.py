import numpy as np
from abm_sfc import Model, scenarios
from dataclasses import replace

# Variant model with two policy switches on the wealth-tax equity leg:
#   redistribute_equity: confiscated equity is handed back to households equally
#                        (a citizens' dividend in stock) instead of kept by state
#   rebate_state_div:    dividends on state-held equity are paid out as extra UBI
class PolicyModel(Model):
    def __init__(self, p, redistribute_equity=False, rebate_state_div=False):
        super().__init__(p)
        self.redistribute_equity = redistribute_equity
        self.rebate_state_div = rebate_state_div
    def step(self):
        # capture state equity before the step to compute its dividend
        super().step()

# Simplest: reimplement the equity-leg behaviour by post-processing each period.
# We subclass and override the equity transfer by re-running the relevant logic.
# To keep it clean we add hooks via small reimplementation:
import numpy as np
from abm_sfc.kinetic import kinetic_step

def run_variant(p, mode, seeds=6):
    """mode in {'keep','redistribute','exempt'}"""
    finals=[]; ginis=[]; govs=[]
    for s in range(seeds):
        pp = replace(p, seed=s)
        m = Model(pp)
        # patch: wrap step to alter equity leg destination
        for _ in range(pp.periods):
            # snapshot before
            eq_before = m.h_eq.copy(); state_before = m.eq_state
            m.step()
            if mode=='redistribute':
                # undo the state's gain of wt_eq, hand it to households equally
                gained = m.eq_state - state_before
                # remove the retained-earnings part that legitimately accrues to state? 
                # in wealth_tax scenarios own_state=0 so all state gain is the tax leg
                m.eq_state -= gained
                m.h_eq += gained / pp.n_agents
        ginis.append(np.mean(m.hist.gini[-50:]))
        govs.append(m.hist.gov_nw[-1]/pp.n_agents)
    return np.mean(ginis), np.mean(govs)

base = scenarios.wealth_tax_ubi()
base = replace(base, periods=300)

print(f"{'policy':28}{'final Gini':>12}{'gov NW / Y':>12}")
g,v = run_variant(base,'keep');           print(f"{'flat wealth tax (keep)':28}{g:12.3f}{v:12.1f}")
g,v = run_variant(base,'redistribute');   print(f"{'redistribute equity':28}{g:12.3f}{v:12.1f}")
ex = replace(base, wealth_exempt=3.0)     # exempt wealth below 3x mean
g,v = run_variant(ex,'keep');             print(f"{'exemption = 3x mean':28}{g:12.3f}{v:12.1f}")
ex2 = replace(base, wealth_exempt=3.0, tax_wealth=0.05)
g,v = run_variant(ex2,'keep');            print(f"{'exemption 3x, tax 5%':28}{g:12.3f}{v:12.1f}")
