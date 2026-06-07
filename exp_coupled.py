import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc import Model, scenarios
from abm_sfc.kinetic import kinetic_step
from dataclasses import replace

# v2 mechanism: couple idiosyncratic equity-return volatility to the capital
# share. As automation raises pi_t, more of value-added compounds through the
# multiplicative (concentrating) channel, so effective sigma rises with pi_t.
class CoupledModel(Model):
    def __init__(self, p, coupling=1.0):
        super().__init__(p); self.coupling=coupling
    def step(self):
        p=self.p
        pi = p.pi0 + (1.0-p.pi0)*self.alpha_t()
        # sigma scales with capital share relative to baseline pi0
        self._sigma_eff = p.bm_sigma * (1.0 + self.coupling*(pi/p.pi0 - 1.0))
        super().step()
    # override the kernel call by temporarily swapping sigma
    def _bm(self):
        pass

# monkeypatch: reimplement the kernel loop using sigma_eff
import abm_sfc.model as M
_orig=M.kinetic_step
def patched(wealth,J,sigma,dt,rng,floor=1e-9):
    return _orig(wealth,J,sigma,dt,rng,floor)
# Instead of patching globally, subclass step to use sigma_eff via closure:
class CoupledModel2(Model):
    def __init__(self,p,coupling=1.0):
        super().__init__(p); self.coupling=coupling
    def step(self):
        p=self.p; pi=p.pi0+(1.0-p.pi0)*self.alpha_t()
        sig=p.bm_sigma*(1.0+self.coupling*(pi/p.pi0-1.0))
        old=p.bm_sigma; p.bm_sigma=sig
        try: super().step()
        finally: p.bm_sigma=old

B=scenarios.laissez_faire()
def traj(auto_max,coupling,seeds=8,periods=400):
    gs=[]
    for s in range(seeds):
        p=replace(B,auto_max=auto_max,auto_speed=0.06,auto_start=80,periods=periods,seed=s,
                  bm_sigma=0.45)
        gs.append(CoupledModel2(p,coupling).run().gini)
    return np.array(gs).mean(0)

print("Coupled model: final Gini vs automation level")
print(f"{'auto_max':>9}{'final Gini':>12}{'t_half':>8}")
trajs={}
for am in [0.0,0.3,0.6,0.9,0.99]:
    g=traj(am,1.0); trajs[am]=g
    final=g[-50:].mean(); g0=g[:20].mean()
    half=g0+0.5*(final-g0); c=int(np.argmax(g>=half))
    print(f"{am:9.2f}{final:12.3f}{c:8d}")

fig,ax=plt.subplots(figsize=(7,5))
for am,g in trajs.items(): ax.plot(g,label=f"auto_max={am}")
ax.axvline(80,ls='--',c='grey',lw=0.8,label='automation onset')
ax.set_title("v2 coupled model: concentration speed tracks automation level")
ax.set_xlabel("period"); ax.set_ylabel("Gini"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0,0.85)
fig.tight_layout(); fig.savefig("figures/coupled_speed.png",dpi=110)
print("saved figures/coupled_speed.png")
