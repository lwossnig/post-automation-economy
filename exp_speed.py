import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc import Model, scenarios
from dataclasses import replace

B = scenarios.laissez_faire()

def gini_traj(auto_max, auto_speed, seeds=8, periods=400):
    gs=[]
    for s in range(seeds):
        p = replace(B, auto_max=auto_max, auto_speed=auto_speed,
                    auto_start=80, periods=periods, seed=s)
        gs.append(Model(p).run().gini)
    return np.array(gs).mean(0)

def summarise(g):
    g=np.asarray(g)
    final=g[-50:].mean()
    g0=g[:20].mean()
    slope=np.max(np.diff(g))            # peak per-period rise
    # time to reach halfway between start and final
    half=g0+0.5*(final-g0)
    cross=np.argmax(g>=half) if np.any(g>=half) else len(g)
    return final, slope, cross

# 1) sweep automation LEVEL (auto_max) at fixed speed
print("Effect of automation LEVEL (auto_speed=0.06):")
print(f"{'auto_max':>9}{'final Gini':>12}{'peak slope':>12}{'t_half':>8}")
levels=[0.0,0.3,0.5,0.7,0.9,0.99]
trajs_level={}
for am in levels:
    g=gini_traj(am,0.06); trajs_level[am]=g
    f,sl,c=summarise(g)
    print(f"{am:9.2f}{f:12.3f}{sl:12.5f}{c:8d}")

# 2) sweep automation SPEED at fixed level
print("\nEffect of automation SPEED (auto_max=0.95):")
print(f"{'auto_speed':>11}{'final Gini':>12}{'peak slope':>12}{'t_half':>8}")
speeds=[0.02,0.04,0.06,0.10,0.20]
trajs_speed={}
for sp in speeds:
    g=gini_traj(0.95,sp); trajs_speed[sp]=g
    f,sl,c=summarise(g)
    print(f"{sp:11.2f}{f:12.3f}{sl:12.5f}{c:8d}")

# plot
fig,ax=plt.subplots(1,2,figsize=(13,5))
for am,g in trajs_level.items():
    ax[0].plot(g,label=f"auto_max={am}")
ax[0].set_title("Wealth Gini vs automation LEVEL"); ax[0].legend(fontsize=8)
for sp,g in trajs_speed.items():
    ax[1].plot(g,label=f"auto_speed={sp}")
ax[1].set_title("Wealth Gini vs automation SPEED"); ax[1].legend(fontsize=8)
for a in ax: a.set_xlabel("period"); a.set_ylabel("Gini"); a.grid(alpha=0.3); a.set_ylim(0,0.7)
fig.tight_layout(); fig.savefig("figures/speed_vs_automation.png",dpi=110)
print("\nsaved figures/speed_vs_automation.png")
