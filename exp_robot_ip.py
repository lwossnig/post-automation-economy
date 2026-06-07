"""Embodied robot-IP rent (Figure 27): robots carry a foreign-held IP markup of the
same order as the AI one. Left: the total foreign-held rent (AI + robot) as a share
of output. Right: foreign ownership over time, showing the hardware robot tax barely
touches the drift while the source levy reaches it. robot_ip defaults to 0.
"""
import numpy as np
from dataclasses import replace
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

T=600
def rent_share(**kw):
    m=ModelV3(replace(S.two_channel_base(), periods=T, seed=0, **kw)); m.run()
    Y=np.array(m.hist.Y)
    return 100*(np.array(m.hist.rent_ai)+np.array(m.hist.rent_robot))/Y
def own_path(**kw):
    m=ModelV3(replace(S.two_channel_base(), periods=T, seed=0, **kw)); fo=[]
    for _ in range(T):
        m.step(); et=m.h_eq.sum()+m.eq_state+m.eq_row; fo.append(100*m.eq_row/et if et>0 else 0)
    return np.array(fo)

t=np.arange(T)
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.6,4.3))

# left: total rent share, baseline (AI only) vs robot-IP (AI + robot)
a1.plot(t, rent_share(), color="#7c2d12", lw=2, label="baseline: AI rent only")
a1.plot(t, rent_share(robot_ip=0.25), color="#1d4e44", lw=2, label="robots also carry IP rent")
a1.axvspan(110,210,color="0.85",alpha=.5,lw=0); a1.text(160,1.5,"AI ramp",ha="center",color="0.4",fontsize=9)
a1.set_title("Total foreign-held IP rent / output"); a1.set_xlabel("period"); a1.set_ylabel("rent / output (%)")
a1.legend(frameon=False,fontsize=8.5); a1.set_ylim(0,29)

# right: foreign ownership under the robot-IP rent and the instruments
a2.plot(t, own_path(), color="#7c2d12", lw=2, label="baseline (no robot rent)")
a2.plot(t, own_path(robot_ip=0.25), color="#3b6ea5", ls="--", lw=2, label="robot-IP rent, untaxed")
a2.plot(t, own_path(robot_ip=0.25, robot_tax=0.15), color="#b5651d", lw=2, label="+ hardware robot tax")
a2.plot(t, own_path(robot_ip=0.25, dst_ai=0.10, tax_repat=0.30), color="#1d4e44", lw=2, label="+ source levy (DST + withholding)")
a2.axvspan(110,210,color="0.85",alpha=.5,lw=0)
a2.set_title("Foreign ownership of capital"); a2.set_xlabel("period"); a2.set_ylabel("foreign ownership (%)")
a2.legend(frameon=False,fontsize=8.5); a2.set_ylim(0,None)

fig.tight_layout(); fig.savefig("figures_v3/p6_robot_ip.png", dpi=110)
print("saved figures_v3/p6_robot_ip.png")
