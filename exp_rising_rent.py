"""Rising-rent extensions (Figure 26): the markup need not stay a fixed share.
markup_power lets pricing power grow with AI automation; cognitive_capture lets
the AI cluster take a growing share of total output (a reduced-form proxy for
AI-native production displacing the human/routine cluster). Both default off.
"""
import numpy as np
from dataclasses import replace
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

T=600
def run(**kw):
    m=ModelV3(replace(S.two_channel_base(), periods=T, seed=0, **kw)); m.run()
    rs=100*np.array(m.hist.rent_ai)/np.array(m.hist.Y)
    fo=[]
    m2=ModelV3(replace(S.two_channel_base(), periods=T, seed=0, **kw))
    for _ in range(T):
        m2.step(); et=m2.h_eq.sum()+m2.eq_state+m2.eq_row; fo.append(100*m2.eq_row/et if et>0 else 0)
    return rs, np.array(fo)

cases=[("flat markup (baseline)","#7c2d12","-",{}),
       ("pricing power rises","#b5651d","-",{"markup_power":1.0}),
       ("AI captures more output","#1d4e44","-",{"cognitive_capture":0.6}),
       ("both (extreme)","#3b6ea5","--",{"markup_power":1.5,"cognitive_capture":0.6})]
t=np.arange(T)
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.6,4.3))
for lab,c,ls,kw in cases:
    rs,fo=run(**kw)
    a1.plot(t,rs,color=c,ls=ls,lw=2,label=lab)
    a2.plot(t,fo,color=c,ls=ls,lw=2,label=lab)
for a in (a1,a2): a.axvspan(110,210,color="0.85",alpha=.5,lw=0)
a1.text(160,2,"AI ramp",ha="center",color="0.4",fontsize=9)
a1.set_title("The AI rent as a share of output"); a1.set_xlabel("period"); a1.set_ylabel("rent / output (%)"); a1.legend(frameon=False,fontsize=8.5); a1.set_ylim(0,None)
a2.set_title("Foreign ownership of capital"); a2.set_xlabel("period"); a2.set_ylabel("foreign ownership (%)"); a2.legend(frameon=False,fontsize=8.5); a2.set_ylim(0,None)
fig.tight_layout(); fig.savefig("figures_v3/p5_rising_rent.png",dpi=110)
print("saved figures_v3/p5_rising_rent.png")
