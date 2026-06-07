"""Compact domestic-vs-foreign owner baseline (Figure 25).
A: who owns the capital. B: inequality at home over time. C: three inequality
measures at the mature steady state (Gini, top-10% and top-1% wealth shares),
so the broad-owner concentration is visible where the Gini alone understates it.
"""
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as S

T, SEEDS = 600, np.arange(4)

def gini(x):
    x=np.sort(np.clip(np.asarray(x,float),0,None));n=len(x);s=x.sum()
    return 0.0 if s<=0 else float((2*np.arange(1,n+1)-n-1).dot(x)/(n*s))
def topshare(x,frac):
    x=np.sort(np.clip(np.asarray(x,float),0,None));k=max(1,int(round(len(x)*frac)))
    return float(x[-k:].sum()/x.sum()) if x.sum()>0 else 0.0

def series(make):
    fs=[]; gs=[]; g_end=[]; t10=[]; t1=[]
    for sd in SEEDS:
        m=ModelV3(replace(make(),periods=T,seed=int(sd))); f1=[]; g1=[]
        for _ in range(T):
            m.step()
            et=m.h_eq.sum()+m.eq_state+m.eq_row
            f1.append(100*m.eq_row/et if et>0 else 0.0); g1.append(gini(m.house_nw()))
        fs.append(f1); gs.append(g1)
        nw=m.house_nw(); g_end.append(gini(nw)); t10.append(topshare(nw,0.10)); t1.append(topshare(nw,0.01))
    return (np.array(fs).mean(0), np.array(gs).mean(0),
            np.mean(g_end), np.mean(t10), np.mean(t1))

f_for,g_for,ge_for,t10_for,t1_for = series(S.two_channel_base)
f_dom,g_dom,ge_dom,t10_dom,t1_dom = series(S.domestic_owner)
_,    g_wt, ge_wt, t10_wt, t1_wt  = series(S.domestic_owner_wealthtax)

t=np.arange(T)
SIE,TEA,BLU="#7c2d12","#1d4e44","#3b6ea5"
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(15.2,4.3))

a1.plot(t,f_for,color=SIE,lw=2,label="foreign owner (UK)")
a1.plot(t,f_dom,color=TEA,lw=2,label="domestic owner (US)")
a1.axvspan(110,210,color="0.85",alpha=.5,lw=0); a1.text(160,4,"AI ramp",ha="center",color="0.4",fontsize=9)
a1.set_title("Who ends up owning the capital"); a1.set_xlabel("period")
a1.set_ylabel("foreign ownership of capital (%)"); a1.legend(frameon=False,fontsize=9); a1.set_ylim(-3,None)

a2.plot(t,g_for,color=SIE,lw=2,label="foreign owner (rent leaves)")
a2.plot(t,g_dom,color=TEA,lw=2,label="domestic owner (rent stays)")
a2.plot(t,g_wt, color=BLU,lw=2,ls="--",label="domestic + wealth tax")
a2.axvspan(110,210,color="0.85",alpha=.5,lw=0)
a2.set_title("Inequality at home over time"); a2.set_xlabel("period")
a2.set_ylabel("Gini of household net worth"); a2.legend(frameon=False,fontsize=9)

labels=["Gini","top 10%\nshare","top 1%\nshare"]
forv=[ge_for,t10_for,t1_for]; domv=[ge_dom,t10_dom,t1_dom]; wtv=[ge_wt,t10_wt,t1_wt]
x=np.arange(3); w=0.26
a3.bar(x-w,forv,w,color=SIE,label="foreign owner")
a3.bar(x,  domv,w,color=TEA,label="domestic owner")
a3.bar(x+w,wtv, w,color=BLU,label="domestic + wealth tax")
a3.set_xticks(x); a3.set_xticklabels(labels); a3.set_ylim(0,0.65)
a3.set_title("Inequality at home, mature steady state"); a3.set_ylabel("fraction")
a3.legend(frameon=False,fontsize=9)
for xi,vals in zip(x,[ (ge_for,ge_dom,ge_wt),(t10_for,t10_dom,t10_wt),(t1_for,t1_dom,t1_wt) ]):
    for dx,v in zip((-w,0,w),vals): a3.text(xi+dx,v+0.012,f"{v:.2f}",ha="center",fontsize=7.5,color="0.25")

fig.tight_layout(); fig.savefig("figures_v3/p4_owner_domicile.png",dpi=110)
print("saved 3-panel figure")
print(f"foreign : Gini {ge_for:.3f} top10 {t10_for:.3f} top1 {t1_for:.3f}")
print(f"domestic: Gini {ge_dom:.3f} top10 {t10_dom:.3f} top1 {t1_dom:.3f}")
print(f"dom+WT  : Gini {ge_wt:.3f} top10 {t10_wt:.3f} top1 {t1_wt:.3f}")
