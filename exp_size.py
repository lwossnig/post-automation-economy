import numpy as np
from abm_sfc import Model, scenarios, gini
from dataclasses import replace

B = scenarios.laissez_faire()

def metrics_for(n, seeds=12, periods=300):
    g=[]; t1=[]
    for s in range(seeds):
        p = replace(B, n_agents=n, periods=periods, seed=s)
        h = Model(p).run()
        # average over last 50 periods (stationary window)
        g.append(np.mean(h.gini[-50:]))
        t1.append(np.mean(h.top1_share[-50:]))
    return np.mean(g), np.std(g), np.mean(t1), np.std(t1)

print(f"{'N':>6} {'Gini mean':>10} {'Gini sd':>8} {'top1% mean':>11} {'top1% sd':>9}")
for n in [250,500,1000,2000,4000,8000]:
    gm,gs,tm,ts = metrics_for(n)
    print(f"{n:6d} {gm:10.3f} {gs:8.4f} {tm*100:11.2f} {ts*100:9.3f}")
