# abm_sfc v2: endogenous-production automating economy

v2 builds on the v1 AB-SFC backbone (same two accounting invariants, still
asserted in tests) and adds the mechanisms from the implementation plan, each as
a switch so v1 behaviour is recoverable as a special case.

## What is new versus v1

| Plan phase | Mechanism | Module / switch |
|---|---|---|
| P1 | Endogenous task-based CES production; K accumulates; r and g endogenous | `production.py`, always on in v2 |
| P2 | Concentration speed couples to the capital share | `kappa` |
| P3 | Endogenous portfolio choice (households buy equity out of saving, book value) | `phi_equity` |
| P4 | Progressive wealth tax; citizens' wealth fund | `wealth_brackets`, `citizens_fund` |
| P5 | Heterogeneous labour and automation-driven displacement | `skill_dispersion`, `displacement` |
| P6 | (planned) Jacobian/eigenvalue stability tooling | not yet implemented |

## Invariants (unchanged guarantee)

Every period, to machine precision and across every switch combination
(asserted in `tests/test_v2.py`):

1. the four sectors' deposit balances sum to zero;
2. the sum of sector net worths equals real assets (capital + firm inventories).

With behavioural investment the firm holds inventories of unsold output, so the
real-asset total is `K + inventories` rather than `K` alone.

## Production block

`Y = A [ (1-I)^{1/e}(A_L L)^{(e-1)/e} + I^{1/e}(A_K K)^{(e-1)/e} ]^{e/(e-1)}`,
with automation index `I` (capital-task share) and elasticity of substitution
`e`. Factor prices are marginal products; `wage_bill + capital_income = Y`
exactly (Euler). The capital share rises endogenously with `I`. Investment is
behavioural: firms invest to close the gap between the net return `r` and a
required return, so capital does not run away. The no-automation baseline is a
verified stationary equilibrium (`r = r_required`, `g = 0`).

## How to run

    pip install numpy scipy matplotlib pytest
    python -m pytest tests/ -q          # 40 tests (19 v1 + 21 v2)
    python run_v2_experiments.py        # full battery -> figures_v2/

## Experiments and findings (12 seeds each, see `figures_v2/results.txt`)

1. Scenario comparison (H1/H2/H3). Flow taxes (income/corporate) leave the Gini
   at ~0.57, indistinguishable from laissez-faire; the wealth tax collapses it
   to ~0.05 and the progressive wealth tax to ~0.01. The stock instrument is
   what de-concentrates, confirming the v1 result now under endogenous output.
2. Concentration speed vs automation (H2). With the capital-share coupling on,
   deeper automation produces both faster and higher condensation (terminal Gini
   0.60 -> 0.77 as the automation ceiling rises). The speed of inequality is a
   function of automation depth.
3. r vs g (H1). The transition shows a capital-deepening boom: `r` rises to
   ~0.17 and `g` to ~0.065, then both fall back to the required return as the new
   capital-intensive steady state is reached. A higher steady-state `r` leaves
   the wealth-tax regime more solvent (it accrues a larger state asset base).
4. Elasticity sweep (H1/H2). `e` governs the labour-share path. The terminal
   Gini is robust to `e` (the kinetic channel sets it), but the wage path is
   highly sensitive.
5. Foreign-ownership leakage (H3). Rest-of-world net worth rises linearly with
   the foreign share (leakage), and government net worth falls correspondingly:
   foreign ownership erodes the domestic base while corporate tax still reaches
   the profit.
6. Equity-purchase dynamics (H4, revised). Important reversal of the v1
   intuition: once production and capital are endogenous, the automation boom
   raises wages in absolute terms, so households are NOT priced out of equity.
   Purchases collapse only because net investment demand ends, not because of
   affordability. The "buyer collapse from poverty" channel does not bind in v2.
7. Policy frontier (H5). Plotting Gini against government solvency, the
   progressive wealth tax and the flat wealth tax dominate (low inequality, high
   solvency); laissez-faire is alone in the insolvent quadrant. The citizens'
   fund lowers inequality relative to state-keep but spends down its surplus by
   rebating dividends, so it trades solvency for equality.
8. Intervention timing (H6). Terminal Gini rises monotonically with the delay
   before the wealth tax is switched on (0.006 at onset t=60 up to 0.281 at
   t=210): early intervention is markedly cheaper, a hysteresis result.
9. Finite-size robustness. Gini is converged from N=250 (0.55) to N=4000 (0.55);
   the top-1% sampling noise falls from 7.1 to 1.9 points. N=2000 is adequate for
   the aggregate metrics with seed replication handling the tail noise.

## Honest caveats

* Magnitudes during the transition are dramatic (output per capita rises ~250x)
  because gross capital deepening over a long horizon compounds. All headline
  results are reported in shares, ratios and Gini, which are scale-free.
* P6 (formal eigenvalue stability) is specified in the plan but not yet built;
  stability is currently read off trajectories and the verified stationary
  baseline.
* Equity is valued at book (no market price / capital gains); P3b in the plan.
* The kinetic concentration channel is exogenous to the economy except through
  the `kappa` coupling; a fully endogenous return-dispersion process is future
  work.
