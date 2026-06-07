# Simulating a Post-Automation Economy

An agent-based, stock-flow-consistent (AB-SFC) macro model of an economy in which
AI and robotics progressively shift income from labour to capital, the ownership of
that capital sits with households, the state, or abroad, and the state may fund a
universal basic income (UBI) by taxing capital. The model is used to ask which
taxation and ownership regimes yield a stable, solvent equilibrium and how wealth
concentrates along the transition.

The current, maintained model is **v3** (`abm_sfc/model_v3.py`). The v1/v2 modules
are kept for provenance and for the carried-over figures cited in the report.

## Paper

This code accompanies the paper *Simulating a Post-Automation Economy* (Leonard Wossnig, 2026).
Preprint: **arXiv:XXXX.XXXXX** *(link to follow)*. The full PDF and LaTeX source are in [`paper_latex/`](paper_latex/).

If you use this model or its results, please cite the paper and this repository (a `CITATION.cff` is included, so GitHub shows a "Cite this repository" button).

## Licence

The code is released under the **MIT Licence** (see [`LICENSE`](LICENSE)). The paper text and figures are released under **CC BY 4.0**, consistent with the arXiv posting.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# run the test suite (144 tests; SFC invariants checked to machine precision)
# a pytest.ini scopes collection to tests/ (the sandbox_equity_price/ tree has same-named
# modules and is run separately by cd-ing into it)
PYTHONPATH=. pytest -q

# reproduce the headline experiments + regenerate all figures and results.txt
PYTHONPATH=. python run_v3_experiments.py        # -> figures_v3/ , results.txt , results.csv
PYTHONPATH=. python regen_v3_figs.py             # -> expA_frontier.png, expB_composition.png

# rebuild the self-contained HTML report (embeds the figures as base64)
PYTHONPATH=. python build_html_v3.py             # -> Automation_Economy_Experiments_v3.html

# regenerate the remaining figures used in the paper and report
PYTHONPATH=. python exp_two_channel.py           # -> expH_income_decomposition.png, expH_foreign_ownership.png
PYTHONPATH=. python exp_debate.py                # -> debate_laffer.png, debate_territory.png
PYTHONPATH=. python exp_owner.py                 # -> p4_owner_domicile.png (Figure 25)
PYTHONPATH=. python exp_rising_rent.py           # -> p5_rising_rent.png (Figure 26)
```

`PYTHONPATH=.` is required because the scripts import the `abm_sfc` package by name;
running a script from inside a subfolder will not put the repo root on the path.

---

## The model in one paragraph

Four sectors carry balance sheets (households as N agents, the firm/capital sector,
the government, the rest of the world). Deposits are a liquid claim that sums to
zero across sectors; equity represents claims on the capital stock K. Output is a
task-based CES aggregate of labour and capital with gross complements (elasticity
< 1), so the capital share rises endogenously as the automation index I climbs.
Output is supply-determined at potential, **investment is the residual that clears
the goods market (S = I)**, and consumption follows a **differential-saving rule**
(Kaldor 1957 / Pasinetti 1962): a high propensity out of labour income, a low one
out of capital income, plus a small wealth effect on equity. That rule pins the
capital-output ratio at a stable interior value. Wealth concentrates through
**heterogeneous, persistent returns to equity** (an AR(1) return type per agent),
not an imposed kinetic kernel. The wealth tax erodes its own base (avoidance) and
triggers capital flight (an endogenous offshore account), both at empirical
elasticities.

Two identities hold every period by construction and are asserted in the tests:
the four sectors' deposits sum to zero, and the sum of sector net worths equals K
(equity claims sum exactly to K; the firm holds no residual cash).

---

## Repository layout

```
abm_sfc/
  model_v3.py        current model: step() is fully inline-commented section by section
  scenarios_v3.py    the 8 headline regimes (REGISTRY) + BASE calibration
  production.py       task-based CES, factor prices (Euler-exact)
  sensitivity.py      Sobol global sensitivity (SALib)
  stability.py        deterministic-skeleton local stability (1-D capital-map slope)
  kinetic.py          Gini and helpers
  model.py, model_v2.py, scenarios.py, scenarios_v2.py, mc.py   legacy v1/v2

run_v3_experiments.py  experiments A-E -> figures_v3/*.png + results.txt
regen_v3_figs.py       frontier + composition figures
exp_two_channel.py     experiment H: AI vs robotic automation, separately taxed
build_html_v3.py       inlines figures_v3/*.png into the HTML report
report_v3_template.html report source (placeholders __IMG_*__)
abm_sfc/production_v4.py nested three-cluster CES (routine+robots, cognitive+AI, top)

tests/                 144 tests: invariants, per-agent no-overdraft, marginal brackets, stateless stability, source-tax, two-channel
figures_v3/            regenerated figures + results.txt + results.csv (machine-readable)
Automation_Economy_Experiments_v3.html   the built report (open in a browser)
```

## The eight scenarios (abm_sfc/scenarios_v3.REGISTRY)

| key | what it is |
|---|---|
| `laissez_faire` | no tax, no UBI, no government (pure-market benchmark) |
| `income_tax_ubi` | 25% corporate / 30% income tax funding a UBI |
| `wealth_tax_ubi` | 25% / 15% + 2% wealth tax funding a UBI |
| `wealth_tax_frictionless` | as above with avoidance and mobility switched off |
| `progressive_wealth` | bracketed wealth tax (1% / 3%) |
| `state_ownership` | state holds 60% of equity, 15% income tax + UBI |
| `citizens_fund` | state ownership rebating its dividends per capita |
| `foreign_ownership` | 60% foreign-held equity |

## Headline findings (periods=600, 10-seed means)

- Stock taxes compress the wealth Gini sharply (~0.63 -> ~0.15); flow taxes do not.
- Solvency turns on tax adequacy, not ownership form: adequately-taxed regimes run
  surpluses and accumulate a sovereign equity fund; under-taxed welfare states
  (15% income tax + UBI) hit an explosive r>g debt path and go insolvent.
- The real economy is locally stable in every regime (capital-map slope < 1); the
  only explosive root is the financial r>g debt root (= 1 + r_debt).
- Automation raises absolute wages massively even as the labour share dips then
  recovers (capital deepening under gross complements): concentration is via
  returns, not affordability.

## Notes / honest boundaries

Equity is valued at book (no market-clearing price); automation is an exogenous
logistic ramp; concentration is household-return-based rather than firm-level
(superstar firms); there is no explicit welfare criterion and no UBI labour-supply
response. These are the natural next steps, not claims the model makes.
