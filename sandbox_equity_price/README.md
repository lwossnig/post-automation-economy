# Equity-price sandbox (Phase 3, item 3) - NOT merged into the shipped model

This extends the shipped model with a market-value layer for the rent-bearing IP.
It is kept here, off the shipped model, because it is a reduced-form valuation whose
discount rate and premium are not yet calibrated, and its one behavioural channel
(Stage 3) adds an unanchored parameter. The shipped model is unchanged (136 tests).

- model under abm_sfc/model_v3.py carries Stages 1-4 (all off by default; bit-identical
  to the shipped model unless equity_price_on / market_clearing_on are set).
- tests/test_equity.py: 12 tests (Stages 1-3 valuation + feedback, Stage 4 reproduction
  + invariants).  Run: PYTHONPATH=. python -m pytest tests/ -q
- PLAN_equity_price.md: the staged plan and test plan.
- STAGE4_FINDINGS.md: the market-clearing negative result.
- exp_ip_value.py: regenerates Figure 24 (run from this dir).
Section 6P of the report writes up the result; Figure 24 is generated here.
