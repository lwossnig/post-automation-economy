# Phase 3, item 3: market-clearing equity price (sandbox plan)

## Why a sandbox
The whole model rests on one accounting identity: the sum of sector net worths
equals the book capital stock K, asserted to machine precision in 136 tests.
A market price for equity decouples the *market value* of equity from its *book*
backing, which restates that identity. We develop and validate here; we merge to
the frozen model only once a generalised invariant suite passes.

Frozen (do not touch until merge): `/home/claude/abm_sfc/`, shipped pkg.
Sandbox (all work here): `/home/claude/equity_sandbox/`.

## The change in one line
Today equity is held at book, so total equity = K and NW_sum = K. We introduce a
price per unit of equity, q (a Tobin's-q-like object), so the *market* value of
equity is q*K, which can differ from K when the AI rent capitalises into the price.

## The accounting problem and the fix
Naively, if sectors value equity at market, NW_sum = q*K + deposits = q*K (deposits
net to zero), so "NW_sum = K" breaks for q != 1.

The standard stock-flow-consistent fix (Godley-Lavoie; the SNA sequence of accounts)
is to separate two ways net worth changes:
  1. **Transactions** (saving, equity bought/sold for deposits) - these still sum to
     zero across sectors, every period. THIS INVARIANT IS UNCHANGED.
  2. **Revaluations** (holding gains from a change in q) - these change net worth
     without any transaction, and are recorded in a revaluation account.

So the identity generalises cleanly:
  - transactions invariant (UNCHANGED): sum of sector deposit flows = 0.
  - stock invariant (GENERALISED): NW_sum = q*K  (book K plus cumulative revaluations).
  - articulation (NEW, must hold each period): dNW_sector = transactions_sector
    + revaluation_sector, where revaluation_sector = dq * (sector's equity units).

The deposit invariant is the load-bearing one for the closure and it does not move.
The equity invariant is the one we generalise, and we add the revaluation account so
the flow-of-funds articulates.

## How q is determined
Two stages, simplest first.
  - **q as no-arbitrage diagnostic (Stage 1).** q clears the required return: the
    equity yield (dividends + expected capital gain over the price) equals the
    deposit rate plus a fixed risk premium. In a steady state this is q ~ D / ((r+rp) - g)
    scaled so q = 1 when the equity return just matches the outside return. Compute it,
    do NOT feed it back into the accounting yet. Sanity: q ~ 1 with no rent; q > 1
    once the AI rent is present (the rent capitalises into the price).
  - **q from market clearing (Stage 4).** Sectors hold a desired share of wealth in
    equity vs deposits as a function of the relative return (Brainard-Tobin portfolio
    equations). q adjusts so desired nominal equity holdings sum to the supply q*K.
    Heaviest piece; do last.

## Behaviour once q is live (Stage 3)
  - The foreign owner acquires equity at the market price q, so a higher q means a
    given amount of reinvested rent buys fewer units: q damps ownership drift (the
    opposite-signed force to the rent capitalising, which is the interesting tension).
  - The wealth tax falls on *market* value q * (units held), not book, so a price
    boom widens the wealth-tax base.
  - Capital gains (dq) accrue to holders as revaluation income; optionally a
    realisation-based capital-gains tax can sit on them later (Phase 4).

## Staged build, each stage test-gated
  - **Stage 0 (DONE).** Copy package to sandbox; confirm the 136-test suite passes
    unchanged (the copy reproduces the frozen model).
  - **Stage 1 (DONE).** Add the IP value as a *diagnostic only* (no feedback): a price series computed
    from dividends/returns, recorded in history. Param `equity_price_on` default
    False -> bit-identical. Validate q is sensible (q ~ 1 no rent, q > 1 with rent).
  - **Stage 2 (DONE).** Add the revaluation account and restate the stock invariant to
    market value (NW_sum = q*K) while keeping transactions summing to zero. Validate
    BOTH invariants to machine precision and the articulation identity each period.
    This is the core accounting change and the main risk; stop and harden here.
  - **Stage 3 (DONE).** Let the valuation affect behaviour: owner buys at q; wealth tax on market value.
    Validate invariants survive the feedback; check the ownership/valuation tension.
  - **Stage 4.** Full market clearing: q set by portfolio demand = supply. Validate.

## Test plan (sandbox `tests/test_equity.py`)
  - **Reproduction.** equity_price_on = False is bit-identical to v3; run the full
    136-test suite against the sandbox unchanged (already green at Stage 0).
  - **Transactions invariant (all stages).** |sum of sector deposit flows| < 1e-9.
  - **Generalised stock invariant (Stage 2+).** |NW_sum - q*K| / (q*K) < 1e-9.
  - **Articulation (Stage 2+).** per sector, |dNW - (transactions + dq*units)| < 1e-9.
  - **q sanity (Stage 1+).** q ~ 1 (within a band) when mu_frac = 0; q strictly
    higher when mu_frac > 0; q rises with the rent.
  - **Behaviour (Stage 3+).** a wealth tax raises more when q is high; a higher q
    slows ownership drift for a given reinvested rent.

## Merge criteria (back to the frozen model)
Merge only when, in the sandbox: (a) the full v3 suite passes with the price off
(bit-identical), (b) both invariants hold to machine precision with the price on
across every scenario, (c) the articulation identity holds each period, and
(d) the new behavioural tests pass. Until then the frozen model and the shipped
package stay exactly as they are.

## Status (sandbox)
Stages 0, 1, 2, 3 complete and validated: sandbox reproduces the frozen model (v3 suite
bit-identical); the IP value capitalises the after-tax rent and behaves sensibly; the
market-value stock invariant (NW_sum = K + V_IP) holds to machine precision each period
alongside the unchanged book and transactions invariants and the articulation identity.
Sandbox suite: 10 equity tests + the full v3 suite (146 total). Stage 3 finding: the
subscription-price channel does NOT bite (the owner's cash re-enters via the secondary
market), so ownership here is set by the reinvest-vs-repatriate split; the valuation
therefore acts through repatriation (a high market value makes the owner take more rent
out), which re-domesticates ownership at a modest output cost, with both invariants
intact. Stage 4 (full market clearing: a price that responds to portfolio demand =
supply, giving the bidding-up dynamic) remains and is the heaviest. Merge to the frozen
model is still a separate gated decision.
