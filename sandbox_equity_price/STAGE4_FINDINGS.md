# Stage 4 (market-clearing equity price): attempt and finding

## What was asked
Attempt the market-clearing price in a fresh copy and validate it against the
current results via a parameter that reproduces the old behaviour.

## What was built
A fresh copy (`equity_sandbox_s4`) carrying Stages 1-3, plus a Stage 4 layer:
a Tobin's-q price on the equity stock (the equity yield capitalised at the
required return, with an optional demand-pressure deviation, damped), applied as
a valuation with capital gains feeding the consumption wealth effect. Gated by
`market_clearing_on` (default False).

## Reproduction (the requested validation): PASSES
- `market_clearing_on = False` is bit-identical to the frozen model: output and
  ownership match exactly and q is exactly 1 (test_stage4_reproduction_bit_identical).
- With the price live, the generalised market-value invariant holds to ~1e-13
  (NW_sum = q*K + V_IP), and the book and deposit invariants are intact
  (test_stage4_market_value_invariant_holds_with_price_on).
- The full v3 suite still passes (146 in the sandbox); the frozen model is untouched.

## Finding: the market-clearing price is degenerate in this model
With the price on, q pins to its floor in every configuration, because the
domestic equity yield (after-tax capital income over K) is slightly NEGATIVE in
the mature economy, even with no rent: the competitive return on capital is
competed away by automation, and the only surplus, the IP rent, is siphoned to the
foreign owner. So:
  - A market price on domestic equity (claims on K) is economically degenerate:
    the capital earns no surplus, so its Tobin's q sits below replacement cost and
    carries no bidding-up dynamic. Worse, the rent LOWERS domestic equity income,
    so q falls as the rent rises, the opposite of the intended story.
  - The asset that actually holds the value is the foreign owner's rent-bearing IP,
    which Stages 1-3 already capitalise (V_IP). But it is singly held, so there is
    no multi-buyer market to clear; "market clearing" on it just returns the
    fundamental, i.e. Stages 1-3.

This is the third time the same structural fact has surfaced: in this economy the
durable surplus is the foreign-held IP rent, not a return on the domestic capital
stock. It is exactly why the paper's instrument is a source levy on the rent.

## Recommendation
Do NOT merge Stage 4. A genuine asset-price / bidding-up dynamic would require
restructuring the model so the rent-bearing firm is a traded asset with competing
buyers, which is a different modelling choice and beyond what the paper needs. The
reduced-form IP capitalisation (Stages 1-3) is the appropriate and sufficient
valuation layer. The reproduction machinery and tests are kept in this sandbox for
the record; the frozen model and shipped package are unchanged.
