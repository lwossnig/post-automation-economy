"""Conservative kinetic-exchange (Bouchaud-Mezard) wealth-reshuffling kernel.

This module implements the *distributional engine* of the model. It is kept
deliberately separate from the stock-flow accounting (see ``model.py``):

  * ``model.py`` evolves each sector's *aggregate* net worth via watertight
    double-entry accounting.
  * this kernel reshuffles a *fixed total* of household equity across
    households, reproducing the wealth-condensation mechanism of
    Bouchaud & Mezard (2000, Physica A 282, 536-545) without creating or
    destroying any wealth.

Continuum limit and stationary distribution
-------------------------------------------
For normalised wealth ``x`` (mean 1) the per-agent SDE used here is, in Ito
form with mean-field pull towards the population mean,

    dx = J (1 - x) dt + sigma * x dW,          Var(dW) = dt.

The stationary Fokker-Planck solution (zero probability current) is the
inverse-gamma density

    P(x) proportional to x^{-(1 + a)} * exp(-b / x),
    a = 1 + 2 J / sigma^2,   b = 2 J / sigma^2,

so the Pareto tail exponent is ``alpha = a = 1 + 2 J / sigma^2`` and, because
1/x ~ Gamma(shape=a, rate=b),

    E[1/x] = a / b = (1 + 2 J / sigma^2) / (2 J / sigma^2).

These two closed-form targets are what ``tests/test_kinetic.py`` checks the
simulator against. Derivation is reproduced in the project notes.
"""
from __future__ import annotations

import numpy as np


def bm_stationary_targets(J: float, sigma: float) -> dict[str, float]:
    """Closed-form stationary targets for the Ito BM kernel above."""
    ratio = 2.0 * J / (sigma ** 2)
    alpha = 1.0 + ratio          # Pareto tail exponent
    e_inv = alpha / ratio        # E[1/x] for the stationary inverse-gamma
    return {"alpha": alpha, "E_inv_x": e_inv, "b": ratio}


def kinetic_step(
    wealth: np.ndarray,
    J: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
    floor: float = 1e-9,
) -> np.ndarray:
    """One Euler-Maruyama step of the conservative BM kernel.

    The total of ``wealth`` is preserved exactly (up to floating point): this
    represents asset *trading / revaluation* among households, which moves
    wealth around but does not change the aggregate stock of claims.
    """
    total = wealth.sum()
    if total <= 0:
        return wealth
    mean = total / wealth.size
    # work in normalised units x = wealth / mean  (mean of x is 1)
    x = wealth / mean
    drift = J * (1.0 - x) * dt
    diffusion = sigma * x * np.sqrt(dt) * rng.standard_normal(x.size)
    x = x + drift + diffusion
    np.maximum(x, floor, out=x)        # wealth stays positive
    x *= 1.0 / x.mean()                # re-impose mean 1 -> conserves the total
    new = x * mean
    # guard against tiny renormalisation drift so the total is exactly preserved
    new *= total / new.sum()
    return new
