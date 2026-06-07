"""Task-based CES production block (v2, phase P1).

Output is a CES aggregate of a labour-task bundle and a capital-task bundle,

    Y = A [ (1-I)^{1/e} (A_L L)^{(e-1)/e} + I^{1/e} (A_K K)^{(e-1)/e} ]^{e/(e-1)}

where I in [0,1] is the automation index (the share of tasks performed by
capital) and e is the elasticity of substitution between the two bundles. This
is the reduced macro form of the Acemoglu-Restrepo task model: raising I shifts
tasks from labour to capital, so the capital share rises *endogenously* rather
than being imposed.

Factor prices are marginal products. Writing

    bL = (1-I)^{1/e} (A_L L)^{(e-1)/e},  bK = I^{1/e} (A_K K)^{(e-1)/e},  Z = bL+bK,

one obtains the clean closed forms

    wage_bill   = w L = Y * bL / Z
    capital_inc = r K = Y * bK / Z
    capital_share pi = bK / Z,   labour_share = bL / Z,

and wage_bill + capital_inc = Y exactly (Euler's theorem; constant returns).
As I -> 1 the labour share -> 0; as I -> 0 it -> 1. The marginal product of
capital is r_gross = capital_inc / K, and the net return is r = r_gross - delta.

e is a central parameter: e > 1 (gross substitutes) lets capital and labour
tasks substitute, so automation can depress the wage level; e < 1 (complements)
means automation raises the wage on the remaining labour tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Production:
    A: float = 1.0          # total factor productivity
    A_L: float = 1.0        # labour-augmenting productivity
    A_K: float = 1.0        # capital-augmenting productivity
    eps: float = 1.5        # elasticity of substitution between task bundles

    def _bundles(self, K: float, L: float, I: float):
        e = self.eps
        I = min(max(I, 1e-9), 1.0 - 1e-9)     # keep interior so both bundles exist
        bL = (1.0 - I) ** (1.0 / e) * (self.A_L * L) ** ((e - 1.0) / e)
        bK = I ** (1.0 / e) * (self.A_K * K) ** ((e - 1.0) / e)
        return bL, bK

    def output(self, K: float, L: float, I: float) -> float:
        e = self.eps
        bL, bK = self._bundles(K, L, I)
        Z = bL + bK
        return self.A * Z ** (e / (e - 1.0))

    def factor_prices(self, K: float, L: float, I: float):
        """Return (wage_bill, capital_income, capital_share, r_gross)."""
        bL, bK = self._bundles(K, L, I)
        Z = bL + bK
        Y = self.A * Z ** (self.eps / (self.eps - 1.0))
        wage_bill = Y * bL / Z
        capital_income = Y * bK / Z
        pi = bK / Z
        r_gross = capital_income / K if K > 0 else 0.0
        return wage_bill, capital_income, pi, r_gross
