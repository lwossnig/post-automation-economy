"""Nested task-based CES production block (v4): AI vs robotic automation.

Two automation channels are modelled as separate task clusters, each pairing a
skill of labour with its own form of capital:

    routine cluster    Q_r = CES( routine labour L_r , robotic capital  K_r )   [e_routine]
    cognitive cluster  Q_c = CES( cognitive labour L_c, AI compute K_ai )       [e_cog]
    top                Y   = A * CES( Q_r , Q_c )                               [e_top]

a_r and a_c are the capital-task shares inside each cluster; they ramp up with
the robotic and AI automation indices respectively (set in the model). Setting
the top elasticity e_top above the within-cluster elasticities makes robotic and
AI capital more substitutable with each other (production shifts between routine-
and cognitive-intensive output) than each is with its own labour, which is the
nesting choice for this model. Robots therefore displace routine labour and AI
displaces cognitive labour, each within its own cluster.

Factor prices are marginal products. For a homothetic nested CES the income
decomposition is exact: each cluster earns a top-level share of Y, and each
factor earns its within-cluster share of that, so the four competitive factor
incomes (w_Lr, w_Lc, ci_Kr, ci_Kai) sum to Y to the cent (Euler's theorem). The
gross return on each capital is ci_Kx / K_x, which the model uses to allocate
investment between the two stocks.

The AI rent (the IP owner's monopoly markup) is NOT applied here; it is a wedge
the model peels from the cognitive cluster's value added (mu_frac * s_c * Y)
before the within-cluster split, so this module stays purely technological and
the rent stays a financial claim. Keeping the two separate is what lets the rent
be taxed differently (DST / withholding) from the competitive capital returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ces_cluster(L: float, K: float, a: float, e: float, A_L: float, A_K: float):
    """One two-input CES cluster.

    Returns (Q, share_L, share_K) where Q is cluster output and share_L/share_K
    are the within-cluster income shares (they sum to 1 and split the cluster's
    value added between its labour and its capital).
    """
    a = min(max(a, 1e-9), 1.0 - 1e-9)
    bL = (1.0 - a) ** (1.0 / e) * (A_L * max(L, 1e-9)) ** ((e - 1.0) / e)
    bK = a ** (1.0 / e) * (A_K * max(K, 1e-9)) ** ((e - 1.0) / e)
    Z = bL + bK
    Q = Z ** (e / (e - 1.0))
    return Q, bL / Z, bK / Z


@dataclass
class NestedProduction:
    A: float = 0.45            # top-level total factor productivity (output scale)
    e_top: float = 1.20        # elasticity between the routine and cognitive clusters
    e_routine: float = 0.60    # elasticity inside the routine cluster (gross complements)
    e_cog: float = 0.60        # elasticity inside the cognitive cluster
    theta: float = 0.50        # top-level weight on the cognitive cluster
    A_Lr: float = 1.0          # routine-labour augmentation
    A_Kr: float = 1.0          # robotic-capital augmentation
    A_Lc: float = 1.0          # cognitive-labour augmentation
    A_Kai: float = 1.0         # AI-compute augmentation

    def decompose(self, K_r, K_ai, L_r, L_c, a_r, a_c):
        """Full technological decomposition at the realised inputs.

        Returns a dict with output Y, the four competitive factor incomes
        (which sum to Y), the top-level cluster income shares s_r/s_c, and the
        gross capital returns r_Kr/r_Kai used for the investment allocation.
        """
        Q_r, shL_r, shK_r = _ces_cluster(L_r, K_r, a_r, self.e_routine, self.A_Lr, self.A_Kr)
        Q_c, shL_c, shK_c = _ces_cluster(L_c, K_ai, a_c, self.e_cog, self.A_Lc, self.A_Kai)
        et = self.e_top
        BR = (1.0 - self.theta) ** (1.0 / et) * Q_r ** ((et - 1.0) / et)
        BC = self.theta ** (1.0 / et) * Q_c ** ((et - 1.0) / et)
        ZT = BR + BC
        Y = self.A * ZT ** (et / (et - 1.0))
        s_r, s_c = BR / ZT, BC / ZT
        # competitive factor incomes (sum to Y exactly)
        w_Lr = Y * s_r * shL_r
        ci_Kr = Y * s_r * shK_r
        w_Lc = Y * s_c * shL_c
        ci_Kai = Y * s_c * shK_c
        return {
            "Y": Y, "s_r": s_r, "s_c": s_c,
            "w_Lr": w_Lr, "ci_Kr": ci_Kr, "w_Lc": w_Lc, "ci_Kai": ci_Kai,
            "r_Kr": ci_Kr / K_r if K_r > 0 else 0.0,
            "r_Kai": ci_Kai / K_ai if K_ai > 0 else 0.0,
        }

    def output(self, K_r, K_ai, L_r, L_c, a_r, a_c) -> float:
        return self.decompose(K_r, K_ai, L_r, L_c, a_r, a_c)["Y"]
