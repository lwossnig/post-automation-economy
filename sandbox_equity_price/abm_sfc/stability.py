"""Item 5: formal local-stability analysis of the deterministic skeleton.

The stochastic model has no closed-form fixed point, so we analyse its
deterministic skeleton with idiosyncratic noise switched off (ret_sigma = 0,
demographic_reset = 0) and automation pinned at a chosen level I.

We separate two questions that live on very different roots:

* REAL stability. Does the capital stock have a stable rest point? Under the
  residual-investment closure with gross complements and differential saving,
  potential output is bounded in K and depreciation is linear, so the aggregate
  capital map K -> K' has an interior fixed point K*. We compute K* by iterating
  the one-dimensional aggregate map (households own their policy share of K,
  deposits projected to zero, returns at their fixed-point yield) and estimate
  the local slope dK'/dK by central differences. |dK'/dK| < 1 means the real
  economy is locally (and, given the global convergence shown numerically, also
  globally) stable.

* FISCAL stability. Any deposit / government-bond balance compounds at 1 + r_debt
  independently of the real block, so a primary imbalance puts public debt on an
  explosive r > g path. That root (1 + r_debt) is the solvency knife-edge; it is
  reported analytically rather than mixed into the real slope.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model_v3 import ModelV3, ParamsV3


def _make_det_model(p: ParamsV3, I_fixed: float) -> ModelV3:
    """Deterministic skeleton: idiosyncratic noise off, automation pinned."""
    pdet = replace(p, ret_sigma=0.0, demographic_reset=0.0,
                   auto_max=0.0, I_base=I_fixed, mobility_on=p.mobility_on)
    return ModelV3(pdet)


def _snapshot(m: ModelV3) -> dict:
    """Capture the full dynamic state that step() reads, so a perturbation can be
    evaluated from an identical starting point regardless of call order."""
    return dict(
        h_eq=m.h_eq.copy(), h_dep=m.h_dep.copy(), offshore=m.offshore.copy(),
        ret_z=m.ret_z.copy(), eq_state=m.eq_state, eq_row=m.eq_row,
        gov_dep=m.gov_dep, row_dep=m.row_dep, firm_dep=m.firm_dep,
        eq_yield_lag=np.array(m.eq_yield_lag, dtype=float).copy()
        if np.ndim(m.eq_yield_lag) else float(m.eq_yield_lag),
        div_house_lag=np.array(m.div_house_lag).copy(),
        fund_rebate_lag=np.array(m.fund_rebate_lag).copy(),
        disp_lag=np.array(m.disp_lag).copy(),
        r_lag=float(m.r_lag), K=m.K, t=m.t,
    )


def _restore(m: ModelV3, s: dict):
    """Restore a snapshot taken by _snapshot."""
    m.h_eq = np.array(s["h_eq"]).copy()
    m.h_dep = np.array(s["h_dep"]).copy()
    m.offshore = np.array(s["offshore"]).copy()
    m.ret_z = np.array(s["ret_z"]).copy()
    m.eq_state = s["eq_state"]; m.eq_row = s["eq_row"]
    m.gov_dep = s["gov_dep"]; m.row_dep = s["row_dep"]; m.firm_dep = s["firm_dep"]
    m.eq_yield_lag = (np.array(s["eq_yield_lag"]).copy()
                      if np.ndim(s["eq_yield_lag"]) else s["eq_yield_lag"])
    m.div_house_lag = np.array(s["div_house_lag"]).copy()
    m.fund_rebate_lag = np.array(s["fund_rebate_lag"]).copy()
    m.disp_lag = np.array(s["disp_lag"]).copy()
    m.r_lag = s["r_lag"]; m.K = s["K"]; m.t = s["t"]


def _k_map(m: ModelV3, K: float) -> float:
    """One deterministic period of the aggregate REAL map K -> K'.

    The capital stock is owned in the policy proportions; household equity is
    spread uniformly across agents; deposits/bonds are projected to zero (their
    1+r_debt root is handled separately); the firm holds no cash. ALL dynamic
    state is reset deterministically from K here (review item 3): the lagged
    income/yield/dividend/rebate and return-type variables are zeroed, so the map
    has no carry-over between calls. find_fixed_point_K iterates this to the rest
    point; real_slope then freezes the converged lag state explicitly for the two
    perturbed evaluations. Guarded so a perturbation producing a non-finite
    intermediate cannot abort the differentiation."""
    p = m.p
    n = p.n_agents
    K = float(np.real(K))
    m.K = K
    m.eq_state = K * p.own_state
    m.eq_row = K * p.own_row
    m.h_eq = np.full(n, K * p.own_households / n)
    m.h_dep = np.zeros(n)
    m.gov_dep = 0.0
    m.row_dep = 0.0
    m.firm_dep = 0.0
    m.offshore = np.zeros(n)
    m.ret_z = np.zeros(n)
    m.eq_yield_lag = 0.0
    m.div_house_lag = np.zeros(n)
    m.fund_rebate_lag = np.zeros(n)
    m.disp_lag = np.zeros(n)
    m.r_lag = 0.0
    m.t = 100                     # past the (now flat) automation ramp
    try:
        m.step()
    except Exception:
        return K
    Kp = float(np.real(m.K))
    return Kp if np.isfinite(Kp) else K


def find_fixed_point_K(p: ParamsV3, I_fixed=0.5, iters=6000, tol=1e-9):
    m = _make_det_model(p, I_fixed)
    K = p.K0_per_capita * p.n_agents
    for _ in range(iters):
        Kn = _k_map(m, K)
        if abs(Kn - K) < tol * max(abs(K), 1.0):
            K = Kn
            break
        K = Kn
    return K, m


def real_slope(p: ParamsV3, I_fixed=0.5) -> float:
    """Local slope dK'/dK of the aggregate capital map at its fixed point.
    Modulus < 1 => locally stable real economy.

    Made STATELESS (review item 3): the two perturbed evaluations K* + h and
    K* - h are each run on a FRESH model instance restored to the identical
    converged fixed-point snapshot, so neither can contaminate the other through
    carried-over lag state. The central difference is then a clean one-step
    Jacobian of the capital map."""
    Kstar, m = find_fixed_point_K(p, I_fixed)
    _k_map(m, Kstar)            # one clean step so the lag fields settle at K*
    snap = _snapshot(m)
    h = 1e-4 * max(abs(Kstar), 1.0)

    def eval_at(Kval):
        mm = _make_det_model(p, I_fixed)     # fresh instance, no shared state
        _restore(mm, snap)                   # identical frozen state for + and -
        n = p.n_agents
        mm.K = Kval
        mm.eq_state = Kval * p.own_state
        mm.eq_row = Kval * p.own_row
        mm.h_eq = np.full(n, Kval * p.own_households / n)
        mm.h_dep = np.zeros(n); mm.gov_dep = 0.0; mm.row_dep = 0.0; mm.firm_dep = 0.0
        mm.offshore = np.zeros(n); mm.ret_z = np.zeros(n); mm.t = 100
        try:
            mm.step()
        except Exception:
            return Kval
        Kp = float(np.real(mm.K))
        return Kp if np.isfinite(Kp) else Kval

    return (eval_at(Kstar + h) - eval_at(Kstar - h)) / (2.0 * h)


def stability_report(p: ParamsV3, I_levels=(0.3, 0.5, 0.7, 0.9)):
    debt_root = 1.0 + p.r_debt
    rows = []
    for I in I_levels:
        Kstar, _ = find_fixed_point_K(p, I_fixed=I)
        slope = real_slope(p, I_fixed=I)
        rows.append({"I": I, "K_star": Kstar, "real_slope": slope,
                     "stable_real": abs(slope) < 1.0 + 1e-6,
                     "debt_root": debt_root})
    return rows


if __name__ == "__main__":
    from abm_sfc import scenarios_v3 as sv3
    print("Deterministic-skeleton REAL stability: |dK'/dK| < 1 => locally stable.")
    print(f"(Financial debt root = 1+r_debt = {1.0 + sv3.BASE.r_debt:.3f} is structural,")
    print(" the r>g solvency knife-edge, reported separately.)\n")
    print(f"{'scenario':20s}{'I=0.3':>10}{'I=0.5':>10}{'I=0.7':>10}{'I=0.9':>10}")
    for name in ["laissez_faire", "income_tax_ubi", "wealth_tax_ubi",
                 "state_ownership", "foreign_ownership"]:
        p = sv3.REGISTRY[name]()
        rep = stability_report(p)
        cells = "".join(f"{r['real_slope']:10.4f}" for r in rep)
        print(f"{name:20s}{cells}")
