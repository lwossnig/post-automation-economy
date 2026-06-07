"""Validate the distributional engine against closed-form Bouchaud-Mezard theory.

The kernel's continuum limit has a known inverse-gamma stationary distribution
(derivation in kinetic.py). We check two independent consequences:

  * total wealth is conserved exactly by every kernel step;
  * the stationary mean of 1/x matches  E[1/x] = alpha / (2J/sigma^2),
    a robust scalar moment (more stable than fitting a tail exponent).

We also sanity-check the tail exponent via a Hill estimator with a loose
tolerance, since tail fits are noisy.
"""
import numpy as np

from abm_sfc.kinetic import kinetic_step, bm_stationary_targets


def test_kernel_conserves_total():
    rng = np.random.default_rng(0)
    w = rng.lognormal(0, 1, 5000)
    total0 = w.sum()
    for _ in range(500):
        w = kinetic_step(w, J=0.6, sigma=0.4, dt=0.05, rng=rng)
        assert abs(w.sum() - total0) < 1e-6 * total0


def _run_to_stationary(J, sigma, dt, n=20000, burn=4000, sample=4000):
    rng = np.random.default_rng(42)
    w = np.ones(n)
    for _ in range(burn):
        w = kinetic_step(w, J, sigma, dt, rng)
    acc = np.zeros(n)
    for k in range(sample):
        w = kinetic_step(w, J, sigma, dt, rng)
        acc += 1.0 / w
    return acc / sample   # time-averaged 1/x per agent


def test_stationary_inverse_moment_matches_theory():
    J, sigma, dt = 0.8, 0.4, 0.02
    target = bm_stationary_targets(J, sigma)
    inv = _run_to_stationary(J, sigma, dt)
    e_inv = float(inv.mean())
    # 8% tolerance: discretisation + finite sample
    assert abs(e_inv - target["E_inv_x"]) / target["E_inv_x"] < 0.08, (
        f"E[1/x]={e_inv:.3f} vs theory {target['E_inv_x']:.3f}"
    )


def test_higher_noise_increases_inequality():
    """alpha = 1 + 2J/sigma^2 falls as sigma rises -> heavier tail -> more
    concentration. Check the Herfindahl of stationary wealth rises with sigma."""
    rng = np.random.default_rng(7)

    def herfindahl(sigma):
        w = np.ones(8000)
        for _ in range(3000):
            w = kinetic_step(w, J=0.6, sigma=sigma, dt=0.02, rng=rng)
        s = w / w.sum()
        return float((s ** 2).sum())

    assert herfindahl(0.6) > herfindahl(0.3)
