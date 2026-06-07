"""Sandbox tests for the market-priced rent-bearing IP (Phase 3, item 3).

Stage 1 is a diagnostic only: capitalise the after-host-tax rent into an IP value
and report the owner's market-value share. No feedback into the accounting yet, so
the v3 suite stays bit-identical (checked separately by running tests/test_v3.py).
"""
import numpy as np
from dataclasses import replace
from abm_sfc.model_v3 import ModelV3
from abm_sfc import scenarios_v3 as _sv3


def _run(periods=400, **o):
    p = replace(_sv3.two_channel_base(), n_agents=400, periods=periods, seed=1)
    for k, v in o.items():
        setattr(p, k, v)
    m = ModelV3(p)
    for _ in range(periods):
        m.step()
    return m


def test_ip_value_zero_without_rent():
    m = _run(mu_frac=0.0)
    assert m.hist.v_ip[-1] == 0.0
    assert abs(m.hist.own_row_mkt[-1] - m.hist.own_row[-1]) < 1e-9


def test_ip_value_positive_and_rises_with_rent():
    lo, hi = _run(mu_frac=0.15), _run(mu_frac=0.40)
    assert lo.hist.v_ip[-1] > 0.0
    assert hi.hist.v_ip[-1] > lo.hist.v_ip[-1]


def test_market_ownership_exceeds_book_with_rent():
    m = _run()
    assert m.hist.own_row_mkt[-1] > m.hist.own_row[-1]


def test_host_tax_lowers_ip_value():
    """The IP is worth the PV of the after-host-tax rent, so a levy reduces it."""
    untaxed = _run()
    taxed = _run(dst_ai=0.10, tax_repat=0.30)
    assert taxed.hist.v_ip[-1] < untaxed.hist.v_ip[-1]


def test_invariants_unchanged_by_diagnostic():
    m = _run(dst_ai=0.10, tax_repat=0.30)
    assert abs((m.h_eq.sum() + m.eq_state + m.eq_row) / m.K - 1.0) < 1e-6
    assert abs(m.deposits_sum()) < 1e-3 * m.K


def test_stage2_market_value_invariant_holds():
    """Generalised stock invariant: market-value net worth of all sectors equals
    the book capital stock plus the IP value, to machine precision, every period."""
    m = _run(dst_ai=0.10, tax_repat=0.30)
    nm = np.array(m.hist.nw_sum_market)
    K = np.array(m.hist.K)
    vip = np.array(m.hist.v_ip)
    rel = np.abs(nm - (K + vip)) / np.maximum(K + vip, 1e-9)
    assert rel.max() < 1e-9


def test_stage2_book_and_transactions_invariants_unchanged():
    """The book stock invariant (NW_sum = K) and the transactions invariant
    (deposits sum to zero) are untouched by the market-value overlay."""
    m = _run(dst_ai=0.10, tax_repat=0.30)
    assert abs((m.h_eq.sum() + m.eq_state + m.eq_row) / m.K - 1.0) < 1e-9
    assert abs(m.deposits_sum()) < 1e-3 * m.K


def test_stage2_articulation_holds():
    """The change in the owner's market net worth decomposes into the transaction
    part (change in book net worth) and the revaluation part (change in IP value)."""
    m = _run()
    rm = np.array(m.hist.row_nw_market)
    vip = np.array(m.hist.v_ip)
    rb = rm - vip                       # implied book net worth of RoW
    d_market = np.diff(rm)
    d_book = np.diff(rb)
    d_reval = np.diff(vip)
    assert np.abs(d_market - (d_book + d_reval)).max() < 1e-9


def test_stage3_off_by_default_bit_identical():
    """equity_price_on = False leaves behaviour exactly as the frozen model."""
    a = ModelV3(replace(_sv3.two_channel_base(), equity_price_on=False, n_agents=300, periods=200, seed=0))
    b = ModelV3(replace(_sv3.two_channel_base(), n_agents=300, periods=200, seed=0))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]
    assert a.eq_row == b.eq_row


def test_stage3_valuation_redomesticates_with_invariants():
    """With the price on, a high valuation raises repatriation and lowers foreign
    ownership relative to the price-off case, and both invariants survive the feedback."""
    off = _run(equity_price_on=False)
    on = _run(equity_price_on=True)
    assert on.hist.exports[-1] > off.hist.exports[-1]      # owner takes more rent out
    assert on.hist.own_row[-1] < off.hist.own_row[-1]      # ownership re-domesticated
    assert abs((on.h_eq.sum() + on.eq_state + on.eq_row) / on.K - 1.0) < 1e-6
    assert abs(on.deposits_sum()) < 1e-3 * on.K
    # market-value invariant still holds under the feedback
    nm = np.array(on.hist.nw_sum_market); K = np.array(on.hist.K); vip = np.array(on.hist.v_ip)
    assert (np.abs(nm - (K + vip)) / np.maximum(K + vip, 1e-9)).max() < 1e-9


def test_stage4_reproduction_bit_identical():
    """The parameter that reproduces the old results: market_clearing_on = False
    keeps q == 1 and is bit-identical to the frozen model, every series."""
    a = ModelV3(replace(_sv3.two_channel_base(), market_clearing_on=False, n_agents=300, periods=200, seed=0))
    b = ModelV3(replace(_sv3.two_channel_base(), n_agents=300, periods=200, seed=0))
    for _ in range(200):
        a.step(); b.step()
    assert a.hist.Y[-1] == b.hist.Y[-1]
    assert a.eq_row == b.eq_row
    assert a.hist.q_mkt[-1] == 1.0


def test_stage4_market_value_invariant_holds_with_price_on():
    """With the market price live, market-value net worth equals q*K plus the IP
    value to machine precision, and the book and deposit invariants are intact."""
    m = _run(market_clearing_on=True, q_portfolio_sensitivity=0.5)
    nm = np.array(m.hist.nw_sum_market); K = np.array(m.hist.K)
    vip = np.array(m.hist.v_ip); q = np.array(m.hist.q_mkt)
    assert (np.abs(nm - (q * K + vip)) / np.maximum(q * K + vip, 1e-9)).max() < 1e-8
    assert abs((m.h_eq.sum() + m.eq_state + m.eq_row) / m.K - 1.0) < 1e-6
    assert abs(m.deposits_sum()) < 1e-3 * m.K
