"""Synthetic checks for the information measures and cross-cost tests.

All data here are generated inside the tests; nothing is written to results/.
"""

from pathlib import Path

import numpy as np
import pytest

from sprint_code.cost_response.cost_info import Pooled, cross_cost, cross_cost_test, own_cost
from sprint_code.detectors.info import binned, cmi, entropy, mi, net_influence
from sprint_code.rfq_logs import design_cost_match, load_runs

DATA = Path(__file__).resolve().parents[1] / "external/dealer_market/data/runs/rfq"


def market(policy, runs=10, T=30, seed=0) -> Pooled:
    """Three bidders, iid uniform costs 1..6; policy(costs_row, rng) -> three quotes."""
    rng = np.random.default_rng(seed)
    costs = rng.integers(1, 7, size=(runs * T, 3))
    bids = np.array([policy(c, rng) for c in costs])
    return Pooled(bids, costs, np.zeros(runs * T, dtype=int), np.repeat(np.arange(runs), T))


def independent(c, rng):
    """Each bidder marks up its own cost by 1-3 ticks."""
    return np.minimum(c + rng.integers(1, 4, 3), 12)


def cover_bid(c, rng):
    """A and B share costs: the higher-cost member of the pair quotes 11 (cover bid); C independent."""
    q = independent(c, rng)
    if c[0] > c[1]:
        q[0] = 11
    elif c[1] > c[0]:
        q[1] = 11
    return q


def test_entropy_and_mi_basics():
    x = np.array([0, 1, 2, 0, 1, 2] * 100)
    assert entropy(x) == pytest.approx(np.log2(3))
    assert mi(x, x) == pytest.approx(np.log2(3))
    assert cmi(x, np.zeros_like(x)) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 3, 20000), rng.integers(0, 3, 20000)
    assert mi(a, b) < 0.002


def test_binning_edges():
    assert binned(np.array([1, 4, 5, 8, 9, 12]), (4, 8)).tolist() == [0, 0, 1, 1, 2, 2]
    assert binned(np.array([1, 3, 4, 6]), (3,)).tolist() == [0, 0, 1, 1]


def test_net_influence_returns_coupled_pair_only():
    rng = np.random.default_rng(2)
    signal = rng.integers(0, 3, 3000)
    noise = lambda: rng.integers(0, 3, 3000)
    bids = np.column_stack([np.where(rng.random(3000) < 0.8, signal, noise()),
                            np.where(rng.random(3000) < 0.8, signal, noise()), noise()])
    state = np.zeros(3000, dtype=int)
    assert net_influence(bids, state)["returned"] == (0, 1)
    assert net_influence(np.column_stack([noise(), noise(), noise()]), state)["returned"] is None


def test_cross_cost_detects_pair_and_not_outsider():
    p = market(cover_bid, runs=10, T=30)
    rng = np.random.default_rng(3)
    assert cross_cost_test(p, 0, 1, rng, K=199)["p_X_crt"] < 0.01
    assert cross_cost_test(p, 1, 0, rng, K=199)["p_X_crt"] < 0.01
    assert cross_cost_test(p, 0, 2, rng, K=199)["p_X_crt"] > 0.01


def test_crt_size_under_null():
    """Without a channel the CRT should reject at about its nominal rate."""
    rejections = []
    for seed in range(60):
        p = market(independent, runs=10, T=30, seed=seed)
        res = cross_cost_test(p, 0, 1, np.random.default_rng(1000 + seed), K=99)
        rejections.append(res["p_X_crt"] <= 0.05)
    assert np.mean(rejections) <= 0.15


def test_plugin_cross_cost_is_biased_up_under_null():
    p = market(independent, runs=10, T=30, seed=5)
    assert cross_cost(p, 0, 1) > 0.0


def test_own_cost_positive_for_markup_policy():
    p = market(independent, runs=10, T=30)
    res = own_cost(p, 0)
    assert res["slope"] > 0.5 and res["contrast"] > 1.0 and res["R"] > 0.1


@pytest.mark.skipif(not DATA.exists(), reason="dealer_market submodule not checked out")
def test_logged_costs_match_simulator_design():
    """Validity of the CRT rests on costs being the simulator's design draws."""
    for f in ("no_talk_rfq_n3_gemini-2.5-flash.jsonl", "talk_rfq_public_gemini-2.5-flash.jsonl"):
        runs = load_runs(DATA / f)
        assert runs and all(design_cost_match(r) == 1.0 for r in runs)
