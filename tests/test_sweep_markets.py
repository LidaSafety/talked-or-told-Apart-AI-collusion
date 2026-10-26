"""Scripted sweep markets. Synthetic only."""

import numpy as np
import pytest

from sprint_code.scripted_controls.sweep_markets import BNE, PAIRS, NullBidder, market, outcome_moments

PLAIN = NullBidder(values=(-1, 0, 1), probs=(0.15, 0.75, 0.10), phi=0.0)
REACTIVE = NullBidder(values=(-1, 0, 1), probs=(0.15, 0.75, 0.10), phi=0.5)


def test_markets_are_valid_and_reproducible():
    for arm in ("S0", "S1", "S2", "S3a", "S3b", "S4"):
        a, truth = market(arm, 11, 40, REACTIVE, p=0.6)
        b, _ = market(arm, 11, 40, REACTIVE, p=0.6)
        assert np.array_equal(a.quotes, b.quotes) and a.quotes.min() >= 1 and a.quotes.max() <= 12
        assert (truth == PAIRS[11 % 3]) == (arm in ("S3a", "S3b", "S4"))
        assert (a.signal is not None) == (arm in ("S1", "S2"))


def test_null_residuals_follow_the_given_distribution():
    run, _ = market("S0", 3, 4000, PLAIN)
    r = (run.quotes - BNE[run.costs - 1]).ravel()
    for v, pr in zip(PLAIN.values, PLAIN.probs):
        assert abs(np.mean(r == v) - pr) < 0.02


def test_reaction_moves_quotes_after_wins_and_losses():
    calm = outcome_moments([market("S0", s, 200, PLAIN)[0] for s in range(10)])
    lively = outcome_moments([market("S0", s, 200, REACTIVE)[0] for s in range(10)])
    gap = lambda m: m["change_after_win"] - m["change_after_loss"]
    assert gap(lively) > gap(calm) + 0.5


def test_coordination_schemes():
    run, (i, j) = market("S3a", 4, 300, PLAIN, p=1.0)
    c, q = run.costs, run.quotes
    assert np.all(q[c[:, i] > c[:, j], i] == 11) and np.all(q[c[:, i] <= c[:, j], j] == 11)
    run, (i, j) = market("S3b", 4, 300, PLAIN, p=1.0)
    assert np.array_equal(run.quotes[:, i], run.quotes[:, j]) and set(run.quotes[:, i]) <= set(range(8, 12))
    run, pair = market("S4", 4, 50, PLAIN)
    assert np.all(run.quotes[:, list(pair)] == 11)


@pytest.mark.parametrize("arm", ["S1", "S2"])
def test_signal_shift_reaches_the_right_bidders(arm):
    run, _ = market(arm, 7, 3000, PLAIN)
    high = run.signal >= 9
    lift = run.quotes[high].mean(axis=0) - run.quotes[~high].mean(axis=0)
    pair = PAIRS[7 % 3]
    for s in range(3):
        expect_lift = arm == "S1" or s in pair
        assert (lift[s] > 1.5) == expect_lift
