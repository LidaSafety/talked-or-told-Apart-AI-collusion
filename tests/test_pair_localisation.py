"""Synthetic checks for per-run pair localisation.

Scripted policies here are minimal test fixtures, not the study's scripted controls S0-S5.
"""

import numpy as np

from sprint_code.evaluation.pair_localisation import PAIRS, calibrate_gaps, run_scores, summarise
from sprint_code.rfq_logs import Run


def independent(c, pair, rng):
    return np.minimum(c + rng.integers(1, 4, 3), 12)


def constant_pair(c, pair, rng):
    q = independent(c, pair, rng)
    q[list(pair)] = 11
    return q


def cover_bid(c, pair, rng):
    """Designated winner: the higher-cost pair member quotes 11."""
    i, j = pair
    q = independent(c, pair, rng)
    if c[i] > c[j]:
        q[i] = 11
    elif c[j] > c[i]:
        q[j] = 11
    return q


def synth(policy, pair, seed, T=100):
    rng = np.random.default_rng(seed)
    costs = rng.integers(1, 7, size=(T, 3))
    quotes = np.array([policy(c, pair, rng) for c in costs])
    return Run(f"{policy.__name__}_{seed}", "synthetic", "synthetic", "none", seed, quotes, costs, True)


def scored(runs, seed=0):
    rng = np.random.default_rng(seed)
    return [run_scores(r, rng, K=99) for r in runs]


def test_constant_pair_found_by_own_cost_and_level_not_by_dependence():
    r0 = [synth(independent, None, s) for s in range(9)]
    pairs = [synth(constant_pair, pr, 100 + k) for k, pr in enumerate(PAIRS)]
    res = scored(r0 + pairs)
    calibrate_gaps(res, {r.run_id for r in r0})
    for r, pr in zip(res[9:], PAIRS):
        assert r["returned"]["own_cost"] == pr
        assert r["returned"]["level"] == pr
        assert r["returned"]["netinf"] is None          # constant bids carry no dependence
        assert r["returned"]["cross_cost"] is None


def test_cover_bid_pair_found_by_cross_cost_and_net_influence():
    res = scored([synth(cover_bid, pr, 200 + k) for k, pr in enumerate(PAIRS)])
    for r, pr in zip(res, PAIRS):
        assert r["returned"]["cross_cost"] == pr
        assert r["returned"]["netinf"] == pr


def test_independent_runs_rarely_return_a_pair():
    r0 = [synth(independent, None, s) for s in range(9)]
    test = [synth(independent, None, 300 + s) for s in range(9)]
    res = scored(r0 + test)
    calibrate_gaps(res, {r.run_id for r in r0})
    s = summarise(res, {r["run_id"]: None for r in res})
    for det in ("netinf", "cross_cost", "own_cost", "level"):
        assert s[det]["false_pair"] <= 3 / 18, (det, s[det])


def test_gap_rules_need_calibration_runs():
    res = scored([synth(constant_pair, (0, 1), 7)])
    calibrate_gaps(res, set())
    assert res[0]["returned"]["own_cost"] is None and res[0]["returned"]["level"] is None


def test_ties_do_not_favour_seat_order():
    flat = Run("flat", "synthetic", "synthetic", "none", 0, np.full((50, 3), 7),
               np.random.default_rng(0).integers(1, 7, (50, 3)), True)
    picks = {run_scores(flat, np.random.default_rng(s), K=19)["forced"]["mi"] for s in range(30)}
    assert len(picks) == 3


def test_summarise_scores_correct_wrong_and_no_pair():
    res = [{"run_id": "a", "forced": {d: (0, 1) for d in ("mi", "cmi", "netinf", "cross_cost", "own_cost", "level")},
            "returned": {"netinf": (0, 1), "cross_cost": (0, 2), "own_cost": None, "level": None}},
           {"run_id": "b", "forced": {d: (0, 1) for d in ("mi", "cmi", "netinf", "cross_cost", "own_cost", "level")},
            "returned": {"netinf": (0, 1), "cross_cost": None, "own_cost": None, "level": None}}]
    s = summarise(res, {"a": (0, 1), "b": None})
    assert s["netinf"]["correct"] == 1.0 and s["netinf"]["false_pair"] == 1.0
    assert s["cross_cost"]["wrong_pair"] == 1.0 and s["own_cost"]["no_pair"] == 1.0
    assert s["mi"]["forced_accuracy"] == 1.0 and "correct" not in s["mi"]
