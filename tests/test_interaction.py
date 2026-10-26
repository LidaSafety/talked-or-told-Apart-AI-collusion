"""Matched-seed two-by-two interaction: per-seed outcomes of the assigned pair and the delta."""

import numpy as np

from sprint_code.analysis.interaction import ARMS, interaction, outcomes_by_seed


def _tables(seeds, price, gamma):
    runs, views, dirs = [], [], []
    for s in seeds:
        for arm in ARMS:
            runs.append({"set": arm, "seed": str(s), "main_seed": "True", "seed_pair": "AC", "price_paid": price[arm][s],
                         "efficiency": 1.0, "profit_pair": 2.0, "profit_outsider": 1.0})
            views.append({"set": arm, "seed": str(s), "view": "public", "gamma_AB": 9.0, "gamma_AC": gamma[arm][s],
                          "gamma_BC": 9.0})
            for b, c in (("A", "C"), ("C", "A"), ("A", "B")):
                dirs.append({"set": arm, "seed": str(s), "bidder": b, "cost_of": c, "X_excess": 0.1 if b + c != "AB" else 5.0})
    return runs, views, dirs


def test_delta_uses_the_seed_pair_and_matches_by_seed():
    seeds = range(10, 16)
    price = {"R0_main": {s: 3.0 for s in seeds}, "R1": {s: 3.2 for s in seeds},
             "R4": {s: 4.0 for s in seeds}, "R5": {s: 3.4 for s in seeds}}
    gamma = {"R0_main": {s: 0.0 for s in seeds}, "R1": {s: 0.2 for s in seeds},
             "R4": {s: 0.0 for s in seeds}, "R5": {s: 0.8 for s in seeds}}
    by_arm = outcomes_by_seed(*_tables(seeds, price, gamma))
    per_seed, summary = interaction(by_arm)
    s = {r["measure"]: r for r in summary}
    assert np.isclose(s["price_paid"]["delta"], (3.4 - 4.0) - (3.2 - 3.0))
    assert np.isclose(s["cmi_pair"]["delta"], 0.6)                       # gamma_AC only, not the larger gamma_AB
    assert np.isclose(s["cross_cost_pair"]["delta"], 0.0)                # the A->B direction is not the pair's
    assert s["price_paid"]["seeds_delta_negative"] == 6 and np.isclose(s["price_paid"]["p_delta"], 2 / 64)
    assert len([r for r in per_seed if r["measure"] == "price_paid"]) == 6


def test_unmatched_seeds_are_dropped():
    seeds = range(10, 13)
    price = {a: {s: 3.0 for s in seeds} for a in ARMS}
    gamma = {a: {s: 0.1 for s in seeds} for a in ARMS}
    runs, views, dirs = _tables(seeds, price, gamma)
    runs = [r for r in runs if not (r["set"] == "R5" and r["seed"] == "12")]
    _, summary = interaction(outcomes_by_seed(runs, views, dirs))
    assert all(r["seeds"] == 2 for r in summary)
