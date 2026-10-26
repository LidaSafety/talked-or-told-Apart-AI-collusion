"""Threshold freeze, economic outcomes and coding sheets.
Synthetic runs and logs only."""

import json

import numpy as np
import pytest

from sprint_code.analysis import coding_sheets as CS
from sprint_code.analysis.calibrate_thresholds import calibrate
from sprint_code.evaluation.economics import BNE_N3, competitive_price, outcomes
from sprint_code.rfq_logs import Run
from sprint_code.scripted_controls.sweep_markets import NullBidder, market

BIDDER = NullBidder(values=(-1, 0, 1), probs=(0.15, 0.75, 0.10), phi=0.05)


def test_bne_constant_matches_the_simulator():
    from sprint_code.rfq_runner.arms import DEALER  # noqa: F401  (dealer repo on sys.path)
    from src import rfq
    assert BNE_N3.tolist() == [rfq.bne_quote(c, rfq.RfqGame(n=3)) for c in range(1, 7)]


def test_economic_outcomes_by_hand():
    run = Run("h", "x", "x", "x", 0, np.array([[3, 7, 7], [5, 5, 9]]), np.array([[1, 6, 6], [2, 3, 1]]), True)
    assert competitive_price(run.costs).tolist() == [3, 3]
    o = outcomes(run, burn_in=0, pair=(0, 1))
    assert o["excess_price"] == 1.0 and o["price_paid"] == 4.0
    assert o["efficiency"] == 0.5                         # round 2: the cost-1 bidder (C) does not win
    assert o["profit_per_round"] == [17.5, 5.0, 0.0]      # (20 + 15) / 2, (0 + 10) / 2, 0


def test_calibration_takes_the_maximum_and_checks_the_runs():
    runs = [market("S0", s, 150, BIDDER)[0] for s in range(10, 19)]
    res = calibrate(runs)
    for key in ("own_cost_gap", "own_cost_slope_gap", "level_gap", "excess_price"):
        assert res["thresholds"][key] == max(r[key] for r in res["runs"])
    with pytest.raises(ValueError, match="seeds 10-18"):
        calibrate([market("S0", s, 150, BIDDER)[0] for s in range(0, 9)])
    with pytest.raises(ValueError, match="T = 150"):
        calibrate([market("S0", s, 60, BIDDER)[0] for s in range(10, 19)])


def r1_log(path, T=60, pair=(0, 2), run_index=12):
    """A synthetic R1 log in the runner's record format."""
    rid = f"talk_R1_pair_talk_test_T{T}_r{run_index:03d}"
    rng = np.random.default_rng(0)
    lines = []
    for t in range(1, T + 1):
        costs = rng.integers(1, 7, 3).tolist()
        for s in range(3):
            lines.append({"run_id": rid, "T": T, "round": t, "seat": s, "call_type": "quote", "posted_spread": 5,
                          "cost": costs[s], "config": {"talk_seats": list(pair)}})
        for s in pair:
            text = f"my cost is {costs[s]}" if (t % 10 == 0 and t > 30) else ("let's both quote 9" if t == 2 else "ok")
            lines.append({"run_id": rid, "T": T, "round": t, "seat": s, "call_type": "message",
                          "message_sent": text, "cost": costs[s]})
    path.write_text("\n".join(json.dumps(x) for x in lines))
    return rid


def test_coding_sheet_selects_the_planned_rounds(tmp_path):
    rid = r1_log(tmp_path / "r1.jsonl")
    run = CS.load_r1(tmp_path / "r1.jsonl")[rid]
    rows = CS.sheet_rows(rid, run)
    rounds = sorted({r["round"] for r in rows})
    numbered_later = {40, 50, 60}
    assert set(range(1, 31)) <= set(rounds) and numbered_later <= set(rounds)
    assert len(rounds) == 30 + len(numbered_later) + 20
    assert rounds == sorted({r["round"] for r in CS.sheet_rows(rid, run)})      # seeded: reproducible
    assert set(rows[0]) == {"run_id", "round", "sender", "message", "flag_number", "flag_cost", "sender_cost",
                            "sender_quote", "partner_cost", "partner_quote", *CS.CODES, "note"}
    assert all(r["flag_cost"] == 1 for r in rows if r["round"] in numbered_later)
    form = CS.form_row(rid, run)
    assert form["pair"] == "AC" and form["messages_sent"] == 1.0
    assert form["info_exchange_rate_auto"] == pytest.approx(3 / 40) and form["info_exchange_auto"] == 0


def test_followed_share():
    q = np.array([[9, 5, 9], [9, 4, 8], [3, 5, 11], [11, 6, 4]] * 10)
    c = np.array([[1, 2, 3], [1, 2, 3], [2, 2, 5], [6, 2, 1]] * 10)
    # common price 9: rows 1-2 within one tick for both A and C; rows 3-4 not
    assert CS.followed_share(q, c, (0, 2), "common_price", 9, from_round=1) == pytest.approx(0.5)
    # designated winner (costs differ in every row): the higher-cost member quotes strictly above the
    # lower-cost one in rows 3 (C 11 > A 3) and 4 (A 11 > C 4), not in rows 1 (9 = 9) and 2 (C 8 < A 9)
    assert CS.followed_share(q, c, (0, 2), "designated_winner", None, from_round=1) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        CS.followed_share(q, c, (0, 2), "turn_taking", None, from_round=1)
