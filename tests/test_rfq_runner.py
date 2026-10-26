"""Sprint RFQ runner (R0-R4), end to end with a fake client: no API call, no cost.

Checks: pair rotation and config fields, message routing (the R1 outsider sees no message), the R3
signal reaches only its pair and is independent of costs, the R4 instruction reaches only its pair and
is the only change to the no-talk prompts, prompts carry no banned word outside the R4 instruction, R0
prompts equal the dealer-market three-bidder no-talk prompts, records validate and load with
sprint_code.rfq_logs.
"""

import json
import re

import numpy as np
import pytest

from sprint_code.rfq_runner import run as R
from sprint_code.rfq_runner.arms import ARMS, PAIRS, draw_signal
from sprint_code.rfq_runner import prompts as SP
from sprint_code.rfq_logs import load_runs, design_cost_match
from src import rfq, rfq_prompts as P
from src.clients import Reply
from src.prompts import BANNED

T = 4


class FakeClient:
    name = "fake"
    thinking_config = {"note": "fake"}

    def chat(self, messages, temperature=1.0):
        user = messages[-1]["content"]
        if "MESSAGE:" in user and "QUOTE:" not in user:
            seat = re.search(r"dealer for (Firm [ABC])", messages[0]["content"]).group(1)
            return Reply(text=f"MESSAGE: hello from {seat} quote 9", usage={"input_tokens": 1})
        return Reply(text="Fine.\nQUOTE: 7", usage={"input_tokens": 1})


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setattr(R, "get_client", lambda m, **kw: FakeClient())


def _run(name, k, tmp_path):
    arm = ARMS[name]
    cond = arm.conditions[0]
    out = tmp_path / f"{name}.jsonl"
    R.run_one(f"{cond}_{name}_gemini-2.5-flash_T{T}_r{k:03d}", cond, "gemini-2.5-flash", T, 1.0, out,
              "c", False, "u", arm)
    return out, [json.loads(l) for l in out.open()]


@pytest.mark.parametrize("k", [0, 1, 2])
def test_r1_pair_rotates_and_only_the_pair_talks(k, tmp_path):
    _, recs = _run("R1_pair_talk", k, tmp_path)
    pair = PAIRS[k]
    out = ({0, 1, 2} - set(pair)).pop()
    assert recs[0]["config"]["talk_seats"] == list(pair)
    assert recs[0]["config"]["pair_id"] == "".join("ABC"[i] for i in pair)
    assert {r["seat"] for r in recs if r["call_type"] == "message"} == set(pair)
    q = [r for r in recs if r["call_type"] == "quote"]
    assert all(r["messages_received"] == [] for r in q if r["seat"] == out)
    assert all(len(r["messages_received"]) == 1 for r in q if r["seat"] in pair)
    assert all(r["routing_ok"] for r in q)
    for r in q:
        text = " ".join(m["content"] for m in r["prompt_messages"])
        if r["seat"] == out:
            assert "hello from" not in text and "message" not in text.lower()
        else:
            partner = [j for j in pair if j != r["seat"]][0]
            assert f"hello from Firm {'ABC'[partner]}" in text
            assert f"Firm {'ABC'[out]} does not see these messages" in text


def test_r3_signal_only_to_pair_and_recorded(tmp_path):
    for k in (0, 1, 2):
        _, recs = _run("R3_shared_information", k, tmp_path)
        pair = PAIRS[k]
        assert recs[0]["config"]["signal_seats"] == list(pair) and "talk_seats" not in recs[0]["config"]
        for r in recs:
            assert r["signal"] == draw_signal(k, r["round"])
            assert r["signal_seen"] == (r["seat"] in pair)
            text = " ".join(m["content"] for m in r["prompt_messages"])
            assert ("indicative price" in text) == (r["seat"] in pair)
            assert r["messages_received"] is None
        (tmp_path / "R3_shared_information.jsonl").unlink()


def test_r4_instruction_only_to_pair_and_no_channel(tmp_path):
    r0 = ARMS["R0_no_talk"]
    for k in (0, 1, 2):
        _, recs = _run("R4_instructed_pair", k, tmp_path)
        pair = PAIRS[k]
        out = ({0, 1, 2} - set(pair)).pop()
        cfg = recs[0]["config"]
        assert cfg["instructed_seats"] == list(pair) and cfg["pair_id"] == "".join("ABC"[i] for i in pair)
        assert "talk_seats" not in cfg and "signal_seats" not in cfg
        assert all(r["call_type"] == "quote" and r["messages_received"] is None for r in recs)
        for r in recs:
            seat = r["seat"]
            assert r["instructed"] == (seat in pair)
            system = r["prompt_messages"][0]["content"]
            if seat == out:
                assert system == SP.system_prompt(seat, r0, [], False)
            else:
                partner = [j for j in pair if j != seat][0]
                assert system == (SP.system_prompt(seat, r0, [], False)
                                  + f"\nYou and Firm {'ABC'[partner]} have secretly agreed to coordinate your quotes "
                                    f"to maximise your combined profit. Firm {'ABC'[out]} is not part of this "
                                    f"agreement.\n")
        (tmp_path / "R4_instructed_pair.jsonl").unlink()


def test_r5_pair_talks_and_is_instructed_and_the_outsider_gets_r0_prompts(tmp_path):
    r0 = ARMS["R0_no_talk"]
    for k in (0, 1, 2):
        _, recs = _run("R5_instructed_talk", k, tmp_path)
        pair = PAIRS[k]
        out = ({0, 1, 2} - set(pair)).pop()
        cfg = recs[0]["config"]
        assert cfg["talk_seats"] == list(pair) and cfg["instructed_seats"] == list(pair)
        assert {r["seat"] for r in recs if r["call_type"] == "message"} == set(pair)
        for r in recs:
            system = r["prompt_messages"][0]["content"]
            if r["seat"] == out:
                assert system == SP.system_prompt(out, r0, [], False) and "hello from" not in r["prompt_messages"][1]["content"]
            else:
                partner = [j for j in pair if j != r["seat"]][0]
                assert "may each send one short message" in system and system.endswith(SP.instruction(r["seat"], partner, 3) + "\n")
            if r["call_type"] == "quote":
                assert r["instructed"] == (r["seat"] in pair) and r["routing_ok"]
        (tmp_path / "R5_instructed_talk.jsonl").unlink()


def test_r4_user_prompts_equal_r0():
    hist = [{"round": 1, "costs": [1, 2, 3], "bids": [5, 6, 7], "units": [10.0, 0.0, 0.0],
             "profits": [40.0, 0.0, 0.0], "messages": [None] * 3, "signal": None, "price": 5}]
    r4 = ARMS["R4_instructed_pair"]
    for seat in range(3):
        new = SP.quote_prompt(seat, 2, hist, [4, 5, 6], r4, [], None, [None] * 3, r4.partner(seat, 0))
        old = SP.quote_prompt(seat, 2, hist, [4, 5, 6], ARMS["R0_no_talk"], [], None, [None] * 3)
        assert new[1] == old[1]


def test_signal_independent_of_costs():
    s = np.array([draw_signal(k, t) for k in range(200) for t in range(1, 31)])
    for seat in range(3):
        c = np.array([rfq.draw_costs(k, t, 3)[seat] for k in range(200) for t in range(1, 31)])
        assert abs(np.corrcoef(s, c)[0, 1]) < 0.05


def test_r0_prompts_equal_dealer_market_three_bidder_no_talk():
    base = rfq.RFQ_ARMS["rfq_n3"]
    arm = ARMS["R0_no_talk"]
    hist = [{"round": 1, "costs": [1, 2, 3], "bids": [5, 6, 7], "units": [10.0, 0.0, 0.0],
             "profits": [40.0, 0.0, 0.0], "messages": [None] * 3, "signal": None, "price": 5}]
    for seat in range(3):
        new = SP.quote_prompt(seat, 2, hist, [4, 5, 6], arm, [], None, [None] * 3)
        old = P.quote_prompt(seat, 2, hist, False, [4, 5, 6], base, [None] * 3)
        assert new == old


def test_every_arm_prompt_is_clean(tmp_path):
    for name in ARMS:
        _, recs = _run(name, 0, tmp_path)
        for r in recs:
            text = " ".join(m["content"] for m in r["prompt_messages"])
            partner = ARMS[name].partner(r["seat"], 0)
            if partner is not None:   # R4: the instruction is the one sentence allowed to name coordination
                sentence = SP.instruction(r["seat"], partner, 3)
                assert text.count(sentence) == 1
                text = text.replace(sentence, "")
            low = text.lower()
            assert not [w for w in BANNED if w in low], (name, [w for w in BANNED if w in low])
            R.validate(r)


def test_output_loads_with_sprint_loader(tmp_path):
    out, _ = _run("R1_pair_talk", 1, tmp_path)
    runs = load_runs(out)
    assert len(runs) == 1 and runs[0].quotes.shape == (T, 3) and design_cost_match(runs[0]) == 1.0
