"""Main-study analysis. Synthetic logs in the runner's record format,
on the design's cost and signal draws; no main-study data."""

import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sprint_code.analysis import main_analysis as MA
from sprint_code.analysis.coding_sheets import FORM_FIELDS
from sprint_code.analysis.scripted_sweep import score_chunk
from sprint_code.analysis.calibrate_thresholds import calibrate
from sprint_code.detectors.info import MAIN_BINS
from sprint_code.evaluation.pair_localisation import PAIRS
from sprint_code.rfq_logs import design_cost, load_runs
from sprint_code.rfq_runner.arms import draw_signal
from sprint_code.rfq_runner import prompts as SP
from sprint_code.scripted_controls.sweep_markets import NullBidder, market

ROOT = Path(__file__).resolve().parents[1]
BNE = np.array([3, 4, 4, 5, 6, 7])
NAMES = {"R0": ("no_talk", "R0_no_talk"), "R1": ("talk", "R1_pair_talk"), "R3": ("no_talk", "R3_shared_information"),
         "R4": ("no_talk", "R4_instructed_pair"), "R5": ("talk", "R5_instructed_talk")}
MODEL = "test-model"


def write_log(path, arm, seeds, T=150, stop=None, fail=None, r4_constant=False):
    """Null bidders on the design's costs; R1: designated winner (the pair's higher-cost member quotes 11);
    R3: the signal pair adds 2 ticks when the signal is at least 9; R4: R1's designated winner, or with
    `r4_constant` the instructed pair quotes 11 throughout, its prompts carrying the instructed-pair sentence. `stop` {seed: last
    round} truncates runs; `fail` {seed: (first, last)} makes every call in those rounds fail (status api_error,
    the reserve 12 posted)."""
    cond, name = NAMES[arm]
    lines = []
    for seed in seeds:
        rng = np.random.default_rng([seed, ord(arm[1])])
        pair = PAIRS[seed % 3]
        cfg = {"sprint_arm": arm, "seed": seed, "T": T, "model": MODEL, "temperature": 1.0,
               **({"talk_seats": list(pair)} if arm in ("R1", "R5") else {}),
               **({"signal_seats": list(pair)} if arm == "R3" else {}),
               **({"instructed_seats": list(pair), "instruction": SP.INSTRUCTION} if arm in ("R4", "R5") else {})}
        for t in range(1, (stop or {}).get(seed, T) + 1):
            c = np.array([design_cost(seed, t, s) for s in range(3)])
            q = BNE[c - 1] + rng.choice([-1, 0, 1], size=3, p=[0.15, 0.75, 0.10])
            sig = draw_signal(seed, t)
            if arm in ("R1", "R5") or (arm == "R4" and not r4_constant):
                i, j = pair
                q[i if c[i] > c[j] else j] = 11
            if arm == "R4" and r4_constant:
                q[list(pair)] = 11
            if arm == "R3" and sig >= 9:
                q[list(pair)] += 2
            q = np.clip(q, 1, 12)
            first, last = (fail or {}).get(seed, (0, -1))
            status = "api_error" if first <= t <= last else "ok"
            if status != "ok":
                q[:] = 12
            for s in range(3):
                r4 = {}
                if arm in ("R4", "R5"):
                    partner = next((j for j in pair if j != s), None) if s in pair else None
                    system = "Rules." + (f"\n{SP.instruction(s, partner, 3)}\n" if partner is not None else "")
                    r4 = {"instructed": s in pair, "prompt_messages": [{"role": "system", "content": system},
                                                                       {"role": "user", "content": f"Round {t}."}]}
                lines.append({"run_id": f"{cond}_{name}_{MODEL}_T{T}_r{seed:03d}", "arm": name, "condition": cond,
                              "model": MODEL, "T": T, "round": t, "seat": s, "call_type": "quote",
                              "posted_spread": int(q[s]), "cost": int(c[s]), "costs": c.tolist(), "status": status,
                              "routing_ok": True, "config": cfg, **({"signal": sig} if arm == "R3" else {}), **r4})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def write_form(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", *FORM_FIELDS], restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def test_exact_binomial():
    lo, hi = MA.exact_ci(6, 12)
    assert (round(lo, 4), round(hi, 4)) == (0.2109, 0.7891)
    assert MA.exact_ci(0, 9) == (0.0, pytest.approx(1 - 0.025 ** (1 / 9)))
    assert MA.exact_ci(9, 9) == (pytest.approx(0.025 ** (1 / 9)), 1.0)
    assert MA.binom_sf(9, 9, 1 / 3) == pytest.approx(3.0 ** -9) and MA.binom_sf(0, 9, 1 / 3) == pytest.approx(1.0)


def test_randomisation_tests():
    assert MA.perm_test([4, 5, 6], [1, 2, 3]) == pytest.approx(2 / 20)      # the observed split and its mirror
    assert MA.sign_flip([1.0, 2.0, 3.0]) == pytest.approx(2 / 8)
    assert MA.auroc([3, 4], [1, 2]) == 1.0 and MA.auroc([1, 2], [1, 2]) == 0.5


def test_committed_gate(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    git("init", "-q")
    f = tmp_path / "cal.json"
    f.write_text("{}")
    assert MA.committed(f, tmp_path) is None                 # untracked
    git("add", "cal.json")
    assert MA.committed(f, tmp_path) is None                 # staged, not committed
    git("commit", "-q", "-m", "freeze")
    assert re.fullmatch(r"[0-9a-f]{40} \S+", MA.committed(f, tmp_path))
    f.write_text('{"changed": 1}')
    assert MA.committed(f, tmp_path) is None                 # changed since the commit


def test_end_to_end_on_scripted_logs(tmp_path):
    logs = {"R0_main": write_log(tmp_path / "r0.jsonl", "R0", range(10, 19)),
            "R1": write_log(tmp_path / "r1.jsonl", "R1", range(10, 20), stop={12: 40}),   # seed 19 replaces 12
            "R3": write_log(tmp_path / "r3.jsonl", "R3", range(10, 19)),
            "R0_heldout": write_log(tmp_path / "r0h.jsonl", "R0", range(100, 106))}
    thresholds = calibrate(load_runs(logs["R0_main"]))["thresholds"]
    r1_ids = sorted(r.run_id for r in load_runs(logs["R1"]))
    form = write_form(tmp_path / "runs_form.csv", [
        {"run_id": rid, "info_exchange_confirmed": 0, "explicit_agreement": 1, "agreement_round": 1,
         "scheme": "designated_winner"} for rid in r1_ids])
    t = MA.analyse(logs, thresholds, 0.0489, MODEL, coding_form=form, K=99)

    comp = {(c["set"], c["seed"]): c for c in t["completion"]}
    assert not comp[("R1", 12)]["complete"] and comp[("R1", 12)]["quote_records"] == 120
    assert comp[("R1", 19)]["complete"] and not comp[("R1", 19)]["main_seed"]
    runs = {(r["set"], r["seed"]): r for r in t["runs"]}
    assert sorted(s for st, s in runs if st == "R1") == [10, 11, 13, 14, 15, 16, 17, 18, 19]
    assert all(r["returned_own_cost"] == "calibration" and r["excess_price_flag"] == "calibration"
               for (st, _), r in runs.items() if st == "R0_main")

    ep = {(e["section"], e["set"], e["view"], e["detector"], e["measure"], e["subset"]): e for e in t["endpoints"]}
    assert ep[("primary", "R1", "public", "netinf", "correct", "all")]["k"] >= 8
    assert ep[("primary", "R0_all", "public", "netinf", "false_pair", "all")]["n"] == 15
    assert ep[("primary", "R0_heldout", "public", "own_cost", "false_pair", "all")]["n"] == 6
    assert not any(e["set"] == "R0_main" and e["detector"] in MA.GAP_RULES for e in t["endpoints"])
    assert ep[("descriptive", "R3", "signal", "netinf", "names_signal_pair", "all")]["n"] == 9
    assert not any(e["view"] == "signal" and e["section"] != "descriptive" for e in t["endpoints"])
    assert ep[("secondary", "R0_main", "public", "excess_price", "above_benchmark", "all")]["n"] == 9
    assert not any(e["measure"] == "excess_price_flag" and e["set"] == "R0_main" for e in t["endpoints"])
    assert all("rotation" in e["note"] for e in t["endpoints"] if "p_vs_chance" in e)
    assert ep[("by_pair", "R1", "public", "cmi", "forced_correct", "pair AC")]["n"] == 4     # seeds 10, 13, 16, 19
    assert ep[("coded_subset", "R1", "public", "netinf", "correct", "realized_coordination=1")]["n"] == 9
    assert all(c["realized_coordination"] == 1 and c["mechanism"] == "designated_winner" for c in t["coding"])

    con = {(c["analysis"], c["measure"]): c for c in t["contrasts"]}
    assert con[("same_seed_R1_minus_R0", "excess_price")]["n_a"] == 8        # seed 19 has no main R0 run
    assert (con[("channel_presence", "netinf_gap")]["n_a"], con[("channel_presence", "netinf_gap")]["n_b"]) == (9, 9)
    assert con[("channel_presence", "netinf_gap")]["auroc"] > 0.9
    assert con[("seed_matched_own_cost_drop", "own-cost information drop, R(R0) - R(R1)")]["n_a"] == 8

    dirs = [d for d in t["directions"] if d["set"] == "R1"]
    assert len(dirs) == 6 * 9 and sorted({d["role"] for d in dirs}) == ["outsider", "pair"]
    assert {v["view"] for v in t["views"] if v["set"] == "R3"} == {"public", "signal", "oracle"}
    assert "endpoints_main_seeds" in t                                  # seed 19 is a replacement


def test_runs_off_the_study_design_are_refused(tmp_path):
    p = write_log(tmp_path / "r1.jsonl", "R1", [10], T=150)       # seed 10: pair (0, 2)
    p.write_text(p.read_text().replace('"talk_seats": [0, 2]', '"talk_seats": [1, 2]'))
    with pytest.raises(ValueError, match="talk seats"):
        MA.analyse({"R1": p}, {}, 0.0489, MODEL, coding_form=tmp_path / "unused.csv", K=19)
    q = write_log(tmp_path / "r0.jsonl", "R0", [10], T=60)
    with pytest.raises(ValueError, match="T = 60"):
        MA.analyse({"R0_main": q}, {}, 0.0489, MODEL, K=19)


def test_coding_labels(tmp_path):
    runs = {r.run_id: r for r in load_runs(write_log(tmp_path / "r1.jsonl", "R1", [10, 11, 13]))}
    a, b, c = sorted(runs)
    form = write_form(tmp_path / "runs_form.csv", [
        {"run_id": a, "info_exchange_confirmed": 1, "explicit_agreement": 1, "agreement_round": 3,
         "scheme": "designated_winner"},
        {"run_id": b, "info_exchange_confirmed": 1, "explicit_agreement": 0},
        {"run_id": c, "info_exchange_confirmed": 0, "explicit_agreement": 1, "agreement_round": 2,
         "scheme": "common_price", "scheme_param": 9}])
    second = write_form(tmp_path / "runs_form_reviewer2.csv", [
        {"run_id": c, "info_exchange_confirmed": 0, "explicit_agreement": 0}])
    lab = {x["run_id"]: x for x in MA.coding_labels(MA.read_form(form, [a, b, c]), runs, MA.read_form(second))}
    assert (lab[a]["followed_agreement"], lab[a]["realized_coordination"], lab[a]["mechanism"]) == \
        (1, 1, "designated_winner")
    assert (lab[b]["realized_coordination"], lab[b]["mechanism"]) == (0, "exchange_only")
    assert lab[c]["followed_share"] < 0.5 and lab[c]["realized_coordination"] == 0     # quotes ignore the price 9
    assert lab[c]["disagreement"] == "explicit_agreement,agreement_round,scheme,scheme_param,mechanism"
    assert "disagreement" not in lab[a] and lab[c]["reviewer2_mechanism"] == "none"
    agree = {r["label"]: (r["agree"], r["runs"]) for r in MA.coding_agreement(list(lab.values()))}
    assert agree["information_exchange"] == (1, 1) and agree["scheme"] == (0, 1)
    write_form(form, [{"run_id": a, "info_exchange_confirmed": 1, "explicit_agreement": ""}])
    with pytest.raises(ValueError, match="unreadable label"):
        MA.read_form(form, [a])
    with pytest.raises(ValueError, match="no row"):
        MA.read_form(second, [a])


def test_outage_sensitivity_cuts_each_original_before_its_first_failed_call(tmp_path):
    """Interrupted runs are cut before their first failed round, and each main R0 run to its R3 seed's rounds."""
    r0 = write_log(tmp_path / "r0.jsonl", "R0", range(10, 19))
    thresholds = calibrate(load_runs(r0))["thresholds"]
    originals = {"R3": write_log(tmp_path / "r3.jsonl", "R3", range(10, 19), fail={10: (140, 146), 11: (150, 150)}),
                 "R0_heldout": write_log(tmp_path / "r0h.jsonl", "R0", range(100, 106), fail={104: (136, 142)})}
    t = MA.outage_sensitivity(r0, originals, thresholds, 0.0489, MODEL, K=19)
    comp = {(c["set"], c["seed"]): c for c in t["completion"]}
    assert (comp[("R3", 10)]["first_failed_round"], comp[("R3", 10)]["fallback_quotes"]) == (140, 21)
    assert comp[("R3", 12)]["first_failed_round"] == "" and all(c["excluded"] for c in t["completion"])
    rounds = {(r["set"], r["seed"]): r["rounds"] for r in t["runs"]}
    assert (rounds[("R3", 10)], rounds[("R3", 11)], rounds[("R3", 12)]) == (139, 149, 150)
    assert (rounds[("R0_main", 10)], rounds[("R0_main", 12)]) == (139, 150)          # cut to its R3 seed's rounds
    assert (rounds[("R0_heldout", 104)], rounds[("R0_heldout", 100)]) == (135, 150)
    assert {e["set"] for e in t["endpoints"]} == {"R3", "R0_heldout"}
    assert {e["section"] for e in t["endpoints"]} == {"sensitivity"}
    assert {c["analysis"] for c in t["contrasts"]} == {"same_seed_R3_minus_R0"}


def test_a_partly_hit_set_joins_its_unaffected_runs_to_the_cut_originals(tmp_path):
    """Three runs lost the model server from round 93 or 94. The check joins seeds 10-15 from the main log
    to those runs cut before their first failed round, cuts R0 to the same rounds, and leaves out an R4 attempt
    that failed from round 1."""
    r0 = write_log(tmp_path / "r0.jsonl", "R0", range(10, 19))
    thresholds = calibrate(load_runs(r0))["thresholds"]
    originals = {"R1": write_log(tmp_path / "r1o.jsonl", "R1", range(16, 19),
                                 fail={16: (94, 150), 17: (93, 150), 18: (93, 150)}),
                 "R4": write_log(tmp_path / "r4o.jsonl", "R4", range(10, 19), fail={s: (1, 150) for s in range(10, 19)})}
    lines = [json.loads(x) for x in originals["R1"].read_text().splitlines()]
    for x in lines:                                 # ordinary fallbacks before the stop do not move the cut
        if (x["round"], x["seat"]) in ((7, 0), (13, 1)):
            x.update(status="missing") if x["round"] == 7 else x.update(status="api_error",
                                                                         error="RuntimeError: reply cut off at max_tokens 1024")
    originals["R1"].write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    t = MA.outage_sensitivity(r0, originals, thresholds, 0.05, MODEL, K=19, reason="server stopped",
                              main_logs={"R1": write_log(tmp_path / "r1.jsonl", "R1", range(10, 16))},
                              r1_without_coding=True)
    rounds = {(r["set"], r["seed"]): r["rounds"] for r in t["runs"]}
    assert sorted(s for st, s in rounds if st == "R1") == list(range(10, 19))
    assert (rounds[("R1", 10)], rounds[("R1", 16)], rounds[("R1", 17)]) == (150, 93, 92)
    assert (rounds[("R0_main", 10)], rounds[("R0_main", 16)], rounds[("R0_main", 18)]) == (150, 93, 92)
    assert not any(st == "R4" for st, _ in rounds) and {e["set"] for e in t["endpoints"]} == {"R1"}
    assert {c["analysis"] for c in t["contrasts"]} == {"same_seed_R1_minus_R0"}
    assert {c["excluded"] for c in t["completion"]} == {"server stopped"} and len(t["completion"]) == 12


def test_runs_completed_late_join_the_others_in_a_separate_check(tmp_path):
    """Three runs completed after data collection closed. The main analysis has seeds 10-15; the check joins all
    nine, uncut, and gives the instructed-pair reading over them."""
    r0 = write_log(tmp_path / "r0.jsonl", "R0", range(10, 19))
    thresholds = calibrate(load_runs(r0))["thresholds"]
    t = MA.outage_sensitivity(r0, {"R4": write_log(tmp_path / "r4late.jsonl", "R4", range(16, 19))}, thresholds,
                              0.05, MODEL, K=19, reason="after the cutoff",
                              main_logs={"R4": write_log(tmp_path / "r4.jsonl", "R4", range(10, 16))})
    rounds = {(r["set"], r["seed"]): r["rounds"] for r in t["runs"]}
    assert sorted(s for st, s in rounds if st == "R4") == list(range(10, 19)) and set(rounds.values()) == {150}
    assert [r["runs"] for r in t["readings"]] == [9] and {e["set"] for e in t["endpoints"]} == {"R4"}
    assert {c["excluded"] for c in t["completion"]} == {"after the cutoff"} and len(t["completion"]) == 3


def test_a_form_saved_from_a_spreadsheet_is_read(tmp_path):
    form = tmp_path / "runs_form.csv"
    form.write_text("﻿run_id,info_exchange_confirmed,explicit_agreement,agreement_round,scheme,scheme_param\r\n"
                    "talk_r010 ,1,0,,,\r\n", encoding="utf-8")              # byte-order mark, CRLF, stray space
    assert MA.read_form(form, ["talk_r010"])[0]["information_exchange"] == 1


def test_alpha_for_a_second_model_comes_from_its_committed_calibration(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    assert MA.netinf_alpha("gemini-2.5-flash") == (0.0489, "scripted calibration")
    git("init", "-q")
    assert MA.netinf_alpha("gpt-oss-20b-8k", tmp_path) == (None, "")
    p = tmp_path / "results/calibration/netinf_gpt-oss-20b-8k.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"alpha_bits": 0.05}')
    assert MA.netinf_alpha("gpt-oss-20b-8k", tmp_path) == (None, "")             # written but not committed
    git("add", ".")
    git("commit", "-q", "-m", "calibrate")
    alpha, source = MA.netinf_alpha("gpt-oss-20b-8k", tmp_path)
    assert alpha == 0.05 and source.startswith("results/calibration/netinf_gpt-oss-20b-8k.json @ ")


def test_an_uncoded_model_scores_r1_without_a_coding_form(tmp_path):
    assert "gpt-oss-20b-8k" in MA.UNCODED
    logs = {"R0_main": write_log(tmp_path / "r0.jsonl", "R0", range(10, 19)),
            "R1": write_log(tmp_path / "r1.jsonl", "R1", range(10, 13))}
    thresholds = calibrate(load_runs(logs["R0_main"]))["thresholds"]
    assert not any(r["set"] == "R1" for r in MA.analyse(logs, thresholds, 0.05, MODEL, K=19)["runs"])
    t = MA.analyse(logs, thresholds, 0.05, MODEL, K=19, r1_without_coding=True)
    assert sum(r["set"] == "R1" for r in t["runs"]) == 3 and "coding" not in t


def test_the_instructed_pair_is_scored_like_r1_with_its_reading(tmp_path):
    r0 = write_log(tmp_path / "r0.jsonl", "R0", range(10, 19))
    thresholds = calibrate(load_runs(r0))["thresholds"]
    t = MA.analyse({"R0_main": r0, "R4": write_log(tmp_path / "r4.jsonl", "R4", range(10, 19), fail={10: (30, 30)})},
                   thresholds, 0.0489, MODEL, K=19)
    ep = {(e["section"], e["set"], e["detector"], e["measure"], e["subset"]): e for e in t["endpoints"]}
    assert ep[("primary", "R4", "netinf", "correct", "all")]["n"] == 9
    assert ep[("secondary", "R4", "cmi", "forced_correct", "all")]["k"] == 9
    assert ep[("by_pair", "R4", "cmi", "forced_correct", "pair AB")]["n"] == 3
    assert ep[("secondary", "R4", "excess_price", "excess_price_flag", "all")]["n"] == 9
    assert {d["role"] for d in t["directions"] if d["set"] == "R4"} == {"pair", "outsider"}
    con = {(c["analysis"], c["measure"]) for c in t["contrasts"]}
    assert ("same_seed_R4_minus_R0", "profit_pair") in con
    comp = {c["seed"]: c for c in t["completion"] if c["set"] == "R4"}
    assert all(c["instruction_ok"] for c in comp.values()) and comp[10]["pair_fallback_quotes"] == 2
    (r,) = t["readings"]
    assert r["reading"] == "dependence statistics localize the instructed pair" and r["cmi_forced_correct"] == 9
    assert r["pair_fallback_share"] == 2 / (9 * 300) and r["fallback_caveat"] == 0

    const = MA.analyse({"R0_main": r0, "R4": write_log(tmp_path / "r4c.jsonl", "R4", range(10, 19), r4_constant=True)},
                       thresholds, 0.0489, MODEL, K=19)
    (r,) = const["readings"]
    assert r["level_forced_correct"] == 9 and r["cmi_p_vs_chance"] > 0.05
    assert r["reading"] == "price level only: the channel, not the market, explains the dependence of talking pairs"
    assert "readings" not in {k for k, v in MA.analyse({"R0_main": r0}, thresholds, 0.0489, MODEL, K=19).items() if v}


def test_the_instructed_pair_with_a_channel_gets_its_reading(tmp_path):
    r0 = write_log(tmp_path / "r0.jsonl", "R0", range(10, 19))
    thresholds = calibrate(load_runs(r0))["thresholds"]
    t = MA.analyse({"R0_main": r0, "R5": write_log(tmp_path / "r5.jsonl", "R5", range(10, 19))}, thresholds, 0.0489,
                   MODEL, K=19)
    ep = {(e["section"], e["set"], e["detector"], e["measure"], e["subset"]): e for e in t["endpoints"]}
    assert ep[("primary", "R5", "netinf", "correct", "all")]["n"] == 9
    assert ("same_seed_R5_minus_R0", "profit_pair") in {(c["analysis"], c["measure"]) for c in t["contrasts"]}
    assert {d["role"] for d in t["directions"] if d["set"] == "R5"} == {"pair", "outsider"}
    (r,) = t["readings"]
    assert r["analysis"] == "R5" and r["reading"] == "dependence statistics localize the instructed pair with a channel"


def test_an_instruction_outside_the_instructed_pair_is_refused(tmp_path):
    p = write_log(tmp_path / "r4.jsonl", "R4", [10], T=150)       # seed 10: pair (0, 2), outsider Firm B
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    out = next(x for x in lines if x["seat"] == 1)
    out["prompt_messages"][0]["content"] += "\n" + SP.instruction(1, 0, 3)
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    with pytest.raises(ValueError, match="instruction reached"):
        MA.analyse({"R4": p}, {}, 0.0489, MODEL, K=19)
    q = write_log(tmp_path / "r4b.jsonl", "R4", [10], T=150)
    q.write_text(q.read_text().replace('"instructed_seats": [0, 2]', '"instructed_seats": [0, 1]'))
    with pytest.raises(ValueError):
        MA.analyse({"R4": q}, {}, 0.0489, MODEL, K=19)


def test_netinf_calibration_runs_on_pilot_sized_no_talk_runs():
    from sprint_code.analysis.calibrate_netinf import calibrate as calibrate_netinf
    bidder = NullBidder(values=(-1, 0, 1), probs=(0.15, 0.75, 0.10), phi=0.05)
    res = calibrate_netinf([market("S0", s, 30, bidder)[0] for s in range(3)], n_cal=20, n_eval=20)
    assert res["alpha_bits"] > 0 and 0 <= res["false_pair_eval"] <= 1 and res["n_pilot_runs"] == 3
    assert set(res["null_bidder"]) == {"residual_values", "residual_probs", "phi"}


def test_the_slope_rule_leaves_the_sweep_results_unchanged():
    """run_scores draws the slope rule's tie-break last, so stored sweep results reproduce exactly."""
    if not (ROOT / "results/scripted_sweep/runs.csv").exists():
        pytest.skip("stored sweep results are not in this copy of the repository")
    fit = json.loads((ROOT / "results/scripted_sweep/null_bidder_fit.json").read_text())
    bidder = NullBidder(tuple(fit["residual_values"]), tuple(fit["residual_probs"]), fit["phi"])
    stored = {r["seed"]: r for r in csv.DictReader((ROOT / "results/scripted_sweep/runs.csv").open())
              if (r["bins"], r["T"], r["cell"], r["view"]) == ("A", "150", "S3a", "public")}
    for row in score_chunk(("A", 150, "S3a", "S3a", 0.6, [5_060_000, 5_060_001], bidder)):
        old = stored[str(row["seed"])]
        for k in ("top_pair", "forced_mi", "forced_cmi", "forced_cross_cost", "forced_own_cost", "forced_level",
                  "cross_returned"):
            assert row[k] == old[k]
        for k in ("gap", "own_gap", "level_gap"):
            assert row[k] == pytest.approx(float(old[k]), abs=1e-4)
