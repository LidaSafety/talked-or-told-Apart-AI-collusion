"""Run the repeated two-dealer market-making game with LLM dealers.

    # smoke (2 runs, T=5, no --confirm needed)
    .venv/bin/python scripts/run_market.py --condition both --runs 1 --T 5 --out-dir data/runs/smoke
    # cost estimate only, no API calls
    .venv/bin/python scripts/run_market.py --condition both --runs 20 --T 30 --estimate-only
    # full design (needs --confirm)
    .venv/bin/python scripts/run_market.py --condition both --runs 20 --T 30 --confirm
    # an arm (decision 2026-10-16, src/arms.py), e.g. a smoke of the spread cap
    .venv/bin/python scripts/run_market.py --arm spread_cap --runs 1 --T 6 --out-dir data/runs/smoke/arms

Before any API call the runner (1) runs the test suite and refuses to start if
it fails, (2) prints a token-cost estimate, (3) refuses more than 2 runs in
total without --confirm. Every agent call is appended to
<out-dir>/<condition>_<model>.jsonl (arms: <condition>_<arm>_<model>.jsonl)
as soon as its round clears.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bots, market, prompts  # noqa: E402
from src.arms import ARMS, SWAP_MODEL, V1_ARM, Arm, event_round  # noqa: E402
from src.clients import PRICES, cost_usd, get_client  # noqa: E402
from src.parse import (FALLBACK_SPREAD, parse_instruction, parse_message, parse_report,  # noqa: E402
                       parse_spread, posted_spread)
from src.schema import validate  # noqa: E402

MAX_RUNS_WITHOUT_CONFIRM = 2
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# Used only when no logged calls exist for the model yet (replaced by measured
# means once a smoke has run). Output includes the rationale; thinking is billed as output.
DEFAULT_TOKENS = {"quote": {"output": 120, "thinking": 700}, "message": {"output": 50, "thinking": 500}}
CHARS_PER_TOKEN_DEFAULT = 4.0


def git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                                    cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True
    return commit, dirty


def make_config(condition: str, model: str, T: int, temperature: float, thinking: dict,
                arm: Arm = V1_ARM) -> dict:
    cfg = {
        "condition": condition, "model": model, "T": T, "temperature": temperature,
        "A": market.A, "B": market.B, "C": market.C, "grid": [market.S_MIN, arm.s_max],
        "prompt_version": prompts.prompt_version(arm=arm), "ticker": prompts.TICKER,
        "history_window": arm.history_window, "message_max_words": arm.message_max_words,
        "fallback_spread": arm.s_max, "thinking_config": thinking,
    }
    if not arm.is_v1:  # absent for v1 runs so their config hash is unchanged
        cfg.update({"arm": arm.name, "discount": arm.discount, "n_dealers": arm.n,
                    "show_rival": arm.show_rival, "known_T": arm.known_T,
                    "strip_numbers": arm.strip_numbers, "outside_spread": arm.game.outside_spread,
                    "designation_every": arm.designation_every, "swap": arm.swap,
                    "swap_model": SWAP_MODEL.get(model) if arm.swap == "model" else None,
                    "challenge": arm.challenge,
                    "event_round": event_round(T) if (arm.swap or arm.challenge or arm.shock) else None})
        if arm.talk_seats:  # seeded pair (decision 2026-10-22, eighth); absent elsewhere so hashes are unchanged
            cfg["talk_seats"] = list(arm.talk_seats)
        if arm.game.info_value is not None:  # Glosten-Milgrom arm (decision 2026-10-17)
            cfg.update({"executor": arm.executor} if arm.executor else {})
            cfg.update({"info_value": arm.game.info_value,
                        "informed_orders": arm.game.informed_orders})
            if arm.name != "gm":  # second GM decision: keys absent for gm so its hash is unchanged
                cfg.update({"hide_d": arm.hide_d, "hide_d_seats": list(arm.hide_d_seats),
                            "shock": arm.shock, "message_fee": arm.message_fee,
                            "state_k": arm.state_k, "irrelevant_ticks": arm.irrelevant_ticks,
                            "no_cot": arm.no_cot, "informed_wording": arm.informed_wording,
                            "inventory_cost": arm.game.inventory_cost})
    return cfg


def chash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def append(path: Path, records: list[dict]) -> None:
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(str(path), threading.Lock())
    with lock:
        with path.open("a") as f:
            for r in records:
                validate(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def call(client, msgs, temperature):
    try:
        rep = client.chat(msgs, temperature=temperature)
        return rep, None
    except Exception as e:  # failures are data
        return None, f"{type(e).__name__}: {str(e)[:300]}"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_one(run_id: str, condition: str, model: str, T: int, temperature: float,
            out_path: Path, commit: str, dirty: bool, arm: Arm = V1_ARM,
            prefix: list[dict] | None = None, arm_at=None, bot_at=None, extra: dict | None = None) -> dict:
    """One run. Fork mode (decision 2026-10-20, forks): `prefix` is the replayed history of a
    finished source run (rounds 1..F, never re-logged); play starts at round F + 1. `arm_at(t)`
    gives the arm whose rules apply in round t (default: `arm`); `bot_at(t)` gives a scripted
    policy for seat 1 in round t, or None for the LLM. `extra` is added to every record."""
    n, talk = arm.n, condition == "talk"
    # gm_no_cot (decision 2026-10-17, seventh): thinking off, Gemini only (get_client refuses others)
    client = get_client(model, thinking_off=True) if arm.no_cot else get_client(model)
    # Each seat: (kind, client-or-policy, model id). A swap replaces seat 1 at the event round.
    seats: list[tuple] = [("llm", client, model)] * n
    since: list[int | None] = [None] * n
    ev = event_round(T) if (arm.swap or arm.challenge or arm.shock) else None
    surcharge: list[int] = [0] * n          # inventory arm: set from the previous round
    cfg = make_config(condition, model, T, temperature, client.thinking_config, arm)
    base = {"run_id": run_id, "condition": condition, "model": model, "temperature": temperature,
            "config_hash": chash(cfg), "config": cfg,
            "git_commit": commit, "git_dirty": dirty, "analysis": "exploratory",
            "prompt_version": prompts.prompt_version(arm=arm), "T": T, "discount": arm.discount,
            "arm": arm.name, **(extra or {})}
    history: list[dict] = [dict(h) for h in (prefix or [])]
    start = len(history) + 1
    pool = ThreadPoolExecutor(max_workers=n)
    n_fallback = 0
    designated: int | None = None

    def seat_base(i):
        kind, obj, mid = seats[i]
        tc = obj.thinking_config if kind == "llm" else {"note": "scripted bot, no model call"}
        return {**base, "thinking_config": tc, "agent": kind if kind == "llm" else mid,
                "seat_model": mid}

    llm_seat1 = seats[1] if n > 1 else None
    for t in range(start, T + 1):
        arm_t = arm_at(t) if arm_at else arm
        if bot_at is not None:
            pol = bot_at(t)
            seats[1] = ("bot", pol[0], pol[1]) if pol else llm_seat1
        if ev is not None and t == ev and arm.swap:
            if arm.swap == "bot":
                seats[1] = ("bot", bots.profitable_undercut(floor=market.benchmarks_for(arm.game).nash_spread),
                            "bot:profitable_undercut")
            elif arm.swap == "model":
                m2 = SWAP_MODEL[model]
                seats[1] = ("llm", get_client(m2), m2)
            elif arm.swap == "fresh":
                since[1] = ev
        msgs_this_round: list[str | None] = [None] * n
        if talk:
            # seats without a channel (gm_n4_pair: Firms C and D) make no message call and log no message record
            speaks = [arm_t.can_talk(i) for i in range(n)]
            mprompts = [prompts.message_prompt(i, t, history, arm=arm_t, T=T, since_round=since[i])
                        if seats[i][0] == "llm" and speaks[i] else [] for i in range(n)]
            results = list(pool.map(
                lambda i: call(seats[i][1], mprompts[i], temperature) if seats[i][0] == "llm" and speaks[i]
                else (None, None), range(n)))
            mrecs = []
            for i, (rep, err) in enumerate(results):
                if not speaks[i]:
                    continue
                is_bot = seats[i][0] != "llm"
                pm = parse_message(rep.text if rep else None, arm.message_max_words, arm.strip_numbers)
                if is_bot:
                    pm["message_status"] = "message_none"
                elif err:
                    pm["message_status"] = "api_error"
                msgs_this_round[i] = pm["message"]
                mrecs.append({**seat_base(i), "round": t, "seat": i, "call_type": "message",
                              "prompt_hash": chash(mprompts[i]), "prompt_messages": mprompts[i],
                              "reply_text": rep.text if rep else "",
                              "reasoning": rep.reasoning if rep else "",
                              "usage": rep.usage if rep else {}, "error": err,
                              "message_sent": pm["message"], "message_status": pm["message_status"],
                              "message_words": pm["message_words"],
                              "message_truncated": pm["message_truncated"],
                              "numbers_removed": pm["numbers_removed"], "timestamp": _now()})
            append(out_path, mrecs)
        challenged = [arm.challenge and ev is not None and t == ev and i == 0 for i in range(n)]
        # message fee (GM paid talk): charged for every message actually sent this round
        fees = [arm.message_fee if (talk and arm.message_fee and msgs_this_round[i]) else 0
                for i in range(n)]
        # shock arm: seat 1's quote at the event round is forced one tick below D (no model call)
        forced = [arm.shock and ev is not None and t == ev and i == 1 for i in range(n)]
        qkind = ["forced" if forced[i] else seats[i][0] for i in range(n)]
        instr: list[dict | None] = [None] * n
        if arm_t.executor:   # decision 2026-10-20 (executor): desk head instructs, executor quotes
            sprompts = [prompts.strategist_prompt(i, t, history, talk, msgs_this_round, arm_t, T=T)
                        if qkind[i] == "llm" else [] for i in range(n)]
            sres = list(pool.map(lambda i: call(seats[i][1], sprompts[i], temperature) if qkind[i] == "llm"
                                 else (None, None), range(n)))
            irecs = []
            for i, (rep, err) in enumerate(sres):
                if qkind[i] != "llm":
                    continue
                pi = parse_instruction(rep.text if rep else None)
                if err:
                    pi["instruction_status"] = "api_error"
                instr[i] = pi
                irecs.append({**seat_base(i), "round": t, "seat": i, "call_type": "instruction",
                              "prompt_hash": chash(sprompts[i]), "prompt_messages": sprompts[i],
                              "reply_text": rep.text if rep else "", "reasoning": rep.reasoning if rep else "",
                              "usage": rep.usage if rep else {}, "error": err, **pi, "timestamp": _now()})
            append(out_path, irecs)
            qprompts = [prompts.executor_prompt(i, t, instr[i]["instruction"] if instr[i] else None, arm_t)
                        if qkind[i] == "llm" else [] for i in range(n)]
        else:
            qprompts = [prompts.quote_prompt(i, t, history, talk, msgs_this_round, arm=arm_t, T=T,
                                             since_round=since[i], challenge=challenged[i],
                                             surcharge_now=surcharge[i])
                        if qkind[i] == "llm" else [] for i in range(n)]
        results = list(pool.map(
            lambda i: call(seats[i][1], qprompts[i], temperature) if qkind[i] == "llm"
            else (None, None), range(n)))
        parsed, spreads = [], []
        for i, (rep, err) in enumerate(results):
            if qkind[i] == "forced":
                f_s = arm.game.info_value - 1
                parsed.append({"parsed_spread": f_s, "status": "ok", "n_spread_lines": 0})
                spreads.append(f_s)
            elif seats[i][0] == "llm":
                p = parse_spread(rep.text if rep else None, arm_t.s_max)
                if err:
                    p["status"] = "api_error"
                parsed.append(p)
                spreads.append(posted_spread(p, arm_t.s_max))
            else:
                b_s = min(arm_t.s_max, int(seats[i][1](i, history)))
                parsed.append({"parsed_spread": b_s, "status": "ok", "n_spread_lines": 0})
                spreads.append(b_s)
        orders, profits = market.clear(spreads, game=arm_t.game, designated=designated,
                                       surcharge=surcharge if arm.game.inventory_cost else None)
        qrecs = []
        for i, (rep, err) in enumerate(results):
            n_fallback += parsed[i]["status"] != "ok"
            rivals = [j for j in range(n) if j != i]
            qrecs.append({**seat_base(i), "round": t, "seat": i, "call_type": "quote",
                          "prompt_hash": chash(qprompts[i]), "prompt_messages": qprompts[i],
                          "reply_text": rep.text if rep else "",
                          "reasoning": rep.reasoning if rep else "",
                          "usage": rep.usage if rep else {}, "error": err,
                          "parsed_spread": parsed[i]["parsed_spread"], "posted_spread": spreads[i],
                          "status": parsed[i]["status"], "n_spread_lines": parsed[i]["n_spread_lines"],
                          "orders": float(orders[i]), "profit": float(profits[i]),
                          "rival_spread": min(spreads[j] for j in rivals),
                          "rival_spreads": [spreads[j] for j in rivals],
                          "message_received": (msgs_this_round[rivals[0]] if talk and n == 2 else None),
                          "messages_received": ([msgs_this_round[j] for j in rivals] if not arm_t.talk_seats
                                                else [msgs_this_round[j] for j in arm_t.peers(i)] or None)
                          if talk else None,
                          "designated": designated, "challenged": challenged[i],
                          "forced": forced[i], "surcharge": surcharge[i], "message_fee_paid": fees[i],
                          "net_profit": float(profits[i]) - fees[i], "round_arm": arm_t.name,
                          "fallback_spread": arm_t.s_max,
                          **({"instruction": instr[i]["instruction"] if instr[i] else None,
                              "executor_report": parse_report(rep.text if rep else None)} if arm_t.executor else {}),
                          **({"agent": "forced:one_below_D",
                              "thinking_config": {"note": "forced quote, no model call"}} if forced[i] else {}),
                          "timestamp": _now()})
        append(out_path, qrecs)
        history.append({"round": t, "spreads": spreads, "orders": orders, "profits": profits,
                        "messages": list(msgs_this_round), "designated": designated,
                        "fees": fees, "surcharge": list(surcharge)})
        if arm.game.inventory_cost:  # the sole filler pays the inventory cost next round
            sf = market.sole_filler(orders)
            surcharge = [arm.game.inventory_cost if i == sf else 0 for i in range(n)]
        k = arm.designation_every
        if k and t % k == 0:  # re-award for the next k rounds; equal means keep
            means = [sum(h["spreads"][i] for h in history[-k:]) / k for i in range(n)]
            best = min(means)
            leaders = [i for i, m_ in enumerate(means) if m_ == best]
            if len(leaders) == 1:
                designated = leaders[0]
        print(f"  {run_id} round {t}: spreads {spreads} statuses "
              f"{[p['status'] for p in parsed]}", flush=True)
    pool.shutdown()
    return {"run_id": run_id, "spreads": [h["spreads"] for h in history], "n_fallback": n_fallback}


# ---------------------------------------------------------------- cost estimate

def _logged_calls(model: str) -> list[dict]:
    recs = []
    for p in (ROOT / "data" / "runs").rglob(f"*_{model}.jsonl"):
        for line in p.open():
            r = json.loads(line)
            if r.get("usage", {}).get("input_tokens"):
                recs.append(r)
    return recs


def estimate(model: str, conditions: list[str], runs: int, T: int,
             discount: float | None = None, arm: Arm | None = None) -> dict:
    """Input tokens: render every prompt the design will send (history filled with
    dummy rounds of the right length) and convert characters to tokens with the
    ratio measured on logged calls. Output/thinking: measured means per call type."""
    logged = _logged_calls(model)
    if logged:
        chars = sum(sum(len(m["content"]) for m in r["prompt_messages"]) for r in logged)
        ratio = chars / sum(r["usage"]["input_tokens"] for r in logged)
        per = {}
        for ct in ("quote", "message"):
            rs = [r for r in logged if r["call_type"] == ct]
            if rs:
                per[ct] = {"output": sum(r["usage"]["output_tokens"] for r in rs) / len(rs),
                           "thinking": sum(r["usage"].get("thinking_tokens", 0) for r in rs) / len(rs)}
            else:
                per[ct] = DEFAULT_TOKENS[ct]
        source = f"measured on {len(logged)} logged calls"
    else:
        per = DEFAULT_TOKENS
        if model == "deepseek-chat":  # deepseek-chat is a non-reasoning model
            per = {k: {**v, "thinking": 0} for k, v in DEFAULT_TOKENS.items()}
        ratio, source = CHARS_PER_TOKEN_DEFAULT, "defaults (no logged calls; thinking 0 for deepseek-chat)"
    if arm is None:
        arm = prompts._arm(None, discount)
    n = arm.n
    dummy_msg = " ".join(["word"] * min(30, arm.message_max_words))
    tot = {"input": 0.0, "output": 0.0, "thinking": 0.0, "calls": 0}
    for cond in conditions:
        talk = cond == "talk"
        hist: list[dict] = []
        for t in range(1, T + 1):
            for seat in range(n):
                kinds = [("quote", prompts.quote_prompt(seat, t, hist, talk, [dummy_msg] * n,
                                                        arm=arm, T=T))]
                if talk and arm.can_talk(seat):
                    kinds.append(("message", prompts.message_prompt(seat, t, hist, arm=arm, T=T)))
                for ct, msgs in kinds:
                    tot["input"] += sum(len(m["content"]) for m in msgs) / ratio * runs
                    tot["output"] += per[ct]["output"] * runs
                    tot["thinking"] += per[ct]["thinking"] * runs
                    tot["calls"] += runs
            s0 = min(arm.s_max, market.BENCH.joint_spread)
            hist.append({"round": t, "spreads": [s0] * n, "orders": [22.0] * n,
                         "profits": [132.0] * n, "messages": [dummy_msg] * n, "designated": None})
    tot["usd"] = cost_usd(model, tot["input"], tot["output"], tot["thinking"])
    tot["source"] = source
    tot["chars_per_token"] = ratio
    return tot


def print_estimate(model, conditions, runs, T, arm=V1_ARM):
    e = estimate(model, conditions, runs, T, arm=arm)
    if arm.executor:   # two calls per seat per round instead of one quote call: an upper-bound scale
        e = {**e, **{k: e[k] * 2 for k in ("calls", "input", "output", "thinking", "usd")}}
        print("  (executor arm: quote-call estimate doubled for the desk-head call)")
    print(f"COST ESTIMATE  model={model}  conditions={conditions}  runs/condition={runs}  T={T}"
          f"  arm={arm.name}")
    print(f"  calls {e['calls']:,}  input {e['input']:,.0f} tok  output {e['output']:,.0f} tok  "
          f"thinking {e['thinking']:,.0f} tok")
    print(f"  prices (USD/1M) {PRICES[model]}  ->  about ${e['usd']:.2f}")
    print(f"  token basis: {e['source']}; chars/token {e['chars_per_token']:.2f}")
    return e


def existing_runs(path: Path, T: int) -> tuple[set[str], set[str]]:
    """(complete, incomplete) run ids already in `path` for this horizon.

    A run is complete when both seats have a quote record for every round 1..T.
    RESUME POLICY: complete runs count toward --runs and are skipped. An
    incomplete run (process killed, machine slept) is NOT continued: its records
    stay in the file as data, analysis excludes it from the measures and counts
    it, and a fresh run id is started in its place.
    """
    if not path.exists():
        return set(), set()
    quotes: dict[str, set] = {}
    seats: dict[str, int] = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("T") != T:
            continue
        q = quotes.setdefault(r["run_id"], set())
        seats[r["run_id"]] = (r.get("config") or {}).get("n_dealers", 2)
        if r["call_type"] == "quote":
            q.add((r["round"], r["seat"]))
    complete = {rid for rid, q in quotes.items()
                if q >= {(t, i) for t in range(1, T + 1) for i in range(seats[rid])}}
    return complete, set(quotes) - complete


def plan_jobs(out_dir: Path, conditions: list[str], model: str, T: int, target: int,
              arm: Arm = V1_ARM):
    jobs, report = [], []
    tag = "" if arm.is_v1 else f"_{arm.name}"
    for cond in conditions:
        path = out_dir / f"{cond}{tag}_{model}.jsonl"
        complete, incomplete = existing_runs(path, T)
        taken = complete | incomplete
        if path.exists():
            taken |= {json.loads(l)["run_id"] for l in path.open()}
        need = max(0, target - len(complete))
        report.append(f"  {cond}: {len(complete)} complete, {len(incomplete)} incomplete "
                      f"(kept as data, not resumed), launching {need} to reach {target}")
        k = 0
        for _ in range(need):
            while (rid := f"{cond}{tag}_{model}_T{T}_r{k:03d}") in taken:
                k += 1
            taken.add(rid)
            jobs.append((rid, cond, path))
    return jobs, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["no_talk", "talk", "both"], default="both")
    ap.add_argument("--model", default="gemini-2.5-flash", choices=sorted(PRICES))
    ap.add_argument("--runs", type=int, default=1,
                    help="TARGET number of complete runs per condition in the output file; "
                         "complete runs already there are skipped (resumable)")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--arm", choices=sorted(ARMS), default="v1", help="src/arms.py")
    ap.add_argument("--discount", choices=sorted(market.DISCOUNT_LEVELS), default=None,
                    help="shorthand for --arm d<level>")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-dir", default="data/runs")
    ap.add_argument("--workers", type=int, default=4, help="runs in parallel")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    arm = ARMS[f"d{a.discount}"] if a.discount else ARMS[a.arm]
    conditions = ["no_talk", "talk"] if a.condition == "both" else [a.condition]
    skipped = [c for c in conditions if c not in arm.conditions]
    conditions = [c for c in conditions if c in arm.conditions]
    if skipped:
        print(f"arm {arm.name} runs only in {arm.conditions}; skipping {skipped}")
    out_dir = ROOT / a.out_dir
    if not arm.is_v1 and Path(a.out_dir).resolve() == (ROOT / "data" / "runs").resolve():
        print("REFUSING: arm runs go in their own directory (e.g. data/runs/arms), "
              "so the v1 analysis never mixes them in.")
        return 2
    jobs, report = plan_jobs(out_dir, conditions, a.model, a.T, a.runs, arm)
    print(f"PLAN  {a.out_dir}  target {a.runs} complete runs per condition, T={a.T}, arm {arm.name}")
    print("\n".join(report))
    n_by_cond = {c: sum(j[1] == c for j in jobs) for c in conditions}
    print("Estimate for the full target:")
    print_estimate(a.model, conditions, a.runs, a.T, arm)
    for c, n in n_by_cond.items():
        if n:
            print(f"Estimate for what this invocation launches ({c}):")
            print_estimate(a.model, [c], n, a.T, arm)
    if a.estimate_only:
        return 0
    if not jobs:
        print("Nothing to launch: target already met.")
        return 0
    if len(jobs) > MAX_RUNS_WITHOUT_CONFIRM and not a.confirm:
        print(f"REFUSING: {len(jobs)} runs to launch; more than {MAX_RUNS_WITHOUT_CONFIRM} "
              f"needs --confirm.")
        return 2
    if not a.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
        if r.returncode != 0:
            print("REFUSING: tests failed; no API call made.")
            return 3
    commit, dirty = git_info()
    if dirty:
        print("WARNING: src/ or scripts/ has uncommitted changes; recorded as git_dirty=true.")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"START {dt.datetime.now().isoformat(timespec='seconds')}  {len(jobs)} runs, "
          f"workers {a.workers}, commit {commit[:7]}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, rid, cond, a.model, a.T, a.temperature, path, commit, dirty,
                          arm)
                for rid, cond, path in jobs]
        for f in futs:
            try:
                res = f.result()
                print(f"DONE {res['run_id']}: fallbacks {res['n_fallback']}  spreads {res['spreads']}",
                      flush=True)
            except Exception as e:  # one run's crash must not hide the others
                print(f"RUN FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"END {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
