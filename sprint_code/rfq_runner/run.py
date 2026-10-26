"""Run the three-bidder RFQ auction under arms R0-R5, adapted from the dealer-market scripts/run_rfq.py.

    # cost estimate only
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R1_pair_talk --runs 2 --T 30 --estimate-only
    # smoke (at most 2 runs without --confirm)
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R1_pair_talk --runs 1 --T 3 --out-dir results/sprint_runs/smoke
    # pilot
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R0_no_talk --runs 2 --T 30 --confirm
    # main study seeds: every arm uses seeds 10-18; held-out R0 uses 100-105 in its own folder
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R0_no_talk --runs 9 --first-seed 10 --T 150 --out-dir results/sprint_runs/main --confirm
    # self-hosted gpt-oss through Ollama (sprint_code/clients.py)
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R0_no_talk --model gpt-oss-20b-8k --temperature 0.7 --runs 1 --T 3 --out-dir results/sprint_runs/smoke
    # instructed pair without a channel
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R4_instructed_pair --model gpt-oss-20b-8k --temperature 0.7 --runs 9 --first-seed 10 --T 150 --out-dir results/sprint_runs/main --workers 3 --confirm
    # instructed pair with a channel
    .venv/bin/python -m sprint_code.rfq_runner.run --arm R5_instructed_talk --runs 9 --first-seed 10 --T 150 --out-dir results/sprint_runs/main --workers 9 --confirm

Same record format and guards as the dealer-market runner: the test suite must pass, a cost
estimate is printed before any call, more than 2 runs need --confirm, complete runs count toward --runs
and interrupted runs are kept as data and never resumed. Costs are the dealer-market common random numbers
(run k of every arm faces the same costs). Additions: `config.talk_seats` (R1 pair, R2 all), `config.signal_seats`
(R3), `config.instructed_seats` and `config.instruction` (R4, R5), `config.pair_id`, `config.seed`,
`config.upstream_sha`; per record `signal` and `signal_seen` (R3), `instructed` (R4, R5);
`messages_received` holds only the seat's peers' messages and is [] for the R1 outsider; a routing check
per quote that no message outside a seat's peers reaches its prompt (`routing_ok`).
Records go to <out-dir>/<condition>_<arm>_<model>.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .arms import ARMS, DEALER, SprintArm, draw_signal, pair_id
from . import prompts as SP

sys.path.insert(0, str(DEALER / "scripts"))
from run_market import (MAX_RUNS_WITHOUT_CONFIRM, _logged_calls, _now, call, chash,  # noqa: E402
                        existing_runs, DEFAULT_TOKENS, CHARS_PER_TOKEN_DEFAULT)
from src import rfq  # noqa: E402
from sprint_code.clients import PRICES, cost_usd, get_client  # noqa: E402  (dealer-market API clients + self-hosted)
from src.parse import parse_message, parse_quote  # noqa: E402
from src.schema import COMMON, MESSAGE, QUOTE, SchemaError, _check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAX_CONCURRENT_RUNS = 20   # never more than about 20 concurrent runs per provider


def git_info() -> tuple[str, bool, str]:
    def g(*a, cwd=ROOT):
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    return (g("rev-parse", "HEAD") or "unknown", bool(g("status", "--porcelain", "--", "sprint_code")),
            g("rev-parse", "HEAD", cwd=DEALER) or "unknown")


def run_index(run_id: str) -> int:
    return int(re.search(r"_r(\d+)$", run_id).group(1))


def make_config(condition: str, model: str, T: int, temperature: float, thinking: dict, arm: SprintArm,
                k: int, upstream: str) -> dict:
    g, p = arm.game, arm.pair(k)
    cfg = {"game": "rfq", "condition": condition, "model": model, "T": T, "temperature": temperature,
           "arm": arm.name, "sprint_arm": arm.label, "n_dealers": g.n, "reserve": g.reserve, "units": g.units,
           "costs": list(g.costs), "grid": [rfq.BID_MIN, g.reserve], "public_costs": False, "price_rule": g.price_rule,
           "disclosure": arm.disclosure, "history_window": arm.history_window,
           "message_max_words": arm.message_max_words,
           "prompt_version": f"{SP.SPRINT_PROMPT_VERSION}+{arm.name}", "fallback_spread": g.reserve,
           "thinking_config": thinking, "cost_draws": "sha256 common random numbers (dealer-market draw_costs)",
           "seed": k, "pair_id": pair_id(p), "upstream_repo": "dealer-market study",
           "upstream_sha": upstream}
    if condition == "talk":
        cfg["talk_seats"] = list(p) if arm.pair_talk else list(range(g.n))
    if arm.signal:
        cfg["signal_seats"] = list(p)
        cfg["signal_values"] = "uniform on 4..11, sha256 'rfq-signal|run|round', independent of costs"
    if arm.instructed:
        cfg["instructed_seats"] = list(p)
        cfg["instruction"] = SP.INSTRUCTION
    return cfg


def validate(rec: dict) -> dict:
    """The dealer-market field checks plus the fields of these arms (its validator only knows its own arms)."""
    _check(rec, COMMON)
    if rec["seat"] not in range(rec["config"]["n_dealers"]):
        raise SchemaError("seat out of range")
    if rec["call_type"] == "quote":
        _check(rec, QUOTE)
        _check(rec, {"cost": int, "costs": list, "price_paid": int})
    elif rec["call_type"] == "message":
        _check(rec, MESSAGE)
    else:
        raise SchemaError(f"bad call_type {rec['call_type']!r}")
    return rec


def append(path: Path, records: list[dict]) -> None:
    import json
    with path.open("a") as f:
        for r in records:
            validate(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def run_one(run_id: str, condition: str, model: str, T: int, temperature: float, out_path: Path,
            commit: str, dirty: bool, upstream: str, arm: SprintArm) -> dict:
    n, talk, k = arm.n, condition == "talk", run_index(run_id)
    client = get_client(model)
    cfg = make_config(condition, model, T, temperature, client.thinking_config, arm, k, upstream)
    base = {"run_id": run_id, "condition": condition, "model": model, "temperature": temperature,
            "config_hash": chash(cfg), "config": cfg, "git_commit": commit, "git_dirty": dirty,
            "analysis": "main", "prompt_version": cfg["prompt_version"], "T": T, "game": "rfq",
            "arm": arm.name, "thinking_config": client.thinking_config, "agent": "llm", "seat_model": model,
            "origin": "rfq_runner"}
    peers = [arm.peers(i, k, talk) for i in range(n)]
    seen = [arm.sees_signal(i, k) for i in range(n)]
    partners = [arm.partner(i, k) for i in range(n)]
    history: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=n)
    n_fallback = 0
    for t in range(1, T + 1):
        g = arm.game
        costs = rfq.draw_costs(k, t, n, g.costs)
        sig = draw_signal(k, t) if arm.signal else None
        sig_i = [sig if seen[i] else None for i in range(n)]
        msgs: list[str | None] = [None] * n
        if talk:
            speakers = [i for i in range(n) if peers[i]]
            mp = {i: SP.message_prompt(i, t, history, costs, arm, peers[i], sig_i[i], partners[i]) for i in speakers}
            res = dict(zip(speakers, pool.map(lambda i: call(client, mp[i], temperature), speakers)))
            recs = []
            for i in speakers:
                rep, err = res[i]
                pm = parse_message(rep.text if rep else None, arm.message_max_words)
                if err:
                    pm["message_status"] = "api_error"
                msgs[i] = pm["message"]
                recs.append({**base, "round": t, "seat": i, "call_type": "message", "prompt_hash": chash(mp[i]),
                             "prompt_messages": mp[i], "reply_text": rep.text if rep else "",
                             "reasoning": rep.reasoning if rep else "", "usage": rep.usage if rep else {},
                             "error": err, "message_sent": pm["message"], "message_status": pm["message_status"],
                             "message_words": pm["message_words"], "message_truncated": pm["message_truncated"],
                             "message_recipients": peers[i], "cost": costs[i],
                             **({"signal": sig, "signal_seen": seen[i]} if arm.signal else {}),
                             "timestamp": _now()})
            append(out_path, recs)
        qp = [SP.quote_prompt(i, t, history, costs, arm, peers[i], sig_i[i], msgs, partners[i]) for i in range(n)]
        # routing check: no message from outside a seat's peers reaches its prompt this round
        routing = [all(not msgs[j] or msgs[j] not in qp[i][1]["content"] for j in range(n)
                       if j != i and j not in peers[i]) for i in range(n)]
        res = list(pool.map(lambda i: call(client, qp[i], temperature), range(n)))
        parsed, bids = [], []
        for rep, err in res:
            pq = parse_quote(rep.text if rep else None, rfq.BID_MIN, g.reserve)
            if err:
                pq["status"] = "api_error"
            parsed.append(pq)
            bids.append(pq["parsed_spread"] if pq["status"] == "ok" else g.reserve)
        units, profits = rfq.clear(bids, costs, g)
        price = min(bids)
        recs = []
        for i, (rep, err) in enumerate(res):
            n_fallback += parsed[i]["status"] != "ok"
            rivals = [j for j in range(n) if j != i]
            recs.append({**base, "round": t, "seat": i, "call_type": "quote", "prompt_hash": chash(qp[i]),
                         "prompt_messages": qp[i], "reply_text": rep.text if rep else "",
                         "reasoning": rep.reasoning if rep else "", "usage": rep.usage if rep else {}, "error": err,
                         "parsed_spread": parsed[i]["parsed_spread"], "posted_spread": bids[i],
                         "status": parsed[i]["status"], "n_spread_lines": parsed[i]["n_spread_lines"],
                         "orders": float(units[i]), "profit": float(profits[i]),
                         "rival_spread": min(bids[j] for j in rivals), "rival_spreads": [bids[j] for j in rivals],
                         "message_received": None,
                         "messages_received": [msgs[j] for j in peers[i]] if talk else None,
                         "message_senders": peers[i] if talk else None,
                         "cost": costs[i], "costs": costs, "price_paid": price, "message_fee_paid": 0,
                         "net_profit": float(profits[i]), "round_arm": arm.name, "routing_ok": routing[i],
                         **({"signal": sig, "signal_seen": seen[i]} if arm.signal else {}),
                         **({"instructed": partners[i] is not None} if arm.instructed else {}),
                         "timestamp": _now()})
        append(out_path, recs)
        if not all(routing):
            raise RuntimeError(f"{run_id} round {t}: a message reached a seat outside its peers")
        history.append({"round": t, "costs": costs, "bids": bids, "units": units, "profits": profits,
                        "messages": list(msgs), "signal": sig, "price": price})
        print(f"  {run_id} round {t}: costs {costs} quotes {bids} signal {sig} "
              f"statuses {[p['status'] for p in parsed]}", flush=True)
    pool.shutdown()
    return {"run_id": run_id, "n_fallback": n_fallback}


def estimate(model: str, cond: str, runs: int, T: int, arm: SprintArm) -> dict:
    logged = _logged_calls(model)
    if logged:
        ratio = (sum(sum(len(m["content"]) for m in r["prompt_messages"]) for r in logged)
                 / sum(r["usage"]["input_tokens"] for r in logged))
        per = {}
        for ct in ("quote", "message"):
            rs = [r for r in logged if r["call_type"] == ct]
            per[ct] = ({"output": sum(r["usage"]["output_tokens"] for r in rs) / len(rs),
                        "thinking": sum(r["usage"].get("thinking_tokens", 0) for r in rs) / len(rs)}
                       if rs else DEFAULT_TOKENS[ct])
    else:
        ratio, per = CHARS_PER_TOKEN_DEFAULT, DEFAULT_TOKENS
    tot = {"input": 0.0, "output": 0.0, "thinking": 0.0, "calls": 0}
    talk, hist, dummy = cond == "talk", [], " ".join(["word"] * 30)
    for t in range(1, T + 1):
        costs = rfq.draw_costs(0, t, arm.n, arm.game.costs)
        sig = draw_signal(0, t) if arm.signal else None
        for s in range(arm.n):
            pe, si, pa = arm.peers(s, 0, talk), (sig if arm.sees_signal(s, 0) else None), arm.partner(s, 0)
            shapes = [("quote", SP.quote_prompt(s, t, hist, costs, arm, pe, si, [dummy] * arm.n, pa))]
            if pe:
                shapes.append(("message", SP.message_prompt(s, t, hist, costs, arm, pe, si, pa)))
            for ct, m in shapes:
                tot["input"] += sum(len(x["content"]) for x in m) / ratio * runs
                tot["output"] += per[ct]["output"] * runs
                tot["thinking"] += per[ct]["thinking"] * runs
                tot["calls"] += runs
        hist.append({"round": t, "costs": costs, "bids": [arm.game.reserve] * arm.n, "units": [5.0] * arm.n,
                     "profits": [40.0] * arm.n, "messages": [dummy] * arm.n, "signal": sig})
    tot["usd"] = cost_usd(model, tot["input"], tot["output"], tot["thinking"])
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--model", default="gemini-2.5-flash", choices=sorted(PRICES))
    ap.add_argument("--runs", type=int, default=1, help="target number of COMPLETE runs (seeds from --first-seed)")
    ap.add_argument("--first-seed", type=int, default=0,
                    help="seed of the first new run (main runs 10-18, held-out R0 100-105)")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-dir", default="results/sprint_runs/pilot")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--estimate-only", action="store_true")
    a = ap.parse_args()
    arm = ARMS[a.arm]
    cond = arm.conditions[0]
    if a.workers > MAX_CONCURRENT_RUNS:
        print(f"REFUSING: more than {MAX_CONCURRENT_RUNS} concurrent runs.")
        return 2
    out_dir = ROOT / a.out_dir
    path = out_dir / f"{cond}_{arm.name}_{a.model}.jsonl"
    complete, incomplete = existing_runs(path, a.T)
    taken = complete | incomplete
    jobs, k = [], a.first_seed
    for _ in range(max(0, a.runs - len(complete))):
        while (rid := f"{cond}_{arm.name}_{a.model}_T{a.T}_r{k:03d}") in taken:
            k += 1
        taken.add(rid)
        jobs.append(rid)
    print(f"PLAN  {a.out_dir}  arm {arm.name} ({cond}), target {a.runs} complete runs, T={a.T}: "
          f"{len(complete)} complete, {len(incomplete)} incomplete (kept, not resumed), launching {len(jobs)}")
    for rid in jobs:
        print(f"  {rid}: pair {pair_id(arm.pair(run_index(rid)))}")
    e = estimate(a.model, cond, a.runs, a.T, arm)
    print(f"COST ESTIMATE  calls {e['calls']:,}  input {e['input']:,.0f}  output {e['output']:,.0f}  "
          f"thinking {e['thinking']:,.0f}  ->  about ${e['usd']:.2f}  (prices {PRICES[a.model]})")
    if a.estimate_only or not jobs:
        return 0
    if len(jobs) > MAX_RUNS_WITHOUT_CONFIRM and not a.confirm:
        print(f"REFUSING: {len(jobs)} runs; more than {MAX_RUNS_WITHOUT_CONFIRM} needs --confirm.")
        return 2
    if subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT).returncode != 0:
        print("REFUSING: tests failed; no API call made.")
        return 3
    commit, dirty, upstream = git_info()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"START {dt.datetime.now().isoformat(timespec='seconds')}  {len(jobs)} runs, commit {commit[:7]}"
          f"{' (dirty)' if dirty else ''}, upstream {upstream[:7]}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, rid, cond, a.model, a.T, a.temperature, path, commit, dirty, upstream, arm)
                for rid in jobs]
        for f in futs:
            try:
                r = f.result()
                print(f"DONE {r['run_id']}: fallbacks {r['n_fallback']}", flush=True)
            except Exception as ex_:
                print(f"RUN FAILED: {type(ex_).__name__}: {ex_}", flush=True)
    print(f"END {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
