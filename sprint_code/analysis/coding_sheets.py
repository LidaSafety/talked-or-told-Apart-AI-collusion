"""Coding sheets for the blind realized-coordination coding of R1 runs.

    .venv/bin/python -m sprint_code.analysis.coding_sheets results/sprint_runs/main/talk_R1_pair_talk_<model>.jsonl

For each R1 run, writes one sheet with the rounds the coder reads: all of rounds 1-30, every later round
whose messages contain a number, and 20 further rounds drawn at random, seeded by run id. A sheet shows only the
messages and the pair members' quotes and costs, with automatic flags and empty code columns. A run-level form
collects the coder's labels. No detector output is read or written here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FIRST_ROUNDS, N_RANDOM, BURN_IN, EXCHANGE_RATE = 30, 20, 20, 0.10
CODES = ("COST", "PRICE", "ALLOCATE", "ACCEPT", "NONE")
FORM_FIELDS = ("info_exchange_confirmed", "explicit_agreement", "agreement_round", "scheme", "scheme_param",
               "other_scheme_check", "ambiguous", "notes")
SCHEMES = ("common_price", "designated_winner")   # other schemes: the coder writes the check before it is computed
SEATS = "ABC"


def load_r1(path: Path) -> dict[str, dict]:
    """run_id -> {T, pair, msg[t][seat], quote[t][seat], cost[t][seat]}; complete runs only."""
    runs: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        run = runs.setdefault(r["run_id"], {"T": r["T"], "pair": None, "msg": defaultdict(dict),
                                            "quote": defaultdict(dict), "cost": defaultdict(dict)})
        t, s = r["round"], r["seat"]
        if r["call_type"] == "message":
            run["msg"][t][s] = (r.get("message_sent") or "").strip()
        elif r["call_type"] == "quote":
            run["quote"][t][s] = r["posted_spread"]
            run["cost"][t][s] = r["cost"]
            run["pair"] = tuple(r["config"]["talk_seats"])
    return {rid: run for rid, run in runs.items()
            if all(len(run["quote"][t]) == 3 for t in range(1, run["T"] + 1))}


def cost_flag(text: str, cost: int) -> bool:
    """The message mentions 'cost' and the sender's current cost as a number."""
    return bool(re.search(r"\bcost", text, re.I)) and str(cost) in re.findall(r"\d+", text)


def select_rounds(run_id: str, T: int, numbered: set[int]) -> list[int]:
    """Rounds 1-30, later rounds with a number in a message, and 20 further rounds seeded by run id."""
    chosen = set(range(1, min(FIRST_ROUNDS, T) + 1)) | {t for t in numbered if t > FIRST_ROUNDS}
    rest = sorted(set(range(1, T + 1)) - chosen)
    rng = np.random.default_rng(int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16))
    chosen |= set(rng.choice(rest, size=min(N_RANDOM, len(rest)), replace=False).tolist())
    return sorted(chosen)


def sheet_rows(run_id: str, run: dict) -> list[dict]:
    i, j = run["pair"]
    numbered = {t for t in run["msg"] if any(re.search(r"\d", m) for m in run["msg"][t].values())}
    rows = []
    for t in select_rounds(run_id, run["T"], numbered):
        for s, partner in ((i, j), (j, i)):
            text = run["msg"][t].get(s, "")
            rows.append({"run_id": run_id, "round": t, "sender": SEATS[s], "message": text,
                         "flag_number": int(bool(re.search(r"\d", text))),
                         "flag_cost": int(cost_flag(text, run["cost"][t][s])),
                         "sender_cost": run["cost"][t][s], "sender_quote": run["quote"][t][s],
                         "partner_cost": run["cost"][t][partner], "partner_quote": run["quote"][t][partner],
                         **{c: "" for c in CODES}, "note": ""})
    return rows


def form_row(run_id: str, run: dict) -> dict:
    """Automatic run-level fields over all rounds, plus empty fields for the coder."""
    i, j = run["pair"]
    T = run["T"]
    sent = np.mean([any(run["msg"][t].get(s) and run["msg"][t][s].upper() != "NONE" for s in (i, j))
                    for t in range(1, T + 1)])
    later = range(BURN_IN + 1, T + 1)
    rate = np.mean([any(cost_flag(run["msg"][t].get(s, ""), run["cost"][t][s]) for s in (i, j)) for t in later])
    return {"run_id": run_id, "pair": SEATS[i] + SEATS[j], "messages_sent": round(float(sent), 4),
            "info_exchange_rate_auto": round(float(rate), 4), "info_exchange_auto": int(rate >= EXCHANGE_RATE),
            **{f: "" for f in FORM_FIELDS}}


def followed_share(quotes: np.ndarray, costs: np.ndarray, pair: tuple[int, int], scheme: str, param: float | None,
                   from_round: int) -> float:
    """Share of post-burn-in rounds from `from_round` (1-based) whose quotes match the recorded scheme.

    common_price: both pair members quote within one tick of `param`. designated_winner: in rounds where
    the pair's costs differ, the higher-cost member quotes above the lower-cost member.
    """
    i, j = pair
    t0 = max(from_round, BURN_IN + 1) - 1
    q, c = quotes[t0:], costs[t0:]
    if scheme == "common_price":
        ok = (np.abs(q[:, i] - param) <= 1) & (np.abs(q[:, j] - param) <= 1)
    elif scheme == "designated_winner":
        differ = c[:, i] != c[:, j]
        high, low = np.where(c[:, i] > c[:, j], i, j), np.where(c[:, i] > c[:, j], j, i)
        rows = np.arange(len(q))
        ok = (q[rows, high] > q[rows, low])[differ]
    else:
        raise ValueError(f"no automatic check for scheme {scheme!r}; the coder writes it in the sheet first")
    return float(np.mean(ok)) if len(ok) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="R1 runner log (JSONL)")
    ap.add_argument("--out", type=Path, default=None, help="default: results/coding/<log stem>/")
    a = ap.parse_args()
    out = a.out or Path("results/coding") / a.log.stem
    out.mkdir(parents=True, exist_ok=True)
    runs = load_r1(a.log)
    for rid, run in sorted(runs.items()):
        rows = sheet_rows(rid, run)
        with (out / f"{rid}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    forms = [form_row(rid, run) for rid, run in sorted(runs.items())]
    with (out / "runs_form.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(forms[0]))
        w.writeheader()
        w.writerows(forms)
    print(f"{len(runs)} complete R1 runs -> {out}/ (one sheet per run + runs_form.csv); no detector output included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
