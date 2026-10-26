"""Load RFQ quote logs into per-run arrays.

Reads the JSONL format written by the RFQ runner (the dealer-market scripts/run_rfq.py and
sprint_code.rfq_runner.run): one record per call, with
call_type "quote" carrying the seat's posted quote (`posted_spread`), its own cost (`cost`) and
every seat's cost this round (`costs`).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COSTS = (1, 2, 3, 4, 5, 6)   # the simulator's cost grid (src/rfq.py)


@dataclass
class Run:
    run_id: str
    arm: str
    condition: str
    model: str
    run_index: int
    quotes: np.ndarray        # (T, n) posted quotes, round order
    costs: np.ndarray         # (T, n) private costs
    status_ok: bool           # every quote parsed without fallback
    signal: np.ndarray | None = None   # (T,) R3 shared signal shown to the signal pair; None when absent
    config: dict | None = None         # the runner's config (talk_seats, signal_seats, pair_id, seed, ...)

    @property
    def T(self) -> int:
        return self.quotes.shape[0]

    @property
    def n(self) -> int:
        return self.quotes.shape[1]

    @property
    def price_paid(self) -> np.ndarray:
        return self.quotes.min(axis=1)


def design_cost(run_index: int, rnd: int, seat: int, costs: tuple[int, ...] = COSTS) -> int:
    """The simulator's common-random-number draw (copied formula from src/rfq.py draw_costs)."""
    h = hashlib.sha256(f"rfq-costs|{run_index}|{rnd}|{seat}".encode()).digest()
    return costs[int.from_bytes(h[:8], "big") % len(costs)]


def load_runs(path: Path, T_expected: int | None = None) -> list[Run]:
    """Complete runs from one JSONL file. A run is complete when every (round, seat) has a quote."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("call_type") == "quote":
            by_run[rec["run_id"]].append(rec)
    runs = []
    for run_id, recs in sorted(by_run.items()):
        T = T_expected or recs[0]["T"]
        n = len(recs[0]["costs"])
        quotes = np.full((T, n), -1, dtype=int)
        costs = np.full((T, n), -1, dtype=int)
        signal = np.full(T, -1, dtype=int) if any("signal" in r for r in recs) else None
        ok = True
        for r in recs:
            t, s = r["round"] - 1, r["seat"]
            if t >= T:
                continue
            quotes[t, s] = r["posted_spread"]
            costs[t, s] = r["cost"]
            if r["costs"][s] != r["cost"]:
                raise ValueError(f"{run_id} round {r['round']} seat {s}: cost {r['cost']} != costs[{s}]")
            if signal is not None and "signal" in r:
                if signal[t] not in (-1, r["signal"]):
                    raise ValueError(f"{run_id} round {r['round']}: seats disagree on the signal")
                signal[t] = r["signal"]
            ok &= r.get("status") == "ok"
        if (quotes < 0).any() or (costs < 0).any() or (signal is not None and (signal < 0).any()):
            continue
        runs.append(Run(run_id=run_id, arm=recs[0]["arm"], condition=recs[0]["condition"],
                        model=recs[0]["model"], run_index=int(run_id.rsplit("_r", 1)[1]),
                        quotes=quotes, costs=costs, status_ok=ok, signal=signal, config=recs[0].get("config")))
    return runs


def design_cost_match(run: Run) -> float:
    """Share of logged costs equal to the simulator's design draw; 1.0 confirms costs came from the design."""
    hits = [run.costs[t, s] == design_cost(run.run_index, t + 1, s)
            for t in range(run.T) for s in range(run.n)]
    return float(np.mean(hits))
