"""Check: do fallback quotes drive the gpt-oss pair results?

    .venv/bin/python -m sprint_code.analysis.fallback_check --model gpt-oss-20b-8k

An unparsed or cut-off reply is played as the reserve quote of 12 in every arm. This rescores the
pair markets (R1, R4, R5) of the main logs with every round that contains a fallback quote removed, using the
main settings, and compares each forced choice and net-influence decision with the main analysis.
Removing a round makes the next round's state come from the round before it, so this is a sensitivity check, not
a replacement analysis. Writes results/main_analysis/<model>/checks/fallback_check.csv, outside the folder that
main_analysis rewrites.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sprint_code.analysis.main_analysis import (BURN_IN, ALPHA_CROSS, FOLDERS, K, LOGS, SETS, decision, netinf_alpha,
                                                pid, run_rng, write_csv)
from sprint_code.detectors.info import MAIN_BINS
from sprint_code.evaluation.pair_localisation import run_scores
from sprint_code.rfq_logs import load_runs

ROOT = Path(__file__).resolve().parents[2]
FORCED = ("mi", "cmi", "cross_cost", "own_cost", "level")


def fallback_rounds(path: Path) -> dict[str, set[int]]:
    """Per run id, the rounds (1-based) in which at least one quote fell back."""
    out: dict[str, set[int]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["call_type"] == "quote" and r.get("status") != "ok":
            out.setdefault(r["run_id"], set()).add(r["round"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-20b-8k")
    a = ap.parse_args()
    alpha, _ = netinf_alpha(a.model)
    primary = {r["run_id"]: r for r in csv.DictReader((ROOT / "results/main_analysis" / a.model / "runs.csv").open())}
    rows = []
    for set_name in ("R1", "R4", "R5"):
        arm, folder = SETS[set_name]
        path = ROOT / FOLDERS[folder] / f"{LOGS[arm]}_{a.model}.jsonl"
        if not path.exists():
            continue
        dropped = fallback_rounds(path)
        for run in load_runs(path):
            keep = np.array([t + 1 not in dropped.get(run.run_id, set()) for t in range(run.T)])
            clean = replace(run, quotes=run.quotes[keep], costs=run.costs[keep])
            s = run_scores(clean, run_rng(run.run_id), bins=MAIN_BINS, burn_in=BURN_IN, alpha=alpha,
                           alpha_cross=ALPHA_CROSS, K=K)
            p = primary[run.run_id]
            row = {"run_id": run.run_id, "set": set_name, "seed": run.run_index, "pair": p["pair"],
                   "rounds_dropped": int((~keep).sum()), "returned_netinf_primary": p["returned_netinf"],
                   "returned_netinf_check": decision(s["returned"]["netinf"])}
            for det in FORCED:
                row[f"forced_{det}_primary"] = p[f"forced_{det}"]
                row[f"forced_{det}_check"] = pid(s["forced"][det])
            rows.append(row)
    out = ROOT / "results/main_analysis" / a.model / "checks" / "fallback_check.csv"
    out.parent.mkdir(exist_ok=True)
    write_csv(out, rows)
    for set_name in ("R1", "R4", "R5"):
        rs = [r for r in rows if r["set"] == set_name]
        if not rs:
            continue
        print(f"{set_name}: {len(rs)} markets, {sum(r['rounds_dropped'] for r in rs)} rounds dropped")
        for det in FORCED:
            print(f"  forced {det:11s} primary {sum(r[f'forced_{det}_primary'] == r['pair'] for r in rs)}/{len(rs)}"
                  f"  without fallback rounds {sum(r[f'forced_{det}_check'] == r['pair'] for r in rs)}/{len(rs)}")
        print(f"  net influence  primary {sum(r['returned_netinf_primary'] == r['pair'] for r in rs)}/{len(rs)}"
              f"  without fallback rounds {sum(r['returned_netinf_check'] == r['pair'] for r in rs)}/{len(rs)}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
