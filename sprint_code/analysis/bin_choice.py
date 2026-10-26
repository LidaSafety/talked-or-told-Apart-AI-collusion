"""Compare bid bins and public-state boundaries on no-talk data only.

    .venv/bin/python -m sprint_code.analysis.bin_choice

Data: the dealer-market rfq_n3 no-talk logs, R0 pilot runs if present, and primary-BNE bids on the design cost draws (theory reference). Never R1/R2/R3.
Writes results/bin_choice/state.csv and bids.csv.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from sprint_code.detectors.info import binned, entropy
from sprint_code.rfq_logs import load_runs
from sprint_code.rfq_runner.arms import DEALER  # noqa: F401  (dealer repo on sys.path)
from src import rfq  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/bin_choice"
STATES = {"8|9": (8,), "3|4": (3,), "4|5": (4,)}
BIDS = {"A 1-4/5-8/9-12": (4, 8), "B 1-4/5-7/8-12": (4, 7), "C 1-3/4-5/6-7/8-12": (3, 5, 7)}
COST_EDGES = (3,)
MAIN_RUNS, MAIN_T, MIN_CELL = 9, 120, 5


def sources() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """name -> list of (quotes (T, n), costs (T, n)) per run."""
    pre = DEALER / "data/runs/rfq"
    out = {}
    for model in ("claude-haiku-5-5", "gemini-2.5-flash"):
        f = pre / f"no_talk_rfq_n3_{model}.jsonl"
        if f.exists():
            out[f"dealer-market rfq_n3 no_talk {model}"] = [(r.quotes, r.costs) for r in load_runs(f)]
    for f in sorted((ROOT / "results/sprint_runs/pilot").glob("no_talk_R0_no_talk_*.jsonl")):
        runs = [r for r in load_runs(f) if r.quotes.shape[0] == 30]
        if runs:
            out[f"sprint R0 pilot {f.stem.split('R0_no_talk_')[1]}"] = [(r.quotes, r.costs) for r in runs]
    g = rfq.RfqGame(n=3)
    bne = []
    for k in range(200):
        c = np.array([rfq.draw_costs(k, t, 3) for t in range(1, 31)])
        bne.append((np.vectorize(lambda x: rfq.bne_quote(int(x), g))(c), c))
    out["theory primary BNE (200 design runs)"] = bne
    return out


def state_series(runs):
    """Previous round's price paid (minimum quote), rounds 2..T, pooled."""
    return np.concatenate([q.min(axis=1)[:-1] for q, _ in runs])


def occupancy_at_main(bi, bj, s) -> float:
    """Share of probability mass in (b_i, b_j, state) cells whose expected count reaches MIN_CELL at
    MAIN_RUNS x (MAIN_T - 1) samples."""
    code = (bi * 10 + bj) * 10 + s
    vals, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(p[p * MAIN_RUNS * (MAIN_T - 1) >= MIN_CELL].sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sources()
    srows, brows = [], []
    for name, runs in srcs.items():
        prev = state_series(runs)
        for sname, edges in STATES.items():
            st = binned(prev, edges)
            srows.append({"source": name, "boundary": sname, "n_rounds": len(st),
                          "state_entropy_bits": round(abs(entropy(st)), 4),
                          "smaller_bin_share": round(float(min(st.mean(), 1 - st.mean())), 4)})
        for bname, bedges in BIDS.items():
            q = np.concatenate([r[0] for r in runs])
            c = np.concatenate([r[1] for r in runs])
            b, cb = binned(q, bedges), binned(c, COST_EDGES)
            h = np.mean([entropy(b[:, i]) for i in range(3)])
            hc = np.mean([entropy(b[:, i], cb[:, i]) - entropy(cb[:, i]) for i in range(3)])
            row = {"source": name, "bins": bname, "H_bid_bits": round(float(h), 4),
                   "H_bid_given_costbin_bits": round(float(hc), 4)}
            for sname, edges in STATES.items():
                occ = []
                for i, j in combinations(range(3), 2):
                    bi = np.concatenate([binned(qq[1:, i], bedges) for qq, _ in runs])
                    bj = np.concatenate([binned(qq[1:, j], bedges) for qq, _ in runs])
                    occ.append(occupancy_at_main(bi, bj, binned(state_series(runs), edges)))
                row[f"occupancy_state_{sname}"] = round(float(np.mean(occ)), 4)
            brows.append(row)
    for fname, rows in (("state.csv", srows), ("bids.csv", brows)):
        with (OUT / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{fname}")
        for r in rows:
            print("  " + " | ".join(str(v) for v in r.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
