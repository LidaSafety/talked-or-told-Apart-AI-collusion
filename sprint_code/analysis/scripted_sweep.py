"""Bins, net-influence threshold and run-length sweep on scripted markets.

    .venv/bin/python -m sprint_code.analysis.scripted_sweep [--workers 8] [--quick]

Fits the null bidder on no-talk Gemini quotes only, scores scripted markets with the detectors,
calibrates net influence's threshold on S0 calibration runs, checks the cross-cost nulls for Type-I error under
feedback, and applies the selection rules for bins and run length. No pair-talk or shared-signal LLM data are read. Writes results/scripted_sweep/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sprint_code.cost_response.cost_info import cross_cost_test, pool_runs
from sprint_code.detectors.info import Bins
from sprint_code.evaluation.pair_localisation import PAIRS, run_scores
from sprint_code.rfq_logs import load_runs
from sprint_code.scripted_controls.sweep_markets import NullBidder, fit_phi, market, outcome_moments, residual_pmf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/scripted_sweep"
NO_TALK = [ROOT / "external/dealer_market/data/runs/rfq/no_talk_rfq_n3_gemini-2.5-flash.jsonl",
           ROOT / "results/sprint_runs/pilot/no_talk_R0_no_talk_gemini-2.5-flash.jsonl"]
BINS = {"A": Bins(bid=(4, 8), cost=(3,), state=(3,)),
        "B": Bins(bid=(4, 7), cost=(3,), state=(3,)),
        "C": Bins(bid=(3, 5, 7), cost=(3,), state=(3,))}
TS = (60, 120, 150, 240)
PS = (0.2, 0.4, 0.6, 0.8, 1.0)
BURN_IN, K, ALPHA_CROSS, STRESS_PHI = 20, 199, 0.05 / 3, 0.5
UNPAIRED = ("S0", "S1", "S2")


def cells(quick: bool) -> list[tuple]:
    """(cell, arm, p, runs, seed base). S0 calibration and evaluation runs use disjoint seeds."""
    n = (lambda k: max(6, k // 20)) if quick else (lambda k: k)
    out = [("S0cal", "S0", 1.0, n(300), 1_000_000), ("S0", "S0", 1.0, n(300), 2_000_000),
           ("S1", "S1", 1.0, n(200), 3_000_000), ("S2", "S2", 1.0, n(200), 4_000_000),
           ("S4", "S4", 1.0, n(100), 7_000_000)]
    for arm, base in (("S3a", 5_000_000), ("S3b", 6_000_000)):
        out += [(arm, arm, p, n(100), base + int(round(p * 10)) * 10_000) for p in PS]
    return out


def pair_name(pr) -> str:
    return "" if pr is None else "".join("ABC"[k] for k in pr)


def score_chunk(task: tuple) -> list[dict]:
    bins_name, T, cell, arm, p, seeds, bidder = task
    rows = []
    for seed in seeds:
        run, truth = market(arm, seed, T, bidder, p)
        for view in (("public", "signal") if arm in ("S1", "S2") else ("public",)):
            rng = np.random.default_rng([seed, T, ord(bins_name), len(view)])
            r = run_scores(run, rng, bins=BINS[bins_name], burn_in=BURN_IN, alpha_cross=ALPHA_CROSS, K=K, view=view)
            g = r["detail"]["gamma"]
            ranked = sorted(PAIRS, key=lambda pr: g[pr], reverse=True)
            rows.append({"bins": bins_name, "T": T, "cell": cell, "arm": arm, "p": p, "seed": seed, "view": view,
                         "truth": pair_name(truth), "signal_pair": pair_name(PAIRS[seed % 3]) if arm == "S2" else "",
                         "top_pair": pair_name(ranked[0]), "gap": g[ranked[0]] - g[ranked[1]],
                         **{f"forced_{d}": pair_name(r["forced"][d]) for d in ("mi", "cmi", "cross_cost", "own_cost", "level")},
                         "cross_returned": pair_name(r["returned"]["cross_cost"]),
                         "own_gap": r["gap"]["own_cost"], "level_gap": r["gap"]["level"]})
    return rows


def typeI_chunk(task: tuple) -> list[dict]:
    """Per-direction cross-cost p-values (CRT and shuffle) on no-coordination runs at T = 150."""
    bins_name, variant, seeds, bidder = task
    rows = []
    for seed in seeds:
        run, _ = market("S0", seed, 150, bidder)
        pooled = pool_runs([run], first_round=BURN_IN + 1, bins=BINS[bins_name])
        rng = np.random.default_rng([seed, ord(bins_name), 1])
        for i in range(3):
            for j in range(3):
                if i != j:
                    t = cross_cost_test(pooled, i, j, rng, K=K, bins=BINS[bins_name], nulls=("crt", "shuffle"))
                    rows.append({"bins": bins_name, "variant": variant, "seed": seed, "i": i, "j": j,
                                 "p_crt": t["p_X_crt"], "p_shuffle": t["p_X_shuffle"]})
    return rows


def chunks(seq, size):
    return [seq[k:k + size] for k in range(0, len(seq), size)]


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for r in rows for k in r))   # paired and unpaired cells have different columns
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})


def summarise(rows: list[dict]) -> tuple[list[dict], dict]:
    """Calibrate net influence on S0cal; detection and false-pair rates per bins, T, arm, p and view."""
    alpha, cal_own, cal_level = {}, {}, {}
    for b in BINS:
        for T in TS:
            cal = [r for r in rows if r["bins"] == b and r["T"] == T and r["cell"] == "S0cal"]
            alpha[(b, T)] = float(np.quantile([r["gap"] for r in cal], 0.95, method="higher"))
            cal_own[(b, T)] = np.array([r["own_gap"] for r in cal])
            cal_level[(b, T)] = np.array([r["level_gap"] for r in cal])
    out = []
    keys = sorted({(r["bins"], r["T"], r["cell"], r["p"], r["view"]) for r in rows if r["cell"] != "S0cal"})
    for b, T, cell, p, view in keys:
        rs = [r for r in rows if (r["bins"], r["T"], r["cell"], r["p"], r["view"]) == (b, T, cell, p, view)]
        a = alpha[(b, T)]
        net = [r["top_pair"] if r["gap"] >= a and r["gap"] > 0 else "" for r in rs]
        # own-cost / level: expected firing over random sets of 9 calibration runs, P(D > max of 9) = F(D-)^9
        fire_own = [float(np.mean(cal_own[(b, T)] < r["own_gap"])) ** 9 for r in rs]
        fire_lvl = [float(np.mean(cal_level[(b, T)] < r["level_gap"])) ** 9 for r in rs]
        row = {"bins": b, "T": T, "cell": cell, "p": p, "view": view, "n": len(rs), "netinf_alpha": a}
        if cell in UNPAIRED:
            row.update({"netinf_false_pair": np.mean([x != "" for x in net]),
                        "cross_false_pair": np.mean([r["cross_returned"] != "" for r in rs]),
                        "own_false_pair": np.mean(fire_own), "level_false_pair": np.mean(fire_lvl)})
            if cell == "S2":
                row["netinf_names_signal_pair"] = np.mean([x == r["signal_pair"] for x, r in zip(net, rs)])
        else:
            hit = lambda x, r: x == r["truth"]
            row.update({"netinf_correct": np.mean([hit(x, r) for x, r in zip(net, rs)]),
                        "cross_correct": np.mean([hit(r["cross_returned"], r) for r in rs]),
                        "own_correct": np.mean([f * hit(r["forced_own_cost"], r) for f, r in zip(fire_own, rs)]),
                        "level_correct": np.mean([f * hit(r["forced_level"], r) for f, r in zip(fire_lvl, rs)]),
                        **{f"forced_acc_{d}": np.mean([hit(r[f"forced_{d}"], r) for r in rs])
                           for d in ("mi", "cmi", "cross_cost", "own_cost", "level")}})
        out.append(row)
    return out, {f"{b}_T{T}": v for (b, T), v in alpha.items()}


def choose(power: list[dict]) -> dict:
    """Selection rules for bins and run length."""
    def score(b, col, arms, ps, T=150):
        rs = [r for r in power if r["bins"] == b and r["T"] == T and r["view"] == "public"
              and r["cell"] in arms and r["p"] in ps]
        return float(np.mean([r[col] for r in rs]))

    def pick(col, arms):
        s = {b: score(b, col, arms, (0.4, 0.6, 0.8)) for b in BINS}
        best = max(s.values())
        near = [b for b in BINS if s[b] >= best - 0.02]
        fewest = min(len(BINS[b].bid) for b in near)
        return max((b for b in near if len(BINS[b].bid) == fewest), key=s.get), s

    dep, dep_scores = pick("netinf_correct", ("S3a", "S3b"))
    cost, cost_scores = pick("cross_correct", ("S3a",))
    t_scores = {T: score(dep, "netinf_correct", ("S3a", "S3b"), (0.6,), T) for T in TS}
    qualifying = [T for T in TS if t_scores[T] >= 0.8]
    T_main = min(qualifying) if qualifying else 150
    return {"bins_bid_dependence": dep, "scores_bid_dependence": dep_scores,
            "bins_cost_responsiveness": cost, "scores_cost_responsiveness": cost_scores,
            "netinf_power_p06_by_T": t_scores, "T_main": T_main,
            "T_rule": "shortest T with mean net-influence detection >= 0.8 on S3a and S3b at p = 0.6; else 150",
            "T_needs_budget_decision": T_main > 150}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="few runs per cell; for a pipeline check only")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    no_talk = [run for f in NO_TALK for run in load_runs(f)]
    values, probs = residual_pmf(no_talk)
    observed = outcome_moments(no_talk)
    phi, grid = fit_phi(observed, values, probs)
    bidder, stress = NullBidder(values, probs, phi), NullBidder(values, probs, STRESS_PHI)
    fit = {"no_talk_runs": len(no_talk), "residual_values": values, "residual_probs": probs,
           "observed": observed, "phi": phi, "stress_phi": STRESS_PHI, "phi_grid": grid}
    (OUT / "null_bidder_fit.json").write_text(json.dumps(fit, indent=2))
    print(f"null bidder: phi = {phi} from {len(no_talk)} no-talk Gemini runs")

    tasks = [(b, T, cell, arm, p, ch, bidder) for b in BINS for T in TS
             for cell, arm, p, n, base in cells(a.quick) for ch in chunks(list(range(base, base + n)), 25)]
    nI = 12 if a.quick else 300
    ttasks = [(b, variant, ch, bd) for b in BINS
              for variant, bd, base in (("fitted", bidder, 2_000_000), ("stress", stress, 8_000_000))
              for ch in chunks(list(range(base, base + nI)), 25)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for part in ex.map(score_chunk, tasks) for r in part]
        trows = [r for part in ex.map(typeI_chunk, ttasks) for r in part]
    rows.sort(key=lambda r: (r["bins"], r["T"], r["cell"], r["p"], r["seed"], r["view"]))
    write_csv(OUT / "runs.csv", rows)
    write_csv(OUT / "typeI_cross_cost.csv", trows)

    power, alpha = summarise(rows)
    write_csv(OUT / "power.csv", power)
    typeI = {}
    for b in BINS:
        for variant in ("fitted", "stress"):
            ts = [r for r in trows if r["bins"] == b and r["variant"] == variant]
            per_run = {}
            for r in ts:
                per_run.setdefault(r["seed"], {})[(r["i"], r["j"])] = r
            fp = {null: float(np.mean([any(d[(i, j)][f"p_{null}"] <= ALPHA_CROSS and d[(j, i)][f"p_{null}"] <= ALPHA_CROSS
                                           for i, j in PAIRS) for d in per_run.values()])) for null in ("crt", "shuffle")}
            typeI[f"{b}_{variant}"] = {"runs": len(per_run),
                                       **{f"reject05_{null}": float(np.mean([r[f"p_{null}"] <= 0.05 for r in ts]))
                                          for null in ("crt", "shuffle")},
                                       **{f"false_pair_{null}": v for null, v in fp.items()}}
    decision = choose(power)
    summary = {"quick": a.quick, "alpha_netinf": alpha, "typeI_cross_cost": typeI, "decision": decision,
               "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["decision"], indent=2, default=float))
    print(f"done in {summary['seconds']} s; wrote {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
