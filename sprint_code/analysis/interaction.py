"""The matched-seed two-by-two interaction of channel and instruction.

    .venv/bin/python -m sprint_code.analysis.interaction --model gemini-2.5-flash

R0 (no talk), R1 (pair talk), R4 (instructed pair) and R5 (instructed pair with talk) share their seeds,
so a seed faces the same costs and the same assigned pair in all four. For each seed and outcome Y,
delta = (Y_R5 - Y_R4) - (Y_R1 - Y_R0): how much more talk changes Y when the pair is instructed than when it is not.
The test is an exact two-sided sign flip over seeds.
Outcomes: price paid, efficiency, pair and outsider profit (as in the same-seed contrasts), and two dependence
statistics of the assigned pair's seats: conditional mutual information on the public state and the mean of its two
cross-cost excesses. Reads the main analysis tables; writes results/main_analysis/<model>/checks/interaction.csv and
interaction_by_seed.csv, outside the folder that main_analysis rewrites.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from sprint_code.analysis.main_analysis import write_csv

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("R0_main", "R1", "R4", "R5")
ECON = ("price_paid", "efficiency", "profit_pair", "profit_outsider")
DEPENDENCE = ("cmi_pair", "cross_cost_pair")


def sign_flip(d) -> float:
    """Exact two-sided sign-flip p-value, as in main_analysis, for differences of values read from its CSVs. Prices,
    efficiency and profits are averages over 130 rounds, so sign patterns can tie the observed sum exactly; with six
    decimals in the CSVs such ties differ by about 1e-6, so they are counted within 1e-4."""
    d = np.asarray(d, float)
    signs = np.array(list(product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(np.abs(signs @ d) >= abs(d.sum()) - 1e-4))


def outcomes_by_seed(runs, views, dirs) -> dict[str, dict[int, dict[str, float]]]:
    """Per arm and seed, the outcomes of the seed's assigned pair (R0: the pair the rotation assigns to that seed)."""
    gamma = {(v["set"], int(v["seed"])): v for v in views if v["view"] == "public"}
    cross: dict[tuple[str, int], dict[tuple[str, str], float]] = {}
    for d in dirs:
        cross.setdefault((d["set"], int(d["seed"])), {})[(d["bidder"], d["cost_of"])] = float(d["X_excess"])
    out: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    for r in runs:
        if r["set"] not in ARMS or r["main_seed"] != "True":
            continue
        seed, (i, j) = int(r["seed"]), r["seed_pair"]
        key = (r["set"], seed)
        row = {m: float(r[m]) for m in ECON}
        row["cmi_pair"] = float(gamma[key][f"gamma_{i}{j}"])
        row["cross_cost_pair"] = (cross[key][(i, j)] + cross[key][(j, i)]) / 2
        out[r["set"]][seed] = row
    return out


def interaction(by_arm) -> tuple[list[dict], list[dict]]:
    """Per-seed deltas and, per outcome, the two talk effects and their difference with sign-flip p-values."""
    seeds = sorted(set.intersection(*(set(by_arm[a]) for a in ARMS)))
    per_seed, summary = [], []
    for m in ECON + DEPENDENCE:
        y = {a: np.array([by_arm[a][s][m] for s in seeds]) for a in ARMS}
        talk, talk_told = y["R1"] - y["R0_main"], y["R5"] - y["R4"]
        delta = talk_told - talk
        per_seed += [{"measure": m, "seed": s, **{a: y[a][k] for a in ARMS}, "talk_effect": talk[k],
                      "talk_effect_instructed": talk_told[k], "delta": delta[k]} for k, s in enumerate(seeds)]
        summary.append({"measure": m, "seeds": len(seeds), **{f"mean_{a}": y[a].mean() for a in ARMS},
                        "talk_effect": talk.mean(), "p_talk_effect": sign_flip(talk),
                        "talk_effect_instructed": talk_told.mean(), "p_talk_effect_instructed": sign_flip(talk_told),
                        "delta": delta.mean(), "p_delta": sign_flip(delta), "seeds_delta_negative": int((delta < 0).sum()),
                        "test": "exact sign flip over seeds, two-sided"})
    return per_seed, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()
    src = ROOT / "results/main_analysis" / a.model
    read = lambda name: list(csv.DictReader((src / name).open()))  # noqa: E731
    by_arm = outcomes_by_seed(read("runs.csv"), read("views.csv"), read("directions.csv"))
    missing = [arm for arm in ARMS if not by_arm[arm]]
    if missing:
        print(f"No {', '.join(missing)} markets for {a.model} yet; nothing written.")
        return 1
    per_seed, summary = interaction(by_arm)
    out = src / "checks"
    out.mkdir(exist_ok=True)
    write_csv(out / "interaction_by_seed.csv", per_seed)
    write_csv(out / "interaction.csv", summary)
    for r in summary:
        print(f"{r['measure']:16s} {r['seeds']} seeds  talk {r['talk_effect']:+.3f} (p = {r['p_talk_effect']:.3f})  "
              f"talk when instructed {r['talk_effect_instructed']:+.3f} (p = {r['p_talk_effect_instructed']:.3f})  "
              f"delta {r['delta']:+.3f} (p = {r['p_delta']:.3f}; {r['seeds_delta_negative']} negative)")
    print(f"wrote {out.relative_to(ROOT)}/interaction.csv and interaction_by_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
