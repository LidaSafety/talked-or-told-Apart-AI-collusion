"""Per-run pair localisation for every detector.

For a three-bidder run, each detector gives a forced-choice pair and, where it has a decision rule, a
decision that may be NO_PAIR (None):

- mi          forced: argmax I(B_i; B_j). Comparator only, no NO_PAIR rule.
- cmi         forced: argmax I(B_i; B_j | S). Comparator only, no NO_PAIR rule.
- netinf      Bonjour net influence: the unique pair with both directed Gamma >= alpha.
- cross_cost  pair score min(excess X(i <- j), excess X(j <- i)); returns the unique pair whose two
              directions both have CRT p <= alpha_cross.
- own_cost    the two seats other than the most cost-responsive one; gap D = R_(3) - R_(2).
- level       the two seats with the highest mean quote; gap L = m_(2) - m_(1).

Gap rules (own_cost, level) are decided by calibrate_gaps against R0 runs. run_scores also gives the
secondary own-cost rule with own-cost slopes in place of R_i (forced["own_cost_slope"],
gap["own_cost_slope"]), outside DETECTORS.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from sprint_code.cost_response.cost_info import cross_cost_test, own_cost, pool_runs
from sprint_code.detectors.info import ALPHA, DEFAULT_BINS, Bins, binned, mi, net_influence

DETECTORS = ("mi", "cmi", "netinf", "cross_cost", "own_cost", "level")
GAP_DETECTORS = ("own_cost", "level")
PAIRS = ((0, 1), (0, 2), (1, 2))


def _outsider(pair: tuple[int, int]) -> int:
    return 3 - sum(pair)


def _pick(scores: dict, rng: np.random.Generator) -> tuple[int, int]:
    """Pair with the highest score; ties are broken at random so no seat order is favoured."""
    best = max(scores.values())
    tied = [p for p, s in scores.items() if s >= best - 1e-12]
    return tied[int(rng.integers(len(tied)))]


def run_scores(run, rng: np.random.Generator, bins: Bins = DEFAULT_BINS, burn_in: int = 0,
               alpha: float = ALPHA, alpha_cross: float = 0.05, K: int = 199, view: str = "public") -> dict:
    """Scores, forced-choice pairs and (except gap rules) decisions for one three-bidder run.

    burn_in rounds are dropped; round 1 is always dropped because it has no previous price. `view` sets
    the conditioning state of the bid-bid measures (cmi, netinf; see pool_runs). Cross-cost always uses
    the public state: an oracle state would condition on the very cost it tests.
    """
    if run.n != 3:
        raise ValueError("pair localisation is defined for three bidders")
    first = max(2, burn_in + 1)
    p = pool_runs([run], first_round=first, bins=bins)
    pv = p if view == "public" else pool_runs([run], first_round=first, bins=bins, view=view)
    b = binned(p.bids, bins.bid)
    forced, returned, gap = {}, {}, {}

    mi_s = {pr: mi(b[:, pr[0]], b[:, pr[1]]) for pr in PAIRS}
    forced["mi"] = _pick(mi_s, rng)

    ni = net_influence(b, pv.state, alpha)
    cmi_s = {pr: ni["gamma"][pr] for pr in PAIRS}
    forced["cmi"] = forced["netinf"] = _pick(cmi_s, rng)
    returned["netinf"] = ni["returned"]

    tests = {(i, j): cross_cost_test(p, i, j, rng, K=K, bins=bins, nulls=("crt",))
             for i in range(3) for j in range(3) if i != j}
    cc_s = {(i, j): min(tests[(i, j)]["X_excess_crt"], tests[(j, i)]["X_excess_crt"]) for i, j in PAIRS}
    forced["cross_cost"] = _pick(cc_s, rng)
    passing = [(i, j) for i, j in PAIRS
               if tests[(i, j)]["p_X_crt"] <= alpha_cross and tests[(j, i)]["p_X_crt"] <= alpha_cross]
    returned["cross_cost"] = passing[0] if len(passing) == 1 else None

    oc = [own_cost(p, i, bins) for i in range(3)]
    R = [o["R"] for o in oc]
    oc_s = {pr: R[_outsider(pr)] - max(R[pr[0]], R[pr[1]]) for pr in PAIRS}
    forced["own_cost"] = _pick(oc_s, rng)
    gap["own_cost"] = oc_s[forced["own_cost"]]

    m = p.bids.mean(axis=0)
    lv_s = {pr: min(m[pr[0]], m[pr[1]]) - m[_outsider(pr)] for pr in PAIRS}
    forced["level"] = _pick(lv_s, rng)
    gap["level"] = lv_s[forced["level"]]

    # drawn last, so every earlier pick uses the same random numbers as before this rule was added
    sl = [o["slope"] for o in oc]
    sl_s = {pr: sl[_outsider(pr)] - max(sl[pr[0]], sl[pr[1]]) for pr in PAIRS}
    forced["own_cost_slope"] = _pick(sl_s, rng)
    gap["own_cost_slope"] = sl_s[forced["own_cost_slope"]]

    detail = {"R": R, "slope": sl, "mean_quote": m.tolist(), "gamma": cmi_s, "Gamma": ni["Gamma"],
              "cross_X": {k: v["X"] for k, v in tests.items()},
              "cross_p_crt": {k: v["p_X_crt"] for k, v in tests.items()},
              "cross_excess": {k: v["X_excess_crt"] for k, v in tests.items()}}
    return {"run_id": run.run_id, "view": view, "forced": forced, "returned": returned, "gap": gap,
            "detail": detail}


def calibrate_gaps(results: list[dict], r0_ids: set[str]) -> None:
    """Decide the gap rules in place: return the forced pair only when the run's gap exceeds the gap
    of every R0 calibration run of the same model and run length (the caller groups runs).

    R0 runs are scored leave-one-out. If the R0 runs are exchangeable, the false-pair rate is at most
    1 / (1 + number of calibration runs). Without calibration runs every gap rule returns NO_PAIR.
    """
    for det in GAP_DETECTORS:
        r0 = {r["run_id"]: r["gap"][det] for r in results if r["run_id"] in r0_ids}
        for r in results:
            ref = [g for rid, g in r0.items() if rid != r["run_id"]]
            r["returned"][det] = r["forced"][det] if ref and r["gap"][det] > max(ref) else None


def summarise(results: list[dict], truth: dict[str, tuple[int, int] | None]) -> dict:
    """Per detector, over runs with a designated pair: forced-choice accuracy and the shares returning
    the correct pair, a wrong pair, or NO_PAIR. Over runs without one: the false-pair rate."""
    out = {}
    paired = [r for r in results if truth[r["run_id"]] is not None]
    unpaired = [r for r in results if truth[r["run_id"]] is None]
    decided = set().union(*(r["returned"] for r in results))
    for det in DETECTORS:
        s = {"n_paired": len(paired), "n_unpaired": len(unpaired)}
        if paired:
            s["forced_accuracy"] = float(np.mean([r["forced"][det] == truth[r["run_id"]] for r in paired]))
        if det in decided:
            if paired:
                dec = [r["returned"][det] for r in paired]
                s["correct"] = float(np.mean([d == truth[r["run_id"]] for d, r in zip(dec, paired)]))
                s["wrong_pair"] = float(np.mean([d is not None and d != truth[r["run_id"]]
                                                 for d, r in zip(dec, paired)]))
                s["no_pair"] = float(np.mean([d is None for d in dec]))
            if unpaired:
                s["false_pair"] = float(np.mean([r["returned"][det] is not None for r in unpaired]))
        out[det] = s
    return out
