"""Per-run measures and the bootstrap. Shared by analysis/ and the bot tests,
so the numbers the tests pin are computed by the same code the paper uses.

Definitions (docs/design.md, Measures):
- window: the last WINDOW rounds (all rounds if the session is shorter);
- mean_spread: mean posted half-spread over both seats in the window;
- mean_market_spread: mean of the tightest spread (s_min) in the window;
- mean_profit: mean per-dealer profit per round in the window;
- delta: Calvano et al. (2020) profit gain from mean_profit. PRIMARY: against
  the most competitive equilibrium in undominated strategies. SECONDARY
  (`delta_most_comp`, always reported beside it): against the most
  competitive equilibrium of all. Decision 2026-10-15, after the smoke;
- share_at_joint: share of window rounds whose market spread s_min is at or
  above the joint-profit-maximising spread;
- undercuts: over the WHOLE session, the number of (round t >= 2, seat i)
  with s_i,t < s_j,t-1 and s_i,t < s_i,t-1, i.e. a dealer cut its own quote
  to below the rival's previous quote (N dealers: below the lowest rival quote).

Arms (decision 2026-10-16): `bench` is the benchmark set of the stage game the
arm is played under (src/market.py benchmarks_for); Δ is computed against it,
so a spread cap's Δ asks whether dealers still sit at the top of what the cap
allows. `mean_market_spread` (s_min, what retail pays) is the measure that is
comparable across arms. `event_measures` gives the pre/post comparison for
the swap and challenge arms.
"""

from __future__ import annotations

import numpy as np

from .market import BENCH, Benchmarks, delta

WINDOW = 10
# Glosten-Milgrom arm (decision 2026-10-17): a message "uses a risk word" if its
# lower-cased text contains any of these. Pre-registered in docs/design.md.
RISK_WORDS = ("inform", "risk", "loss", "lose", "losing", "protect", "safe", "cover", "value move")


def uses_risk_word(message: str | None) -> bool:
    low = (message or "").lower()
    return any(w in low for w in RISK_WORDS)
N_BOOT = 10_000
BOOT_SEED = 20261023


def run_measures(rounds: list[dict], bench: Benchmarks | None = None) -> dict:
    """rounds: [{round, spreads:[s0,s1,...], profits:[p0,p1,...]}, ...] sorted by round."""
    b = BENCH if bench is None else bench
    rounds = sorted(rounds, key=lambda r: r["round"])
    w = rounds[-WINDOW:]
    spreads = np.array([r["spreads"] for r in w], dtype=float)
    profits = np.array([r["profits"] for r in w], dtype=float)
    smin = spreads.min(axis=1)
    mean_profit = float(profits.mean())
    n = spreads.shape[1]
    undercuts = 0
    for prev, cur in zip(rounds, rounds[1:]):
        for i in range(n):
            rival_prev = min(prev["spreads"][j] for j in range(n) if j != i)
            if cur["spreads"][i] < rival_prev and cur["spreads"][i] < prev["spreads"][i]:
                undercuts += 1
    return {
        "n_rounds": len(rounds),
        "window": len(w),
        "mean_spread": float(spreads.mean()),
        "mean_market_spread": float(smin.mean()),
        "mean_profit": mean_profit,
        "delta": delta(mean_profit, bench=b),
        "delta_most_comp": delta(mean_profit, nash_profit=b.most_comp_nash_profit, bench=b),
        "share_at_joint": float((smin >= b.joint_spread).mean()),
        "undercuts": undercuts,
    }


PRE_ROUNDS = 5


def event_measures(rounds: list[dict], event: int, bench: Benchmarks | None = None,
                   seat: int = 0) -> dict:
    """Swap / challenge arms: the PRE_ROUNDS rounds before `event` against the rounds from
    `event` on. retention: share of post rounds whose market spread is at least the
    pre median; recovery: rounds after the event until the market spread is back at
    or above the pre median (None if never); seat_held: `seat` posted at the event
    round the same quote it posted the round before (the challenged seat is 0)."""
    b = BENCH if bench is None else bench
    rounds = sorted(rounds, key=lambda r: r["round"])
    pre = [r for r in rounds if event - PRE_ROUNDS <= r["round"] < event]
    post = [r for r in rounds if r["round"] >= event]
    if not pre or not post:
        return {}
    smin_pre = np.array([min(r["spreads"]) for r in pre], float)
    smin_post = np.array([min(r["spreads"]) for r in post], float)
    med = float(np.median(smin_pre))
    rec = next((k for k, r in enumerate(post) if min(r["spreads"]) >= med), None)
    by_round = {r["round"]: r for r in rounds}
    held = (by_round[event]["spreads"][seat] == by_round[event - 1]["spreads"][seat]
            if event in by_round and event - 1 in by_round else None)
    mp = lambda rs: float(np.mean([r["profits"] for r in rs]))
    return {
        "pre_market_spread": float(smin_pre.mean()), "post_market_spread": float(smin_post.mean()),
        "pre_delta": delta(mp(pre), bench=b), "post_delta": delta(mp(post), bench=b),
        "retention": float((smin_post >= med).mean()), "recovery_rounds": rec,
        "seat_held": held,
    }


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile 95% CI, resampling RUNS with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a 95% CI; runs resampled independently within each condition."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = ma - mb
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
