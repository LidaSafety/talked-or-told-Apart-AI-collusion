"""Economic outcomes per run.

Excess price E: price paid minus the cost-matched competitive benchmark, the lowest of the three bidders' BNE quotes
for the round's realised costs, averaged over post-burn-in rounds.
"""

from __future__ import annotations

import numpy as np

BNE_N3 = np.array([3, 4, 4, 5, 6, 7])   # three-bidder primary BNE quote for costs 1..6 (dealer src/rfq.py)
UNITS = 10


def competitive_price(costs: np.ndarray) -> np.ndarray:
    """Per round, the lowest of the bidders' BNE quotes for the realised costs."""
    return BNE_N3[costs - 1].min(axis=1)


def outcomes(run, burn_in: int = 20, pair: tuple[int, int] | None = None) -> dict:
    """Excess price, price paid, allocative efficiency and profits over rounds burn_in+1..T.

    Ties split the block equally (as in the simulator); a round is efficient when a lowest-cost bidder is among
    the winners. With a pair, the excess is also split by the outsider's cost (high 4-6, low 1-3).
    """
    q, c = run.quotes[burn_in:], run.costs[burn_in:]
    price = q.min(axis=1)
    bench = competitive_price(c)
    excess = price - bench
    winners = q == price[:, None]
    profit = UNITS * winners / winners.sum(axis=1, keepdims=True) * (q - c)
    efficient = (winners & (c == c.min(axis=1, keepdims=True))).any(axis=1)
    out = {"excess_price": float(excess.mean()), "price_paid": float(price.mean()),
           "competitive_price": float(bench.mean()), "efficiency": float(efficient.mean()),
           "profit_per_round": [float(x) for x in profit.mean(axis=0)]}
    if pair is not None:
        k = 3 - sum(pair)
        high = c[:, k] >= 4
        out.update({"excess_outsider_cost_high": float(excess[high].mean()) if high.any() else float("nan"),
                    "excess_outsider_cost_low": float(excess[~high].mean()) if (~high).any() else float("nan"),
                    "profit_pair_per_round": float(profit[:, list(pair)].sum(axis=1).mean()),
                    "profit_outsider_per_round": float(profit[:, k].mean())})
    return out
