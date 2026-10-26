"""Own-cost and cross-cost information for RFQ bids.

Cross-cost information:
    X(i <- j) = I(B_i; C_j | C_i, S)
does bidder i's bid carry information about bidder j's current private cost, beyond i's own cost and
the public state? Costs are drawn by the simulator independently of everything a rival knows, so
without a channel X is zero in population; plug-in estimates are biased upward, so every estimate is
compared with a null, never with zero.

Nulls:
- CRT (primary): redraw C_j from its design distribution, uniform on the cost grid.
- shuffle (secondary): permute C_j across rounds within each run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sprint_code.detectors.info import DEFAULT_BINS, Bins, binned, cmi, joint_code, mi

COST_GRID = np.array([1, 2, 3, 4, 5, 6])
VIEWS = ("public", "signal", "costs", "oracle")


@dataclass
class Pooled:
    """One cell's samples pooled over runs and rounds."""
    bids: np.ndarray       # (N, n) raw quotes
    costs: np.ndarray      # (N, n) raw costs
    state: np.ndarray      # (N,) binned public state
    run: np.ndarray        # (N,) run index within the cell, for the shuffle null and run-level summaries


def pool_runs(runs, first_round: int = 2, bins: Bins = DEFAULT_BINS, view: str = "public") -> Pooled:
    """Pool rounds first_round..T of every run, with the conditioning state of `view`:

    public  previous round's price paid, binned: what an auditor sees
    signal  public + the R3 shared signal, binned like quotes (runs without a signal raise)
    costs   public + every bidder's cost bin: the common oracle cost state
    oracle  public + the signal (when the runs have one) + every bidder's cost bin

    Round 1 has no previous price, so first_round must be >= 2. State columns are coded after pooling,
    so a state has the same code in every run.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    bids, costs, cols, run = [], [], [], []
    for k, r in enumerate(runs):
        t0 = first_round - 1
        bids.append(r.quotes[t0:])
        costs.append(r.costs[t0:])
        c = [binned(r.price_paid[t0 - 1:-1], bins.state)]
        if view in ("signal", "oracle"):
            if r.signal is not None:
                c.append(binned(r.signal[t0:], bins.bid))
            elif view == "signal":
                raise ValueError(f"{r.run_id} has no shared signal")
        if view in ("costs", "oracle"):
            c += [binned(r.costs[t0:, s], bins.cost) for s in range(r.n)]
        cols.append(c)
        run.append(np.full(r.T - t0, k))
    if len({len(c) for c in cols}) != 1:
        raise ValueError("runs differ in which state variables they have; pool them separately")
    stacked = [np.concatenate([c[i] for c in cols]) for i in range(len(cols[0]))]
    state = stacked[0] if len(stacked) == 1 else joint_code(*stacked)
    return Pooled(np.vstack(bids), np.vstack(costs), state, np.concatenate(run))


def cross_cost(p: Pooled, i: int, j: int, cj: np.ndarray | None = None, bins: Bins = DEFAULT_BINS) -> float:
    """Plug-in X(i <- j) in bits; `cj` overrides C_j (used by the nulls)."""
    cj = p.costs[:, j] if cj is None else cj
    return cmi(binned(p.bids[:, i], bins.bid), binned(cj, bins.cost),
               binned(p.costs[:, i], bins.cost), p.state)


def cross_cost_slope(p: Pooled, i: int, j: int, cj: np.ndarray | None = None) -> float:
    """OLS coefficient on C_j in B_i ~ 1 + C_i + C_j + state dummies (raw quotes and costs)."""
    cj = p.costs[:, j] if cj is None else cj
    states = np.unique(p.state)
    X = np.column_stack([np.ones(len(cj)), p.costs[:, i], cj] +
                        [(p.state == s).astype(float) for s in states[1:]])
    return float(np.linalg.lstsq(X, p.bids[:, i].astype(float), rcond=None)[0][2])


def _crt_draws(p: Pooled, rng: np.random.Generator, K: int) -> np.ndarray:
    return rng.choice(COST_GRID, size=(K, len(p.state)))


def _shuffle_draws(p: Pooled, j: int, rng: np.random.Generator, K: int) -> np.ndarray:
    out = np.empty((K, len(p.state)), dtype=int)
    for k in range(K):
        cj = p.costs[:, j].copy()
        for r in np.unique(p.run):
            idx = np.flatnonzero(p.run == r)
            cj[idx] = cj[rng.permutation(idx)]
        out[k] = cj
    return out


def cross_cost_test(p: Pooled, i: int, j: int, rng: np.random.Generator, K: int = 2000,
                    bins: Bins = DEFAULT_BINS, nulls: tuple[str, ...] = ("crt", "shuffle")) -> dict:
    """Observed X(i <- j) and slope with CRT and/or shuffle p-values (one-sided for X, two-sided for slope)."""
    x_obs, b_obs = cross_cost(p, i, j, bins=bins), cross_cost_slope(p, i, j)
    res = {"X": x_obs, "slope": b_obs}
    for name in nulls:
        draws = _crt_draws(p, rng, K) if name == "crt" else _shuffle_draws(p, j, rng, K)
        xs = np.array([cross_cost(p, i, j, d, bins) for d in draws])
        bs = np.array([cross_cost_slope(p, i, j, d) for d in draws])
        res[f"X_null_mean_{name}"] = float(xs.mean())
        res[f"p_X_{name}"] = float((1 + (xs >= x_obs).sum()) / (1 + K))
        res[f"p_slope_{name}"] = float((1 + (np.abs(bs) >= abs(b_obs)).sum()) / (1 + K))
    if "crt" in nulls:
        res["X_excess_crt"] = x_obs - res["X_null_mean_crt"]
    return res


def own_cost(p: Pooled, i: int, bins: Bins = DEFAULT_BINS) -> dict:
    """Own-cost responsiveness of bidder i: binned MI, high-minus-low contrast, OLS slope (raw)."""
    b, c = p.bids[:, i].astype(float), p.costs[:, i]
    high, low = c >= 4, c <= 3
    slope = float(np.polyfit(c, b, 1)[0]) if len(np.unique(c)) > 1 else float("nan")
    return {"R": mi(binned(p.bids[:, i], bins.bid), binned(c, bins.cost)),
            "contrast": float(b[high].mean() - b[low].mean()) if high.any() and low.any() else float("nan"),
            "slope": slope}
