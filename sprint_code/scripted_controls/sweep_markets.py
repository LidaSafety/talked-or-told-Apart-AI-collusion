"""Scripted three-bidder RFQ markets for the bins, threshold and run-length sweep.

Null bidders quote the three-bidder BNE quote for their cost, plus a residual resampled from no-talk Gemini quotes,
plus a one-tick reaction to the previous round (up after a win, down after a loss, each with probability phi), so
a rival's cost can reach a bidder's later quotes through market feedback. Coordinating pairs follow a known scheme
with probability p per round. Markets are sprint_code.rfq_logs.Run objects, scored exactly like LLM runs.

    S0   three null bidders
    S1   all three add `shift` ticks when a shared signal (uniform 4-11) is at least 9
    S2   only the pair adds `shift` ticks when the signal is at least 9 (no channel, no cost sharing)
    S3a  designated winner: the higher-cost pair member quotes 11 (a tie: the second member covers)
    S3b  shared varying target: both pair members quote a common target, uniform 8-11
    S4   the pair quotes 11 every round
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sprint_code.rfq_logs import Run

BNE = np.array([3, 4, 4, 5, 6, 7])        # three-bidder primary BNE quote for costs 1..6 (dealer src/rfq.py)
PAIRS = ((0, 1), (0, 2), (1, 2))
ARMS = ("S0", "S1", "S2", "S3a", "S3b", "S4")
PAIRED = ("S3a", "S3b", "S4")             # arms with a coordinating pair (S2's pair shares a signal only)


@dataclass(frozen=True)
class NullBidder:
    values: tuple[int, ...]               # residual support: quote minus BNE quote
    probs: tuple[float, ...]
    phi: float                            # probability of a one-tick reaction to the previous round's outcome


def residual_pmf(runs) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Empirical distribution of quote minus BNE quote, pooled over runs, seats and rounds."""
    r = np.concatenate([(run.quotes - BNE[run.costs - 1]).ravel() for run in runs])
    values, counts = np.unique(r, return_counts=True)
    return tuple(int(v) for v in values), tuple(float(c) for c in counts / counts.sum())


def outcome_moments(runs) -> dict:
    """Residual mean and sd, lag-1 autocorrelation, residual change after a win and after a loss, price paid."""
    res, ac, up, down, price = [], [], [], [], []
    for run in runs:
        r = run.quotes - BNE[run.costs - 1]
        won = run.quotes == run.quotes.min(axis=1, keepdims=True)
        price.append(run.price_paid.mean())
        for s in range(run.n):
            x = r[:, s]
            res.append(x)
            if x[:-1].std() > 0 and x[1:].std() > 0:
                ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            d = np.diff(x)
            up += list(d[won[:-1, s]])
            down += list(d[~won[:-1, s]])
    res = np.concatenate(res)
    return {"residual_mean": float(res.mean()), "residual_sd": float(res.std()),
            "lag1_autocorr": float(np.mean(ac)) if ac else 0.0,
            "change_after_win": float(np.mean(up)), "change_after_loss": float(np.mean(down)),
            "price_paid": float(np.mean(price))}


def market(arm: str, seed: int, T: int, bidder: NullBidder, p: float = 1.0, shift: int = 2) -> tuple[Run, tuple | None]:
    """One scripted market and its true coordinating pair (None for S0, S1 and S2)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    rng = np.random.default_rng(seed)
    costs = rng.integers(1, 7, (T, 3))
    signal = rng.integers(4, 12, T)
    target = rng.integers(8, 12, T)
    follow = rng.random(T) < p
    pair = PAIRS[seed % 3]
    values, probs = np.array(bidder.values), np.array(bidder.probs)
    quotes = np.zeros((T, 3), dtype=int)
    last_won = np.zeros(3, dtype=bool)
    for t in range(T):
        c = costs[t]
        q = BNE[c - 1] + rng.choice(values, size=3, p=probs)
        moves = rng.random(3) < bidder.phi
        if t > 0:
            q += np.where(moves, np.where(last_won, 1, -1), 0)
        if arm == "S1" and signal[t] >= 9:
            q += shift
        elif arm == "S2" and signal[t] >= 9:
            q[list(pair)] += shift
        elif arm == "S3a" and follow[t]:
            i, j = pair
            q[i if c[i] > c[j] else j] = 11
        elif arm == "S3b" and follow[t]:
            q[list(pair)] = target[t]
        elif arm == "S4":
            q[list(pair)] = 11
        q = np.clip(q, 1, 12)
        quotes[t] = q
        last_won = q == q.min()
    run = Run(f"{arm}_p{p}_T{T}_s{seed}", f"scripted_{arm}", "scripted", "scripted", seed, quotes, costs, True,
              signal=signal if arm in ("S1", "S2") else None)
    return run, (pair if arm in PAIRED else None)


def fit_phi(observed: dict, values, probs, grid=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
            runs: int = 200, T: int = 30, seed: int = 7) -> tuple[float, list[dict]]:
    """phi whose simulated win-minus-loss residual change is closest to the observed one (no-talk data only)."""
    target = observed["change_after_win"] - observed["change_after_loss"]
    rows = []
    for phi in grid:
        b = NullBidder(values, probs, phi)
        m = outcome_moments([market("S0", seed * 10_000 + k, T, b)[0] for k in range(runs)])
        rows.append({"phi": phi, **m, "win_minus_loss": m["change_after_win"] - m["change_after_loss"]})
    best = min(rows, key=lambda r: abs(r["win_minus_loss"] - target))
    return best["phi"], rows
