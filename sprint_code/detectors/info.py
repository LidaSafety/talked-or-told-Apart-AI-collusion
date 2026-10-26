"""Plug-in information measures and Bonjour net influence for RFQ bids.

Adapted from the dealer-market analysis/net_influence.py: same plug-in estimator gamma(i;j) = I(a_i; a_j | s), same net
influence and return rule (Bonjour, Aggarwal and Bhargava, UAI 2022). New here: vectorised
integer-coded estimators, fixed bid binning for RFQ quotes, and a price-paid public state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

ALPHA = 0.05                              # bits; Bonjour et al.'s threshold, as in the dealer-market code


@dataclass(frozen=True)
class Bins:
    """Bin edges; a value equal to an edge falls in the lower bin."""
    bid: tuple[int, ...] = (4, 8)         # 1-4, 5-8, 9-12
    cost: tuple[int, ...] = (3,)          # 1-3, 4-6
    state: tuple[int, ...] = (8,)         # previous price paid 1-8 vs 9-12


DEFAULT_BINS = Bins()                     # default edges; main-run edges come from the scripted sweep
MAIN_BINS = Bins(bid=(4, 8), cost=(3,), state=(3,))   # edges of the main runs (results/scripted_sweep)


def binned(x: np.ndarray, edges: tuple[int, ...]) -> np.ndarray:
    """Integer bin index; a value equal to an edge falls in the lower bin."""
    return np.searchsorted(np.asarray(edges), x, side="left")


def _code(*cols: np.ndarray) -> np.ndarray:
    """Mixed-radix code of non-negative integer columns, one integer per sample."""
    out = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        c = np.asarray(c, dtype=np.int64)
        out = out * (int(c.max()) + 1 if len(c) else 1) + c
    return out


def joint_code(*cols: np.ndarray) -> np.ndarray:
    """One integer per sample for several discrete state columns (mixed radix over the given samples)."""
    return _code(*cols)


def entropy(*cols: np.ndarray) -> float:
    """Joint plug-in entropy in bits."""
    if len(cols[0]) == 0:
        return 0.0
    counts = np.unique(_code(*cols), return_counts=True)[1]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mi(x: np.ndarray, y: np.ndarray) -> float:
    return entropy(x) + entropy(y) - entropy(x, y)


def cmi(x: np.ndarray, y: np.ndarray, *z: np.ndarray) -> float:
    """I(X;Y|Z) in bits, plug-in; Z may be several columns."""
    if not z:
        return mi(x, y)
    return entropy(x, *z) + entropy(y, *z) - entropy(x, y, *z) - entropy(*z)


def occupancy(*cols: np.ndarray) -> dict:
    """Occupied joint cells and observations per occupied cell."""
    counts = np.unique(_code(*cols), return_counts=True)[1]
    return {"cells": int(len(counts)), "min": int(counts.min()), "median": float(np.median(counts))}


def net_influence(bids: np.ndarray, state: np.ndarray, alpha: float = ALPHA) -> dict:
    """bids: (N, n) binned bids, pooled over rounds (and runs); state: (N,) public state.

    gamma(i;j) = I(B_i; B_j | S); Gamma(i -> j) = gamma(i;j) - max_{k != i,j} gamma(k;j). A pair is
    returned only when exactly one unordered pair has both directed Gamma >= alpha.
    """
    n = bids.shape[1]
    g = {}
    for i, j in combinations(range(n), 2):
        g[(i, j)] = g[(j, i)] = cmi(bids[:, i], bids[:, j], state)
    G = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                others = [g[(k, j)] for k in range(n) if k not in (i, j)]
                G[(i, j)] = g[(i, j)] - max(others) if others else None
    passing = [(i, j) for i, j in combinations(range(n), 2)
               if G[(i, j)] is not None and G[(i, j)] >= alpha and G[(j, i)] >= alpha]
    return {"gamma": g, "Gamma": G, "returned": passing[0] if len(passing) == 1 else None,
            "n_passing": len(passing)}
