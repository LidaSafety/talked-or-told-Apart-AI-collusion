"""Scripted dealers for no-API sanity checks of the game and the measures.

A policy is f(seat, history) -> spread, where history is the list of past
round dicts (round, spreads, orders, profits). `play` runs the same clearing
code the LLM runner uses.
"""

from __future__ import annotations

from .market import BENCH, S_MAX, S_MIN, clear


def play(policies, T: int) -> list[dict]:
    history: list[dict] = []
    for t in range(1, T + 1):
        spreads = [int(p(i, history)) for i, p in enumerate(policies)]
        orders, profits = clear(spreads)
        history.append({"round": t, "spreads": spreads, "orders": orders, "profits": profits})
    return history


def always_undercut(start: int = S_MAX):
    """Quote one tick below the rival's last quote, floored at the grid minimum
    (reaches the most competitive equilibrium, the secondary benchmark)."""
    def f(seat, history):
        if not history:
            return start
        return max(S_MIN, history[-1]["spreads"][1 - seat] - 1)
    return f


def profitable_undercut(start: int = S_MAX, floor: int = S_MIN + 1):
    """Undercut by one tick only while the undercut still earns a positive margin
    (reaches the undominated equilibrium, the primary benchmark). `floor`: the
    lowest quote it posts; the runner passes the equilibrium spread of the arm's game."""
    def f(seat, history):
        if not history:
            return start
        return max(floor, history[-1]["spreads"][1 - seat] - 1)
    return f


def hold(spread: int = BENCH.joint_spread):
    return lambda seat, history: spread


def grim_trigger(spread: int = BENCH.joint_spread, punish: int = BENCH.nash_spread):
    """Hold `spread` until the rival has ever quoted below it, then `punish` forever."""
    def f(seat, history):
        if any(h["spreads"][1 - seat] < spread for h in history):
            return punish
        return spread
    return f


def defect_once(spread: int = BENCH.joint_spread, at_round: int = 5):
    """Hold `spread`, undercut by one tick at `at_round`, then hold again."""
    def f(seat, history):
        return spread - 1 if len(history) + 1 == at_round else spread
    return f
