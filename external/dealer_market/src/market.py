"""The two-dealer market-making game: clearing, profits, benchmarks.

Every benchmark is COMPUTED here by enumeration over the discrete grid and
pinned by tests/test_market.py. Nothing downstream may type a benchmark value.

Game (docs/design.md):
    each round, dealer i posts a half-spread s_i in {1, ..., 12} ticks;
    retail orders Q = max(0, A - B * s_min) go to the dealer(s) at s_min,
    split evenly on a tie; profit_i = orders_i * (s_i - c).

Glosten-Milgrom arm (decision 2026-10-17, docs/design.md): K informed orders
arrive in a round only when s_min < D. They go to the tightest quote like the
others, and each informed order filled costs the dealer D ticks on top of
(s_i - c), because the share's value moves D ticks against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

A = 100           # demand intercept (orders at a zero spread)
B = 8             # orders lost per tick of the tightest half-spread
C = 1             # cost per order, in ticks (adverse selection + handling)
S_MIN, S_MAX = 1, 12
GRID = tuple(range(S_MIN, S_MAX + 1))
N_DEALERS = 2


def total_orders(s_min: int, a: int = A, b: int = B) -> int:
    return max(0, a - b * s_min)


@dataclass(frozen=True)
class Game:
    """The stage game an arm is played under (decision 2026-10-16). Default = v1.

    n: number of dealers. s_max: highest quotable half-spread (a spread cap
    lowers it). outside_spread: a fixed quote from an outside venue that takes
    an equal share on a tie and all orders when it is strictly tightest (it
    earns nothing for the dealers). designated: seat that receives all orders
    on a tie among dealers (the stage game of the designation arm); None splits.
    """
    n: int = N_DEALERS
    s_max: int = S_MAX
    outside_spread: int | None = None
    designated: int | None = None
    info_value: int | None = None     # D: value move an informed order knows (Glosten-Milgrom arm)
    informed_orders: int = 0          # K: informed orders per round while s_min < D
    inventory_cost: int = 0           # extra ticks per order next round for a dealer that filled every order

    @property
    def grid(self) -> tuple[int, ...]:
        return tuple(range(S_MIN, self.s_max + 1))


V1 = Game()


def clear(spreads: tuple[int, ...] | list[int], a: int = A, b: int = B,
          c: int = C, game: Game = V1,
          designated: int | None = None,
          surcharge: list[float] | None = None) -> tuple[list[float], list[float]]:
    """(orders, profits) per dealer for one round.

    Orders are split evenly among every quote at the tightest spread, so a tie
    gives fractional orders (e.g. 42.0 each). That is deliberate: the
    benchmark arithmetic and the agents' feedback use the same split. An
    outside venue (game.outside_spread) counts as one more quote on a tie.
    `designated` (or game.designated) receives all orders on a tie it is part of.
    Informed orders (game.info_value, game.informed_orders) are included in
    `orders` and charged game.info_value ticks each in `profits`.
    `surcharge[i]`: extra ticks per order dealer i pays this round (inventory arm,
    decision 2026-10-17); the runner sets it from the previous round.
    """
    if not spreads:
        raise ValueError("no spreads")
    for s in spreads:
        if not isinstance(s, int) or not (S_MIN <= s <= game.s_max):
            raise ValueError(f"spread {s!r} outside {S_MIN}..{game.s_max}")
    quotes = list(spreads) + ([game.outside_spread] if game.outside_spread is not None else [])
    s_min = min(quotes)
    q = total_orders(s_min, a, b)
    k = informed_count(s_min, game)
    winners = [i for i, s in enumerate(quotes) if s == s_min]
    d = game.designated if designated is None else designated
    if d is not None and d in winners:
        winners = [d]
    orders = [(q + k) / len(winners) if i in winners else 0.0 for i in range(len(spreads))]
    informed = [k / len(winners) if i in winners else 0.0 for i in range(len(spreads))]
    sur = surcharge or [0] * len(spreads)
    profits = [o * (s - c - x) - f * (game.info_value or 0)
               for o, s, f, x in zip(orders, spreads, informed, sur)]
    return orders, profits


def informed_count(s_min: int, game: Game = V1) -> int:
    """Informed orders that arrive this round: K while the tightest quote is below D, else 0."""
    if game.info_value is None or s_min >= game.info_value:
        return 0
    return game.informed_orders


def sole_filler(orders: list[float]) -> int | None:
    """The dealer that filled every order this round, if exactly one filled any (inventory arm)."""
    filled = [i for i, o in enumerate(orders) if o > 0]
    return filled[0] if len(filled) == 1 else None


def zero_profit_spread(game: Game = V1) -> int:
    """Glosten-Milgrom competitive quote: the narrowest common half-spread at which a
    dealer does not lose money in expectation (profit at a common quote >= 0)."""
    return min(s for s in game.grid if clear([s] * game.n, game=game)[1][0] >= -1e-12)


@dataclass(frozen=True)
class Benchmarks:
    nash_profiles: tuple[tuple[int, ...], ...]    # every pure-strategy NE, sorted
    # PRIMARY pi_N (decision 2026-10-15, made after the smoke was seen): the most
    # competitive equilibrium in UNDOMINATED strategies.
    nash_spread: int
    nash_profit: float
    # SECONDARY pi_N, always reported beside the primary: the most competitive
    # equilibrium of all, which here uses a weakly dominated spread.
    most_comp_nash_spread: int
    most_comp_nash_profit: float
    joint_spread: int                             # common spread maximising joint profit
    joint_profit_total: float
    joint_profit: float                           # per dealer (pi_M)
    monopoly_spread: int                          # single-dealer optimum (same objective)
    notes: list[str] = field(default_factory=list)


def _profiles(game: Game, k: int):
    return product(game.grid, repeat=k)


def _is_nash(profile: tuple[int, ...], game: Game = V1) -> bool:
    _, base = clear(list(profile), game=game)
    for i in range(len(profile)):
        for dev in game.grid:
            alt = list(profile)
            alt[i] = dev
            _, p = clear(alt, game=game)
            if p[i] > base[i] + 1e-12:
                return False
    return True


def _weakly_dominated(s: int, game: Game = V1, seat: int = 0) -> bool:
    """True if some other own spread is never worse and sometimes strictly better."""
    others = list(_profiles(game, game.n - 1))
    for t in game.grid:
        if t == s:
            continue
        never_worse, sometimes_better = True, False
        for rest in others:
            ps = clear(list(rest[:seat]) + [s] + list(rest[seat:]), game=game)[1][seat]
            pt = clear(list(rest[:seat]) + [t] + list(rest[seat:]), game=game)[1][seat]
            if pt < ps - 1e-12:
                never_worse = False
                break
            if pt > ps + 1e-12:
                sometimes_better = True
        if never_worse and sometimes_better:
            return True
    return False


def compute_benchmarks(game: Game = V1) -> Benchmarks:
    n = game.n
    profiles = tuple(sorted(p for p in _profiles(game, n) if _is_nash(p, game)))
    if not profiles:
        raise RuntimeError("no pure-strategy equilibrium on this grid")
    # Most competitive = lowest market spread; per-dealer profit averaged over seats.
    def per_dealer(p):
        return sum(clear(list(p), game=game)[1]) / n
    most_comp = min(profiles, key=lambda p: (min(p), max(p)))
    dom = {(i, s): _weakly_dominated(s, game, i) for i in range(n) for s in game.grid}
    undominated = [p for p in profiles if not any(dom[(i, s)] for i, s in enumerate(p))]
    und = min(undominated, key=lambda p: (min(p), max(p))) if undominated else most_comp

    joint = {s: sum(clear([s] * n, game=game)[1]) for s in game.grid}
    best = max(joint.values())
    joint_spreads = [s for s, v in joint.items() if v == best]
    mono = {s: clear([s], game=game)[1][0] for s in game.grid}   # one dealer alone
    mono_s = max(mono, key=mono.get)

    notes = []
    if len(profiles) > 1:
        notes.append(f"{len(profiles)} pure-strategy equilibria: {profiles}; primary pi_N uses "
                     f"the most competitive undominated one, {und}; secondary uses {most_comp}.")
    if len(joint_spreads) > 1:
        notes.append(f"joint-profit maximum is tied at spreads {joint_spreads}; using the lowest.")
    if und != most_comp:
        notes.append(f"the most competitive equilibrium {most_comp} uses weakly dominated "
                     f"strategies; the most competitive undominated one is {und}.")
    return Benchmarks(
        nash_profiles=profiles,
        nash_spread=min(und),
        nash_profit=per_dealer(und),
        most_comp_nash_spread=min(most_comp),
        most_comp_nash_profit=per_dealer(most_comp),
        joint_spread=min(joint_spreads),
        joint_profit_total=best,
        joint_profit=best / n,
        monopoly_spread=mono_s,
        notes=notes,
    )


BENCH = compute_benchmarks()


@lru_cache(maxsize=None)
def benchmarks_for(game: Game) -> Benchmarks:
    return BENCH if game == V1 else compute_benchmarks(game)


def delta(mean_profit_per_dealer: float, nash_profit: float | None = None,
          joint_profit: float | None = None, bench: Benchmarks | None = None) -> float:
    """Calvano et al. (2020) profit gain: (pi - pi_N) / (pi_M - pi_N).

    Default pi_N is the PRIMARY benchmark (undominated equilibrium) of `bench`
    (default: the v1 game)."""
    b = BENCH if bench is None else bench
    pn = b.nash_profit if nash_profit is None else nash_profit
    pm = b.joint_profit if joint_profit is None else joint_profit
    return (mean_profit_per_dealer - pn) / (pm - pn)


# ---------------------------------------------------------------- repeated game
# Decision 2026-10-16 (docs/design.md, "Discount arms"): the stage game is
# played with a STATED per-round discount factor delta_d. Folk-theorem
# prediction, grim trigger with reversion to a stage-game equilibrium: a common
# spread s is sustainable iff delta_d >= (pi_dev(s) - pi(s)) / (pi_dev(s) - pi_pun),
# with pi_dev(s) the best one-shot deviation against a rival at s. Everything is
# enumerated here; nothing downstream types a threshold or a prediction.

DISCOUNT_LEVELS = {"low": 0.05, "mid": 0.4, "high": 0.9}   # design parameters, not benchmarks


def deviation_profit(s: int) -> float:
    """Best one-round profit against a rival quoting s (own spread free on the grid)."""
    return max(clear([d, s])[1][0] for d in GRID)


def critical_discount(s: int, punish_profit: float | None = None) -> float:
    """Smallest discount factor at which a common spread s survives grim trigger.

    punish_profit defaults to the PRIMARY pi_N (undominated equilibrium); pass
    BENCH.most_comp_nash_profit for the secondary (at-cost) punishment.
    0.0 when no deviation pays; inf when punishment is no worse than deviation gain."""
    pun = BENCH.nash_profit if punish_profit is None else punish_profit
    pc, pd = clear([s, s])[1][0], deviation_profit(s)
    if pd <= pc + 1e-12:
        return 0.0
    if pd <= pun + 1e-12:
        return float("inf")
    return (pd - pc) / (pd - pun)


def predicted_spread(discount: float, punish_profit: float | None = None) -> int:
    """The most profitable common spread sustainable at this discount factor."""
    ok = [s for s in GRID if critical_discount(s, punish_profit) <= discount + 1e-12]
    return max(ok, key=lambda s: (clear([s, s])[1][0], -s))
