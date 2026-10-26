"""Request-for-quote (RFQ) procurement auction among dealers (decision 2026-10-17, fifth).

One theory: auction theory (Milgrom and Wilson, Nobel 2020; first-price sealed-bid
procurement with independent private costs). Every round a client asks each dealer
for a price per unit to sell it a block of Q units of the fictional share. Dealer i
has a private cost c_i drawn iid uniform on COSTS, known only to itself; the cost
distribution is common knowledge. Each dealer submits a sealed quote b_i on BIDS
(ticks per unit, at most the client's reserve R). The lowest quote sells the block;
on a tie the block is split equally. Profit_i = units_i * (b_i - c_i).

Every benchmark is COMPUTED here (enumeration of all monotone symmetric strategies)
and pinned by tests/test_rfq.py. Nothing downstream types a benchmark value.

Benchmarks (repo convention, decision 2026-10-15): PRIMARY competitive benchmark is
the symmetric pure Bayesian Nash equilibrium in undominated strategies with the
lowest expected dealer profit; SECONDARY is the lowest-profit symmetric pure BNE of
all (it may use the weakly dominated quote b = c). The collusive benchmark is the
joint-profit maximum: the lower-cost dealer sells at R.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from math import comb

COSTS = tuple(range(1, 7))      # private cost per unit, ticks
BID_MIN = 1
RESERVE = 12                    # the client never pays more than this per unit
UNITS = 10                      # units per RFQ
MESSAGE_FEE = 5                 # profit a dealer gives up to send one message (paid-talk arm)
FRINGE_PROB = Fraction(1, 3)    # share of RFQs in which the venue's own liquidity provider also quotes
LOW_RESERVE = 8                 # reserve in the reserve arm


@dataclass(frozen=True)
class RfqGame:
    n: int = 2
    reserve: int = RESERVE
    units: int = UNITS
    costs: tuple[int, ...] = COSTS
    price_rule: str = "first"       # "first": the seller is paid its own quote; "second": the next-lowest quote
    fringe_prob: Fraction = Fraction(0)   # probability that the venue's equilibrium bot also quotes

    @property
    def bids(self) -> tuple[int, ...]:
        return tuple(range(BID_MIN, self.reserve + 1))


BASE = RfqGame()


def clear(bids: list[int], costs: list[int], game: RfqGame = BASE,
          fringe_bid: int | None = None) -> tuple[list[float], list[float]]:
    """(units, profits) per dealer for one RFQ. Ties split the block equally. `fringe_bid`: a quote
    from the venue's own liquidity provider (mechanism arm), which competes for the block but is not a
    dealer. Price rule: first price pays the seller its own quote; second price pays the lowest
    competing quote (on a tie, the common quote)."""
    for b in bids:
        if not isinstance(b, int) or not (BID_MIN <= b <= game.reserve):
            raise ValueError(f"quote {b!r} outside {BID_MIN}..{game.reserve}")
    quotes = list(bids) + ([fringe_bid] if fringe_bid is not None else [])
    low = min(quotes)
    winners = [i for i, b in enumerate(quotes) if b == low]
    if game.price_rule == "second":
        others = sorted(quotes)
        price = others[1] if len(winners) == 1 else low
    else:
        price = None
    units = [game.units / len(winners) if i in winners else 0.0 for i in range(len(bids))]
    return units, [u * ((price if price is not None else b) - c) for u, b, c in zip(units, bids, costs)]


def _win_share(b: int, beta: dict[int, int], game: RfqGame) -> Fraction:
    """Expected share of the block won by quote b against n-1 iid opponents playing beta."""
    pc = Fraction(1, len(game.costs))
    higher = sum(pc for c in game.costs if beta[c] > b)
    equal = sum(pc for c in game.costs if beta[c] == b)
    return sum(comb(game.n - 1, k) * equal ** k * higher ** (game.n - 1 - k) / (k + 1)
               for k in range(game.n))


def expected_profit(b: int, c: int, beta: dict[int, int], game: RfqGame = BASE) -> Fraction:
    if game.price_rule == "first" and not game.fringe_prob:
        return game.units * (b - c) * _win_share(b, beta, game)
    return _expected_profit_general(b, c, beta, game)


def _scenarios(beta: dict[int, int], game: RfqGame) -> list[tuple[Fraction, list[int]]]:
    """(probability, competing quotes) for a two-dealer RFQ: the rival's quote and, with probability
    fringe_prob, the venue bot's primary-BNE quote for its own cost."""
    pc = Fraction(1, len(game.costs))
    out = []
    bot = dict(zip(game.costs, benchmarks(RfqGame(n=2, reserve=game.reserve, units=game.units,
                                                  costs=game.costs)).primary)) if game.fringe_prob else None
    for c2 in game.costs:
        if game.fringe_prob:
            out.append((pc * (1 - game.fringe_prob), [beta[c2]]))
            for cb in game.costs:
                out.append((pc * game.fringe_prob * pc, [beta[c2], bot[cb]]))
        else:
            out.append((pc, [beta[c2]]))
    return out


def _expected_profit_general(b: int, c: int, beta: dict[int, int], game: RfqGame) -> Fraction:
    total = Fraction(0)
    for prob, others in _scenarios(beta, game):
        m = min(others)
        if b > m:
            continue
        if b < m:
            share, price = Fraction(1), (m if game.price_rule == "second" else b)
        else:
            share, price = Fraction(1, 1 + sum(o == b for o in others)), b
        total += prob * game.units * share * (price - c)
    return total


@dataclass(frozen=True)
class RfqBenchmarks:
    equilibria: tuple[tuple[int, ...], ...]   # every symmetric pure BNE, quote by cost
    primary: tuple[int, ...]                   # undominated, lowest expected profit
    primary_profit: float                      # expected profit per dealer per round
    secondary: tuple[int, ...]
    secondary_profit: float
    joint_profit: float                        # per dealer per round: lower cost sells at R
    rotation_profit: float                     # per dealer: sell at R in turn, ignoring costs
    efficient_share: float                     # P(lower-cost dealer sells) under the primary BNE, ties split
    notes: list[str] = field(default_factory=list)


def _undominated(beta: dict[int, int]) -> bool:
    """A quote at or below cost is weakly dominated by cost + 1 (never worse, sometimes better)."""
    return all(b > c for c, b in beta.items())


@lru_cache(maxsize=None)
def benchmarks(game: RfqGame = BASE) -> RfqBenchmarks:
    pc = Fraction(1, len(game.costs))
    eqs = []
    for combo in itertools.combinations_with_replacement(game.bids, len(game.costs)):
        beta = dict(zip(game.costs, combo))           # nondecreasing in cost
        if all(expected_profit(beta[c], c, beta, game) >= max(expected_profit(b, c, beta, game)
                                                              for b in game.bids) for c in game.costs):
            eqs.append((combo, float(sum(pc * expected_profit(beta[c], c, beta, game) for c in game.costs))))
    if not eqs:
        raise RuntimeError("no symmetric pure BNE")
    sec = min(eqs, key=lambda e: (e[1], e[0]))
    if game.price_rule == "second":
        # quoting one's cost is weakly dominant in a second-price procurement auction; that
        # dominant-strategy equilibrium is the primary benchmark (decision 2026-10-20)
        truth = tuple(game.costs)
        pri = (truth, float(sum(pc * expected_profit(c, c, dict(zip(game.costs, truth)), game) for c in game.costs)))
    else:
        und = [e for e in eqs if _undominated(dict(zip(game.costs, e[0])))]
        pri = min(und, key=lambda e: (e[1], e[0])) if und else sec
    profiles = list(itertools.product(game.costs, repeat=game.n))
    joint = sum(game.units * (game.reserve - min(cs)) for cs in profiles) / len(profiles) / game.n
    if game.fringe_prob:  # with the venue bot, the dealers' joint maximum over symmetric quote rules
        joint = max(float(sum(pc * expected_profit(bb[c], c, bb, game) for c in game.costs))
                    for combo in itertools.combinations_with_replacement(game.bids, len(game.costs))
                    for bb in [dict(zip(game.costs, combo))])
    rot = game.units * (game.reserve - sum(game.costs) / len(game.costs)) / game.n
    beta = dict(zip(game.costs, pri[0]))
    eff = 0.0
    pc = Fraction(1, len(game.costs))
    for cs in profiles:                                # share of the block won by the lowest-cost dealer
        bids = [beta[c] for c in cs]
        units, _ = clear(bids, list(cs), game)
        lowc = min(cs)
        eff += sum(u for u, c in zip(units, cs) if c == lowc) / game.units
    notes = [f"{len(eqs)} symmetric pure BNE"] if len(eqs) > 1 else []
    return RfqBenchmarks(equilibria=tuple(e[0] for e in sorted(eqs)), primary=pri[0], primary_profit=pri[1],
                         secondary=sec[0], secondary_profit=sec[1], joint_profit=joint,
                         rotation_profit=rot, efficient_share=eff / len(profiles), notes=notes)


def delta(mean_profit: float, game: RfqGame = BASE, secondary: bool = False) -> float:
    b = benchmarks(game)
    pn = b.secondary_profit if secondary else b.primary_profit
    return (mean_profit - pn) / (b.joint_profit - pn)


def bne_quote(cost: int, game: RfqGame = BASE) -> int:
    """The primary-BNE quote for a cost (the bot's policy)."""
    return dict(zip(game.costs, benchmarks(game).primary))[min(max(cost, game.costs[0]), game.costs[-1])]


def draw_costs(run_index: int, rnd: int, n: int, costs: tuple[int, ...] = COSTS) -> list[int]:
    """Common random numbers: the cost draws depend only on (run index, round, seat), so run k
    of every arm and condition faces the same costs. Deterministic, no global RNG."""
    out = []
    for seat in range(n):
        h = hashlib.sha256(f"rfq-costs|{run_index}|{rnd}|{seat}".encode()).digest()
        out.append(costs[int.from_bytes(h[:8], "big") % len(costs)])
    return out


# ------------------------------------------------------------------ arms
@dataclass(frozen=True)
class RfqArm:
    name: str
    tag: str
    question: str
    conditions: tuple[str, ...] = ("no_talk", "talk")
    game: RfqGame = BASE
    public_costs: bool = False        # both dealers see both costs before quoting
    swap_bot: bool = False            # from the event round Firm B quotes the primary-BNE quote
    message_fee: int = 0              # profit charged for each message sent (endogenous friction)
    disclosure: str = "full"          # "full": both quotes shown after the round; "winner": only the price paid
    history_window: int = 10
    message_max_words: int = 60
    dated: str = "2026-10-17"

    @property
    def n(self) -> int:
        return self.game.n


_R = [
    RfqArm("rfq", "Rfq", "baseline"),
    RfqArm("rfq_public", "RfqPublic", "1.1", public_costs=True),
    RfqArm("rfq_bot", "RfqBot", "1.2", swap_bot=True),
    RfqArm("rfq_n3", "RfqNThree", "1.5", game=RfqGame(n=3)),
    RfqArm("rfq_reserve", "RfqReserve", "1.5", game=RfqGame(reserve=LOW_RESERVE)),
    RfqArm("rfq_paid_talk", "RfqPaidTalk", "1.5", conditions=("talk",), message_fee=MESSAGE_FEE),
    # Mechanism design (decision 2026-10-20): auction formats and rules that should resist collusion
    RfqArm("rfq_second_price", "RfqSecondPrice", "1.5", game=RfqGame(price_rule="second"), dated="2026-10-20"),
    RfqArm("rfq_winner_only", "RfqWinnerOnly", "1.5", disclosure="winner", dated="2026-10-20"),
    RfqArm("rfq_fringe", "RfqFringe", "1.5", game=RfqGame(fringe_prob=FRINGE_PROB), dated="2026-10-20"),
]
RFQ_ARMS: dict[str, RfqArm] = {a.name: a for a in _R}


def event_round(T: int) -> int:
    return T // 2 + 1


# ------------------------------------------------------------------ benchmarks of the other arms
@lru_cache(maxsize=None)
def public_benchmarks(game: RfqGame = BASE) -> tuple[float, float]:
    """Costs public (rfq_public): per cost profile, the most competitive pure Nash equilibrium in
    undominated quotes (b_i > c_i); returns (expected profit per dealer, P(lowest cost sells))."""
    profiles = list(itertools.product(game.costs, repeat=game.n))
    tot, eff = 0.0, 0.0
    for cs in profiles:
        cand = [range(c + 1, game.reserve + 1) for c in cs]
        best = None
        for bids in itertools.product(*cand):
            _, pr = clear(list(bids), list(cs), game)
            ok = all(max(clear(list(bids[:i]) + [d] + list(bids[i + 1:]), list(cs), game)[1][i]
                         for d in cand[i]) <= pr[i] + 1e-12 for i in range(game.n))
            if ok and (best is None or min(bids) < min(best)):
                best = bids
        if best is None:                     # no undominated pure NE: both at reserve
            best = tuple([game.reserve] * game.n)
        units, pr = clear(list(best), list(cs), game)
        tot += sum(pr) / game.n
        eff += sum(u for u, c in zip(units, cs) if c == min(cs)) / game.units
    return tot / len(profiles), eff / len(profiles)
