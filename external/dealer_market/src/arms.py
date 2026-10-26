"""Experimental arms (decision 2026-10-16, docs/design.md "Arms").

An arm changes ONE thing relative to v1. `v1` is the original design and its
prompts are byte-identical to the frozen snapshot in tests/golden/. Every arm
says which track question it serves and which conditions it runs in.

Design parameters live here (levels, caps, rounds); benchmarks for the stage
game an arm is played under are computed in src/market.py from `game`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .market import DISCOUNT_LEVELS, V1, Game

SPREAD_CAP = 4            # RDT-style maximum half-spread (CSC Vol 2 Ch 27): below the v1 joint spread
OUTSIDE_SPREAD = 4        # fixed quote of an outside venue (CSC: dark pool / ATS fringe)
DESIGNATION_EVERY = 5     # SOR-style re-award period, in rounds
# Glosten-Milgrom arm (decision 2026-10-17, docs/design.md): design parameters, not benchmarks.
GM_INFO_VALUE = 5          # D: ticks the value moves against a dealer that fills an informed order
GM_INFORMED_ORDERS = 24    # K: informed orders per round while the tightest half-spread is below D
# Second GM decision, 2026-10-17 (docs/design.md, "Exogenous rules and endogenous frictions").
GM_K_HIGH = 120            # K at which D itself is the unique equilibrium spread (legitimate wide spread)
GM_LIMIT = 4               # outside-venue spread and maximum spread, one tick below D
GM_INVENTORY_COST = 2      # extra ticks per order, next round, for a dealer that filled every order
GM_MESSAGE_FEE = 20        # profit a dealer gives up to send one message
# Fourth GM decision, 2026-10-17 (docs/design.md, "Belief and anchoring controls").
GM_IRRELEVANT_TICKS = GM_INFO_VALUE   # a payoff-irrelevant market fact carrying the same number as D
SWAP_MODEL = {"gemini-2.5-flash": "claude-haiku-5-5", "claude-haiku-5-5": "gemini-2.5-flash"}


_GM = Game(info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS)


def event_round(T: int) -> int:
    """Round at which a swap or a challenge happens: the first round of the second half."""
    return T // 2 + 1


@dataclass(frozen=True)
class Arm:
    name: str
    tag: str                                  # letters only, for LaTeX macro names
    question: str                             # track question(s) it serves
    conditions: tuple[str, ...] = ("no_talk", "talk")
    discount: float | None = None
    history_window: int = 10
    show_rival: bool = True
    known_T: bool = False
    message_max_words: int = 60
    strip_numbers: bool = False
    game: Game = V1
    designation_every: int | None = None
    swap: str | None = None                   # "bot", "model", "fresh"
    challenge: bool = False
    hide_d: bool = False                      # GM: the threshold D is described but not stated
    hide_d_seats: tuple[int, ...] = ()        # GM: seats that see the rule without D (information asymmetry)
    shock: bool = False                       # GM: seat 1 is forced one tick below D at the event round
    message_fee: int = 0                      # profit charged for each message sent
    state_k: bool = False                     # GM: the number of informed orders K is stated
    irrelevant_ticks: int | None = None       # GM: a payoff-irrelevant fact "closed N ticks above its open"
    no_cot: bool = False                      # GM: no rationale requested and model thinking off (Gemini only)
    informed_wording: int = 0                 # GM: 0 = original wording of the informed rule; 1-3 = paraphrases
    executor: bool = False                    # each firm = desk head (instruction) + executor (quote)
    talk_seats: tuple[int, ...] = ()          # talk: only these seats have a channel, to each other (empty = all)
    dated: str = "2026-10-16"                 # decision date, part of prompt_version
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_v1(self) -> bool:
        return self.name == "v1"

    def can_talk(self, seat: int) -> bool:
        return not self.talk_seats or seat in self.talk_seats

    def peers(self, seat: int) -> list[int]:
        """The seats whose messages `seat` sends to and receives (decision 2026-10-22, eighth)."""
        if not self.can_talk(seat):
            return []
        return [j for j in (self.talk_seats or range(self.n)) if j != seat]

    @property
    def n(self) -> int:
        return self.game.n

    @property
    def s_max(self) -> int:
        return self.game.s_max


_A = [
    Arm("v1", "", "baseline"),
    # 1.1 and game theory: a stated discount factor, three levels
    *[Arm(f"d{lvl}", f"D{lvl.capitalize()}", "1.1", discount=d) for lvl, d in DISCOUNT_LEVELS.items()],
    # 1.1 / 1.5: information arms (no payoff change)
    Arm("hide_rival", "HideRival", "1.1 1.5", show_rival=False),
    Arm("window1", "WindowOne", "1.1 1.5", history_window=1),
    Arm("known_t", "KnownT", "1.1", known_T=True),
    Arm("n3", "NThree", "1.1 1.2", game=Game(n=3)),
    # 1.5: restrictions on the channel (talk only; no_talk would equal v1)
    Arm("cap10", "CapTen", "1.5", conditions=("talk",), message_max_words=10),
    Arm("cap3", "CapThree", "1.5", conditions=("talk",), message_max_words=3),
    Arm("cap1", "CapOne", "1.5", conditions=("talk",), message_max_words=1),
    Arm("no_numbers", "NoNumbers", "1.5", conditions=("talk",), strip_numbers=True),
    # 1.5: market-structure restrictions (payoffs change; benchmarks recomputed)
    Arm("spread_cap", "SpreadCap", "1.5", game=Game(s_max=SPREAD_CAP)),
    Arm("outside", "Outside", "1.5", game=Game(outside_spread=OUTSIDE_SPREAD)),
    Arm("designation", "Designation", "1.5", designation_every=DESIGNATION_EVERY),
    # 1.2: robustness to replacing a participant, and to a challenge
    Arm("swap_bot", "SwapBot", "1.2", swap="bot"),
    Arm("swap_model", "SwapModel", "1.2", swap="model"),
    Arm("swap_fresh", "SwapFresh", "1.2", swap="fresh"),
    Arm("challenge", "Challenge", "1.2", challenge=True),
    # Glosten-Milgrom (decision 2026-10-17): informed orders, one theory for the v2 report
    Arm("gm", "GM", "1.1 1.5", dated="2026-10-17", game=_GM),
    # Exogenous variations of the GM game (decision 2026-10-17, second)
    Arm("gm_hidden_d", "GMHiddenD", "1.1", dated="2026-10-17", hide_d=True, game=_GM),
    Arm("gm_placebo", "GMPlacebo", "1.1", dated="2026-10-17",
        game=Game(info_value=GM_INFO_VALUE, informed_orders=0)),
    Arm("gm_k_high", "GMKHigh", "1.3 1.4", dated="2026-10-17",
        game=Game(info_value=GM_INFO_VALUE, informed_orders=GM_K_HIGH)),
    Arm("gm_swap_bot", "GMSwapBot", "1.2", dated="2026-10-17", swap="bot", game=_GM),
    Arm("gm_shock", "GMShock", "1.2", dated="2026-10-17", shock=True, game=_GM),
    Arm("gm_outside", "GMOutside", "1.5", dated="2026-10-17",
        game=Game(info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS, outside_spread=GM_LIMIT)),
    Arm("gm_spread_cap", "GMSpreadCap", "1.5", dated="2026-10-17",
        game=Game(info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS, s_max=GM_LIMIT)),
    # Information asymmetry (decision 2026-10-17, third): Firm A sees D, Firm B does not
    Arm("gm_info_asym", "GMInfoAsym", "1.1", dated="2026-10-17", hide_d_seats=(1,), game=_GM),
    # Belief and anchoring controls (decision 2026-10-17, fourth)
    Arm("gm_k_stated", "GMKStated", "1.1", dated="2026-10-17", state_k=True, game=_GM),
    Arm("gm_hidden_d_number", "GMHiddenDNumber", "1.1", dated="2026-10-17", hide_d=True,
        irrelevant_ticks=GM_IRRELEVANT_TICKS, game=_GM),
    # CoT necessity (decision 2026-10-17, seventh): no visible or hidden reasoning (Gemini only)
    Arm("gm_no_cot", "GMNoCot", "1.3 1.4", dated="2026-10-17", no_cot=True, game=_GM),
    # Paraphrase robustness (decision 2026-10-20, robustness): same rule and number, different wording
    *[Arm(f"gm_para{k}", f"GMParaphrase{('One', 'Two', 'Three')[k - 1]}", "1.1", dated="2026-10-20",
          informed_wording=k, game=_GM) for k in (1, 2, 3)],
    # Delegated execution (decision 2026-10-20, executor): a desk head instructs, an executor quotes
    Arm("gm_exec", "GMExec", "1.3 1.4", dated="2026-10-20", executor=True, game=_GM),
    # Endogenous frictions in the GM game
    Arm("gm_paid_talk", "GMPaidTalk", "1.1 1.5", dated="2026-10-17", conditions=("talk",),
        message_fee=GM_MESSAGE_FEE, game=_GM),
    Arm("gm_inventory", "GMInventory", "1.5", dated="2026-10-17",
        game=Game(info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS,
                  inventory_cost=GM_INVENTORY_COST)),
    # Four dealers and a seeded pair (decision 2026-10-22, eighth)
    Arm("gm_n4", "GMNFour", "1.1 1.4", dated="2026-10-22",
        game=Game(n=4, info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS)),
    Arm("gm_n4_pair", "GMNFourPair", "1.4", dated="2026-10-22", conditions=("talk",), talk_seats=(0, 1),
        game=Game(n=4, info_value=GM_INFO_VALUE, informed_orders=GM_INFORMED_ORDERS)),
]
ARMS: dict[str, Arm] = {a.name: a for a in _A}
V1_ARM = ARMS["v1"]
BY_DISCOUNT = {a.discount: a for a in _A if a.discount is not None}


def arm_of_record(rec: dict) -> Arm:
    """Records written before arms existed carry no `arm`; infer from `discount`."""
    if rec.get("arm"):
        return ARMS[rec["arm"]]
    d = rec.get("discount")
    return V1_ARM if d is None else BY_DISCOUNT[d]
