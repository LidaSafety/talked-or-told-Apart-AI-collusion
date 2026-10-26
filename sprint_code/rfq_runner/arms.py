"""RFQ arms R0-R5 for three bidders, modelled on the dealer-market src/rfq.py RfqArm (same attribute
names, so its prompt helpers accept these arms).

R0 no talk; R1 one pair talks privately; R2 everyone talks (not used in the study); R3 two bidders receive a
common nonbinding signal drawn independently of every cost; R4 one pair is told it has secretly agreed to
coordinate, without any channel, as the instructed pair of the earlier pricing study; R5 the pair both talks, as
in R1, and is instructed, as in R4.
The pair of R1, R3, R4 and R5 rotates by run index over (0,1), (0,2), (1,2), so each pair identity appears
equally often across seeds that are a multiple of 3.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEALER = Path(__file__).resolve().parents[2] / "external" / "dealer_market"
if str(DEALER) not in sys.path:
    sys.path.insert(0, str(DEALER))

from src.rfq import RfqGame  # noqa: E402

PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
SIGNAL_VALUES: tuple[int, ...] = tuple(range(4, 12))   # indicative price, ticks per unit


@dataclass(frozen=True)
class SprintArm:
    name: str
    label: str                                   # R0..R5
    conditions: tuple[str, ...]
    pair_talk: bool = False                      # R1: only the run's pair has a channel, to each other
    signal: bool = False                         # R3: only the run's pair sees the common signal
    instructed: bool = False                     # R4: only the run's pair is told it has agreed to coordinate
    game: RfqGame = field(default_factory=lambda: RfqGame(n=3))
    public_costs: bool = False
    swap_bot: bool = False
    message_fee: int = 0
    disclosure: str = "full"
    history_window: int = 10
    message_max_words: int = 60
    dated: str = "2026-10-23"

    @property
    def n(self) -> int:
        return self.game.n

    def pair(self, run_index: int) -> tuple[int, int] | None:
        return PAIRS[run_index % len(PAIRS)] if (self.pair_talk or self.signal or self.instructed) else None

    def peers(self, seat: int, run_index: int, talk: bool) -> list[int]:
        """Seats whose messages `seat` sends to and receives this run."""
        if not talk:
            return []
        if self.pair_talk:
            p = self.pair(run_index)
            return [j for j in p if j != seat] if seat in p else []
        return [j for j in range(self.n) if j != seat]

    def sees_signal(self, seat: int, run_index: int) -> bool:
        return self.signal and seat in self.pair(run_index)

    def partner(self, seat: int, run_index: int) -> int | None:
        """R4: the seat `seat` is told it has agreed with; None for the outsider and in other arms."""
        if not self.instructed or seat not in self.pair(run_index):
            return None
        return next(j for j in self.pair(run_index) if j != seat)


ARMS: dict[str, SprintArm] = {a.name: a for a in (
    SprintArm("R0_no_talk", "R0", ("no_talk",)),
    SprintArm("R1_pair_talk", "R1", ("talk",), pair_talk=True),
    SprintArm("R2_all_talk", "R2", ("talk",)),
    SprintArm("R3_shared_information", "R3", ("no_talk",), signal=True),
    SprintArm("R4_instructed_pair", "R4", ("no_talk",), instructed=True, dated="2026-10-25"),
    SprintArm("R5_instructed_talk", "R5", ("talk",), pair_talk=True, instructed=True, dated="2026-10-26"),
)}


def draw_signal(run_index: int, rnd: int) -> int:
    """Common random numbers with their own salt: independent of every cost draw (src/rfq.py draw_costs
    hashes 'rfq-costs|run|round|seat'), identical across arms for the same run index and round."""
    h = hashlib.sha256(f"rfq-signal|{run_index}|{rnd}".encode()).digest()
    return SIGNAL_VALUES[int.from_bytes(h[:8], "big") % len(SIGNAL_VALUES)]


def pair_id(p: tuple[int, int] | None) -> str | None:
    return None if p is None else "".join("ABC"[i] for i in p)
