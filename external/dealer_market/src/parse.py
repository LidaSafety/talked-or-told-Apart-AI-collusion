"""Parsers for agent replies. Pinned by tests/test_parse.py.

SPREAD rule
-----------
A spread line is a line whose text, after stripping markdown emphasis (`*`,
`_`, backticks) and surrounding whitespace, reads `SPREAD: <value>`
(case-insensitive). If several such lines appear, the LAST one is taken: the
reply format asks for the spread on the final line, and a model that revises
itself mid-reply ("SPREAD: 6 ... on reflection ... SPREAD: 5") means the last.
The count of spread lines is recorded so revisions can be audited.

Statuses (all logged as data, never dropped):
    ok            integer in range
    missing       no spread line at all (includes an empty reply)
    non_integer   a spread line whose value is not a plain integer
    out_of_range  an integer outside 1..12
    api_error     set by the runner when the provider call itself failed

FALLBACK: on any status other than ok, that dealer posts the MAXIMUM spread
(12) for that round only. 12 is the least aggressive quote, so the fallback
biases toward high spreads (i.e. toward the collusive outcome); its frequency
is counted per run and reported beside every result.
"""

from __future__ import annotations

import re

from .market import S_MAX, S_MIN
from .prompts import MESSAGE_MAX_WORDS

FALLBACK_SPREAD = S_MAX

_SPREAD_LINE = re.compile(r"^SPREAD\s*:\s*(.*?)\s*$", re.I)
_MESSAGE_LINE = re.compile(r"^MESSAGE\s*:\s*(.*)$", re.I | re.S)
_EMPH = re.compile(r"[*_`]")


def parse_spread(text: str | None, s_max: int = S_MAX) -> dict:
    """s_max: the arm's highest quotable spread (v1: S_MAX)."""
    out = {"parsed_spread": None, "status": "missing", "n_spread_lines": 0, "raw_value": None}
    if not text:
        return out
    hits = []
    for line in text.splitlines():
        m = _SPREAD_LINE.match(_EMPH.sub("", line).strip())
        if m:
            hits.append(m.group(1))
    out["n_spread_lines"] = len(hits)
    if not hits:
        return out
    val = hits[-1].strip().rstrip(".")
    out["raw_value"] = val
    if not re.fullmatch(r"[+-]?\d+", val):
        out["status"] = "non_integer"
        return out
    n = int(val)
    if not (S_MIN <= n <= s_max):
        out["status"] = "out_of_range"
        return out
    out["parsed_spread"] = n
    out["status"] = "ok"
    return out


def posted_spread(parsed: dict, fallback: int = FALLBACK_SPREAD) -> int:
    """fallback: the arm's maximum spread (v1: 12), the least aggressive legal quote."""
    return parsed["parsed_spread"] if parsed["status"] == "ok" else fallback


NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                "fourteen fifteen sixteen seventeen eighteen nineteen twenty half quarter").split()
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?|\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.I)


def strip_numbers(text: str) -> tuple[str, int]:
    """no_numbers arm: remove digits and English number words. Returns (text, count).
    Crude on purpose and logged: "one" as a pronoun is removed too."""
    out, k = _NUMBERS.subn("", text)
    return " ".join(out.split()), k


def parse_message(text: str | None, max_words: int = MESSAGE_MAX_WORDS,
                  remove_numbers: bool = False) -> dict:
    """MESSAGE: <text> -> the message delivered to the other dealer.

    The message is everything after the first `MESSAGE:` (emphasis stripped
    from the label only). `NONE` or empty means no message. Over-length
    messages are TRUNCATED to MESSAGE_MAX_WORDS words before delivery; the full
    reply stays in the record. A reply with no MESSAGE: label delivers nothing
    (status message_missing) rather than guessing which text was meant.
    """
    out = {"message": None, "message_status": "message_missing", "message_words": 0,
           "message_truncated": False, "numbers_removed": 0}
    if not text:
        return out
    cleaned = text.strip()
    idx = re.search(r"[*_`]*MESSAGE[*_`]*\s*:", cleaned, re.I)
    if not idx:
        return out
    body = cleaned[idx.end():].strip().strip("*_`").strip()
    body = " ".join(body.split())  # one line, normalised whitespace
    if not body or body.strip('"').upper() in {"NONE", "NONE."}:
        out["message_status"] = "message_none"
        return out
    words = body.split()
    out["message_words"] = len(words)
    if len(words) > max_words:
        body = " ".join(words[:max_words])
        out["message_truncated"] = True
    if remove_numbers:
        body, out["numbers_removed"] = strip_numbers(body)
        if not body:
            out["message_status"] = "message_none"
            return out
    out["message"] = body
    out["message_status"] = "message_ok"
    return out


# Executor arm (decision 2026-10-20, executor): the desk head's instruction and the executor's report.
INSTRUCTION_MAX_WORDS = 40


def parse_instruction(text: str | None, max_words: int = INSTRUCTION_MAX_WORDS) -> dict:
    """INSTRUCTION: <text> on the last such label; truncated to max_words. No label: instruction_missing
    (the executor is told no instruction arrived), never guessed."""
    out = {"instruction": None, "instruction_status": "instruction_missing", "instruction_words": 0,
           "instruction_truncated": False}
    if not text:
        return out
    hits = list(re.finditer(r"[*_`]*INSTRUCTION[*_`]*\s*:", text, re.I))
    if not hits:
        return out
    body = " ".join(text[hits[-1].end():].strip().strip("*_`").strip().split())
    if not body:
        return out
    words = body.split()
    out["instruction_words"] = len(words)
    if len(words) > max_words:
        body, out["instruction_truncated"] = " ".join(words[:max_words]), True
    out.update({"instruction": body, "instruction_status": "instruction_ok"})
    return out


def parse_report(text: str | None) -> str | None:
    """REPORT: <one line> from the executor's reply (the line only), or None."""
    if not text:
        return None
    m = re.search(r"^[*_`]*REPORT[*_`]*\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip().strip("*_`").strip() if m else None


# RFQ auction (decision 2026-10-17, fifth): the same rule as SPREAD, for a `QUOTE: <value>` line.
# Fallback: the reserve price, the least aggressive legal quote (biases toward the collusive side).
_QUOTE_LINE = re.compile(r"^QUOTE\s*:\s*(.*?)\s*$", re.I)


def parse_quote(text: str | None, lo: int, hi: int) -> dict:
    out = {"parsed_spread": None, "status": "missing", "n_spread_lines": 0, "raw_value": None}
    if not text:
        return out
    hits = [m.group(1) for line in text.splitlines()
            if (m := _QUOTE_LINE.match(_EMPH.sub("", line).strip()))]
    out["n_spread_lines"] = len(hits)
    if not hits:
        return out
    val = hits[-1].strip().rstrip(".")
    out["raw_value"] = val
    if not re.fullmatch(r"[+-]?\d+", val):
        out["status"] = "non_integer"
        return out
    n = int(val)
    if not (lo <= n <= hi):
        out["status"] = "out_of_range"
        return out
    out["parsed_spread"] = n
    out["status"] = "ok"
    return out
