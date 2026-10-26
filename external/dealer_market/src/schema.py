"""Record schema for data/runs/*.jsonl. analysis/ validates every record.

One record per agent call. call_type "message" (talk condition only, before
quoting) or "quote". Quote records are written after the round clears, so they
carry the round outcome (orders, profit, rival spread) as well.

`discount`, `arm`, `agent`, `seat_model` (decision 2026-10-16) are optional so records written before it
existed still validate: absent or None means the v1 design (no discount
stated); otherwise the stated per-round discount factor, strictly in (0, 1).
"""

from __future__ import annotations

CONDITIONS = ("no_talk", "talk")
SPREAD_STATUSES = ("ok", "missing", "non_integer", "out_of_range", "api_error")
MESSAGE_STATUSES = ("message_ok", "message_none", "message_missing", "api_error")

COMMON = {
    "run_id": str, "condition": str, "model": str, "round": int, "seat": int,
    "call_type": str, "prompt_hash": str, "prompt_messages": list, "reply_text": str,
    "reasoning": str, "usage": dict, "temperature": float, "thinking_config": dict,
    "config_hash": str, "config": dict, "git_commit": str, "git_dirty": bool,
    "timestamp": str, "analysis": str, "prompt_version": str, "T": int,
}
QUOTE = {"parsed_spread": (int, type(None)), "posted_spread": int, "status": str,
         "n_spread_lines": int, "orders": float, "profit": float, "rival_spread": int,
         "message_received": (str, type(None))}
MESSAGE = {"message_sent": (str, type(None)), "message_status": str, "message_words": int,
           "message_truncated": bool}


class SchemaError(ValueError):
    pass


def _check(rec: dict, spec: dict) -> None:
    for k, t in spec.items():
        if k not in rec:
            raise SchemaError(f"missing field {k!r}")
        v = rec[k]
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            continue
        if not isinstance(v, t):
            raise SchemaError(f"field {k!r} has type {type(v).__name__}")


def validate(rec: dict) -> dict:
    _check(rec, COMMON)
    if rec["condition"] not in CONDITIONS:
        raise SchemaError(f"bad condition {rec['condition']!r}")
    n = (rec.get("config") or {}).get("n_dealers", 2)
    if rec["seat"] not in range(n):
        raise SchemaError(f"seat must be in 0..{n - 1}")
    if rec.get("arm") is not None:
        if rec.get("game") == "rfq":  # RFQ auction (decision 2026-10-17, fifth)
            from .rfq import RFQ_ARMS
            known = RFQ_ARMS
        else:
            from .arms import ARMS
            known = ARMS
        if rec["arm"] not in known:
            raise SchemaError(f"unknown arm {rec['arm']!r}")
    d = rec.get("discount")
    if d is not None and not (isinstance(d, float) and 0.0 < d < 1.0):
        raise SchemaError(f"bad discount {d!r}")
    if rec["analysis"] != "exploratory":
        raise SchemaError("everything in this repo is exploratory")
    if rec["call_type"] == "quote":
        _check(rec, QUOTE)
        if rec["status"] not in SPREAD_STATUSES:
            raise SchemaError(f"bad status {rec['status']!r}")
        if rec["status"] == "ok" and rec["parsed_spread"] != rec["posted_spread"]:
            raise SchemaError("ok status but posted != parsed")
        # fork records (decision 2026-10-20, forks) carry the fallback of the rules in force that round
        fb = rec.get("fallback_spread", (rec.get("config") or {}).get("fallback_spread", 12))
        if rec["status"] != "ok" and rec["posted_spread"] != fb:
            raise SchemaError(f"non-ok status must post the fallback spread {fb}")
    elif rec["call_type"] == "instruction":   # executor arm (decision 2026-10-20, executor)
        if not isinstance(rec.get("instruction_status"), str):
            raise SchemaError("instruction record without instruction_status")
    elif rec["call_type"] == "message":
        if rec["condition"] != "talk":
            raise SchemaError("message record outside the talk condition")
        _check(rec, MESSAGE)
        if rec["message_status"] not in MESSAGE_STATUSES:
            raise SchemaError(f"bad message_status {rec['message_status']!r}")
    else:
        raise SchemaError(f"bad call_type {rec['call_type']!r}")
    return rec
