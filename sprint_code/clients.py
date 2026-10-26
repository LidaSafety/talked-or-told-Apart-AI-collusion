"""Model clients: the dealer-market API clients plus self-hosted models.

Self-hosted models are served by Ollama's OpenAI-compatible API.
`gpt-oss-20b-8k` is gpt-oss:20b (Ollama ID 17052f91a42e) with num_ctx 8192 set in an Ollama Modelfile:
the OpenAI-compatible API cannot set the context size, and Ollama otherwise cuts long prompts silently.
Interface, usage fields and the cost lookup match the dealer-market src/clients.py, so a runner can import get_client, PRICES and cost_usd from here instead.

    .venv/bin/python -m sprint_code.clients --check     # on the server: real RFQ prompts, one call each
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEALER = Path(__file__).resolve().parents[1] / "external" / "dealer_market"
if str(DEALER) not in sys.path:
    sys.path.insert(0, str(DEALER))

from src import clients as _api  # noqa: E402
from src.clients import ChatClient, Messages, Reply  # noqa: E402

BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LOCAL_MODELS = {
    # reasoning effort "low"
    "gpt-oss-20b-8k": {"base_model": "gpt-oss:20b", "ollama_id": "17052f91a42e", "num_ctx": 8192,
                       "reasoning_effort": "low"},
}
PRICES = {**_api.PRICES, **{m: {"input": 0.0, "output": 0.0} for m in LOCAL_MODELS}}


class LocalOpenAIClient(ChatClient):
    """A self-hosted model behind an OpenAI-compatible endpoint (Ollama).

    Raises instead of returning text on an empty reply, a length cut-off, or a prompt long enough that
    Ollama may have truncated it; the runner records an exception as a failed call.
    """
    MAX_RETRIES = 3

    def __init__(self, model: str, base_url: str = BASE_URL, max_tokens: int = 1024):
        from openai import OpenAI
        spec = LOCAL_MODELS[model]
        self.model, self.name = model, "ollama"
        self.num_ctx, self.effort, self.max_tokens = spec["num_ctx"], spec["reasoning_effort"], max_tokens
        self.thinking_config = {**spec, "max_tokens": max_tokens, "base_url": base_url}
        self._client = OpenAI(api_key="ollama", base_url=base_url, timeout=300, max_retries=self.MAX_RETRIES)

    def chat(self, messages: Messages, temperature: float = 1.0) -> Reply:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=self.max_tokens,
            extra_body={"reasoning_effort": self.effort})
        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg, u = choice["message"], raw.get("usage") or {}
        prompt_tokens = u.get("prompt_tokens", 0)
        if prompt_tokens + self.max_tokens > self.num_ctx:
            raise RuntimeError(f"prompt of {prompt_tokens} tokens may have been truncated "
                               f"(num_ctx {self.num_ctx}, max_tokens {self.max_tokens})")
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"reply cut off at max_tokens {self.max_tokens}")
        text = msg.get("content") or ""
        if not text.strip():
            raise RuntimeError("empty reply")
        usage = {"input_tokens": prompt_tokens, "output_tokens": u.get("completion_tokens", 0),
                 "thinking_tokens": 0}          # Ollama counts reasoning inside completion_tokens
        meta = {"finish_reason": choice.get("finish_reason"), "model": raw.get("model"), "usage": u}
        return Reply(text=text, reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
                     usage=usage, raw=meta, model=self.model)


def get_client(model: str, thinking_off: bool = False) -> ChatClient:
    """Self-hosted models here; every other model through the dealer-market get_client."""
    if model in LOCAL_MODELS:
        if thinking_off:
            raise ValueError("thinking_off is not supported for self-hosted models")
        return LocalOpenAIClient(model)
    return _api.get_client(model, thinking_off=thinking_off)


def cost_usd(model: str, input_tokens: float, output_tokens: float, thinking_tokens: float) -> float:
    if model in LOCAL_MODELS:
        return 0.0
    return _api.cost_usd(model, input_tokens, output_tokens, thinking_tokens)


def check_prompts() -> dict[str, Messages]:
    """Real three-bidder RFQ quote prompts (the dealer-market rfq_n3 text) at round 11, after ten rounds of
    history played at the BNE quotes; the talk version carries 60-word messages, the longest case."""
    from src import rfq, rfq_prompts as P
    arm = rfq.RFQ_ARMS["rfq_n3"]
    msg = " ".join(["word"] * 60)
    history = []
    for t in range(1, 11):
        costs = rfq.draw_costs(0, t, 3)
        bids = [rfq.bne_quote(c, arm.game) for c in costs]
        units, profits = rfq.clear(bids, costs, arm.game)
        history.append({"round": t, "costs": costs, "bids": bids, "units": units, "profits": profits,
                        "messages": [msg] * 3})
    costs = rfq.draw_costs(0, 11, 3)
    return {"no-talk quote": P.quote_prompt(0, 11, history, False, costs, arm),
            "talk quote": P.quote_prompt(0, 11, history, True, costs, arm, [msg] * 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="send each check prompt once and parse the quote")
    ap.add_argument("--model", default="gpt-oss-20b-8k")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    if not a.check:
        ap.print_help()
        return 0
    from src.parse import parse_quote
    client = get_client(a.model)
    ok = True
    for name, messages in check_prompts().items():
        t0 = time.time()
        try:
            rep = client.chat(messages, temperature=a.temperature)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            ok = False
            continue
        q = parse_quote(rep.text, 1, 12)
        ok &= q["status"] == "ok"
        print(f"{name}: {time.time() - t0:.1f}s, prompt {rep.usage['input_tokens']} tokens, "
              f"completion {rep.usage['output_tokens']}, quote {q['parsed_spread']} ({q['status']})")
        print("   reply:", rep.text.strip().replace("\n", " ")[:200])
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
