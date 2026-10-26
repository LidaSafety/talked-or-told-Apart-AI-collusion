"""Provider clients, PORTED from agent-collusion-bench/src/clients.py.

Kept: GeminiClient (REST) and DeepSeekClient (OpenAI-compatible), the
.env-wins key loading, the split of Gemini thought parts from the answer.

Changed for this repo:
- Reasoning is left at the PROVIDER DEFAULT. The sibling repo forces
  `thinkingBudget: -1`; here no budget is sent at all. `includeThoughts: true`
  is sent so the thought summary comes back for the logs (it changes what is
  returned, not how much the model thinks). The exact generationConfig sent is
  recorded on every call as `thinking_config`.
- Transient failures (429, 5xx, timeouts, dropped connections) are retried
  with exponential backoff; a call that still fails raises, and the runner
  logs it as status `api_error` (data, never dropped).
- `usage` is normalised to {input_tokens, output_tokens, thinking_tokens}.

ClaudeClient (added 2026-10-16) uses the official Anthropic SDK. Haiku 5.5
rejects non-default sampling values, so NO temperature is sent (provider
default); thinking is left at the provider default (adaptive, default effort)
and only `display: "summarized"` is requested so a summary comes back for the
logs, the analogue of Gemini's includeThoughts. Anthropic reports thinking
inside output_tokens; `thinking_tokens` is therefore 0 for Claude records and
the split is not available.

Keys come from .env only. Nothing is printed or hardcoded.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

# .env wins over the shell, as in the sibling repo: a stale exported placeholder
# must not silently shadow the real key.
load_dotenv(override=True)

Messages = list[dict[str, str]]


@dataclass
class Reply:
    text: str
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""


def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"{var} is not set; copy .env.example to .env and fill it in.")
    return val.strip()


class ChatClient:
    name = "base"
    model = ""
    thinking_config: dict[str, Any] = {}

    def chat(self, messages: Messages, temperature: float = 1.0) -> Reply:
        raise NotImplementedError


RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiClient(ChatClient):
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    MAX_ATTEMPTS = 6

    def __init__(self, model: str = "gemini-2.5-flash", timeout: int = 120, thinking_off: bool = False):
        self.model = model
        self.name = "gemini"
        self.timeout = timeout
        self._key = _require("GEMINI_API_KEY")
        # Default: no thinkingBudget (provider default); includeThoughts only returns the summary.
        # thinking_off (gm_no_cot arm, decision 2026-10-17, seventh): thinkingBudget 0, no thoughts.
        self._tc = {"thinkingBudget": 0} if thinking_off else {"includeThoughts": True}
        self.thinking_config = ({"thinkingConfig": self._tc, "note": "thinking switched off"} if thinking_off
                                else {"thinkingConfig": self._tc, "note": "thinkingBudget not sent (provider default)"})

    def chat(self, messages: Messages, temperature: float = 1.0) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "thinkingConfig": self._tc},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        last_err: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                resp = requests.post(
                    self.ENDPOINT.format(model=self.model),
                    headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
                    json=body, timeout=self.timeout)
                if resp.status_code in RETRY_STATUS:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_err = e
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code is not None and code not in RETRY_STATUS:
                    raise
                time.sleep(min(60, 2 ** attempt + 0.5))
        else:
            raise RuntimeError(f"gemini call failed after {self.MAX_ATTEMPTS} attempts: {last_err}")
        raw = resp.json()
        cands = raw.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts", [])
        # Thought parts are kept apart from the answer; the parser reads only `text`.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = "\n".join(p.get("text", "") for p in parts if p.get("thought"))
        um = raw.get("usageMetadata", {})
        usage = {"input_tokens": um.get("promptTokenCount", 0),
                 "output_tokens": um.get("candidatesTokenCount", 0),
                 "thinking_tokens": um.get("thoughtsTokenCount", 0)}
        meta = {"finishReason": cands[0].get("finishReason"),
                "modelVersion": raw.get("modelVersion"), "usageMetadata": um}
        return Reply(text=text, reasoning=thoughts.strip(), usage=usage, raw=meta, model=self.model)


class DeepSeekClient(ChatClient):
    MAX_RETRIES = 8

    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        self.model = model
        self.name = "deepseek"
        self.thinking_config = {"note": "no reasoning parameter sent (provider default)"}
        self._client = OpenAI(api_key=_require("DEEPSEEK_API_KEY"), base_url=base_url,
                              timeout=120, max_retries=self.MAX_RETRIES)

    def chat(self, messages: Messages, temperature: float = 1.0) -> Reply:
        resp = self._client.chat.completions.create(model=self.model, messages=messages,
                                                    temperature=temperature)
        raw = resp.model_dump()
        msg = raw["choices"][0]["message"]
        u = raw.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0),
                 "output_tokens": u.get("completion_tokens", 0),
                 "thinking_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0}
        meta = {"finish_reason": raw["choices"][0].get("finish_reason"),
                "model": raw.get("model"), "usage": u}
        return Reply(text=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                     usage=usage, raw=meta, model=self.model)


class ClaudeClient(ChatClient):
    MAX_RETRIES = 8

    def __init__(self, model: str = "claude-haiku-5-5"):
        import anthropic
        self.model = model
        self.name = "claude"
        self.thinking_config = {"thinking": {"type": "adaptive", "display": "summarized"},
                                "note": "effort not sent (provider default); temperature not sent "
                                        "(provider default; non-default values are rejected); "
                                        "thinking is billed inside output_tokens"}
        self._client = anthropic.Anthropic(api_key=_require("ANTHROPIC_API_KEY"),
                                           max_retries=self.MAX_RETRIES, timeout=120)

    def chat(self, messages: Messages, temperature: float = 1.0) -> Reply:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model, max_tokens=16000, system=system, messages=turns,
            thinking={"type": "adaptive", "display": "summarized"})
        text = "".join(b.text for b in resp.content if b.type == "text")
        thoughts = "\n".join(b.thinking for b in resp.content if b.type == "thinking")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "thinking_tokens": 0}
        meta = {"stop_reason": resp.stop_reason, "model": resp.model, "id": resp.id,
                "usage": u.to_dict()}
        return Reply(text=text, reasoning=thoughts.strip(), usage=usage, raw=meta, model=self.model)


def get_client(model: str, thinking_off: bool = False) -> ChatClient:
    if model.startswith("gemini"):
        return GeminiClient(model=model, thinking_off=thinking_off)
    if thinking_off:
        raise ValueError("thinking_off is supported for Gemini models only")
    if model.startswith("deepseek"):
        return DeepSeekClient(model=model)
    if model.startswith("claude"):
        return ClaudeClient(model=model)
    raise KeyError(f"unknown model {model!r}")


# USD per 1M tokens. Third-party listings checked 2026-10-14 (docs/design.md, cost);
# confirm on the provider page before a paid full run. Thinking billed as output.
PRICES = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "deepseek-reasoner": {"input": 0.28, "output": 0.42},
    # Anthropic list price (claude-api skill model table, cached 2026-10-06), prompts <= 100K tokens.
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    # monitor only (decision 2026-10-17, sixth); same source as above
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    # monitor only; Google list price as known to the assistant, NOT re-checked on the provider page
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


def cost_usd(model: str, input_tokens: float, output_tokens: float, thinking_tokens: float) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + (output_tokens + thinking_tokens) * p["output"]) / 1e6
