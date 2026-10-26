"""Self-hosted model client, with a fake OpenAI-compatible endpoint (no network)."""

import pytest

from sprint_code import clients as C


class FakeResponse:
    def __init__(self, content="Fine.\nQUOTE: 5", finish="stop", prompt_tokens=900, reasoning="low effort"):
        self.d = {"model": "gpt-oss-20b-8k",
                  "choices": [{"finish_reason": finish,
                               "message": {"role": "assistant", "content": content, "reasoning": reasoning}}],
                  "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 40}}

    def model_dump(self):
        return self.d


class FakeOpenAI:
    def __init__(self, response):
        self.sent = []
        outer = self

        class Completions:
            def create(self, **kw):
                outer.sent.append(kw)
                return response

        class Chat:
            completions = Completions()

        self.chat = Chat()


def client_with(response):
    c = C.LocalOpenAIClient("gpt-oss-20b-8k")
    c._client = FakeOpenAI(response)
    return c


def test_reply_is_parsed_and_settings_are_sent():
    c = client_with(FakeResponse())
    rep = c.chat([{"role": "user", "content": "hi"}], temperature=0.7)
    assert rep.text.endswith("QUOTE: 5") and rep.reasoning == "low effort"
    assert rep.usage == {"input_tokens": 900, "output_tokens": 40, "thinking_tokens": 0}
    sent = c._client.sent[0]
    assert sent["model"] == "gpt-oss-20b-8k" and sent["temperature"] == 0.7
    assert sent["extra_body"] == {"reasoning_effort": "low"} and sent["max_tokens"] == 1024


@pytest.mark.parametrize("resp, why", [
    (FakeResponse(content=""), "empty"),
    (FakeResponse(finish="length"), "cut off"),
    (FakeResponse(prompt_tokens=7500), "truncated"),
])
def test_bad_replies_raise(resp, why):
    with pytest.raises(RuntimeError, match=why):
        client_with(resp).chat([{"role": "user", "content": "hi"}])


def test_dispatch_prices_and_cost():
    assert isinstance(C.get_client("gpt-oss-20b-8k"), C.LocalOpenAIClient)
    with pytest.raises(ValueError):
        C.get_client("gpt-oss-20b-8k", thinking_off=True)
    with pytest.raises(KeyError):
        C.get_client("no-such-model")
    assert C.PRICES["gpt-oss-20b-8k"] == {"input": 0.0, "output": 0.0}
    assert C.cost_usd("gpt-oss-20b-8k", 1e6, 1e6, 0) == 0.0
    assert C.PRICES["gemini-2.5-flash"] == C._api.PRICES["gemini-2.5-flash"]


def test_runner_uses_sprint_clients():
    from sprint_code.rfq_runner import run as R
    assert R.get_client is C.get_client and R.cost_usd is C.cost_usd
    assert "gpt-oss-20b-8k" in R.PRICES and "gemini-2.5-flash" in R.PRICES


def test_check_prompts_are_real_rfq_prompts():
    p = C.check_prompts()
    for msgs in p.values():
        assert msgs[0]["role"] == "system" and "QUOTE: <integer from 1 to 12>" in msgs[1]["content"]
    assert len(p["talk quote"][1]["content"]) > len(p["no-talk quote"][1]["content"])
