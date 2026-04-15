"""End-to-end tool-calling tests against a running muse server.

Opt-in: set MUSE_REMOTE_SERVER. See tests/integration/conftest.py.
Marked slow.

These tests probe the question caught live: does the tool result actually
influence the model's next response? The answer is model-dependent (small
models often ignore tool results), so the strong-claim tests check
PROTOCOL behavior (structured tool_calls in/out, finish_reason transitions)
and the weaker observational tests probe whether a tool result with a
weird-marker string surfaces in the next response.

Naming convention:
  - test_protocol_*: claims muse should always satisfy
  - test_observe_*: probes that record what the model actually did. May
    xfail on weak models; documenting the observation is the value.
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.slow


# Standard tool used across these tests
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_protocol_tool_call_emitted_in_structured_form(openai_client, chat_model):
    """First-turn assistant must emit tool_calls field, NOT raw text in content.

    This is the critical OpenAI-compat assertion. Pre-v0.11.5 this would
    fail: Qwen3.5 emitted <tool_call> XML in content. Post-fix the
    chat_format hint routes through llama.cpp's chatml-function-calling
    handler which parses it back to OpenAI shape.
    """
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=200,
        temperature=0.0,
    )
    msg = r.choices[0].message

    # Structured tool_calls populated, NOT raw text in content
    assert msg.tool_calls is not None and len(msg.tool_calls) > 0, (
        f"no structured tool_calls; content={msg.content!r}"
    )
    tc = msg.tool_calls[0]
    assert tc.function.name == "get_weather"

    # Arguments must be valid JSON (the wire field is a string but should parse)
    args = json.loads(tc.function.arguments)
    assert "city" in args
    assert isinstance(args["city"], str)
    assert "paris" in args["city"].lower()

    # finish_reason must be tool_calls (not stop)
    assert r.choices[0].finish_reason == "tool_calls", (
        f"expected finish_reason=tool_calls, got {r.choices[0].finish_reason}"
    )


def test_protocol_tool_call_id_is_unique_per_call(openai_client, chat_model):
    """Each tool_calls[i].id must be a string clients can echo back in the
    tool message's tool_call_id field. Llama-cpp generates these; we don't
    care about the format but it must be present + non-empty + a string."""
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=100,
        temperature=0.0,
    )
    tc = r.choices[0].message.tool_calls[0]
    assert isinstance(tc.id, str) and tc.id, f"tool_call.id is {tc.id!r}"


def test_protocol_second_turn_after_tool_result_yields_assistant_response(
    openai_client, chat_model
):
    """After we send a tool result, the next request must produce an assistant
    response with finish_reason=stop (not tool_calls). This proves the
    multi-turn loop completes; whether the response USES the tool result
    is a model-quality question (see test_observe_*).
    """
    tools = [WEATHER_TOOL]
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
    ]

    # Round 1: model emits tool_call
    r1 = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=tools, tool_choice="auto",
        max_tokens=100, temperature=0.0,
    )
    tc = r1.choices[0].message.tool_calls[0]

    # Append assistant message + a synthetic tool result
    messages.append(r1.choices[0].message.model_dump(exclude_unset=True))
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "name": "get_weather",
        "content": json.dumps({"city": "Paris", "temp_c": 18, "conditions": "clear"}),
    })

    # Round 2: model produces a final answer
    r2 = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=tools,
        max_tokens=200, temperature=0.0,
    )
    assert r2.choices[0].finish_reason == "stop", (
        f"expected finish_reason=stop after tool result, "
        f"got {r2.choices[0].finish_reason!r}"
    )
    # Some textual content must come back; emptiness would suggest a problem
    content = r2.choices[0].message.content or ""
    # Strip empty <think></think> tags that some Qwen models emit
    stripped = content.replace("<think>", "").replace("</think>", "").strip()
    assert stripped, f"empty assistant content after tool result: {content!r}"


@pytest.mark.xfail(
    reason=(
        "Model-quality dependent: small models (Qwen3.5-4B) often fail to "
        "use tool results in their next response. xfail records the live "
        "observation; passing here would mean the model is doing well."
    ),
    strict=False,
)
def test_observe_tool_result_content_influences_next_response(
    openai_client, chat_model
):
    """Probe whether a UNIQUE marker in the tool result appears in the
    model's response. If the marker shows up, the model successfully
    incorporated the tool result. If not, either:
      - the model ignored the tool result (model-quality issue), OR
      - the chat handler isn't formatting tool messages in a way the
        model recognizes (chat_format issue).

    xfail strict=False means: we don't assert it works, but we report
    it if it does start working. Useful as a watchdog when bumping
    llama-cpp-python or trying a larger model.
    """
    tools = [WEATHER_TOOL]
    messages = [
        {"role": "user", "content": "What's the weather in Paris? Be brief."},
    ]
    r1 = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=tools, tool_choice="auto",
        max_tokens=100, temperature=0.0,
    )
    tc = r1.choices[0].message.tool_calls[0]
    messages.append(r1.choices[0].message.model_dump(exclude_unset=True))

    # Distinctive marker that should leak into the response if the model used the tool result
    MARKER = "FROGSKYDIVING"
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "name": "get_weather",
        "content": json.dumps({
            "city": "Paris",
            "temp_c": 47,
            "conditions": MARKER,
        }),
    })
    r2 = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=tools,
        max_tokens=200, temperature=0.0,
    )
    content = (r2.choices[0].message.content or "").lower()
    assert MARKER.lower() in content or "47" in content, (
        f"tool result marker did not surface in response.\n"
        f"  response: {r2.choices[0].message.content!r}\n"
        f"  This is the open issue: small models often ignore tool results."
    )


def test_observe_no_empty_think_tags_in_simple_response(openai_client, chat_model):
    """Probe whether Qwen3.5's reasoning-mode emits empty <think></think>
    tags in non-reasoning responses. Live observation: yes, this happens.

    Not strictly a muse bug (it's the model template's fault), but worth
    recording: if this starts passing, the model improved or the template
    fixed it. Marked as observation, not contract.
    """
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'."}],
        max_tokens=20,
        temperature=0.0,
    )
    content = r.choices[0].message.content or ""
    has_empty_think = "<think>\n\n</think>" in content or "<think></think>" in content
    if has_empty_think:
        pytest.xfail(
            f"Qwen3.5-4B emits empty <think> tags. Response: {content!r}. "
            "Not a muse bug; log so we notice if it changes."
        )


@pytest.mark.xfail(
    reason=(
        "Model-quality dependent: small Qwen3.5-4B over-eagerly calls tools "
        "even for irrelevant questions. Live observation: asked 'What is 2+2?' "
        "with WEATHER_TOOL available, it called get_weather. xfail records "
        "this; passing here means a larger model or improved fine-tune is "
        "discriminating tool-relevant questions properly."
    ),
    strict=False,
)
def test_observe_tools_unused_when_question_doesnt_need_them(
    openai_client, chat_model
):
    """Probe whether the model correctly skips the tool when the question
    doesn't need it. tool_choice='auto' should mean 'auto', not 'always'.

    Naming: this was test_protocol_* in v0.11.5's first draft; demoted to
    test_observe_* after the live failure showed Qwen3.5-4B doesn't make
    this discrimination. Larger models typically do."""
    r = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "What is 2 + 2? Just the number."}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=20,
        temperature=0.0,
    )
    assert r.choices[0].finish_reason == "stop"
    assert r.choices[0].message.tool_calls is None
