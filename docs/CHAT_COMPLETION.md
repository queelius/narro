# Chat Completion modality

Modality tag: `chat/completion`. HTTP path: `POST /v1/chat/completions`.
OpenAI-shape request and response. v0.10.0 ships one runtime,
`LlamaCppModel`, which serves any GGUF model in-process via
`llama-cpp-python`.

## Endpoint

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen3-8b-gguf-q4-k-m",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 200,
  "stream": false
}
```

Response (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1742000000,
  "model": "qwen3-8b-gguf-q4-k-m",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Paris."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 24, "completion_tokens": 2, "total_tokens": 26}
}
```

## Streaming

`stream: true` switches the response to Server-Sent Events with the
OpenAI `chat.completion.chunk` shape:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","model":"...","choices":[{"index":0,"delta":{"content":"Pa"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","model":"...","choices":[{"index":0,"delta":{"content":"ris."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Tokens dispatch as the model produces them (producer thread + asyncio
queue; no server-side buffering). Last line is the literal `[DONE]`
sentinel.

## Tool calling

Pass `tools` and optionally `tool_choice` per OpenAI's spec:

```json
{
  "model": "qwen3-8b-gguf-q4-k-m",
  "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

The model's response message will include a `tool_calls` array when it
decides to call a function:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
    }
  ]
}
```

Tool-call quality is the **model's** responsibility, not muse's. muse
passes `tools` through verbatim to llama-cpp-python's
`create_chat_completion`, which in turn relies on the model's chat
template for tool formatting.

### Tool-use behavior matrix

This is what we've actually observed end-to-end against muse, not
guesses. The two halves of the round-trip (call out, result back) are
asymmetric in current llama-cpp-python:

| Capability | What works | What's spotty |
|---|---|---|
| Model emits `tool_calls` | Qwen3.5 (all sizes), Hermes-3, Functionary, Llama-3.1+ | base Llama-2, fine-tunes without tool-aware templates |
| Tool result is incorporated into next response | Qwen3.5-9B and larger, Hermes-3, Functionary | Qwen3.5-4B and smaller (often ignores the result) |

**Why the asymmetry:** llama-cpp-python's `chatml-function-calling`
chat handler parses tool calls *out* of model responses correctly, but
formats tool result messages going *into* the model in a way some
chat templates (notably Qwen's) don't always recognize. Larger models
tolerate the format mismatch via in-context inference; smaller ones
don't. Upstream tracking:
[abetlen/llama-cpp-python#2063](https://github.com/abetlen/llama-cpp-python/issues/2063).

**Recommendation for tool-use today:** use models 9B+ for any agent
loop that depends on tool-result interpretation. The bundled curated
list (`muse models list --modality chat/completion`) includes
`qwen3.5-9b-q4` and `qwen3.5-27b-q4` for this reason.

### supports_tools sniffing

When pulling a GGUF, the resolver sniffs `tokenizer_config.json` for
`{% if tools %}` markers and writes `capabilities.supports_tools` into
the manifest. Surfaced as `supports_tools` in `/v1/models`. This is a
hint, not a gate; muse never refuses tool requests.

## Structured output

Pass `response_format` per OpenAI's spec:

```json
{
  "messages": [{"role": "user", "content": "List 3 colors as JSON."}],
  "response_format": {"type": "json_object"}
}
```

Or with a JSON schema:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {"schema": {"type": "object", "properties": {...}}}
  }
}
```

llama-cpp-python implements both via its grammar engine. Quality
depends on the model's instruction following.

## Sampling parameters

All optional, all forwarded to the backend:

| Field | Type | Default | Notes |
|---|---|---|---|
| `temperature` | float | model default | 0 = greedy |
| `top_p` | float | model default | nucleus sampling |
| `max_tokens` | int | model default (often unlimited) | hard cap |
| `stop` | str or list | none | strings to stop at |
| `seed` | int | random | for reproducibility |
| `logprobs` | bool | false | include token logprobs |
| `top_logprobs` | int | none | top-k alternatives per position |
| `frequency_penalty` | float | 0 | per OpenAI spec |
| `presence_penalty` | float | 0 | per OpenAI spec |
| `repeat_penalty` | float | 1.0 | llama.cpp specific (not in OpenAI) |

`extra_body` (dict) lets you pass any other backend-specific kwarg
through unmodified.

## Python client

```python
from muse.modalities.chat_completion import ChatClient

c = ChatClient()  # MUSE_SERVER env or http://localhost:8000

# Non-streaming
r = c.chat(
    model="qwen3-8b-gguf-q4-k-m",
    messages=[{"role": "user", "content": "Capital of France?"}],
)
print(r["choices"][0]["message"]["content"])

# Streaming
for chunk in c.chat_stream(
    model="qwen3-8b-gguf-q4-k-m",
    messages=[{"role": "user", "content": "Tell me a story"}],
):
    delta = chunk["choices"][0]["delta"].get("content", "")
    print(delta, end="", flush=True)
```

## Using the OpenAI SDK against muse

Because the wire contract is OpenAI-compatible, the OpenAI Python SDK
works against muse with no modifications:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

resp = client.chat.completions.create(
    model="qwen3-8b-gguf-q4-k-m",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

Same for streaming, tool calls, structured output. Most clients
written against OpenAI's API (Cline, Aider, Continue, LibreChat) work
out of the box by pointing them at muse's URL.

## What's NOT supported in v0.10.0

- **Vision input** (image content blocks in user messages). GGUF vision
  models work via llama.cpp's mmproj feature, but the route doesn't
  yet wire that up. Coming in a future release.
- **Reasoning content blocks** (o1-style). Open models like DeepSeek-R1
  emit `<think>...</think>` in plain text, which passes through
  unchanged but isn't surfaced as a separate content block.
- **`n > 1`** (multiple choices per request). Routes accept it but
  the backend ignores; you always get one choice.
- **Anthropic-dialect `/v1/messages` endpoint.** Anthropic SDK v0.40+
  can be pointed at OpenAI-compatible endpoints, so most users don't
  need this. A future release may add a translation adapter.

## Pulling a chat model

Two ways:

```bash
# Resolver path (recommended for GGUFs)
muse pull hf://Qwen/Qwen3-8B-GGUF@q4_k_m

# Then start the server
muse serve
```

For GGUF variants, `@variant` is required. List options with `muse search`:

```bash
muse search qwen3 --modality chat/completion --max-size-gb 10
```

The resolver synthesizes a MANIFEST referencing `LlamaCppModel`,
creates a per-model venv with `llama-cpp-python` installed, downloads
just the chosen GGUF file (skipping other quants in the same repo to
save bandwidth), and persists the manifest in `~/.muse/catalog.json`.

## Implementation notes

- `LlamaCppModel` runs in-process inside the per-model venv. This
  matches the rest of muse's worker-per-venv shape. There is no Ollama
  proxy mode; if you want to federate to Ollama instead, write an
  `ollama://` resolver that returns a different runtime class.
- `n_gpu_layers=-1` by default (offload everything that fits on GPU).
  Override per-pull by editing the persisted manifest's
  `capabilities.n_gpu_layers`, or by passing as a kwarg in the
  loader's call to `load_backend`.
- Default `context_length` is 8192 if the manifest doesn't override
  it. The GGUF header carries its own context length and llama-cpp
  respects it; the manifest value just sets the active KV-cache size.
