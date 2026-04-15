# Integration tests

End-to-end tests against a **real running muse server**. Slow (real GPU
inference) and opt-in (don't run by default).

## Running

```bash
# Point at any reachable muse server
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/ -v

# All integration tests are also `@pytest.mark.slow`, so the default
# `-m "not slow"` lane skips them too. Both gates must clear.
MUSE_REMOTE_SERVER=http://localhost:8000 pytest tests/integration/ -v -m slow

# Test against a different chat model (default: qwen3.5-4b-q4)
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 \
MUSE_CHAT_MODEL_ID=qwen3.5-9b-q4 \
pytest tests/integration/ -v
```

## Env vars

| Var | Purpose | Default |
| --- | --- | --- |
| `MUSE_REMOTE_SERVER` | URL of the muse server to test | unset (skips all integration tests) |
| `MUSE_CHAT_MODEL_ID` | Chat model id for chat + tool tests | `qwen3.5-4b-q4` |

## What gets skipped automatically

Each test self-skips when the precondition isn't met, so you can leave
`MUSE_REMOTE_SERVER` set permanently in your env without breaking the
fast lane:

| Skip reason | When |
| --- | --- |
| `MUSE_REMOTE_SERVER not set` | env var unset |
| `muse server not reachable` | no response on `/health` within 5s |
| `non-muse /health body` | `/health` doesn't have `modalities` and `models` |
| `model not loaded` | the test requires e.g. `qwen3.5-4b-q4` but it's not in `/health.models` |

Add new "model gate" fixtures in `conftest.py` via `require_model_fixture("model-id")`.

## Test layout

- `test_remote_chat.py`: non-streaming + streaming chat, JSON mode, response.model field, deterministic decoding
- `test_remote_tools.py`: tool calling protocol + multi-turn loop + observational probes (xfail-style) for whether the model actually USES tool results

## Naming convention

- `test_protocol_*`: claims muse should always satisfy (will fail loudly)
- `test_observe_*`: probes that record what a particular model actually did. May `xfail` on weak models. Useful as watchdogs.

## Why not just hit the server in unit tests?

Two reasons:

1. **Speed**: chat responses take 1-30 seconds. Unit tests run in milliseconds.
2. **Flakiness**: GPU availability, model download state, network latency. Integration tests fail for environmental reasons; we don't want them blocking PRs.

The unit tests in `tests/modalities/chat_completion/test_routes_messages_passthrough.py` cover the muse-side contract (every field forwards to the backend untouched). These integration tests cover the *combination* of muse + llama-cpp-python + a specific model.
