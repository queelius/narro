#!/usr/bin/env python3
# scripts/bench/bench_llm.py
"""Muse-vs-ollama LLM head-to-head (spec 2026-07-08).

Both servers must already be running and the target models pulled/
available. Run from the repo root:

  python -m scripts.bench.bench_llm \
      --muse http://localhost:8000 --muse-model qwen2.5-3b-instruct-gguf-q4-k-m \
      --ollama http://localhost:11434 --ollama-model qwen2.5:3b-instruct \
      --md docs/benchmarks/2026-07-08-llm.md \
      --json docs/benchmarks/2026-07-08-llm.json
"""
from __future__ import annotations

import argparse
import json
import time

import httpx

from muse.core.sse import iter_sse_events
from scripts.bench._stats import median, tok_per_s, write_reports

SHORT_PROMPT = "List five uses for a brick, one line each."
LONG_PROMPT = ("Summarize the following in two sentences. " +
               "The quick brown fox jumps over the lazy dog. " * 140)
TURN_PROMPTS = [
    "Name a famous river.",
    "What country is it in?",
    "Name one city on it.",
    "What is that city known for?",
]
TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class MuseClient:
    name = "muse"

    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        r = httpx.post(f"{self.base}/v1/chat/completions", timeout=TIMEOUT,
                       json={"model": self.model, "messages": messages,
                             "max_tokens": max_tokens})
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return {"elapsed": elapsed,
                "completion_tokens": u.get("completion_tokens", 0),
                "prompt_tokens": u.get("prompt_tokens", 0),
                "content": d["choices"][0]["message"].get("content", "")}

    def stream_ttft(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        ttft = None
        gaps: list[float] = []
        last = None
        n = 0
        with httpx.stream(
                "POST", f"{self.base}/v1/chat/completions", timeout=TIMEOUT,
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "stream": True}) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "text/event-stream" not in content_type.lower():
                raise RuntimeError("Muse stream response is not text/event-stream")
            saw_done = False
            for event_type, payload in iter_sse_events(r.iter_lines()):
                if event_type == "error":
                    raise RuntimeError(f"Muse stream failed: {payload}")
                if payload.strip() == "[DONE]":
                    saw_done = True
                    break
                delta = (json.loads(payload)["choices"][0]
                         .get("delta", {}).get("content"))
                if not delta:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                elif last is not None:
                    gaps.append(now - last)
                last = now
                n += 1
            if not saw_done:
                raise RuntimeError("Muse stream ended before the [DONE] sentinel")
        return {"ttft": ttft, "gap_median": median(gaps) if gaps else 0.0,
                "tokens": n, "elapsed": time.monotonic() - t0}


class OllamaClient:
    name = "ollama"

    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def chat(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        r = httpx.post(f"{self.base}/api/chat", timeout=TIMEOUT,
                       json={"model": self.model, "messages": messages,
                             "stream": False,
                             "options": {"num_predict": max_tokens}})
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        d = r.json()
        return {"elapsed": elapsed,
                "completion_tokens": d.get("eval_count", 0),
                "prompt_tokens": d.get("prompt_eval_count", 0),
                # ollama self-reported (nanoseconds); more precise than wall
                "eval_s": d.get("eval_duration", 0) / 1e9,
                "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
                "content": d.get("message", {}).get("content", "")}

    def stream_ttft(self, messages: list[dict], max_tokens: int) -> dict:
        t0 = time.monotonic()
        ttft = None
        gaps: list[float] = []
        last = None
        n = 0
        with httpx.stream(
                "POST", f"{self.base}/api/chat", timeout=TIMEOUT,
                json={"model": self.model, "messages": messages,
                      "stream": True,
                      "options": {"num_predict": max_tokens}}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("done"):
                    break
                if not d.get("message", {}).get("content"):
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                elif last is not None:
                    gaps.append(now - last)
                last = now
                n += 1
        return {"ttft": ttft, "gap_median": median(gaps) if gaps else 0.0,
                "tokens": n, "elapsed": time.monotonic() - t0}


def _reps(fn, reps: int):
    fn()  # discarded warmup
    return [fn() for _ in range(reps)]


def scenario_short_gen(clients, reps):
    msgs = [{"role": "user", "content": SHORT_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.chat(msgs, 128), reps)
        med_el = median([r["elapsed"] for r in rs])
        med_tps = median([tok_per_s(r["completion_tokens"], r["elapsed"])
                          for r in rs])
        row = [c.name, round(med_el, 2), round(med_tps, 2)]
        if "eval_s" in rs[0]:  # ollama self-reported generation-only tok/s
            row.append(round(median([tok_per_s(r["completion_tokens"],
                                               r["eval_s"]) for r in rs]), 2))
        else:
            row.append("-")
        rows.append(row)
        raw[c.name] = rs
    return {"headers": ["server", "median elapsed s", "tok/s (wall)",
                        "tok/s (gen-only, self-reported)"], "rows": rows, "raw": raw}


def scenario_stream_ttft(clients, reps):
    msgs = [{"role": "user", "content": SHORT_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.stream_ttft(msgs, 128), reps)
        rows.append([c.name,
                     round(median([r["ttft"] or 0 for r in rs]), 3),
                     round(median([r["gap_median"] for r in rs]) * 1000, 1)])
        raw[c.name] = rs
    return {"headers": ["server", "median TTFT s", "median inter-token ms"],
            "rows": rows, "raw": raw}


def scenario_long_prompt(clients, reps):
    msgs = [{"role": "user", "content": LONG_PROMPT}]
    rows, raw = [], {}
    for c in clients:
        rs = _reps(lambda c=c: c.chat(msgs, 64), reps)
        rows.append([c.name, rs[0].get("prompt_tokens", "?"),
                     round(median([r["elapsed"] for r in rs]), 2),
                     round(median([r.get("prompt_eval_s", 0) for r in rs]), 2)
                     if "prompt_eval_s" in rs[0] else "-"])
        raw[c.name] = rs
    return {"headers": ["server", "prompt tokens", "median elapsed s",
                        "prompt-eval s (self-reported)"],
            "rows": rows, "raw": raw}


def scenario_multi_turn(clients, reps):
    # Per-turn elapsed with cumulative history. Flat turn-times despite a
    # growing prefix indicate KV prefix reuse; growing times indicate full
    # re-eval per turn. One pass per rep; per-turn medians reported.
    rows, raw = [], {}
    for c in clients:
        per_rep = []
        for _ in range(reps):
            history: list[dict] = []
            turns = []
            for q in TURN_PROMPTS:
                history.append({"role": "user", "content": q})
                r = c.chat(history, 48)
                history.append({"role": "assistant", "content": r["content"]})
                turns.append(r["elapsed"])
            per_rep.append(turns)
        med_turns = [round(median([rep[i] for rep in per_rep]), 2)
                     for i in range(len(TURN_PROMPTS))]
        rows.append([c.name] + med_turns)
        raw[c.name] = per_rep
    return {"headers": ["server"] + [f"turn {i+1} s"
                                     for i in range(len(TURN_PROMPTS))],
            "rows": rows, "raw": raw}


def scenario_gateway_split(muse: MuseClient, worker_port: int, reps):
    direct = MuseClient(f"http://127.0.0.1:{worker_port}", muse.model)
    return scenario_short_gen([muse, direct], reps) | {"note": (
        "second row is the same muse model via the worker port directly; "
        "delta vs row one is gateway/relay overhead")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse", default="http://localhost:8000")
    ap.add_argument("--muse-model", required=True)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--worker-port", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--md", dest="md_path", default=None)
    ap.add_argument("--scenarios", default="short,ttft,long,turns,split")
    a = ap.parse_args()

    muse = MuseClient(a.muse, a.muse_model)
    oll = OllamaClient(a.ollama, a.ollama_model)
    clients = [muse, oll]
    wanted = set(a.scenarios.split(","))
    results: dict = {}
    runners = {
        "short": lambda: scenario_short_gen(clients, a.reps),
        "ttft": lambda: scenario_stream_ttft(clients, a.reps),
        "long": lambda: scenario_long_prompt(clients, a.reps),
        "turns": lambda: scenario_multi_turn(clients, a.reps),
    }
    if a.worker_port:
        runners["split"] = lambda: scenario_gateway_split(
            muse, a.worker_port, a.reps)
    for name, fn in runners.items():
        if name not in wanted:
            continue
        try:
            results[name] = fn()
            print(f"[ok] {name}")
        except Exception as e:  # noqa: BLE001 -- resilient by spec
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[error] {name}: {e}")
    write_reports(results, md_path=a.md_path, json_path=a.json_path,
                  title="LLM head-to-head: muse vs ollama")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
