#!/usr/bin/env python3
"""Narro TTS pipeline benchmark.

Measures startup, cold inference, and warm inference.

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --runs 10          # more iterations
    python benchmarks/bench.py --no-compile       # disable torch.compile
"""

import argparse
import gc
import statistics
import time

# Suppress all warnings for clean output
import warnings
warnings.simplefilter("ignore")

import torch


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_ms(val):
    """Format a time value in ms."""
    return f"{val * 1000:>8.1f} ms"


def fmt_stats(times):
    """Format a list of times as mean +/- std."""
    if len(times) == 1:
        return fmt_ms(times[0])
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    return f"{mean * 1000:>8.1f} +/- {std * 1000:>5.1f} ms"


TEXTS = {
    "short":  "Hello, this is a test.",
    "medium": "The quick brown fox jumps over the lazy dog near the river bank. "
              "She sells seashells by the seashore on a warm summer afternoon.",
    "long":   "In a hole in the ground there lived a hobbit. Not a nasty, dirty, "
              "wet hole, filled with the ends of worms and an oozy smell, nor yet "
              "a dry, bare, sandy hole with nothing in it to sit down on or to eat. "
              "It was a hobbit-hole, and that means comfort.",
}


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(num_runs, compile_flag):
    """Run full benchmark suite."""
    from narro.tts import Narro, TOKEN_SIZE, SAMPLE_RATE

    print(f"\n{'=' * 70}")
    print(f"  compile={compile_flag}  |  runs={num_runs}")
    print(f"{'=' * 70}")

    # ── Startup ──────────────────────────────────────────────────────────
    gc.collect()
    torch.manual_seed(42)

    t0 = time.perf_counter()
    tts = Narro(
        compile=compile_flag,
        quantize=True,
    )
    startup_time = time.perf_counter() - t0
    print(f"\n  Startup (model load + warmup): {fmt_ms(startup_time)}")

    # ── Cold inference (first run) ───────────────────────────────────────
    print(f"\n  {'Text':<10} {'Cold (1st run)':<25} {'Warm (mean +/- std)':<30} {'Tokens':<8}")
    print(f"  {'─' * 10} {'─' * 25} {'─' * 30} {'─' * 8}")

    for label, text in TEXTS.items():
        # Cold run
        gc.collect()
        t0 = time.perf_counter()
        result = tts.infer(text)
        cold_time = time.perf_counter() - t0
        audio_samples = result.shape[0]

        # Warm runs
        warm_times = []
        for _ in range(num_runs):
            gc.collect()
            t0 = time.perf_counter()
            tts.infer(text)
            warm_times.append(time.perf_counter() - t0)

        tokens = audio_samples // TOKEN_SIZE
        print(f"  {label:<10} {fmt_ms(cold_time):<25} {fmt_stats(warm_times):<30} {tokens:<8}")

    # ── Batch inference ──────────────────────────────────────────────────
    all_texts = list(TEXTS.values())
    batch_times = []
    for _ in range(num_runs):
        gc.collect()
        t0 = time.perf_counter()
        tts.infer_batch(all_texts)
        batch_times.append(time.perf_counter() - t0)

    print(f"\n  Batch (3 texts):              {fmt_stats(batch_times)}")

    # ── Real-time factor ─────────────────────────────────────────────────
    medium_result = tts.infer(TEXTS["medium"])
    audio_duration = medium_result.shape[0] / SAMPLE_RATE

    medium_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tts.infer(TEXTS["medium"])
        medium_times.append(time.perf_counter() - t0)
    mean_infer = statistics.mean(medium_times)
    rtf = mean_infer / audio_duration if audio_duration > 0 else float('inf')

    print(f"\n  Real-time factor (medium):     {rtf:.2f}x  "
          f"({audio_duration:.1f}s audio in {mean_infer * 1000:.0f}ms)")

    # ── Cleanup ──────────────────────────────────────────────────────────
    del tts
    gc.collect()

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Narro TTS benchmark")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of warm inference runs (default: 5)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    run_benchmark(args.runs, not args.no_compile)


if __name__ == "__main__":
    main()
