#!/usr/bin/env python3
"""Benchmark narro server with ~5 min of varied prose.

Measures wall-clock time, audio duration, and RTF (Real-Time Factor)
for both non-streaming and streaming endpoints.

Usage:
    python benchmarks/long_bench.py http://gpu-box:8000
    python benchmarks/long_bench.py http://localhost:8000
    python benchmarks/long_bench.py http://localhost:8000 --streaming
"""

import argparse
import base64
import json
import sys
import time

import requests

from narro.tts import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Corpus: ~5 minutes of varied prose styles
# ---------------------------------------------------------------------------

PARAGRAPHS = [
    # --- Narrative (Dickens, public domain) ---
    (
        "narrative",
        "It was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
        "incredulity, it was the season of Light, it was the season of Darkness, it was "
        "the spring of hope, it was the winter of despair, we had everything before us, "
        "we had nothing before us, we were all going direct to Heaven, we were all going "
        "direct the other way."
    ),
    (
        "narrative",
        "There were a king with a large jaw and a queen with a plain face, on the throne "
        "of England; there were a king with a large jaw and a queen with a fair face, on "
        "the throne of France. In both countries it was clearer than crystal to the lords "
        "of the State preserves of loaves and fishes, that things in general were settled "
        "for ever."
    ),
    (
        "narrative",
        "The night was cold and wet, but beyond the dripping and the chilling, no unusual "
        "circumstance attended the departure. The coach lumbered up the hill, with its "
        "three passengers. The guard suspected the passengers, the passengers suspected "
        "one another and the guard, and the coachman was sure of nothing but the horses."
    ),

    # --- Technical / expository ---
    (
        "technical",
        "The transformer architecture processes input sequences through multi-head "
        "self-attention layers, each computing scaled dot-product attention over queries, "
        "keys, and values. The attention weights are computed as the softmax of the "
        "inner product between query and key vectors, divided by the square root of the "
        "key dimension. This mechanism allows the model to attend to different positions "
        "in the input simultaneously, capturing both local and long-range dependencies."
    ),
    (
        "technical",
        "In survival analysis, the hazard function describes the instantaneous rate of "
        "failure at time t, given survival up to that point. For a Weibull distribution "
        "with shape parameter k and scale parameter lambda, the hazard is monotonically "
        "increasing when k is greater than one, constant when k equals one, and decreasing "
        "when k is less than one. This flexibility makes the Weibull family widely used in "
        "reliability engineering and medical research."
    ),
    (
        "technical",
        "Encrypted search systems face a fundamental tension between query efficiency and "
        "information leakage. Deterministic encryption enables fast equality searches but "
        "reveals frequency patterns. Randomized schemes hide frequency but require linear "
        "scans or specialized index structures. The optimal tradeoff depends on the threat "
        "model, the query workload, and the adversary's background knowledge about the "
        "data distribution."
    ),

    # --- Dialogue ---
    (
        "dialogue",
        '"I don\'t think you understand what we\'re dealing with here," said the professor, '
        'removing her glasses. "This isn\'t just another anomaly in the data. The signal '
        'has been repeating every forty-seven minutes for the last three weeks." She paused, '
        'then added quietly, "And it\'s getting stronger."'
    ),
    (
        "dialogue",
        '"But that\'s impossible," replied Chen, leaning forward. "We checked every '
        "instrument twice. The calibration logs are clean. There's no way a natural source "
        'could produce that kind of periodicity." He looked around the room. "Unless '
        "someone is telling us we're not alone.\""
    ),
    (
        "dialogue",
        'The captain studied the readout and frowned. "All hands, this is the bridge. We '
        "are altering course to bearing two-seven-zero. Engineering, I need maximum output "
        "from the reactor within fifteen minutes. Navigation, plot an intercept trajectory "
        'and send it to my console." He turned to his first officer. "This is either the '
        'discovery of the century or the worst mistake of our careers."'
    ),

    # --- Numbers, abbreviations, special characters ---
    (
        "numbers",
        "The experiment ran from January 15th, 2024 through March 3rd, 2025, enrolling "
        "12,847 participants across 47 countries. The treatment group received 250mg doses "
        "every 8 hours, while the control group received a placebo. Results showed a 34.7% "
        "improvement in the primary endpoint, with a p-value of 0.003 and a 95% confidence "
        "interval of 28.1% to 41.3%."
    ),
    (
        "numbers",
        "Dr. R. J. Thompson of MIT presented findings at the IEEE conference in Washington, "
        "D.C. The study cost approximately $4.2 million over 3 years. CPU utilization "
        "peaked at 97.3% during the 48-hour stress test, processing 1.5 million requests "
        "per second. Memory usage held steady at 16GB out of 64GB available."
    ),

    # --- Descriptive / atmospheric ---
    (
        "descriptive",
        "There was a steaming mist in all the hollows, and it had roamed in its forlornness "
        "up the hill, like an evil spirit, seeking rest and finding none. A clammy and "
        "intensely cold mist, it made its slow way through the air in ripples that visibly "
        "followed and overspread one another, as the waves of an unwholesome sea might do. "
        "It was dense enough to shut out everything from the light of the coach lamps but "
        "these its own workings, and a few yards of road."
    ),
    (
        "descriptive",
        "The valley stretched out below them in the grey light of early morning. Frost "
        "clung to every blade of grass, turning the meadows into fields of silver. A thin "
        "ribbon of smoke rose from a chimney in the village, the only sign of life in the "
        "vast stillness. Somewhere in the distance, a church bell began to toll, its sound "
        "carrying across the frozen landscape with a clarity that seemed almost unreal."
    ),
    (
        "descriptive",
        "The library was enormous, stretching up three stories with a spiral staircase "
        "connecting the galleries. Thousands of leather-bound volumes lined the walls, their "
        "spines catching the afternoon light that filtered through tall arched windows. The "
        "air smelled of old paper and wood polish. A fire crackled in the stone hearth at "
        "the far end, and two high-backed chairs faced it, casting long shadows across the "
        "oriental carpet."
    ),

    # --- Instructional / procedural ---
    (
        "instructional",
        "To configure the system, first create a configuration file in your home directory. "
        "Set the server address to the hostname of your primary node, and specify the port "
        "number. Next, generate an authentication token using the command-line tool. Copy "
        "the token into the configuration file under the credentials section. Finally, "
        "restart the service and verify connectivity by running the built-in health check."
    ),

    # --- Philosophical / abstract ---
    (
        "philosophical",
        "The question of machine consciousness resists easy answers. If we define "
        "consciousness as subjective experience, we face the hard problem: no amount of "
        "functional description tells us what it is like to be a system processing "
        "information. Yet if we define it purely by behavior, then sufficiently sophisticated "
        "programs might qualify. The philosophical zombie argument cuts both ways, leaving "
        "us uncertain whether the lights are on behind any pair of digital eyes."
    ),
    (
        "philosophical",
        "Consider the ship of Theseus, rebuilt plank by plank over decades until no original "
        "material remains. Is it the same ship? The puzzle generalizes to neural networks "
        "fine-tuned on new data, to human brains replacing their neurons over a lifetime, "
        "and to institutions that outlast every founding member. Identity, it seems, is not "
        "a property of matter but of pattern, and patterns persist only as long as something "
        "cares to maintain them."
    ),
]

TEXT = "\n\n".join(text for _, text in PARAGRAPHS)


def style_breakdown():
    """Return {style: (count, total_chars)} for the corpus."""
    breakdown = {}
    for style, text in PARAGRAPHS:
        count, chars = breakdown.get(style, (0, 0))
        breakdown[style] = (count + 1, chars + len(text))
    return breakdown


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_nonstreaming(server_url, text):
    """POST full text, receive complete WAV. Returns (elapsed_s, audio_bytes)."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/audio/speech",
        json={"input": text, "response_format": "wav"},
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return elapsed, resp.content


def bench_streaming(server_url, text):
    """POST text with stream=true, consume SSE chunks. Returns (ttfb_s, total_s, audio_bytes)."""
    t0 = time.perf_counter()
    ttfb = None
    audio_chunks = []

    resp = requests.post(
        f"{server_url}/v1/audio/speech",
        json={"input": text, "response_format": "wav", "stream": True},
        timeout=600,
        stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8', errors='replace')
        if line_str.startswith('data: '):
            payload = line_str[6:]
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get('type') == 'speech.audio.done':
                break
            audio_b64 = chunk.get('audio')
            if audio_b64:
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                audio_chunks.append(audio_b64)

    total = time.perf_counter() - t0
    raw = b''.join(base64.b64decode(c) for c in audio_chunks)
    return ttfb or total, total, raw


def audio_duration_from_wav(wav_bytes):
    """Compute duration in seconds from raw WAV bytes (16-bit PCM, 32kHz)."""
    audio_samples = (len(wav_bytes) - 44) // 2  # 44-byte header, 16-bit samples
    return audio_samples / SAMPLE_RATE


def audio_duration_from_pcm(pcm_bytes):
    """Compute duration from raw 16-bit PCM at 32kHz."""
    return (len(pcm_bytes) // 2) / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark narro TTS server (~5 min speech)")
    parser.add_argument("server_url", help="Server URL (e.g. http://localhost:8000)")
    parser.add_argument("--streaming", action="store_true", help="Also run streaming benchmark")
    parser.add_argument("--save", action="store_true", help="Save output audio to benchmarks/")
    args = parser.parse_args()

    server_url = args.server_url.rstrip('/')

    # Health check
    print(f"Server: {server_url}")
    resp = requests.get(f"{server_url}/health", timeout=5)
    resp.raise_for_status()
    health = resp.json()
    print(f"Device: {health['device']}  Model: {health['model']}")
    print()

    # Corpus stats
    words = len(TEXT.split())
    chars = len(TEXT)
    print(f"Corpus: {chars:,} chars, {words:,} words, {len(PARAGRAPHS)} paragraphs")
    print()
    print("  Style breakdown:")
    for style, (count, style_chars) in sorted(style_breakdown().items()):
        print(f"    {style:<16} {count} para, {style_chars:>5} chars")
    print()

    # --- Non-streaming benchmark ---
    print("=" * 60)
    print("NON-STREAMING (full WAV response)")
    print("=" * 60)

    elapsed, wav_bytes = bench_nonstreaming(server_url, TEXT)
    audio_s = audio_duration_from_wav(wav_bytes)
    rtf = elapsed / audio_s if audio_s > 0 else float('inf')

    print(f"  Audio size:  {len(wav_bytes):,} bytes ({len(wav_bytes) / 1024 / 1024:.1f} MB)")
    print(f"  Audio:       {audio_s:.1f}s ({audio_s / 60:.1f} min)")
    print(f"  Wall-clock:  {elapsed:.1f}s")
    print(f"  RTF:         {rtf:.4f}")
    print(f"  Speed:       {audio_s / elapsed:.1f}x real-time" if elapsed > 0 else "")
    print()

    if args.save:
        out_path = "benchmarks/long_bench_output.wav"
        with open(out_path, 'wb') as f:
            f.write(wav_bytes)
        print(f"  Saved: {out_path}")
        print()

    # --- Per-style breakdown ---
    print("Per-style breakdown:")
    print(f"  {'Style':<16} {'Chars':>6} {'Audio(s)':>9} {'Time(s)':>8} {'RTF':>7} {'Speed':>7}")
    print(f"  {'-'*16} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")

    for style in sorted(set(s for s, _ in PARAGRAPHS)):
        style_text = "\n\n".join(t for s, t in PARAGRAPHS if s == style)
        t, wav = bench_nonstreaming(server_url, style_text)
        dur = audio_duration_from_wav(wav)
        style_rtf = t / dur if dur > 0 else float('inf')
        print(f"  {style:<16} {len(style_text):>6} {dur:>9.1f} {t:>8.1f} {style_rtf:>7.4f} {dur/t:>6.1f}x")

    print()

    # --- Streaming benchmark ---
    if args.streaming:
        print("=" * 60)
        print("STREAMING (SSE chunked PCM)")
        print("=" * 60)

        ttfb, total, pcm_bytes = bench_streaming(server_url, TEXT)
        audio_s_stream = audio_duration_from_pcm(pcm_bytes)
        rtf_stream = total / audio_s_stream if audio_s_stream > 0 else float('inf')

        print(f"  Audio:       {audio_s_stream:.1f}s ({audio_s_stream / 60:.1f} min)")
        print(f"  TTFB:        {ttfb * 1000:.0f} ms")
        print(f"  Wall-clock:  {total:.1f}s")
        print(f"  RTF:         {rtf_stream:.4f}")
        print(f"  Speed:       {audio_s_stream / total:.1f}x real-time" if total > 0 else "")
        print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Device:     {health['device']}")
    print(f"  Input:      {words:,} words ({chars:,} chars)")
    print(f"  Audio:      {audio_s:.1f}s ({audio_s / 60:.1f} min)")
    print(f"  RTF:        {rtf:.4f} (non-streaming)")
    print(f"  Throughput: {audio_s / elapsed:.1f}x real-time" if elapsed > 0 else "")
    if args.streaming:
        print(f"  TTFB:       {ttfb * 1000:.0f} ms (streaming)")
        print(f"  RTF:        {rtf_stream:.4f} (streaming)")


if __name__ == "__main__":
    main()
