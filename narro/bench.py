"""Benchmark framework for Narro TTS performance measurement.

Usage:
    from narro.bench import run_benchmark, format_table
    results = run_benchmark(device='cpu', compile=False, quantize=False, num_runs=3)
    print(format_table(results))
"""

import time
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Benchmark corpus
# ---------------------------------------------------------------------------

BENCH_CORPUS = {
    'short': "Hello world, how are you today?",
    'medium': (
        "The quick brown fox jumps over the lazy dog near the river bank. "
        "It was a bright cold day in April, and the clocks were striking thirteen."
    ),
    'long': (
        "In the beginning God created the heavens and the earth. "
        "The earth was without form and void, and darkness was over the face of the deep. "
        "And the Spirit of God was hovering over the face of the waters. "
        "And God said, Let there be light, and there was light. "
        "And God saw that the light was good. "
        "And God separated the light from the darkness."
    ),
    'blog': (
        "Dr. Smith at MIT published a landmark paper in 2026 showing that AI inference costs "
        "have dropped by over $80B globally since 2020. The study (conducted across 47 countries) "
        "found that open-source models like Llama and Mistral now account for 38% of enterprise "
        "deployments. For more details, see https://example.com/study. "
        "His conclusion: we're entering a new era of accessible, on-device AI."
    ),
}

# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    device: str = 'auto',
    compile: bool = True,
    quantize: bool = False,
    num_runs: int = 3,
    num_threads: Optional[int] = None,
) -> dict:
    """Run the benchmark suite and return structured results.

    Creates a Narro instance with the given parameters and measures
    preprocessing, encode, and decode time for each corpus text.

    Args:
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        compile: Whether to enable torch.compile.
        quantize: Whether to enable INT8 quantization.
        num_runs: Number of timed runs per corpus entry (best is reported).
        num_threads: Number of CPU threads (None = auto).

    Returns:
        Dict with keys: timestamp, device, compile, quantize, num_runs, texts.
        Each entry in 'texts' has: label, text, chars, runs (list of per-run
        dicts with preprocess_ms, encode_ms, decode_ms, total_ms, tokens, rtf).
    """
    from .tts import Narro

    tts = Narro(
        compile=compile,
        quantize=quantize,
        num_threads=num_threads,
        device=device,
    )
    resolved_device = tts.device

    text_results = []
    for label, text in BENCH_CORPUS.items():
        runs = []
        for _ in range(num_runs):
            # Preprocessing
            t0 = time.perf_counter()
            tts._preprocess_text([text])
            preprocess_ms = (time.perf_counter() - t0) * 1000

            # Encode
            t0 = time.perf_counter()
            encoded = tts.encode(text)
            encode_ms = (time.perf_counter() - t0) * 1000

            tokens = encoded.total_tokens
            audio_duration_s = encoded.estimated_duration

            # Decode
            t0 = time.perf_counter()
            tts.decode(encoded)
            decode_ms = (time.perf_counter() - t0) * 1000

            total_ms = preprocess_ms + encode_ms + decode_ms
            inference_s = total_ms / 1000.0
            rtf = inference_s / audio_duration_s if audio_duration_s > 0 else float('inf')

            runs.append({
                'preprocess_ms': preprocess_ms,
                'encode_ms': encode_ms,
                'decode_ms': decode_ms,
                'total_ms': total_ms,
                'tokens': tokens,
                'audio_s': audio_duration_s,
                'rtf': rtf,
            })

        text_results.append({
            'label': label,
            'text': text,
            'chars': len(text),
            'runs': runs,
        })

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'device': resolved_device,
        'compile': compile,
        'quantize': quantize,
        'num_runs': num_runs,
        'texts': text_results,
    }


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------

def format_table(results: dict) -> str:
    """Format benchmark results as a human-readable table.

    Args:
        results: Dict returned by run_benchmark().

    Returns:
        Multi-line string with device/compile settings header and a results table.
    """
    device = results.get('device', 'unknown')
    compile_ = results.get('compile', False)
    quantize = results.get('quantize', False)
    num_runs = results.get('num_runs', 1)
    timestamp = results.get('timestamp', '')

    lines = []
    lines.append(f"Narro Benchmark Results")
    lines.append(f"  Timestamp : {timestamp}")
    lines.append(f"  Device    : {device}")
    lines.append(f"  Compile   : {compile_}")
    lines.append(f"  Quantize  : {quantize}")
    lines.append(f"  Runs      : {num_runs} (best reported)")
    lines.append("")

    # Column headers
    col_label   = "Text"
    col_chars   = "Chars"
    col_tokens  = "Tokens"
    col_audio   = "Audio(s)"
    col_encode  = "Encode(ms)"
    col_decode  = "Decode(ms)"
    col_total   = "Total(ms)"
    col_rtf     = "RTF"

    header = (
        f"  {col_label:<8}  {col_chars:>6}  {col_tokens:>7}  {col_audio:>9}"
        f"  {col_encode:>10}  {col_decode:>10}  {col_total:>9}  {col_rtf:>6}"
    )
    separator = "  " + "-" * (len(header) - 2)
    lines.append(header)
    lines.append(separator)

    for entry in results.get('texts', []):
        label = entry['label']
        chars = entry['chars']
        runs = entry['runs']

        # Report best (minimum total_ms) run
        best = min(runs, key=lambda r: r['total_ms'])
        tokens    = best['tokens']
        audio_s   = best['audio_s']
        encode_ms = best['encode_ms']
        decode_ms = best['decode_ms']
        total_ms  = best['total_ms']
        rtf       = best['rtf']

        lines.append(
            f"  {label:<8}  {chars:>6}  {tokens:>7}  {audio_s:>9.2f}"
            f"  {encode_ms:>10.1f}  {decode_ms:>10.1f}  {total_ms:>9.1f}  {rtf:>6.3f}"
        )

    lines.append("")
    return "\n".join(lines)
