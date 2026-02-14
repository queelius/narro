"""Benchmark tests for Narro TTS pipeline timing.

Skipped by default. Run with:
    pytest tests/test_benchmarks.py -v -k benchmark --no-header -rN
"""

import time

import pytest
import torch

from narro.tts import Narro, SAMPLE_RATE
from narro.vocos.decoder import SopranoDecoder


# ---------------------------------------------------------------------------
# Real benchmarks (skipped by default — require model download)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Benchmark: run with pytest -k benchmark --no-header -rN")
class TestDecoderBenchmark:
    """Decoder-only benchmarks (no model download needed)."""

    def test_decoder_throughput(self):
        """Measure decoder throughput at various batch sizes and sequence lengths."""
        import statistics
        decoder = SopranoDecoder(num_input_channels=512)
        decoder.eval()

        print("\n  Decoder throughput:")
        print(f"  {'Batch':<8} {'SeqLen':<8} {'Mean (ms)':<12} {'Std (ms)':<12} {'Tok/s':<10}")
        print(f"  {'---':<8} {'---':<8} {'---':<12} {'---':<12} {'---':<10}")

        for batch_size in [1, 4]:
            for seq_len in [10, 50, 100]:
                x = torch.randn(batch_size, 512, seq_len)

                # Warmup
                with torch.inference_mode():
                    for _ in range(3):
                        decoder(x)

                # Timed runs
                times = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        decoder(x)
                    times.append(time.perf_counter() - t0)

                mean_s = statistics.mean(times)
                std_s = statistics.stdev(times) if len(times) > 1 else 0
                tok_per_sec = (batch_size * seq_len) / mean_s

                print(f"  {batch_size:<8} {seq_len:<8} "
                      f"{mean_s * 1000:<12.1f} {std_s * 1000:<12.1f} "
                      f"{tok_per_sec:<10.0f}")


@pytest.mark.skip(reason="Benchmark: requires model download")
class TestEndToEndBenchmark:
    """Full pipeline benchmarks (require model download)."""

    def test_pipeline_throughput(self):
        """Measure end-to-end pipeline throughput."""
        import statistics
        text = "The quick brown fox jumps over the lazy dog near the river bank."
        num_runs = 5

        t0 = time.perf_counter()
        tts = Narro(compile=False, quantize=True)
        startup = time.perf_counter() - t0

        # Cold
        t0 = time.perf_counter()
        result = tts.infer(text)
        cold = time.perf_counter() - t0

        # Warm
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            tts.infer(text)
            times.append(time.perf_counter() - t0)

        audio_duration = result.shape[0] / SAMPLE_RATE
        mean_infer = statistics.mean(times)
        rtf = mean_infer / audio_duration if audio_duration > 0 else float('inf')

        print(f"\n  Startup: {startup * 1000:.0f} ms")
        print(f"  Cold: {cold * 1000:.0f} ms")
        print(f"  Warm: {mean_infer * 1000:.0f} +/- {statistics.stdev(times) * 1000:.0f} ms")
        print(f"  RTF: {rtf:.2f}x ({audio_duration:.1f}s audio)")

        del tts
