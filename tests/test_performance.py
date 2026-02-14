"""Performance and optimization tests for Narro TTS."""

import time

import numpy as np
import pytest
import torch
from torch import nn

from narro.vocos.modules import ChannelsFirstLayerNorm, ConvNeXtBlock
from narro.vocos.models import VocosBackbone
from narro.vocos.decoder import SopranoDecoder
from narro.vocos.heads import ISTFTHead
from narro.vocos.migrate_weights import (
    migrate_decoder_weights,
    is_migrated,
    load_with_migration,
)


# ---------------------------------------------------------------------------
# Phase 1 tests: hallucination detector
# ---------------------------------------------------------------------------

class TestHallucinationDetector:
    """Test the vectorized hallucination detector."""

    def _make_tts_stub(self):
        """Create a Narro-like object with only the detector method."""
        from narro.tts import Narro
        # Build a minimal stub without __init__ side-effects
        obj = object.__new__(Narro)
        return obj

    def test_no_hallucination(self):
        """Normal varied hidden states should not trigger detection."""
        tts = self._make_tts_stub()
        # 30 distinct hidden states (random => large L1 diffs)
        hidden_state = [torch.randn(512) * 1000 for _ in range(30)]
        assert tts.hallucination_detector(hidden_state) is False

    def test_hallucination_detected(self):
        """Repeated identical hidden states should trigger detection."""
        tts = self._make_tts_stub()
        base = torch.randn(512)
        # 50 near-identical states => long run of small diffs
        hidden_state = [base + torch.randn(512) * 1e-6 for _ in range(50)]
        assert tts.hallucination_detector(hidden_state) is True

    def test_short_sequence_skipped(self):
        """Sequences shorter than MAX_RUNLENGTH should return False immediately."""
        from narro.tts import MAX_RUNLENGTH
        tts = self._make_tts_stub()
        hidden_state = [torch.randn(512) for _ in range(MAX_RUNLENGTH)]
        assert tts.hallucination_detector(hidden_state) is False

    def test_mixed_sequence(self):
        """A sequence with some similar and some different states."""
        tts = self._make_tts_stub()
        base = torch.randn(512)
        hidden_state = []
        # 10 varied states, then 10 similar, then 10 varied => no detection
        for _ in range(10):
            hidden_state.append(torch.randn(512) * 1000)
        for _ in range(10):
            hidden_state.append(base + torch.randn(512) * 1e-6)
        for _ in range(10):
            hidden_state.append(torch.randn(512) * 1000)
        assert tts.hallucination_detector(hidden_state) is False

    def test_edge_just_above_threshold(self):
        """Exactly MAX_RUNLENGTH+1 similar states should trigger."""
        from narro.tts import MAX_RUNLENGTH, DIFF_THRESHOLD
        tts = self._make_tts_stub()
        # Need enough states and enough consecutive similar ones
        hidden_state = []
        # Start with a few varied
        for _ in range(5):
            hidden_state.append(torch.randn(512) * 1000)
        base = torch.zeros(512)
        # MAX_RUNLENGTH + 2 identical states => MAX_RUNLENGTH + 1 diffs below threshold
        for _ in range(MAX_RUNLENGTH + 2):
            hidden_state.append(base.clone())
        assert tts.hallucination_detector(hidden_state) is True


# ---------------------------------------------------------------------------
# Phase 2 tests: ChannelsFirstLayerNorm
# ---------------------------------------------------------------------------

class TestChannelsFirstLayerNorm:
    """Test ChannelsFirstLayerNorm equivalence with standard LayerNorm."""

    def test_equivalence_with_standard_layernorm(self):
        """ChannelsFirstLayerNorm should match transpose->LayerNorm->transpose."""
        dim = 768
        torch.manual_seed(42)

        # Create both norms with the same weights
        cf_norm = ChannelsFirstLayerNorm(dim, eps=1e-6)
        std_norm = nn.LayerNorm(dim, eps=1e-6)

        # Copy weights: std (dim,) -> cf (1, dim, 1)
        with torch.no_grad():
            cf_norm.weight.copy_(std_norm.weight.unsqueeze(0).unsqueeze(2))
            cf_norm.bias.copy_(std_norm.bias.unsqueeze(0).unsqueeze(2))

        x = torch.randn(2, dim, 100)

        # Standard: transpose -> norm -> transpose
        expected = std_norm(x.transpose(1, 2)).transpose(1, 2)
        actual = cf_norm(x)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape should match input shape."""
        dim = 512
        norm = ChannelsFirstLayerNorm(dim)
        x = torch.randn(4, dim, 50)
        y = norm(x)
        assert y.shape == x.shape

    def test_weight_shapes(self):
        """Weight and bias should be (1, dim, 1)."""
        dim = 256
        norm = ChannelsFirstLayerNorm(dim)
        assert norm.weight.shape == (1, dim, 1)
        assert norm.bias.shape == (1, dim, 1)

    def test_normalized_output_stats(self):
        """Output should have approximately zero mean and unit variance per channel."""
        dim = 512
        norm = ChannelsFirstLayerNorm(dim)
        x = torch.randn(8, dim, 200) * 5 + 3  # Non-zero mean, non-unit variance
        y = norm(x)
        # Mean over channel dim should be near zero
        channel_mean = y.mean(dim=1)
        assert channel_mean.abs().mean() < 0.1


# ---------------------------------------------------------------------------
# Phase 2 tests: updated ConvNeXtBlock
# ---------------------------------------------------------------------------

class TestConvNeXtBlockPerformance:
    """Test ConvNeXtBlock forward pass with optimized layers."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, channels, seq_len = 2, 512, 100
        block = ConvNeXtBlock(
            dim=channels,
            intermediate_dim=channels * 3,
            layer_scale_init_value=1e-6,
        )
        block.eval()

        x = torch.randn(batch_size, channels, seq_len)
        with torch.no_grad():
            y = block(x)

        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_residual_connection(self):
        """Test that residual connection is working."""
        batch_size, channels, seq_len = 1, 512, 50
        block = ConvNeXtBlock(
            dim=channels,
            intermediate_dim=channels * 3,
            layer_scale_init_value=0.0,  # gamma is None
        )
        block.eval()

        x = torch.randn(batch_size, channels, seq_len)
        with torch.no_grad():
            y = block(x)

        assert not torch.allclose(x, y), "Residual connection should modify input"

    def test_conv1d_weights_shape(self):
        """Test that Conv1d layers have correct weight shapes."""
        dim, intermediate_dim = 512, 1536
        block = ConvNeXtBlock(
            dim=dim,
            intermediate_dim=intermediate_dim,
            layer_scale_init_value=1e-6,
        )

        assert block.pwconv1.weight.shape == (intermediate_dim, dim, 1)
        assert block.pwconv2.weight.shape == (dim, intermediate_dim, 1)
        assert block.pwconv1.bias.shape == (intermediate_dim,)
        assert block.pwconv2.bias.shape == (dim,)

    def test_gamma_shape(self):
        """Test that gamma parameter has correct shape for broadcasting."""
        dim = 512
        block = ConvNeXtBlock(
            dim=dim,
            intermediate_dim=dim * 3,
            layer_scale_init_value=1e-6,
        )

        assert block.gamma.shape == (dim, 1)

    def test_uses_channels_first_layernorm(self):
        """ConvNeXtBlock should use ChannelsFirstLayerNorm, not nn.LayerNorm."""
        block = ConvNeXtBlock(dim=64, intermediate_dim=192, layer_scale_init_value=1e-6)
        assert isinstance(block.norm, ChannelsFirstLayerNorm)


# ---------------------------------------------------------------------------
# Phase 2 tests: updated VocosBackbone
# ---------------------------------------------------------------------------

class TestVocosBackbonePerformance:
    """Test VocosBackbone with optimized ConvNeXt blocks."""

    def test_forward_shape(self):
        """Test backbone forward pass produces correct shape."""
        batch_size, input_channels, seq_len = 2, 512, 100
        dim = 768
        backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=dim * 3,
            num_layers=4,
        )
        backbone.eval()

        x = torch.randn(batch_size, input_channels, seq_len)
        with torch.no_grad():
            y = backbone(x)

        assert y.shape == (batch_size, dim, seq_len)

    def test_multiple_layers(self):
        """Test backbone with multiple ConvNeXt layers."""
        batch_size, input_channels, seq_len = 1, 512, 50
        dim = 768
        num_layers = 8
        backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=dim * 3,
            num_layers=num_layers,
        )
        backbone.eval()

        assert len(backbone.convnext) == num_layers

        x = torch.randn(batch_size, input_channels, seq_len)
        with torch.no_grad():
            y = backbone(x)

        assert y.shape == (batch_size, dim, seq_len)

    def test_uses_channels_first_layernorm(self):
        """VocosBackbone should use ChannelsFirstLayerNorm for both norms."""
        backbone = VocosBackbone(
            input_channels=64, dim=128, intermediate_dim=384, num_layers=2,
        )
        assert isinstance(backbone.norm, ChannelsFirstLayerNorm)
        assert isinstance(backbone.final_layer_norm, ChannelsFirstLayerNorm)


# ---------------------------------------------------------------------------
# Phase 2 tests: ISTFTHead Conv1d
# ---------------------------------------------------------------------------

class TestISTFTHeadConv1d:
    """Test ISTFTHead uses Conv1d instead of Linear."""

    def test_out_is_conv1d(self):
        """ISTFTHead.out should be Conv1d(kernel_size=1)."""
        head = ISTFTHead(dim=768, n_fft=2048, hop_length=512)
        assert isinstance(head.out, nn.Conv1d)
        assert head.out.kernel_size == (1,)

    def test_forward_shape(self):
        """ISTFTHead forward should produce (B, T) audio."""
        head = ISTFTHead(dim=768, n_fft=2048, hop_length=512)
        head.eval()
        # Input: (B, C, T) = (1, 768, 100)
        x = torch.randn(1, 768, 100)
        with torch.no_grad():
            audio = head(x)
        assert audio.dim() == 2
        assert audio.shape[0] == 1


# ---------------------------------------------------------------------------
# Phase 2 tests: decoder end-to-end
# ---------------------------------------------------------------------------

class TestDecoderPerformance:
    """Test SopranoDecoder performance."""

    def test_decoder_forward_shape(self):
        """Test decoder forward pass produces audio output."""
        batch_size, channels, seq_len = 1, 512, 10
        decoder = SopranoDecoder(num_input_channels=channels)
        decoder.eval()

        x = torch.randn(batch_size, channels, seq_len)
        with torch.no_grad():
            audio = decoder(x)

        assert audio.dim() == 2
        assert audio.shape[0] == batch_size

    def test_decoder_batch(self):
        """Test decoder with batch_size > 1."""
        batch_size, channels, seq_len = 4, 512, 10
        decoder = SopranoDecoder(num_input_channels=channels)
        decoder.eval()

        x = torch.randn(batch_size, channels, seq_len)
        with torch.no_grad():
            audio = decoder(x)

        assert audio.dim() == 2
        assert audio.shape[0] == batch_size


# ---------------------------------------------------------------------------
# Weight migration tests (extended for Phase 2)
# ---------------------------------------------------------------------------

class TestWeightMigration:
    """Test weight migration utility."""

    def test_migrate_pwconv_weights(self):
        """Test migration of pwconv Linear weights to Conv1d format."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
            "convnext.0.pwconv1.bias": torch.randn(1536),
            "convnext.0.pwconv2.weight": torch.randn(512, 1536),
            "convnext.0.pwconv2.bias": torch.randn(512),
            "convnext.0.gamma": torch.randn(512),
            "other.weight": torch.randn(100, 100),
        }

        migrated = migrate_decoder_weights(state_dict)

        assert migrated["convnext.0.pwconv1.weight"].shape == (1536, 512, 1)
        assert migrated["convnext.0.pwconv2.weight"].shape == (512, 1536, 1)
        assert migrated["convnext.0.pwconv1.bias"].shape == (1536,)
        assert migrated["convnext.0.pwconv2.bias"].shape == (512,)
        assert migrated["convnext.0.gamma"].shape == (512, 1)
        assert migrated["other.weight"].shape == (100, 100)

    def test_migrate_layernorm_weights(self):
        """Test migration of LayerNorm weights to ChannelsFirstLayerNorm format."""
        state_dict = {
            "norm.weight": torch.randn(768),
            "norm.bias": torch.randn(768),
            "final_layer_norm.weight": torch.randn(768),
            "final_layer_norm.bias": torch.randn(768),
        }

        migrated = migrate_decoder_weights(state_dict)

        # LayerNorm (dim,) -> ChannelsFirstLayerNorm (1, dim, 1)
        assert migrated["norm.weight"].shape == (1, 768, 1)
        assert migrated["norm.bias"].shape == (1, 768, 1)
        assert migrated["final_layer_norm.weight"].shape == (1, 768, 1)
        assert migrated["final_layer_norm.bias"].shape == (1, 768, 1)

    def test_migrate_istft_head_weights(self):
        """Test migration of ISTFTHead Linear weights to Conv1d format."""
        out_dim = 2050  # n_fft + 2
        state_dict = {
            "head.out.weight": torch.randn(out_dim, 768),
            "head.out.bias": torch.randn(out_dim),
        }

        migrated = migrate_decoder_weights(state_dict)

        assert migrated["head.out.weight"].shape == (out_dim, 768, 1)
        assert migrated["head.out.bias"].shape == (out_dim,)

    def test_is_migrated_false(self):
        """Test detection of non-migrated weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
        }
        assert not is_migrated(state_dict)

    def test_is_migrated_false_layernorm(self):
        """Test detection of non-migrated LayerNorm weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),
            "norm.weight": torch.randn(768),  # Not migrated
        }
        assert not is_migrated(state_dict)

    def test_is_migrated_false_istft(self):
        """Test detection of non-migrated ISTFTHead weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),
            "norm.weight": torch.randn(1, 768, 1),
            "head.out.weight": torch.randn(2050, 768),  # Not migrated
        }
        assert not is_migrated(state_dict)

    def test_is_migrated_true(self):
        """Test detection of fully migrated weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),
            "norm.weight": torch.randn(1, 768, 1),
            "head.out.weight": torch.randn(2050, 768, 1),
        }
        assert is_migrated(state_dict)

    def test_load_with_migration_already_migrated(self):
        """Test that already migrated weights are not double-migrated."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512, 1),
            "convnext.0.gamma": torch.randn(512, 1),
            "norm.weight": torch.randn(1, 768, 1),
            "head.out.weight": torch.randn(2050, 768, 1),
        }

        result = load_with_migration(state_dict)

        assert result["convnext.0.pwconv1.weight"].shape == (1536, 512, 1)
        assert result["convnext.0.gamma"].shape == (512, 1)
        assert result["norm.weight"].shape == (1, 768, 1)
        assert result["head.out.weight"].shape == (2050, 768, 1)

    def test_load_with_migration_needs_migration(self):
        """Test migration when loading old format weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
            "convnext.0.gamma": torch.randn(512),
            "norm.weight": torch.randn(768),
            "head.out.weight": torch.randn(2050, 768),
        }

        result = load_with_migration(state_dict)

        assert result["convnext.0.pwconv1.weight"].shape == (1536, 512, 1)
        assert result["convnext.0.gamma"].shape == (512, 1)
        assert result["norm.weight"].shape == (1, 768, 1)
        assert result["head.out.weight"].shape == (2050, 768, 1)

    def test_migration_idempotent(self):
        """Running migrate twice should not change already-migrated weights."""
        state_dict = {
            "convnext.0.pwconv1.weight": torch.randn(1536, 512),
            "norm.weight": torch.randn(768),
            "head.out.weight": torch.randn(2050, 768),
        }
        first = migrate_decoder_weights(state_dict)
        second = migrate_decoder_weights(first)

        for key in first:
            assert first[key].shape == second[key].shape
            torch.testing.assert_close(first[key], second[key])

    def test_full_decoder_load_after_migration(self):
        """Integration: migrate old-format state dict and load into decoder."""
        decoder = SopranoDecoder(num_input_channels=512)
        # Extract current state dict (already in new format)
        new_state_dict = decoder.state_dict()

        # Simulate old format by un-migrating
        old_state_dict = {}
        for key, value in new_state_dict.items():
            if '.pwconv1.weight' in key or '.pwconv2.weight' in key:
                old_state_dict[key] = value.squeeze(2) if value.dim() == 3 else value
            elif '.gamma' in key:
                old_state_dict[key] = value.squeeze(1) if value.dim() == 2 else value
            elif ('.norm.weight' in key or '.norm.bias' in key or
                  '.final_layer_norm.weight' in key or '.final_layer_norm.bias' in key):
                old_state_dict[key] = value.squeeze(0).squeeze(-1) if value.dim() == 3 else value
            elif key.endswith('.out.weight'):
                old_state_dict[key] = value.squeeze(2) if value.dim() == 3 else value
            else:
                old_state_dict[key] = value

        # Migrate and load
        migrated = load_with_migration(old_state_dict)
        decoder2 = SopranoDecoder(num_input_channels=512)
        decoder2.load_state_dict(migrated)

        # Verify forward pass works
        x = torch.randn(1, 512, 10)
        with torch.no_grad():
            audio = decoder2(x)
        assert audio.dim() == 2


# ---------------------------------------------------------------------------
# Benchmarks (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests (skipped by default, run with pytest -m benchmark)."""

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_convnext_block_benchmark(self):
        """Benchmark ConvNeXtBlock forward pass."""
        batch_size, channels, seq_len = 4, 512, 200
        block = ConvNeXtBlock(
            dim=channels,
            intermediate_dim=channels * 3,
            layer_scale_init_value=1e-6,
        )
        block.eval()

        x = torch.randn(batch_size, channels, seq_len)

        with torch.no_grad():
            for _ in range(10):
                block(x)

        iterations = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                block(x)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iterations) * 1000
        print(f"\nConvNeXtBlock avg time: {avg_time_ms:.3f} ms")
        assert avg_time_ms < 50

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_backbone_benchmark(self):
        """Benchmark VocosBackbone forward pass."""
        batch_size, input_channels, seq_len = 1, 512, 100
        dim = 768
        backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=dim * 3,
            num_layers=8,
        )
        backbone.eval()

        x = torch.randn(batch_size, input_channels, seq_len)

        with torch.no_grad():
            for _ in range(5):
                backbone(x)

        iterations = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                backbone(x)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iterations) * 1000
        print(f"\nVocosBackbone avg time: {avg_time_ms:.3f} ms")
        assert avg_time_ms < 200

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_decoder_benchmark(self):
        """Benchmark full decoder at various sequence lengths."""
        decoder = SopranoDecoder(num_input_channels=512)
        decoder.eval()

        torch.manual_seed(42)
        for batch_size in [1, 4]:
            for seq_len in [10, 50, 100]:
                x = torch.randn(batch_size, 512, seq_len)
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        decoder(x)
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    with torch.no_grad():
                        decoder(x)
                    times.append(time.perf_counter() - start)
                mean_ms = np.mean(times) * 1000
                std_ms = np.std(times) * 1000
                tokens_per_sec = seq_len / np.mean(times)
                print(f"\nDecoder bs={batch_size} seq={seq_len}: "
                      f"{mean_ms:.1f} +/- {std_ms:.1f} ms "
                      f"({tokens_per_sec:.0f} tokens/sec)")
