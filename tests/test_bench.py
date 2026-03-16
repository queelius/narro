"""Tests for narro.bench — pure-function tests only (no model loading)."""

import pytest
from narro.bench import BENCH_CORPUS, format_table


# ---------------------------------------------------------------------------
# TestBenchCorpus
# ---------------------------------------------------------------------------

class TestBenchCorpus:
    """Verify the benchmark corpus has the expected shape and content."""

    REQUIRED_KEYS = {'short', 'medium', 'long', 'blog'}

    def test_has_required_keys(self):
        assert set(BENCH_CORPUS.keys()) >= self.REQUIRED_KEYS

    def test_values_are_non_empty_strings(self):
        for key, value in BENCH_CORPUS.items():
            assert isinstance(value, str), f"{key!r} value is not a str"
            assert len(value) > 0, f"{key!r} value is empty"

    def test_short_is_shortest(self):
        """'short' should be shorter than 'medium', 'long', and 'blog'."""
        short_len = len(BENCH_CORPUS['short'])
        for key in ('medium', 'long', 'blog'):
            assert short_len < len(BENCH_CORPUS[key]), (
                f"'short' ({short_len}) is not shorter than '{key}' ({len(BENCH_CORPUS[key])})"
            )

    def test_blog_contains_difficult_tts_cases(self):
        """Blog text should exercise tricky TTS cases."""
        blog = BENCH_CORPUS['blog']
        # Year
        assert '2026' in blog or any(c.isdigit() for c in blog), \
            "blog text should contain digits (years, numbers)"
        # Dollar amount
        assert '$' in blog, "blog text should contain a dollar amount"
        # Abbreviation
        assert '.' in blog, "blog text should contain abbreviations with periods"


# ---------------------------------------------------------------------------
# TestFormatTable
# ---------------------------------------------------------------------------

class TestFormatTable:
    """Verify format_table output structure without running the model."""

    def _make_results(self, device='cpu', compile=False, quantize=False, num_runs=1):
        """Build a minimal synthetic results dict."""
        texts = []
        for label, text in BENCH_CORPUS.items():
            texts.append({
                'label': label,
                'text': text,
                'chars': len(text),
                'runs': [{
                    'preprocess_ms': 1.0,
                    'encode_ms': 100.0,
                    'decode_ms': 50.0,
                    'total_ms': 151.0,
                    'tokens': 20,
                    'audio_s': 1.28,
                    'rtf': 0.118,
                }],
            })
        return {
            'timestamp': '2026-03-16T00:00:00+00:00',
            'device': device,
            'compile': compile,
            'quantize': quantize,
            'num_runs': num_runs,
            'texts': texts,
        }

    def test_returns_string(self):
        output = format_table(self._make_results())
        assert isinstance(output, str)

    def test_contains_device(self):
        output = format_table(self._make_results(device='cpu'))
        assert 'cpu' in output

    def test_contains_all_corpus_labels(self):
        output = format_table(self._make_results())
        for label in BENCH_CORPUS:
            assert label in output, f"Expected corpus label {label!r} in table output"

    def test_contains_column_headers(self):
        output = format_table(self._make_results())
        for header in ('Text', 'Chars', 'Tokens', 'Audio', 'Encode', 'Decode', 'Total', 'RTF'):
            assert header in output, f"Expected column header {header!r} in table output"

    def test_compile_flag_shown(self):
        output_true = format_table(self._make_results(compile=True))
        output_false = format_table(self._make_results(compile=False))
        assert 'True' in output_true
        assert 'False' in output_false

    def test_non_empty_output(self):
        output = format_table(self._make_results())
        assert len(output.strip()) > 0

    def test_empty_texts_list(self):
        """format_table should not crash with an empty texts list."""
        results = {
            'timestamp': '2026-03-16T00:00:00+00:00',
            'device': 'cpu',
            'compile': False,
            'quantize': False,
            'num_runs': 1,
            'texts': [],
        }
        output = format_table(results)
        assert isinstance(output, str)
        assert 'cpu' in output
