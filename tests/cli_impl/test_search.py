"""Tests for `muse search` via run_search()."""
from unittest.mock import MagicMock, patch

from muse.cli_impl.search import run_search
from muse.core.resolvers import SearchResult


def test_run_search_filters_by_modality(capsys):
    results = [
        SearchResult(
            uri="hf://Qwen/Qwen3-8B-GGUF@q4_k_m",
            model_id="qwen3-8b-gguf-q4-k-m",
            modality="chat/completion",
            size_gb=4.5, downloads=1000,
            license="apache-2.0",
            description="Qwen3 8B Q4_K_M",
        ),
    ]
    with patch("muse.cli_impl.search.search", return_value=results) as mock_search:
        run_search(query="qwen3", modality="chat/completion", limit=10, sort="downloads")
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["modality"] == "chat/completion"
        assert call_kwargs["limit"] == 10
        assert call_kwargs["sort"] == "downloads"

    out = capsys.readouterr().out
    assert "hf://Qwen/Qwen3-8B-GGUF@q4_k_m" in out
    assert "4.5 GB" in out


def test_run_search_emits_helpful_message_when_no_results(capsys):
    with patch("muse.cli_impl.search.search", return_value=[]):
        rc = run_search(query="bogus", modality="chat/completion", limit=10, sort="downloads")
    out = capsys.readouterr().out
    assert rc == 0
    assert "no results" in out.lower()


def test_run_search_size_filter_client_side(capsys):
    """--max-size-gb filters post-hoc (resolver returns everything)."""
    results = [
        SearchResult(uri="hf://a@q4", model_id="a-q4", modality="chat/completion",
                     size_gb=4.5, downloads=1, license=None, description=""),
        SearchResult(uri="hf://b@q8", model_id="b-q8", modality="chat/completion",
                     size_gb=12.0, downloads=1, license=None, description=""),
    ]
    with patch("muse.cli_impl.search.search", return_value=results):
        run_search(query="x", modality="chat/completion", limit=10,
                   sort="downloads", max_size_gb=10.0)
    out = capsys.readouterr().out
    assert "hf://a@q4" in out
    assert "hf://b@q8" not in out


def test_run_search_size_filter_keeps_unknown_size(capsys):
    """Rows with size_gb=None pass the filter (we can't know they're too big)."""
    results = [
        SearchResult(uri="hf://known@q4", model_id="x", modality="chat/completion",
                     size_gb=4.5, downloads=1, license=None, description=""),
        SearchResult(uri="hf://unknown", model_id="y", modality="embedding/text",
                     size_gb=None, downloads=1, license=None, description=""),
    ]
    with patch("muse.cli_impl.search.search", return_value=results):
        run_search(query="x", limit=10, sort="downloads", max_size_gb=5.0)
    out = capsys.readouterr().out
    assert "hf://known@q4" in out
    assert "hf://unknown" in out


def test_run_search_returns_2_on_resolver_error(capsys):
    from muse.core.resolvers import ResolverError
    with patch("muse.cli_impl.search.search", side_effect=ResolverError("bad backend")):
        rc = run_search(query="x", modality=None, limit=10, sort="downloads")
    assert rc == 2
    err = capsys.readouterr().err
    assert "bad backend" in err


def test_run_search_passes_backend_kwarg():
    """If caller specifies backend, it's forwarded to resolvers.search."""
    with patch("muse.cli_impl.search.search", return_value=[]) as mock_search:
        run_search(query="x", limit=10, sort="downloads", backend="hf")
    assert mock_search.call_args.kwargs["backend"] == "hf"
