"""Tests for LlamaCppModel (mocks llama_cpp.Llama; no real GGUF loaded)."""
from unittest.mock import MagicMock, patch

import pytest


def _with_existing_gguf(tmp_path, filename="fake.gguf"):
    """Helper: write a placeholder file so the Model constructor's
    existence check passes. Returns (local_dir, gguf_filename)."""
    p = tmp_path / filename
    p.write_bytes(b"placeholder")
    return tmp_path, filename


def test_llama_cpp_loads_gguf_path(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path, "qwen3-8b-q4_k_m.gguf")
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        m = LlamaCppModel(
            model_id="qwen3-8b-q4",
            hf_repo="Qwen/Qwen3-8B-GGUF",
            local_dir=str(local_dir),
            gguf_file=gguf,
            context_length=8192,
        )
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model_path"] == str(local_dir / gguf)
        assert kwargs["n_ctx"] == 8192
        assert m.model_id == "qwen3-8b-q4"


def test_llama_cpp_chat_passes_openai_response_through(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "qwen3-8b-q4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        from muse.modalities.chat_completion.protocol import ChatResult
        m = LlamaCppModel(
            model_id="qwen3-8b-q4",
            hf_repo="x", local_dir=str(local_dir), gguf_file=gguf,
        )
        result = m.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert isinstance(result, ChatResult)
        assert result.model_id == "qwen3-8b-q4"
        assert result.choices[0].message["content"] == "hello"
        assert result.usage["total_tokens"] == 5
        call_kwargs = fake_llama.create_chat_completion.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs["temperature"] == 0.7


def test_llama_cpp_chat_stream_translates_chunks(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = iter([
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                         "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 0,
             "model": "x", "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]},
        ])
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        from muse.modalities.chat_completion.protocol import ChatChunk
        m = LlamaCppModel(
            model_id="x", hf_repo="x", local_dir=str(local_dir), gguf_file=gguf,
        )
        chunks = list(m.chat_stream(messages=[{"role": "user", "content": "hi"}]))
        assert len(chunks) == 3
        assert all(isinstance(c, ChatChunk) for c in chunks)
        assert chunks[0].delta == {"role": "assistant"}
        assert chunks[1].delta == {"content": "hi"}
        assert chunks[2].finish_reason == "stop"
        assert fake_llama.create_chat_completion.call_args.kwargs["stream"] is True


def test_llama_cpp_forwards_tools_and_tool_choice(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        fake_llama = MagicMock()
        fake_llama.create_chat_completion.return_value = {
            "id": "c", "object": "chat.completion", "created": 0,
            "model": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        mock_cls.return_value = fake_llama
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        m = LlamaCppModel(model_id="x", hf_repo="x", local_dir=str(local_dir), gguf_file=gguf)
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        m.chat(messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice="auto")
        ck = fake_llama.create_chat_completion.call_args.kwargs
        assert ck["tools"] == tools
        assert ck["tool_choice"] == "auto"


def test_llama_cpp_raises_clear_error_when_deps_missing(tmp_path):
    """Host without llama-cpp-python: constructor raises informative RuntimeError."""
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch(
        "muse.modalities.chat_completion.runtimes.llama_cpp.Llama",
        new=None,
    ):
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            LlamaCppModel(
                model_id="x", hf_repo="x",
                local_dir=str(local_dir), gguf_file=gguf,
            )


def test_llama_cpp_accepts_chat_template_override(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        LlamaCppModel(
            model_id="x", hf_repo="x", local_dir=str(local_dir),
            gguf_file=gguf, chat_template="chatml",
        )
        assert mock_cls.call_args.kwargs.get("chat_format") == "chatml"


def test_llama_cpp_n_gpu_layers_default_is_all(tmp_path):
    local_dir, gguf = _with_existing_gguf(tmp_path)
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        LlamaCppModel(model_id="x", hf_repo="x", local_dir=str(local_dir), gguf_file=gguf)
        assert mock_cls.call_args.kwargs.get("n_gpu_layers") == -1


def test_llama_cpp_raises_on_missing_gguf_file(tmp_path):
    """If the gguf_file doesn't exist at local_dir/gguf_file, raise clearly."""
    with patch("muse.modalities.chat_completion.runtimes.llama_cpp.Llama") as mock_cls:
        mock_cls.return_value = MagicMock()
        from muse.modalities.chat_completion.runtimes.llama_cpp import LlamaCppModel
        with pytest.raises(FileNotFoundError, match="GGUF file not found"):
            LlamaCppModel(
                model_id="x", hf_repo="x",
                local_dir=str(tmp_path), gguf_file="does-not-exist.gguf",
            )
