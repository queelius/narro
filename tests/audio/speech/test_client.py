"""Tests for muse.audio.speech.client — HTTP client for remote Muse TTS server."""
import json
from unittest.mock import patch, MagicMock

import pytest

from muse.audio.speech.client import SpeechClient


class TestSpeechClient:
    def test_health_returns_dict(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "device": "cuda"}
        with patch('muse.audio.speech.client.requests.get', return_value=mock_resp):
            client = SpeechClient("http://localhost:8000")
            result = client.health()
            assert result["status"] == "ok"

    def test_infer_returns_audio_bytes(self):
        fake_wav = b"RIFF" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav
        mock_resp.headers = {"content-type": "audio/wav"}
        with patch('muse.audio.speech.client.requests.post', return_value=mock_resp):
            client = SpeechClient("http://localhost:8000")
            audio = client.infer("Hello world")
            assert audio == fake_wav

    def test_generate_with_alignment(self):
        fake_wav = b"RIFF" + b"\x00" * 100
        alignment = [{"paragraph": 0, "start": 0.0, "end": 1.5}]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav
        mock_resp.headers = {
            "content-type": "audio/wav",
            "x-alignment": json.dumps(alignment),
        }
        with patch('muse.audio.speech.client.requests.post', return_value=mock_resp):
            client = SpeechClient("http://localhost:8000")
            audio, align = client.generate_with_alignment(
                ["Paragraph one.", "Paragraph two."], out_path="/dev/null",
            )
            assert align == alignment

    def test_server_unreachable_raises(self):
        import requests as req_lib
        with patch('muse.audio.speech.client.requests.get',
                   side_effect=req_lib.ConnectionError("refused")):
            client = SpeechClient("http://localhost:9999")
            with pytest.raises(ConnectionError):
                client.health()

    def test_env_var_fallback(self, monkeypatch):
        """SpeechClient should read MUSE_SERVER when server_url is unset."""
        monkeypatch.setenv("MUSE_SERVER", "http://custom:1234")
        client = SpeechClient()
        assert client.server_url == "http://custom:1234"

    def test_env_var_not_used_when_explicit_url_given(self, monkeypatch):
        """Explicit server_url always wins over MUSE_SERVER env var."""
        monkeypatch.setenv("MUSE_SERVER", "http://custom:1234")
        client = SpeechClient("http://explicit:5678")
        assert client.server_url == "http://explicit:5678"

    def test_default_url_when_no_env_and_no_arg(self, monkeypatch):
        """Falls back to localhost:8000 when no arg and no env var."""
        monkeypatch.delenv("MUSE_SERVER", raising=False)
        client = SpeechClient()
        assert client.server_url == "http://localhost:8000"
