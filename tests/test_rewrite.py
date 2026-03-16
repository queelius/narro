"""Tests for narro.rewrite — LLM paragraph rewriting."""
from unittest.mock import patch, MagicMock
from narro.rewrite import rewrite_paragraphs

class TestRewriteParagraphs:
    def test_returns_same_number_of_paragraphs(self):
        paragraphs = ["Paragraph one.", "Paragraph two."]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Rewritten paragraph."}}]}
        with patch('narro.rewrite.requests.post', return_value=mock_resp):
            result = rewrite_paragraphs(paragraphs, api_url="http://localhost/v1")
            assert len(result) == 2

    def test_empty_input_returns_empty(self):
        result = rewrite_paragraphs([], api_url="http://localhost/v1")
        assert result == []

    def test_passes_api_key_in_header(self):
        paragraphs = ["Hello."]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hi."}}]}
        with patch('narro.rewrite.requests.post', return_value=mock_resp) as mock_post:
            rewrite_paragraphs(paragraphs, api_url="http://x/v1", api_key="sk-test")
            headers = mock_post.call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer sk-test"

    def test_falls_back_on_error(self):
        paragraphs = ["Original text."]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        with patch('narro.rewrite.requests.post', return_value=mock_resp):
            result = rewrite_paragraphs(paragraphs, api_url="http://localhost/v1")
            assert result == ["Original text."]
