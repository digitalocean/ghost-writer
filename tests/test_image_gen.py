"""Tests for image_gen.py — feature image generation."""
import base64
import pytest
from unittest.mock import patch, MagicMock

from image_gen import generate_feature_image, _build_image_prompt


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
FAKE_B64 = base64.b64encode(FAKE_PNG).decode()


# =====================================================================
# _build_image_prompt
# =====================================================================

class TestBuildImagePrompt:
    def test_uses_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  A serene landscape  ")
        result = _build_image_prompt("AI in Healthcare", "ai, health", mock_llm)
        assert result == "A serene landscape"

    def test_llm_failure_returns_fallback(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("timeout")
        result = _build_image_prompt("My Title", "tags", mock_llm)
        assert "My Title" in result
        assert "Professional blog header" in result

    def test_fallback_strips_html(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("fail")
        result = _build_image_prompt("<b>Bold Title</b>", "", mock_llm)
        assert "<b>" not in result
        assert "Bold Title" in result


# =====================================================================
# generate_feature_image
# =====================================================================

class TestGenerateFeatureImage:
    @patch("image_gen.requests.post")
    def test_success_returns_bytes(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": FAKE_B64}]},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="test-key")
        assert result == FAKE_PNG
        mock_post.assert_called_once()

        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["model"] == "openai-gpt-image-1"
        assert call_json["size"] == "1536x1024"
        assert call_json["output_format"] == "png"

    @patch("image_gen.requests.post")
    def test_api_error_returns_none(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=500,
            text='{"error": "internal server error"}',
        )
        mock_post.return_value.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("500 Server Error")
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="test-key")
        assert result is None

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_MODEL_ACCESS_KEY", raising=False)
        mock_llm = MagicMock()
        result = generate_feature_image("Title", "tags", mock_llm, api_key="")
        assert result is None

    @patch("image_gen.requests.post")
    def test_uses_env_api_key(self, mock_post, monkeypatch):
        monkeypatch.setenv("GRADIENT_MODEL_ACCESS_KEY", "env-key")
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": FAKE_B64}]},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        generate_feature_image("Title", "tags", mock_llm)
        auth_header = mock_post.call_args.kwargs["headers"]["Authorization"]
        assert auth_header == "Bearer env-key"

    @patch("image_gen.requests.post")
    def test_bad_json_returns_none(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
        mock_post.return_value.raise_for_status = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="key")
        assert result is None

    @patch("image_gen.requests.post")
    def test_invalid_png_bytes_returns_none(self, mock_post):
        not_png = b"JFIF" + b"\x00" * 100
        fake_b64 = base64.b64encode(not_png).decode()
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": fake_b64}]},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="key")
        assert result is None

    @patch("image_gen.requests.post")
    def test_http_error_returns_none(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=429,
            text='{"error": "rate limited"}',
        )
        mock_post.return_value.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("429 Too Many Requests")
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="key")
        assert result is None

    @patch("image_gen.requests.post")
    def test_timeout_returns_none(self, mock_post):
        mock_post.side_effect = __import__("requests").exceptions.Timeout("timed out")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="key")
        assert result is None

    @patch("image_gen.requests.post")
    def test_missing_b64_json_key_returns_none(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"url": "https://example.com/img.png"}]},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="prompt")

        result = generate_feature_image("Title", "tags", mock_llm, api_key="key")
        assert result is None
