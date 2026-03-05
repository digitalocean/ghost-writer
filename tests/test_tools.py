"""Tests for tools.py — search_web, publish_to_blog, set_draft_content, clean_article_body."""
import pytest
from unittest.mock import patch, MagicMock

from tools import search_web, publish_to_blog, set_draft_content, clean_article_body, _draft_store


# =====================================================================
# set_draft_content
# =====================================================================

class TestSetDraftContent:
    def test_stores_long_content(self):
        content = "x" * 501
        set_draft_content(content)
        assert _draft_store["content"] == content

    def test_ignores_short_content(self):
        set_draft_content("x" * 500)
        assert "content" not in _draft_store

    def test_ignores_empty(self):
        set_draft_content("")
        assert "content" not in _draft_store

    def test_overwrites_previous(self):
        set_draft_content("a" * 501)
        set_draft_content("b" * 501)
        assert _draft_store["content"] == "b" * 501


# =====================================================================
# clean_article_body
# =====================================================================

class TestCleanArticleBody:
    def test_strips_code_fences(self):
        body = '```html\n<p>Hello world.</p>\n```'
        assert clean_article_body(body) == "<p>Hello world.</p>"

    def test_strips_triple_backtick_no_lang(self):
        body = '```\n<p>Content here.</p>\n```'
        assert clean_article_body(body) == "<p>Content here.</p>"

    def test_strips_preamble_let_me_write(self):
        body = 'I have everything I need. Let me write the full deep-dive post now!\n---\n<p>Actual content.</p>'
        result = clean_article_body(body)
        assert "Let me write" not in result
        assert "I have everything I need" not in result
        assert "<p>Actual content.</p>" in result

    def test_strips_ill_now_write(self):
        body = "I'll now write the article.\n<p>Body text.</p>"
        result = clean_article_body(body)
        assert "I'll now write" not in result
        assert "<p>Body text.</p>" in result

    def test_strips_here_is_the_blog_post(self):
        body = "Here's the complete blog post:\n<p>The real article.</p>"
        result = clean_article_body(body)
        assert "Here's the complete blog post" not in result
        assert "<p>The real article.</p>" in result

    def test_strips_trailing_text_after_html(self):
        body = '<p>Article content.</p>\n\nLet me know if you want changes!'
        result = clean_article_body(body)
        assert result == "<p>Article content.</p>"
        assert "Let me know" not in result

    def test_strips_horizontal_rules(self):
        body = '---\n<p>Content.</p>\n---'
        result = clean_article_body(body)
        assert "---" not in result
        assert "<p>Content.</p>" in result

    def test_preserves_clean_html(self):
        body = '<p>Intro paragraph.</p>\n<h2>Section</h2>\n<p>More content.</p>'
        assert clean_article_body(body) == body

    def test_no_html_returns_as_is(self):
        body = "Plain text with no HTML tags at all."
        assert clean_article_body(body) == body

    def test_complex_realistic_artifact(self):
        body = (
            "I have everything I need. Let me write the full deep-dive post now!\n"
            "---\n"
            "```html\n"
            "<p>Real content starts here.</p>\n"
            "<h2>A Great Section</h2>\n"
            "<p>More details.</p>\n"
            "```\n"
        )
        result = clean_article_body(body)
        assert "Let me write" not in result
        assert "---" not in result
        assert "```" not in result
        assert "<p>Real content starts here.</p>" in result
        assert "<h2>A Great Section</h2>" in result
        assert "<p>More details.</p>" in result

    def test_strips_ill_craft(self):
        body = "Let me craft the article now.\n<p>Content.</p>"
        result = clean_article_body(body)
        assert "craft" not in result
        assert "<p>Content.</p>" in result

    def test_empty_string(self):
        assert clean_article_body("") == ""


# =====================================================================
# search_web
# =====================================================================

class TestSearchWeb:
    @patch("tools.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "Result 1", "body": "Snippet 1", "href": "https://a.com"},
            {"title": "Result 2", "body": "Snippet 2", "href": "https://b.com"},
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = search_web.invoke("test query")
        assert "Result 1" in result
        assert "Result 2" in result
        assert "https://a.com" in result

    @patch("tools.DDGS")
    def test_empty_results(self, mock_ddgs_cls):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = []
        mock_ddgs_cls.return_value = mock_ctx

        result = search_web.invoke("nothing here")
        assert "No search results found" in result

    @patch("tools.DDGS")
    def test_exception_returns_error(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("network down")

        result = search_web.invoke("fail query")
        assert "Search temporarily unavailable" in result
        assert "network down" in result


# =====================================================================
# publish_to_blog
# =====================================================================

class TestPublishToBlog:
    def test_no_content_no_draft_returns_error(self):
        result = publish_to_blog.invoke({"title": "T", "content": "", "tags": ""})
        assert "No draft content available" in result

    def test_no_title_returns_error(self):
        _draft_store["content"] = "x" * 600
        result = publish_to_blog.invoke({"title": "", "content": "", "tags": ""})
        assert "No title provided" in result

    def test_missing_env_vars(self, monkeypatch):
        monkeypatch.delenv("BLOG_TYPE", raising=False)
        monkeypatch.delenv("BLOG_URL", raising=False)
        monkeypatch.delenv("BLOG_API_KEY", raising=False)
        result = publish_to_blog.invoke({"title": "T", "content": "x" * 600, "tags": ""})
        assert "BLOG_TYPE" in result
        assert "BLOG_URL" in result
        assert "BLOG_API_KEY" in result

    @patch("tools.get_blog_client")
    def test_successful_publish(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:secret")

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {
            "title": "My Title",
            "url": "https://example.com/post",
            "id": "123",
        }
        mock_get_client.return_value = mock_client

        result = publish_to_blog.invoke({"title": "My Title", "content": "x" * 600, "tags": "a, b"})
        assert "published successfully" in result
        assert "My Title" in result
        assert _draft_store == {}  # cleared after publish

    @patch("tools.get_blog_client")
    def test_uses_auto_captured_draft(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:secret")

        draft_content = "x" * 600
        _draft_store["content"] = draft_content

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {"title": "T", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        publish_to_blog.invoke({"title": "T", "content": "", "tags": ""})
        mock_client.publish_post.assert_called_once_with("T", draft_content, None)

    @patch("tools.get_blog_client")
    def test_publish_error(self, mock_get_client, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:secret")

        mock_client = MagicMock()
        mock_client.publish_post.side_effect = Exception("API failure")
        mock_get_client.return_value = mock_client

        result = publish_to_blog.invoke({"title": "T", "content": "x" * 600, "tags": ""})
        assert "Error publishing" in result
