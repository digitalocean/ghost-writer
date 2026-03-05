"""Tests for agent.py — Agent class, callbacks, and helpers."""
import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from agent import Agent, DraftCaptureCallback, LoggingCallback
from tools import _draft_store, set_draft_content


# =====================================================================
# _extract_metadata
# =====================================================================

class TestExtractMetadata:
    """Test the regex-based title/tags/body parser."""

    def test_standard_format(self, mock_agent):
        article = (
            "TITLE: My Great Post\n"
            "TAGS: python, ai\n"
            "\n"
            "<p>Body content here.</p>"
        )
        title, tags, body = mock_agent._extract_metadata(article)
        assert title == "My Great Post"
        assert tags == "python, ai"
        assert "<p>Body content here.</p>" in body
        assert "TITLE:" not in body
        assert "TAGS:" not in body

    def test_missing_title_falls_back_to_h1(self, mock_agent):
        article = "TAGS: tech\n\n<h1>Heading Title</h1>\n<p>Body</p>"
        title, tags, body = mock_agent._extract_metadata(article)
        assert title == "Heading Title"
        assert tags == "tech"

    def test_missing_title_falls_back_to_markdown_heading(self, mock_agent):
        article = "# Markdown Title\n\n<p>Body</p>"
        title, tags, body = mock_agent._extract_metadata(article)
        assert title == "Markdown Title"

    def test_missing_title_and_no_heading_defaults(self, mock_agent):
        article = "<p>Just body text, no title anywhere.</p>"
        title, tags, body = mock_agent._extract_metadata(article)
        assert title == "Untitled Blog Post"

    def test_missing_tags_returns_empty(self, mock_agent):
        article = "TITLE: Only Title\n\n<p>Body</p>"
        title, tags, body = mock_agent._extract_metadata(article)
        assert title == "Only Title"
        assert tags == ""

    def test_body_stripping(self, mock_agent):
        article = "TITLE: T\nTAGS: t\n<p>clean body</p>"
        _, _, body = mock_agent._extract_metadata(article)
        assert body.strip() == "<p>clean body</p>"

    def test_h1_with_nested_tags(self, mock_agent):
        article = '<h1><strong>Bold Title</strong></h1>\n<p>Body</p>'
        title, _, _ = mock_agent._extract_metadata(article)
        assert title == "Bold Title"


# =====================================================================
# _clean_text
# =====================================================================

class TestCleanText:
    def test_em_dash_replaced(self, mock_agent):
        assert mock_agent._clean_text("word\u2014another") == "word, another"

    def test_en_dash_replaced(self, mock_agent):
        assert mock_agent._clean_text("word\u2013another") == "word, another"

    def test_double_hyphen_replaced(self, mock_agent):
        assert mock_agent._clean_text("word -- another") == "word, another"

    def test_no_double_commas(self, mock_agent):
        assert ",," not in mock_agent._clean_text("a,\u2014b")

    def test_normal_hyphens_preserved(self, mock_agent):
        assert mock_agent._clean_text("well-known") == "well-known"

    def test_multiple_em_dashes(self, mock_agent):
        result = mock_agent._clean_text("one\u2014two\u2014three")
        assert "\u2014" not in result
        assert result == "one, two, three"

    def test_empty_string(self, mock_agent):
        assert mock_agent._clean_text("") == ""

    def test_html_with_em_dash(self, mock_agent):
        html = "<p>AI is transforming healthcare\u2014and fast.</p>"
        result = mock_agent._clean_text(html)
        assert "\u2014" not in result
        assert "<p>" in result


# =====================================================================
# _is_failed_article
# =====================================================================

class TestIsFailedArticle:
    def test_max_iterations_marker(self, mock_agent):
        assert mock_agent._is_failed_article("Agent stopped due to max iterations")

    def test_iteration_limit_marker(self, mock_agent):
        assert mock_agent._is_failed_article("Agent stopped due to iteration limit")

    def test_parse_error_marker(self, mock_agent):
        assert mock_agent._is_failed_article("Could not parse LLM output")

    def test_case_insensitive(self, mock_agent):
        assert mock_agent._is_failed_article("AGENT STOPPED DUE TO MAX ITERATIONS")

    def test_normal_article(self, mock_agent):
        assert not mock_agent._is_failed_article("<p>A real article about technology.</p>")

    def test_empty_string(self, mock_agent):
        assert not mock_agent._is_failed_article("")


# =====================================================================
# _pick_topic
# =====================================================================

class TestPickTopic:
    def test_single_topic(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI")
        assert mock_agent._pick_topic() == "AI"

    def test_comma_separated(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI, Cloud, DevOps")
        with patch("agent.random.choice", side_effect=lambda lst: lst[1]):
            assert mock_agent._pick_topic() == "Cloud"

    def test_empty_falls_back(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "")
        assert mock_agent._pick_topic() == "General"

    def test_whitespace_only_falls_back(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "   ,  ,  ")
        assert mock_agent._pick_topic() == "General"


# =====================================================================
# _search_trends
# =====================================================================

class TestSearchTrends:
    @patch("duckduckgo_search.DDGS")
    def test_returns_formatted_trends(self, mock_ddgs_cls, mock_agent):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "Trend A", "body": "Details A"},
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = mock_agent._search_trends("AI")
        assert "Trend A" in result
        assert "Details A" in result

    @patch("duckduckgo_search.DDGS")
    def test_search_failure(self, mock_ddgs_cls, mock_agent):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.side_effect = Exception("timeout")
        mock_ddgs_cls.return_value = mock_ctx

        result = mock_agent._search_trends("AI")
        assert result == "No trend data available."


# =====================================================================
# _brainstorm_topic
# =====================================================================

class TestBrainstormTopic:
    @patch("agent.random.choice", return_value="Deep dive or explainer")
    def test_returns_idea_with_format(self, _mock_choice, mock_agent):
        mock_agent._search_trends = MagicMock(return_value="- Trend: AI is hot")
        mock_response = MagicMock()
        mock_response.content = "  How to build an AI agent in 2026  "
        mock_agent.llm.invoke.return_value = mock_response

        result = mock_agent._brainstorm_topic("AI")
        assert "How to build an AI agent in 2026" in result
        assert "Deep dive or explainer" in result

    @patch("agent.random.choice", return_value="Listicle (e.g. '7 Ways to...', 'Top 10...')")
    def test_llm_failure_falls_back(self, _mock_choice, mock_agent):
        mock_agent._search_trends = MagicMock(return_value="trends")
        mock_agent.llm.invoke.side_effect = Exception("LLM down")

        result = mock_agent._brainstorm_topic("Cloud")
        assert "Cloud" in result
        assert "Listicle" in result


# =====================================================================
# _load_recent_titles
# =====================================================================

class TestLoadRecentTitles:
    @patch("agent.get_blog_client")
    def test_populates_titles(self, mock_get_client, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:aabb")

        mock_client = MagicMock()
        mock_client.get_recent_posts.return_value = [
            {"title": "Post A"},
            {"title": "Post B"},
        ]
        mock_get_client.return_value = mock_client

        mock_agent._recent_titles = []
        mock_agent._load_recent_titles()
        assert mock_agent._recent_titles == ["Post A", "Post B"]

    def test_missing_env_vars_skips(self, mock_agent, monkeypatch):
        monkeypatch.delenv("BLOG_TYPE", raising=False)
        mock_agent._recent_titles = []
        mock_agent._load_recent_titles()
        assert mock_agent._recent_titles == []

    @patch("agent.get_blog_client")
    def test_api_error_leaves_titles_empty(self, mock_get_client, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:aabb")

        mock_get_client.side_effect = Exception("connection refused")

        mock_agent._recent_titles = []
        mock_agent._load_recent_titles()
        assert mock_agent._recent_titles == []


# =====================================================================
# process_message
# =====================================================================

class TestProcessMessage:
    def test_returns_output_and_updates_history(self, mock_agent):
        mock_agent.chat_executor.invoke.return_value = {"output": "Hello there!"}
        result = mock_agent.process_message("Hi")
        assert result == "Hello there!"
        assert len(mock_agent.chat_history) == 2

    def test_long_output_captures_draft(self, mock_agent):
        long_output = "x" * 600
        mock_agent.chat_executor.invoke.return_value = {"output": long_output}
        mock_agent.process_message("write something")
        assert _draft_store.get("content") == long_output

    def test_short_output_no_draft(self, mock_agent):
        mock_agent.chat_executor.invoke.return_value = {"output": "short"}
        mock_agent.process_message("hi")
        assert "content" not in _draft_store

    def test_history_truncated_at_30(self, mock_agent):
        mock_agent.chat_history = [MagicMock()] * 29
        mock_agent.chat_executor.invoke.return_value = {"output": "ok"}
        mock_agent.process_message("msg")
        # 29 existing + 2 new = 31 → truncated to 30
        assert len(mock_agent.chat_history) == 30

    def test_em_dashes_stripped_from_output(self, mock_agent):
        mock_agent.chat_executor.invoke.return_value = {
            "output": "AI is great\u2014really great"
        }
        result = mock_agent.process_message("tell me about AI")
        assert "\u2014" not in result
        assert "AI is great" in result

    def test_exception_returns_error_string(self, mock_agent):
        mock_agent.chat_executor.invoke.side_effect = Exception("boom")
        result = mock_agent.process_message("fail")
        assert "Error processing message" in result
        assert "boom" in result


# =====================================================================
# generate_and_publish
# =====================================================================

class TestGenerateAndPublish:
    def _setup_happy_path(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:aabb")

        mock_agent._brainstorm_topic = MagicMock(return_value="Write about AI agents")
        mock_agent.write_executor.invoke.return_value = {
            "output": (
                "TITLE: AI Agents in 2026\n"
                "TAGS: ai, agents\n\n"
                "<p>" + "x" * 300 + "</p>"
            )
        }
        return mock_agent

    @patch("agent.generate_feature_image", return_value=None)
    @patch("agent.get_blog_client")
    def test_happy_path(self, mock_get_client, _mock_img, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        mock_client = MagicMock()
        mock_client.publish_post.return_value = {
            "title": "AI Agents in 2026",
            "url": "https://example.com/ai",
            "id": "99",
        }
        mock_get_client.return_value = mock_client

        result = mock_agent.generate_and_publish()
        assert "published successfully" in result
        assert "AI Agents in 2026" in result
        assert "AI Agents in 2026" in mock_agent._recent_titles

    def test_failed_article_skipped(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI")
        mock_agent._brainstorm_topic = MagicMock(return_value="idea")
        mock_agent.write_executor.invoke.return_value = {
            "output": "Agent stopped due to max iterations"
        }

        result = mock_agent.generate_and_publish()
        assert "Skipped" in result

    def test_short_body_skipped(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI")
        mock_agent._brainstorm_topic = MagicMock(return_value="idea")
        mock_agent.write_executor.invoke.return_value = {
            "output": "TITLE: Short\nTAGS: x\n\n<p>tiny</p>"
        }

        result = mock_agent.generate_and_publish()
        assert "too short" in result

    def test_missing_env_vars_returns_error(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI")
        monkeypatch.delenv("BLOG_TYPE", raising=False)
        monkeypatch.delenv("BLOG_URL", raising=False)
        monkeypatch.delenv("BLOG_API_KEY", raising=False)

        mock_agent._brainstorm_topic = MagicMock(return_value="idea")
        mock_agent.write_executor.invoke.return_value = {
            "output": "TITLE: T\nTAGS: t\n\n<p>" + "x" * 300 + "</p>"
        }

        result = mock_agent.generate_and_publish()
        assert "Missing required environment variables" in result

    @patch("agent.generate_feature_image", return_value=None)
    @patch("agent.get_blog_client")
    def test_recent_titles_capped(self, mock_get_client, _mock_img, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        mock_agent._recent_titles = [f"Title {i}" for i in range(Agent.MAX_RECENT_TITLES)]

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {"title": "New", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        mock_agent.generate_and_publish()
        assert len(mock_agent._recent_titles) == Agent.MAX_RECENT_TITLES

    @patch("agent.generate_feature_image")
    @patch("agent.get_blog_client")
    def test_feature_image_attached_ghost(self, mock_get_client, mock_gen_image, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        mock_gen_image.return_value = b"\x89PNG-fake-image-bytes"

        mock_client = MagicMock()
        mock_client.__class__ = type("GhostClient", (), {})
        mock_client.upload_image.return_value = "https://example.com/images/feature.png"
        mock_client.publish_post.return_value = {"title": "AI", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        with patch("agent.GhostClient", type(mock_client)):
            mock_agent.generate_and_publish()

        mock_client.upload_image.assert_called_once()
        call_kwargs = mock_client.publish_post.call_args.kwargs
        assert call_kwargs.get("feature_image_url") == "https://example.com/images/feature.png"

    @patch("agent.generate_feature_image")
    @patch("agent.get_blog_client")
    def test_image_gen_failure_still_publishes(self, mock_get_client, mock_gen_image, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        mock_gen_image.return_value = None

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {"title": "AI", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        result = mock_agent.generate_and_publish()
        assert "published successfully" in result
        mock_client.upload_image.assert_not_called()

    @patch("agent.generate_feature_image")
    @patch("agent.get_blog_client")
    def test_image_upload_failure_still_publishes(self, mock_get_client, mock_gen_image, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        mock_gen_image.return_value = b"\x89PNG-fake"

        from blog_clients import GhostClient as _GhostClient
        mock_client = MagicMock(spec=_GhostClient)
        mock_client.upload_image.side_effect = Exception("upload failed")
        mock_client.publish_post.return_value = {"title": "AI", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        result = mock_agent.generate_and_publish()
        assert "published successfully" in result
        call_kwargs = mock_client.publish_post.call_args.kwargs
        assert "feature_image_url" not in call_kwargs

    @patch("agent.generate_feature_image", return_value=None)
    @patch("agent.get_blog_client")
    def test_thinking_artifacts_stripped_before_publish(self, mock_get_client, _mock_img, mock_agent, monkeypatch):
        self._setup_happy_path(mock_agent, monkeypatch)
        preamble = "I have everything I need. Let me write the full deep-dive post now!"
        mock_agent.write_executor.invoke.return_value = {
            "output": (
                "TITLE: AI Agents in 2026\n"
                "TAGS: ai, agents\n\n"
                f"{preamble}\n---\n```html\n"
                "<p>" + "x" * 300 + "</p>\n"
                "```"
            )
        }

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {"title": "AI Agents in 2026", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        mock_agent.generate_and_publish()

        call_args = mock_client.publish_post.call_args
        published_body = call_args[0][1]
        assert "Let me write" not in published_body
        assert "---" not in published_body
        assert "```" not in published_body
        assert "<p>" in published_body

    @patch("agent.generate_feature_image", return_value=None)
    @patch("agent.get_blog_client")
    def test_em_dashes_stripped_before_publish(self, mock_get_client, _mock_img, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TYPE", "ghost")
        monkeypatch.setenv("BLOG_URL", "https://example.com")
        monkeypatch.setenv("BLOG_API_KEY", "id:aabb")

        mock_agent._brainstorm_topic = MagicMock(return_value="AI idea")
        body_with_dashes = "x" * 200 + "\u2014" + "y" * 100
        mock_agent.write_executor.invoke.return_value = {
            "output": (
                "TITLE: AI\u2014The Future\n"
                "TAGS: ai\n\n"
                f"<p>{body_with_dashes}</p>"
            )
        }

        mock_client = MagicMock()
        mock_client.publish_post.return_value = {"title": "AI, The Future", "url": "", "id": "1"}
        mock_get_client.return_value = mock_client

        mock_agent.generate_and_publish()

        call_args = mock_client.publish_post.call_args
        published_title = call_args[0][0]
        published_body = call_args[0][1]
        assert "\u2014" not in published_title
        assert "\u2014" not in published_body

    def test_exception_returns_error(self, mock_agent, monkeypatch):
        monkeypatch.setenv("BLOG_TOPIC", "AI")
        mock_agent._brainstorm_topic = MagicMock(side_effect=Exception("kaboom"))

        result = mock_agent.generate_and_publish()
        assert "Error in autonomous publish" in result
        assert "kaboom" in result


# =====================================================================
# DraftCaptureCallback
# =====================================================================

class TestDraftCaptureCallback:
    def _make_response(self, content):
        """Build a mock LLM response with the given message content."""
        msg = MagicMock()
        msg.content = content
        gen = MagicMock()
        gen.message = msg
        gen.text = content if isinstance(content, str) else ""
        response = MagicMock()
        response.generations = [[gen]]
        return response

    def test_long_string_captured(self):
        cb = DraftCaptureCallback()
        cb.on_llm_end(self._make_response("x" * 600))
        assert _draft_store.get("content") == "x" * 600

    def test_short_string_not_captured(self):
        cb = DraftCaptureCallback()
        cb.on_llm_end(self._make_response("short"))
        assert "content" not in _draft_store

    def test_list_content_joined_and_captured(self):
        content = [
            {"type": "text", "text": "a" * 300},
            {"type": "text", "text": "b" * 300},
        ]
        cb = DraftCaptureCallback()
        cb.on_llm_end(self._make_response(content))
        assert "content" in _draft_store
        # "\n".join produces 300 + 1 (newline) + 300 = 601
        assert len(_draft_store["content"]) == 601

    def test_list_content_too_short(self):
        content = [{"type": "text", "text": "small"}]
        cb = DraftCaptureCallback()
        cb.on_llm_end(self._make_response(content))
        assert "content" not in _draft_store

    def test_malformed_response_handled(self):
        response = MagicMock()
        response.generations = []
        cb = DraftCaptureCallback()
        cb.on_llm_end(response)  # should not raise
        assert "content" not in _draft_store

    def test_fallback_to_gen_text(self):
        """When msg.content is not a string > 500 and not a list, fall back to gen.text."""
        msg = MagicMock()
        msg.content = 42  # not a string or list
        gen = MagicMock()
        gen.message = msg
        gen.text = "y" * 600
        response = MagicMock()
        response.generations = [[gen]]

        cb = DraftCaptureCallback()
        cb.on_llm_end(response)
        assert _draft_store.get("content") == "y" * 600


# =====================================================================
# LoggingCallback (smoke tests)
# =====================================================================

class TestLoggingCallback:
    def test_on_tool_start(self):
        cb = LoggingCallback()
        cb.on_tool_start({"name": "search_web"}, "test input")

    def test_on_tool_end(self):
        cb = LoggingCallback()
        cb.on_tool_end("result output text")

    def test_on_tool_error(self):
        cb = LoggingCallback()
        cb.on_tool_error(Exception("oops"))
