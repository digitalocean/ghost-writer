"""
LangChain tools for the ghost writer agent.
"""
import os
import re
import logging
from typing import Optional
from langchain_core.tools import tool
from duckduckgo_search import DDGS

from blog_clients import get_blog_client, GhostClient

logger = logging.getLogger("ghost-writer")

_draft_store: dict = {}

_LLM_PREAMBLE_PATTERNS = [
    re.compile(r"^.*?Let me write.*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*?I'll (?:now |start )?\s*write.*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*?I have everything I need.*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*?Here(?:'s| is) the (?:complete |full |final )?(?:blog |article|post).*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*?(?:I'll|Let me) (?:craft|create|draft|compose).*?$", re.MULTILINE | re.IGNORECASE),
]


def clean_article_body(body: str) -> str:
    """Strip LLM thinking artifacts, code fences, and preamble from article HTML."""
    body = re.sub(r"```+(?:html|HTML)?\s*\n?", "", body)
    body = re.sub(r"^-{3,}\s*$", "", body, flags=re.MULTILINE)

    for pat in _LLM_PREAMBLE_PATTERNS:
        body = pat.sub("", body)

    first_tag = re.search(
        r"<(?:p|h[1-6]|div|ul|ol|section|article|blockquote|table)\b",
        body, re.IGNORECASE,
    )
    if first_tag:
        body = body[first_tag.start():]

    last_tag = None
    for m in re.finditer(
        r"</(?:p|h[1-6]|div|ul|ol|section|article|blockquote|table)>",
        body, re.IGNORECASE,
    ):
        last_tag = m
    if last_tag:
        body = body[:last_tag.end()]

    return body.strip()


def set_draft_content(content: str):
    """Auto-capture LLM output as draft content for publishing."""
    if len(content) > 500:
        _draft_store["content"] = content
        logger.info(f"Draft auto-captured ({len(content)} chars)")


@tool
def search_web(query: str) -> str:
    """
    Search the web for research material on a given topic.
    Returns the top 5 results with titles, snippets, and URLs.
    Use short, simple queries (2-4 words) for best results.

    Args:
        query: The search query to research. Keep it short and simple.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            # Retry once with a simplified query (first 4 words)
            words = query.split()
            if len(words) > 4:
                short_query = " ".join(words[:4])
                logger.info(f"Retrying search with shorter query: {short_query}")
                with DDGS() as ddgs:
                    results = list(ddgs.text(short_query, max_results=5))

        if not results:
            return (
                f"No search results found for: {query}. "
                "Search may be temporarily unavailable. "
                "Proceed to write the article using your own knowledge."
            )

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r.get('title', 'No title')}**\n"
                f"   {r.get('body', 'No snippet')}\n"
                f"   URL: {r.get('href', 'No URL')}"
            )
        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return (
            f"Search temporarily unavailable: {str(e)}. "
            "Proceed to write the article using your own knowledge."
        )


@tool
def publish_to_blog(title: str = "", content: str = "", tags: str = "") -> str:
    """
    Publish a blog post to the configured blog (Ghost or WordPress).
    The article content is saved automatically from your previous response,
    so you only need to provide the title and optionally tags.
    In interactive chat mode, only call this AFTER the user has explicitly
    confirmed they want to publish.

    Args:
        title: The title of the blog post.
        content: Optional. Auto-filled from your last draft if omitted.
        tags: Comma-separated list of tags for the post.
    """
    if not content and _draft_store.get("content"):
        content = _draft_store["content"]
        logger.info(f"Using auto-captured draft ({len(content)} chars)")

    if not content:
        return "Error: No draft content available. Please write the full blog post first, then call publish_to_blog."

    content = clean_article_body(content)

    if not title:
        return "Error: No title provided. Please provide a title for the blog post."

    blog_type = os.getenv("BLOG_TYPE", "")
    blog_url = os.getenv("BLOG_URL", "")
    api_key = os.getenv("BLOG_API_KEY", "")

    if not all([blog_type, blog_url, api_key]):
        missing = []
        if not blog_type:
            missing.append("BLOG_TYPE")
        if not blog_url:
            missing.append("BLOG_URL")
        if not api_key:
            missing.append("BLOG_API_KEY")
        return f"Error: Missing required environment variables: {', '.join(missing)}"

    try:
        client = get_blog_client(blog_type, blog_url, api_key)

        image_kwargs = {}
        image_bytes = None

        try:
            from image_gen import generate_feature_image
            from langchain_openai import ChatOpenAI

            gradient_key = os.getenv("GRADIENT_MODEL_ACCESS_KEY", "")
            gradient_model = os.getenv("GRADIENT_MODEL", "anthropic-claude-4.6-sonnet")
            if gradient_key:
                llm = ChatOpenAI(
                    model=gradient_model, api_key=gradient_key,
                    base_url="https://inference.do-ai.run/v1/",
                )
                image_bytes = generate_feature_image(title, tags, llm, api_key=gradient_key)
                if not image_bytes:
                    logger.warning("Image generation returned no data, publishing without image")
            else:
                logger.info("No GRADIENT_MODEL_ACCESS_KEY set, skipping image generation")
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)

        if image_bytes:
            try:
                slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
                filename = f"{slug}.png"
                if isinstance(client, GhostClient):
                    image_url = client.upload_image(image_bytes, filename)
                    image_kwargs["feature_image_url"] = image_url
                    logger.info(f"Feature image uploaded to Ghost: {image_url}")
                else:
                    media_id = client.upload_image(image_bytes, filename)
                    image_kwargs["featured_media_id"] = media_id
                    logger.info(f"Feature image uploaded to WordPress: media_id={media_id}")
            except Exception as e:
                logger.error(f"Image upload to blog failed: {e}", exc_info=True)

        result = client.publish_post(title, content, tags or None, **image_kwargs)
        _draft_store.clear()
        return (
            f"Blog post published successfully!\n"
            f"Title: {result.get('title', title)}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"ID: {result.get('id', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"Publish error: {e}", exc_info=True)
        return f"Error publishing blog post: {str(e)}"
