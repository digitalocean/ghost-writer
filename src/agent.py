"""
Ghost Writer Agent - LangChain agent logic.
"""
import os
import re
import random
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

from tools import search_web, publish_to_blog, set_draft_content, clean_article_body
from blog_clients import get_blog_client, GhostClient
from image_gen import generate_feature_image

logger = logging.getLogger("ghost-writer")


class LoggingCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Tool call: {tool_name} | Input length: {len(input_str)}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        logger.info(f"Tool result ({len(output)} chars)")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"Tool error: {type(error).__name__}")


class DraftCaptureCallback(BaseCallbackHandler):
    """Fallback: auto-capture the LLM's text output as draft content.

    The primary draft capture happens in process_message after the agent
    finishes. This callback is an additional safety net.
    """

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)

            if msg:
                content = msg.content
                if isinstance(content, str) and len(content) > 500:
                    set_draft_content(clean_article_body(content))
                    return
                if isinstance(content, list):
                    text = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if len(text) > 500:
                        set_draft_content(clean_article_body(text))
                        return

            text = getattr(gen, "text", "")
            if isinstance(text, str) and len(text) > 500:
                set_draft_content(clean_article_body(text))
        except (IndexError, AttributeError) as e:
            logger.debug(f"DraftCaptureCallback: could not extract content: {e}")


CHAT_SYSTEM_PROMPT = """You are a professional blog author and editor. Your expertise spans the following topics: {blog_topic}.

You help the user brainstorm, research, write, and publish blog posts. Follow these rules:

CONVERSATION FLOW:
1. When the user suggests a topic or idea, discuss it with them. Ask clarifying questions, suggest angles, and explore the idea together.
2. When the user is ready, research the topic using search_web to gather current information and data points.
3. Write a well-structured, engaging blog post with:
   - A compelling title
   - An engaging introduction that hooks the reader
   - Well-organized sections with clear headers
   - Data, examples, and insights from your research
   - A strong conclusion with a call to action
   - Proper HTML formatting for the blog platform
4. Present the FULL draft article to the user in the chat.
5. Ask the user: "Would you like me to publish this, or would you like any changes?"
6. Only call publish_to_blog AFTER the user explicitly confirms (e.g. "publish it", "looks good", "go ahead", "yes"). You only need to pass the title and tags — the article content is saved automatically from your previous response.
7. If the user requests changes, revise the draft and present the updated version for review.

CRITICAL: NEVER call publish_to_blog without explicit user confirmation. Always show the draft first and wait for approval.

STYLE RULES:
- NEVER use em-dashes (—) or double hyphens (--). Use commas, periods, semicolons, colons, or parentheses instead.
- Write in a professional but approachable tone. Use SEO best practices: include relevant keywords naturally, use descriptive headers, and write compelling meta descriptions."""

ARTICLE_FORMATS = [
    "How-to guide or tutorial",
    "Listicle (e.g. '7 Ways to...', 'Top 10...')",
    "Deep dive or explainer",
    "Opinion or thought leadership piece",
    "Comparison or 'vs' article",
    "Beginner's guide or introduction",
    "Case study or real-world example",
    "Myth-busting or common mistakes article",
    "Trends and predictions",
    "Q&A or FAQ-style article",
]

WRITE_SYSTEM_PROMPT = """You are a professional blog author. Your expertise spans the following topics: {blog_topic}.

TODAY'S DATE: {current_date}. You are writing in {current_year}. All references to dates, years, trends, and events MUST be accurate to the current year. Do NOT reference {previous_year} or earlier as if they are current — they are in the past.

Your task is to research and write a complete blog post. Follow these steps:
1. Run UP TO 3 search_web calls to gather current information. Use short, simple queries (2-4 words work best).
2. After your searches, IMMEDIATELY write the complete blog post. Do NOT keep searching if results are sparse — use your own expertise to fill gaps.

IMPORTANT: You MUST write the article even if searches return no results. Your knowledge is sufficient. Do NOT exceed 3 search calls total.

Pick a specific subtopic, niche angle, or unique thesis rather than writing a broad overview. Surprise the reader with an unexpected take.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
TITLE: Your Blog Post Title Here
TAGS: tag1, tag2, tag3

<p>Your article body in HTML starts here...</p>
<h2>Section headers</h2>
<p>More content...</p>

The article should have an engaging introduction, well-organized sections with clear headers (<h2>, <h3>), data and insights from research, and a strong conclusion. Use proper HTML formatting (<p>, <h2>, <ul>, <li>, <strong>, etc.). Do NOT include the title in the HTML body.

STYLE RULES:
- NEVER use em-dashes (—) or double hyphens (--). Use commas, periods, semicolons, colons, or parentheses instead.
- Write in a professional but approachable tone. Use SEO best practices."""

BRAINSTORM_PROMPT = """You are a blog content strategist. Today's date is {current_date}.

Given the broad topic "{topic}", current trend research, and a list of recently published articles, generate ONE specific, timely blog post idea.

CURRENT TRENDS AND NEWS:
{trends}

RECENTLY PUBLISHED (do NOT repeat these topics, titles, or angles):
{recent_titles}

REQUIRED FORMAT: {article_format}

Rules:
- The idea must be clearly different from every recently published title above.
- Ground the idea in a specific current event, trend, tool, or development from the trend research.
- All date references must use {current_year}, never {previous_year}.
- Be specific: include a concrete angle, audience, or thesis — not a vague category.

Respond with ONLY the article idea as a single sentence. No preamble, no explanation."""


class Agent:
    MAX_RECENT_TITLES = 20

    def __init__(self):
        self.name = "Ghost Writer Agent"
        self.chat_history = []
        self._recent_titles: List[str] = []

        self._blog_topic = os.getenv("BLOG_TOPIC", "General")
        gradient_key = os.getenv("GRADIENT_MODEL_ACCESS_KEY", "")
        gradient_model = os.getenv("GRADIENT_MODEL", "anthropic-claude-4.6-sonnet")

        self.tools = [search_web, publish_to_blog]

        if not gradient_key:
            raise ValueError("GRADIENT_MODEL_ACCESS_KEY must be set")

        self.llm = ChatOpenAI(
            model=gradient_model,
            temperature=0.7,
            max_tokens=16384,
            api_key=gradient_key,
            base_url="https://inference.do-ai.run/v1/",
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", CHAT_SYSTEM_PROMPT.format(blog_topic=self._blog_topic)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        chat_agent = create_tool_calling_agent(self.llm, self.tools, chat_prompt)
        self.chat_executor = AgentExecutor(
            agent=chat_agent,
            tools=self.tools,
            verbose=True,
            callbacks=[LoggingCallback(), DraftCaptureCallback()],
        )

        now = datetime.now()
        write_system = WRITE_SYSTEM_PROMPT.format(
            blog_topic=self._blog_topic,
            current_date=now.strftime("%B %Y"),
            current_year=now.year,
            previous_year=now.year - 1,
        )

        write_tools = [search_web]
        write_prompt = ChatPromptTemplate.from_messages([
            ("system", write_system),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        write_agent = create_tool_calling_agent(self.llm, write_tools, write_prompt)
        self.write_executor = AgentExecutor(
            agent=write_agent,
            tools=write_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=25,
            early_stopping_method="generate",
            callbacks=[LoggingCallback()],
        )

        self._load_recent_titles()

    def _load_recent_titles(self):
        """Seed _recent_titles from the blog so we survive restarts."""
        blog_type = os.getenv("BLOG_TYPE", "")
        blog_url = os.getenv("BLOG_URL", "")
        api_key = os.getenv("BLOG_API_KEY", "")
        if not all([blog_type, blog_url, api_key]):
            return
        try:
            client = get_blog_client(blog_type, blog_url, api_key)
            posts = client.get_recent_posts(self.MAX_RECENT_TITLES)
            self._recent_titles = [p["title"] for p in posts]
            logger.info(f"Loaded {len(self._recent_titles)} recent titles from blog")
        except Exception as e:
            logger.warning(f"Could not load recent titles from blog: {e}")

    def process_message(self, message_text: str) -> str:
        """Process an interactive chat message."""
        try:
            logger.info(f"Chat message ({len(message_text)} chars)")
            result = self.chat_executor.invoke({
                "input": message_text,
                "chat_history": self.chat_history,
            })

            output = self._clean_text(result["output"])

            if len(output) > 500:
                set_draft_content(output)

            self.chat_history.append(HumanMessage(content=message_text))
            self.chat_history.append(AIMessage(content=output))

            if len(self.chat_history) > 30:
                self.chat_history = self.chat_history[-30:]

            return output
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return f"Error processing message: {str(e)}"

    def _pick_topic(self) -> str:
        """Randomly select one topic from the BLOG_TOPIC env var."""
        raw = os.getenv("BLOG_TOPIC", "General")
        topics = [t.strip() for t in raw.split(",") if t.strip()]
        return random.choice(topics) if topics else "General"

    def _search_trends(self, topic: str) -> str:
        """Search the web for current trends and news on the topic."""
        from duckduckgo_search import DDGS

        now = datetime.now()
        queries = [
            f"{topic} latest trends {now.year}",
            f"{topic} news {now.strftime('%B %Y')}",
        ]
        all_results = []
        try:
            with DDGS() as ddgs:
                for q in queries:
                    results = list(ddgs.text(q, max_results=5))
                    for r in results:
                        all_results.append(
                            f"- {r.get('title', '')}: {r.get('body', '')}"
                        )
        except Exception as e:
            logger.warning(f"Trend search failed: {e}")
            return "No trend data available."

        return "\n".join(all_results) if all_results else "No trend data available."

    def _brainstorm_topic(self, topic: str) -> str:
        """Use trends + recent titles to brainstorm a specific, timely article idea."""
        trends = self._search_trends(topic)
        now = datetime.now()

        recent = "\n".join(f"- {t}" for t in self._recent_titles) if self._recent_titles else "(none yet)"
        article_format = random.choice(ARTICLE_FORMATS)

        prompt = BRAINSTORM_PROMPT.format(
            topic=topic,
            trends=trends,
            recent_titles=recent,
            article_format=article_format,
            current_date=now.strftime("%B %d, %Y"),
            current_year=now.year,
            previous_year=now.year - 1,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            idea = response.content.strip()
            logger.info(f"Brainstormed idea: {idea} | format: {article_format}")
            return f"{idea}\n\nWrite it as a {article_format}."
        except Exception as e:
            logger.warning(f"Brainstorm failed, falling back to broad topic: {e}")
            return f"Write a {article_format} about {topic}."

    def generate_and_publish(self) -> str:
        """Autonomously generate and publish a blog post (used by scheduler)."""
        try:
            topic = self._pick_topic()
            logger.info(f"Autonomous write phase started | topic: {topic}")

            specific_idea = self._brainstorm_topic(topic)
            logger.info(f"Writing article from brainstormed idea")

            prompt = (
                f"Research and write a complete, engaging blog post based on this idea:\n\n"
                f"{specific_idea}"
            )

            result = self.write_executor.invoke({"input": prompt})
            article = result["output"]
            logger.info(f"Article written ({len(article)} chars)")

            if self._is_failed_article(article):
                logger.warning(f"Article generation failed, skipping publish: {article[:200]}")
                return "Skipped: article generation did not produce valid content."

            title, tags, body = self._extract_metadata(article)
            title = self._clean_text(title)
            body = self._clean_text(body)

            if len(body.strip()) < 200:
                logger.warning(f"Article body too short ({len(body.strip())} chars), skipping publish")
                return "Skipped: generated article body was too short to publish."

            blog_type = os.getenv("BLOG_TYPE", "")
            blog_url = os.getenv("BLOG_URL", "")
            api_key = os.getenv("BLOG_API_KEY", "")

            if not all([blog_type, blog_url, api_key]):
                missing = [v for v, val in [("BLOG_TYPE", blog_type), ("BLOG_URL", blog_url), ("BLOG_API_KEY", api_key)] if not val]
                return f"Error: Missing required environment variables: {', '.join(missing)}"

            client = get_blog_client(blog_type, blog_url, api_key)

            image_kwargs = {}
            image_bytes = generate_feature_image(title, tags, self.llm)
            if image_bytes:
                logger.info(f"Feature image generated ({len(image_bytes)} bytes), uploading...")
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
            else:
                logger.warning("Image generation returned no data, publishing without image")

            pub_result = client.publish_post(title, body, tags or None, **image_kwargs)
            logger.info(f"Autonomous publish complete: {pub_result}")

            self._recent_titles.append(title)
            if len(self._recent_titles) > self.MAX_RECENT_TITLES:
                self._recent_titles = self._recent_titles[-self.MAX_RECENT_TITLES:]

            return (
                f"Blog post published successfully!\n"
                f"Title: {pub_result.get('title', title)}\n"
                f"URL: {pub_result.get('url', 'N/A')}\n"
                f"ID: {pub_result.get('id', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Autonomous publish error: {e}", exc_info=True)
            return f"Error in autonomous publish: {str(e)}"

    def _extract_metadata(self, article: str) -> Tuple[str, str, str]:
        """Extract title, tags, and clean article body from the write agent's output."""
        title = "Untitled Blog Post"
        tags = ""
        body = article

        title_match = re.search(r'^TITLE:\s*(.+)$', article, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        tags_match = re.search(r'^TAGS:\s*(.+)$', article, re.MULTILINE)
        if tags_match:
            tags = tags_match.group(1).strip()

        if title_match or tags_match:
            body = re.sub(r'^TITLE:.*\n?', '', body, count=1, flags=re.MULTILINE)
            body = re.sub(r'^TAGS:.*\n?', '', body, count=1, flags=re.MULTILINE)
            body = body.strip()

        if not title_match:
            h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', article, re.IGNORECASE | re.DOTALL)
            if h1_match:
                title = re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
            else:
                md_match = re.search(r'^#\s+(.+)$', article, re.MULTILINE)
                if md_match:
                    title = md_match.group(1).strip()

        body = clean_article_body(body)
        return title, tags, body

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove em-dashes and other AI-characteristic punctuation."""
        text = text.replace("—", ", ")
        text = text.replace("–", ", ")
        text = text.replace(" -- ", ", ")
        text = re.sub(r",\s*,", ",", text)
        return text

    _FAILURE_MARKERS = [
        "agent stopped due to max iterations",
        "agent stopped due to iteration limit",
        "could not parse llm output",
    ]

    def _is_failed_article(self, article: str) -> bool:
        """Check if the article output is an error message rather than real content."""
        lower = article.strip().lower()
        return any(marker in lower for marker in self._FAILURE_MARKERS)
