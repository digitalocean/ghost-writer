"""Feature image generation using DigitalOcean Gradient's GPT-IMAGE-1."""
import os
import re
import base64
import logging
from typing import Optional

import requests

logger = logging.getLogger("ghost-writer")

GRADIENT_IMAGE_URL = "https://inference.do-ai.run/v1/images/generations"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

IMAGE_PROMPT_TEMPLATE = (
    "Create a short visual description (1-2 sentences) for a blog post "
    "feature image. The image should NOT contain any text, words, or letters. "
    "Focus on visual concepts, scenes, objects, or abstract compositions that "
    "evoke the topic.\n\n"
    "Blog title: {title}\n"
    "Tags: {tags}\n\n"
    "Respond with ONLY the visual description, nothing else."
)


def _build_image_prompt(title: str, tags: str, llm) -> str:
    """Use the LLM to generate a concise image prompt from the article metadata."""
    from langchain_core.messages import HumanMessage

    prompt = IMAGE_PROMPT_TEMPLATE.format(title=title, tags=tags or "general")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"LLM image prompt generation failed, using fallback: {e}")
        clean_title = re.sub(r"<[^>]+>", "", title)
        return f"Professional blog header image representing: {clean_title}"


def generate_feature_image(
    title: str, tags: str, llm, api_key: Optional[str] = None
) -> Optional[bytes]:
    """Generate a feature image for a blog post.

    Returns PNG image bytes on success, None on failure.
    Image generation is best-effort and should never block publishing.
    """
    api_key = api_key or os.getenv("GRADIENT_MODEL_ACCESS_KEY", "")
    if not api_key:
        logger.warning("No GRADIENT_MODEL_ACCESS_KEY set, skipping image generation")
        return None

    try:
        image_prompt = _build_image_prompt(title, tags, llm)
        logger.info(f"Generating feature image with prompt: {image_prompt[:100]}...")

        resp = requests.post(
            GRADIENT_IMAGE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai-gpt-image-1",
                "prompt": image_prompt,
                "size": "1536x1024",
                "quality": "auto",
                "output_format": "png",
                "n": 1,
            },
            timeout=120,
        )

        if resp.status_code != 200:
            body_snippet = resp.text[:500] if resp.text else "(empty body)"
            logger.error(
                f"Image API returned HTTP {resp.status_code}: {body_snippet}"
            )
            resp.raise_for_status()

        data = resp.json()
        b64_image = data.get("data", [{}])[0].get("b64_json")
        if not b64_image:
            logger.error(
                f"Image API response missing b64_json. Keys: {list(data.keys())}"
            )
            return None

        image_bytes = base64.b64decode(b64_image)

        if not image_bytes.startswith(PNG_MAGIC):
            logger.error(
                f"Generated image is not valid PNG "
                f"(first 8 bytes: {image_bytes[:8]!r})"
            )
            return None

        logger.info(f"Feature image generated ({len(image_bytes)} bytes)")
        return image_bytes

    except requests.Timeout:
        logger.error("Image API request timed out (120s)")
        return None
    except requests.RequestException as e:
        logger.error(f"Image API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Feature image generation failed: {e}", exc_info=True)
        return None
