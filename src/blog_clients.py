"""
Blog CMS API clients for Ghost and WordPress.
"""
import os
import time
import logging
import requests
from typing import Optional, List
from base64 import b64encode

import jwt

logger = logging.getLogger("ghost-writer")


class GhostClient:
    """Client for the Ghost Admin API using JWT authentication."""

    def __init__(self, blog_url: str, api_key: str):
        self.blog_url = blog_url.rstrip("/")
        self.api_url = f"{self.blog_url}/ghost/api/admin"
        key_parts = api_key.split(":")
        if len(key_parts) != 2:
            raise ValueError("Ghost API key must be in 'id:secret' format")
        self.key_id = key_parts[0]
        self.key_secret = bytes.fromhex(key_parts[1])

    def _make_token(self) -> str:
        iat = int(time.time())
        header = {"alg": "HS256", "typ": "JWT", "kid": self.key_id}
        payload = {"iat": iat, "exp": iat + 300, "aud": "/admin/"}
        return jwt.encode(payload, self.key_secret, algorithm="HS256", headers=header)

    def get_recent_posts(self, limit: int = 20) -> List[dict]:
        """Fetch titles of recently published posts."""
        token = self._make_token()
        headers = {"Authorization": f"Ghost {token}"}
        params = {"limit": limit, "fields": "title", "order": "published_at desc"}
        try:
            resp = requests.get(f"{self.api_url}/posts/", params=params, headers=headers)
            resp.raise_for_status()
            return [{"title": p["title"]} for p in resp.json().get("posts", [])]
        except Exception as e:
            logger.warning(f"Failed to fetch recent Ghost posts: {e}")
            return []

    def upload_image(self, image_bytes: bytes, filename: str = "feature.png") -> str:
        """Upload an image to Ghost and return the public URL."""
        token = self._make_token()
        headers = {"Authorization": f"Ghost {token}"}
        files = {
            "file": (filename, image_bytes, "image/png"),
            "purpose": (None, "image"),
            "ref": (None, filename),
        }
        resp = requests.post(f"{self.api_url}/images/upload/", files=files, headers=headers)
        resp.raise_for_status()
        return resp.json()["images"][0]["url"]

    def publish_post(self, title: str, content: str, tags: Optional[str] = None,
                     feature_image_url: Optional[str] = None) -> dict:
        token = self._make_token()
        headers = {"Authorization": f"Ghost {token}", "Content-Type": "application/json"}

        tag_list = []
        if tags:
            tag_list = [{"name": t.strip()} for t in tags.split(",")]

        post = {
            "title": title,
            "html": content,
            "status": "published",
            "tags": tag_list,
        }
        if feature_image_url:
            post["feature_image"] = feature_image_url

        post_data = {"posts": [post]}

        resp = requests.post(f"{self.api_url}/posts/?source=html", json=post_data, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        post = result["posts"][0]
        return {"url": post.get("url", ""), "id": post.get("id", ""), "title": post.get("title", "")}


class WordPressClient:
    """Client for the WordPress REST API using application password auth."""

    def __init__(self, blog_url: str, api_key: str):
        self.blog_url = blog_url.rstrip("/")
        self.api_url = f"{self.blog_url}/wp-json/wp/v2"
        key_parts = api_key.split(":")
        if len(key_parts) != 2:
            raise ValueError("WordPress API key must be in 'username:application_password' format")
        self.username = key_parts[0]
        self.password = key_parts[1]

    def _auth_header(self) -> dict:
        credentials = b64encode(f"{self.username}:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}", "Content-Type": "application/json"}

    def get_recent_posts(self, limit: int = 20) -> List[dict]:
        """Fetch titles of recently published posts."""
        headers = self._auth_header()
        params = {"per_page": limit, "orderby": "date", "_fields": "title"}
        try:
            resp = requests.get(f"{self.api_url}/posts", params=params, headers=headers)
            resp.raise_for_status()
            return [{"title": p["title"]["rendered"]} for p in resp.json()]
        except Exception as e:
            logger.warning(f"Failed to fetch recent WordPress posts: {e}")
            return []

    def upload_image(self, image_bytes: bytes, filename: str = "feature.png") -> int:
        """Upload an image to WordPress media library and return the attachment ID."""
        headers = self._auth_header()
        headers["Content-Type"] = "image/png"
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        resp = requests.post(f"{self.api_url}/media", data=image_bytes, headers=headers)
        resp.raise_for_status()
        return resp.json()["id"]

    def publish_post(self, title: str, content: str, tags: Optional[str] = None,
                     featured_media_id: Optional[int] = None) -> dict:
        headers = self._auth_header()

        post_data = {
            "title": title,
            "content": content,
            "status": "publish",
        }

        if featured_media_id:
            post_data["featured_media"] = featured_media_id

        if tags:
            tag_names = [t.strip() for t in tags.split(",")]
            tag_ids = []
            for tag_name in tag_names:
                tag_resp = requests.post(
                    f"{self.api_url}/tags",
                    json={"name": tag_name},
                    headers=headers,
                )
                if tag_resp.status_code in (200, 201):
                    tag_ids.append(tag_resp.json()["id"])
                elif tag_resp.status_code == 400 and "term_exists" in tag_resp.text:
                    existing = requests.get(
                        f"{self.api_url}/tags", params={"search": tag_name}, headers=headers
                    )
                    if existing.ok and existing.json():
                        tag_ids.append(existing.json()[0]["id"])
            if tag_ids:
                post_data["tags"] = tag_ids

        resp = requests.post(f"{self.api_url}/posts", json=post_data, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        return {"url": result.get("link", ""), "id": result.get("id", ""), "title": result.get("title", {}).get("rendered", "")}


def get_blog_client(blog_type: str, blog_url: str, api_key: str):
    """Factory function that returns the appropriate blog client."""
    blog_type_lower = blog_type.lower().strip()
    if blog_type_lower == "ghost":
        return GhostClient(blog_url, api_key)
    elif blog_type_lower == "wordpress":
        return WordPressClient(blog_url, api_key)
    else:
        raise ValueError(f"Unsupported blog type: {blog_type}. Must be 'Ghost' or 'Wordpress'.")
