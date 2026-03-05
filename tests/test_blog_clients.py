"""Tests for blog_clients.py — Ghost and WordPress API clients."""
import jwt
import pytest
import responses
from blog_clients import GhostClient, WordPressClient, get_blog_client


# ── Valid hex secret used across Ghost tests ──────────────────────────
GHOST_KEY_ID = "abc123"
GHOST_SECRET_HEX = "aabbccdd"
GHOST_API_KEY = f"{GHOST_KEY_ID}:{GHOST_SECRET_HEX}"
BLOG_URL = "https://example.com"


# =====================================================================
# GhostClient — constructor
# =====================================================================

class TestGhostClientInit:
    def test_valid_key(self):
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        assert client.key_id == GHOST_KEY_ID
        assert client.key_secret == bytes.fromhex(GHOST_SECRET_HEX)
        assert client.api_url == f"{BLOG_URL}/ghost/api/admin"

    def test_trailing_slash_stripped(self):
        client = GhostClient(f"{BLOG_URL}/", GHOST_API_KEY)
        assert client.blog_url == BLOG_URL

    def test_invalid_key_no_colon(self):
        with pytest.raises(ValueError, match="id:secret"):
            GhostClient(BLOG_URL, "no-colon-here")

    def test_invalid_key_too_many_parts(self):
        with pytest.raises(ValueError, match="id:secret"):
            GhostClient(BLOG_URL, "a:b:c")


# =====================================================================
# GhostClient — JWT token
# =====================================================================

class TestGhostMakeToken:
    def test_token_structure(self):
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        token = client._make_token()

        decoded = jwt.decode(
            token,
            client.key_secret,
            algorithms=["HS256"],
            audience="/admin/",
        )
        assert decoded["aud"] == "/admin/"
        assert decoded["exp"] - decoded["iat"] == 300

        header = jwt.get_unverified_header(token)
        assert header["kid"] == GHOST_KEY_ID
        assert header["alg"] == "HS256"


# =====================================================================
# GhostClient — get_recent_posts
# =====================================================================

class TestGhostGetRecentPosts:
    @responses.activate
    def test_returns_titles(self):
        responses.get(
            f"{BLOG_URL}/ghost/api/admin/posts/",
            json={"posts": [{"title": "Post A"}, {"title": "Post B"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        posts = client.get_recent_posts(2)
        assert posts == [{"title": "Post A"}, {"title": "Post B"}]

    @responses.activate
    def test_api_error_returns_empty(self):
        responses.get(
            f"{BLOG_URL}/ghost/api/admin/posts/",
            status=500,
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        assert client.get_recent_posts() == []


# =====================================================================
# GhostClient — upload_image
# =====================================================================

class TestGhostUploadImage:
    @responses.activate
    def test_upload_returns_url(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/images/upload/",
            json={"images": [{"url": "https://example.com/content/images/feature.png", "ref": "feature.png"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        url = client.upload_image(b"\x89PNG-fake", "feature.png")
        assert url == "https://example.com/content/images/feature.png"

    @responses.activate
    def test_upload_api_error_raises(self):
        responses.post(f"{BLOG_URL}/ghost/api/admin/images/upload/", status=500)
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        with pytest.raises(Exception):
            client.upload_image(b"\x89PNG-fake")


# =====================================================================
# GhostClient — publish_post
# =====================================================================

class TestGhostPublishPost:
    @responses.activate
    def test_publish_with_tags(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/posts/?source=html",
            json={"posts": [{"url": "https://example.com/p", "id": "1", "title": "My Post"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        result = client.publish_post("My Post", "<p>body</p>", "tag1, tag2")
        assert result == {"url": "https://example.com/p", "id": "1", "title": "My Post"}

        body = responses.calls[0].request.body
        import json
        payload = json.loads(body)
        assert payload["posts"][0]["tags"] == [{"name": "tag1"}, {"name": "tag2"}]

    @responses.activate
    def test_publish_without_tags(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/posts/?source=html",
            json={"posts": [{"url": "", "id": "2", "title": "No Tags"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        result = client.publish_post("No Tags", "<p>body</p>")
        assert result["id"] == "2"

        import json
        payload = json.loads(responses.calls[0].request.body)
        assert payload["posts"][0]["tags"] == []

    @responses.activate
    def test_publish_with_feature_image(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/posts/?source=html",
            json={"posts": [{"url": "", "id": "3", "title": "Img Post"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        result = client.publish_post("Img Post", "<p>body</p>",
                                     feature_image_url="https://example.com/img.png")
        assert result["id"] == "3"

        import json
        payload = json.loads(responses.calls[0].request.body)
        assert payload["posts"][0]["feature_image"] == "https://example.com/img.png"

    @responses.activate
    def test_publish_without_feature_image_omits_field(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/posts/?source=html",
            json={"posts": [{"url": "", "id": "4", "title": "No Img"}]},
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        client.publish_post("No Img", "<p>body</p>")

        import json
        payload = json.loads(responses.calls[0].request.body)
        assert "feature_image" not in payload["posts"][0]

    @responses.activate
    def test_publish_api_error_raises(self):
        responses.post(
            f"{BLOG_URL}/ghost/api/admin/posts/?source=html",
            status=500,
        )
        client = GhostClient(BLOG_URL, GHOST_API_KEY)
        with pytest.raises(Exception):
            client.publish_post("Title", "<p>body</p>")


# =====================================================================
# WordPressClient — constructor
# =====================================================================

WP_API_KEY = "admin:app-pass"

class TestWordPressClientInit:
    def test_valid_key(self):
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        assert client.username == "admin"
        assert client.password == "app-pass"
        assert client.api_url == f"{BLOG_URL}/wp-json/wp/v2"

    def test_trailing_slash_stripped(self):
        client = WordPressClient(f"{BLOG_URL}/", WP_API_KEY)
        assert client.blog_url == BLOG_URL

    def test_invalid_key(self):
        with pytest.raises(ValueError, match="username:application_password"):
            WordPressClient(BLOG_URL, "nocolon")


# =====================================================================
# WordPressClient — auth header
# =====================================================================

class TestWordPressAuthHeader:
    def test_basic_auth_format(self):
        import base64
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        header = client._auth_header()
        expected = base64.b64encode(b"admin:app-pass").decode()
        assert header["Authorization"] == f"Basic {expected}"
        assert header["Content-Type"] == "application/json"


# =====================================================================
# WordPressClient — get_recent_posts
# =====================================================================

class TestWordPressGetRecentPosts:
    @responses.activate
    def test_returns_titles(self):
        responses.get(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json=[
                {"title": {"rendered": "WP Post 1"}},
                {"title": {"rendered": "WP Post 2"}},
            ],
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        posts = client.get_recent_posts(2)
        assert posts == [{"title": "WP Post 1"}, {"title": "WP Post 2"}]

    @responses.activate
    def test_api_error_returns_empty(self):
        responses.get(f"{BLOG_URL}/wp-json/wp/v2/posts", status=500)
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        assert client.get_recent_posts() == []


# =====================================================================
# WordPressClient — upload_image
# =====================================================================

class TestWordPressUploadImage:
    @responses.activate
    def test_upload_returns_media_id(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/media",
            json={"id": 55},
            status=201,
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        media_id = client.upload_image(b"\x89PNG-fake", "feature.png")
        assert media_id == 55

        req = responses.calls[0].request
        assert req.headers["Content-Type"] == "image/png"
        assert "feature.png" in req.headers["Content-Disposition"]

    @responses.activate
    def test_upload_api_error_raises(self):
        responses.post(f"{BLOG_URL}/wp-json/wp/v2/media", status=500)
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        with pytest.raises(Exception):
            client.upload_image(b"\x89PNG-fake")


# =====================================================================
# WordPressClient — publish_post
# =====================================================================

class TestWordPressPublishPost:
    @responses.activate
    def test_publish_with_new_tags(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/tags",
            json={"id": 10},
            status=201,
        )
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json={"link": "https://wp.example.com/p", "id": 42, "title": {"rendered": "WP Title"}},
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        result = client.publish_post("WP Title", "<p>content</p>", "python")
        assert result == {"url": "https://wp.example.com/p", "id": 42, "title": "WP Title"}

    @responses.activate
    def test_publish_with_existing_tag_fallback(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/tags",
            json={"code": "term_exists", "data": {"term_id": 7}},
            status=400,
        )
        responses.get(
            f"{BLOG_URL}/wp-json/wp/v2/tags",
            json=[{"id": 7}],
        )
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json={"link": "", "id": 43, "title": {"rendered": "WP"}},
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        result = client.publish_post("WP", "<p>body</p>", "existing-tag")
        assert result["id"] == 43

    @responses.activate
    def test_publish_with_featured_media(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json={"link": "", "id": 45, "title": {"rendered": "With Image"}},
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        result = client.publish_post("With Image", "<p>body</p>", featured_media_id=55)
        assert result["id"] == 45

        import json
        payload = json.loads(responses.calls[0].request.body)
        assert payload["featured_media"] == 55

    @responses.activate
    def test_publish_without_featured_media_omits_field(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json={"link": "", "id": 46, "title": {"rendered": "No Media"}},
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        client.publish_post("No Media", "<p>body</p>")

        import json
        payload = json.loads(responses.calls[0].request.body)
        assert "featured_media" not in payload

    @responses.activate
    def test_publish_without_tags(self):
        responses.post(
            f"{BLOG_URL}/wp-json/wp/v2/posts",
            json={"link": "", "id": 44, "title": {"rendered": "No Tags"}},
        )
        client = WordPressClient(BLOG_URL, WP_API_KEY)
        result = client.publish_post("No Tags", "<p>body</p>")
        assert len(responses.calls) == 1  # no tag API calls


# =====================================================================
# get_blog_client factory
# =====================================================================

class TestGetBlogClient:
    def test_ghost(self):
        client = get_blog_client("ghost", BLOG_URL, GHOST_API_KEY)
        assert isinstance(client, GhostClient)

    def test_wordpress(self):
        client = get_blog_client("wordpress", BLOG_URL, WP_API_KEY)
        assert isinstance(client, WordPressClient)

    def test_case_insensitive(self):
        assert isinstance(get_blog_client("Ghost", BLOG_URL, GHOST_API_KEY), GhostClient)
        assert isinstance(get_blog_client("WORDPRESS", BLOG_URL, WP_API_KEY), WordPressClient)

    def test_whitespace_stripped(self):
        assert isinstance(get_blog_client("  ghost  ", BLOG_URL, GHOST_API_KEY), GhostClient)

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported blog type"):
            get_blog_client("medium", BLOG_URL, "key")
