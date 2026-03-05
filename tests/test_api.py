"""Tests for __main__.py — FastAPI routes and JSON-RPC endpoint."""
import os
import sys
import importlib.util
import pytest
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")


def _load_app_module():
    """Load src/__main__.py as 'ghost_writer_app' so it doesn't collide with pytest's __main__."""
    mod_name = "ghost_writer_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(SRC_DIR, "__main__.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client():
    """Create a TestClient with chat enabled and scheduler disabled."""
    with patch.dict(os.environ, {
        "GRADIENT_MODEL_ACCESS_KEY": "test-key",
        "BLOG_TOPIC": "Technology",
        "ENABLE_CHAT": "true",
        "BLOGS_PER_DAY": "0",
    }):
        with patch("agent.ChatOpenAI"), \
             patch("agent.create_tool_calling_agent"), \
             patch("agent.AgentExecutor"), \
             patch("agent.Agent._load_recent_titles"):

            mod = _load_app_module()

            from fastapi.testclient import TestClient
            yield TestClient(mod.app), mod


@pytest.fixture()
def client_chat_disabled():
    """Create a TestClient with chat disabled."""
    with patch.dict(os.environ, {
        "GRADIENT_MODEL_ACCESS_KEY": "test-key",
        "BLOG_TOPIC": "Technology",
        "ENABLE_CHAT": "false",
        "BLOGS_PER_DAY": "0",
    }):
        with patch("agent.ChatOpenAI"), \
             patch("agent.create_tool_calling_agent"), \
             patch("agent.AgentExecutor"), \
             patch("agent.Agent._load_recent_titles"):

            mod = _load_app_module()

            from fastapi.testclient import TestClient
            yield TestClient(mod.app), mod


# =====================================================================
# GET /
# =====================================================================

class TestGetRoot:
    def test_chat_enabled_serves_html(self, client):
        tc, _ = client
        resp = tc.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"]

    def test_chat_disabled_serves_autonomous_splash(self, client_chat_disabled):
        tc, _ = client_chat_disabled
        resp = tc.get("/")
        assert resp.status_code == 200
        assert "Autonomous Mode" in resp.text


# =====================================================================
# GET /status
# =====================================================================

class TestGetStatus:
    def test_returns_status(self, client):
        tc, _ = client
        resp = tc.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "Ghost Writer"
        assert "chat_enabled" in data
        assert "blog_url" in data
        assert "scheduler_running" in data


# =====================================================================
# POST / (JSON-RPC)
# =====================================================================

class TestPostRpc:
    def _rpc_payload(self, method="message/send", text="Hello"):
        return {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": method,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }

    def test_message_send_returns_task(self, client):
        tc, mod = client
        mod.agent.process_message = MagicMock(return_value="response text")
        resp = tc.post("/", json=self._rpc_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-1"
        assert data["result"]["status"]["state"] == "completed"
        assert data["result"]["artifacts"][0]["parts"][0]["text"] == "response text"

    def test_chat_disabled_returns_403(self, client_chat_disabled):
        tc, _ = client_chat_disabled
        resp = tc.post("/", json=self._rpc_payload())
        assert resp.status_code == 403

    def test_unknown_method_returns_404(self, client):
        tc, _ = client
        resp = tc.post("/", json=self._rpc_payload(method="task/cancel"))
        assert resp.status_code == 404
