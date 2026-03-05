import logging
import uuid
import os
import time
import click
import uvicorn
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Any, Dict, Union, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from agent import Agent
from models import (
    JsonRpcRequest, JsonRpcResponse, Message, Task,
    TaskStatus, Artifact, ArtifactPart,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ghost-writer")

app = FastAPI()


def _get_api_key() -> str:
    return os.getenv("GW_API_KEY", "")


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        timestamps = self._hits[key]
        self._hits[key] = [t for t in timestamps if now - t < self.window]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True


_rate_limiter = _RateLimiter(
    max_requests=int(os.getenv("GW_RATE_LIMIT", "20")),
    window_seconds=60,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "POST":
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
            )
    response = await call_next(request)
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require Bearer token on protected endpoints when GW_API_KEY is set."""
    api_key = _get_api_key()
    if api_key:
        protected = (
            request.method == "POST"
            or request.url.path == "/status"
        )
        if protected:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {api_key}":
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

agent = Agent()
scheduler = AsyncIOScheduler()
last_auto_publish: Optional[str] = None


def is_chat_enabled() -> bool:
    return os.getenv("ENABLE_CHAT", "false").lower() in ("true", "1")


AUTONOMOUS_MODE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ghost Writer Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
        }
        .container {
            text-align: center;
            animation: fadeIn 0.8s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .robot-icon {
            width: 120px;
            height: 120px;
            margin: 0 auto 32px;
            opacity: 0.9;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
        }
        .subtitle {
            font-size: 1.15rem;
            opacity: 0.7;
            font-weight: 400;
        }
    </style>
</head>
<body>
    <div class="container">
        <svg class="robot-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="8" width="16" height="12" rx="2" />
            <line x1="12" y1="8" x2="12" y2="4" />
            <circle cx="12" cy="3" r="1" />
            <circle cx="9" cy="13" r="1" fill="currentColor" />
            <circle cx="15" cy="13" r="1" fill="currentColor" />
            <path d="M9 17h6" />
            <line x1="2" y1="12" x2="4" y2="12" />
            <line x1="20" y1="12" x2="22" y2="12" />
        </svg>
        <h1>Ghost Writer</h1>
        <p class="subtitle">Autonomous Mode</p>
    </div>
</body>
</html>"""


def scheduled_publish():
    """Called by APScheduler to autonomously generate and publish a post."""
    global last_auto_publish
    blog_url = os.getenv("BLOG_URL", "")
    logger.info(f"Autonomous publish started | Blog: {blog_url}")
    try:
        result = agent.generate_and_publish()
        last_auto_publish = datetime.now().isoformat()

        title = "Unknown"
        for line in result.splitlines():
            if line.startswith("Title:"):
                title = line.split("Title:", 1)[1].strip()
                break

        logger.info(f"Autonomous post published | Title: {title} | Blog: {blog_url}")
    except Exception as e:
        logger.error(f"Autonomous publish failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    if is_chat_enabled():
        logger.info("Mode: Chat (interactive UI enabled)")
    else:
        logger.info("Mode: Autonomous (chat disabled)")

    blog_url = os.getenv("BLOG_URL", "")
    if blog_url:
        logger.info(f"Blog: {blog_url}")

    try:
        blogs_per_day = int(os.getenv("BLOGS_PER_DAY", "1"))
    except ValueError:
        logger.warning("Invalid BLOGS_PER_DAY value, defaulting to 1")
        blogs_per_day = 1
    blogs_per_day = max(0, min(blogs_per_day, 24))
    if blogs_per_day > 0:
        interval_hours = 24.0 / blogs_per_day
        scheduler.add_job(
            scheduled_publish,
            trigger=IntervalTrigger(hours=interval_hours),
            id="blog_publisher",
            name="Autonomous Blog Publisher",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            f"Scheduler started: {blogs_per_day} posts/day "
            f"(every {interval_hours:.1f} hours)"
        )


@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI or autonomous mode splash screen."""
    if not is_chat_enabled():
        return HTMLResponse(content=AUTONOMOUS_MODE_HTML)

    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Ghost Writer Agent</h1><p>Chat interface not found.</p>")


@app.get("/status")
async def get_status():
    """Return current agent config and scheduler status."""
    return {
        "agent": "Ghost Writer",
        "chat_enabled": is_chat_enabled(),
        "blog_url": os.getenv("BLOG_URL", ""),
        "blog_type": os.getenv("BLOG_TYPE", ""),
        "blog_topic": os.getenv("BLOG_TOPIC", ""),
        "blogs_per_day": os.getenv("BLOGS_PER_DAY", "1"),
        "scheduler_running": scheduler.running,
        "last_auto_publish": last_auto_publish,
    }


@app.post("/")
async def handle_rpc(request: JsonRpcRequest):
    """Handle JSON-RPC requests."""
    if request.method == "message/send":
        if not is_chat_enabled():
            raise HTTPException(
                status_code=403,
                detail="Chat is disabled. Set ENABLE_CHAT=true to enable.",
            )
        try:
            params = request.params
            message_data = params.get("message", {})
            user_message = Message(**message_data)

            input_text = ""
            for part in user_message.parts:
                if part.kind == "text" and part.text:
                    input_text += part.text

            logger.info(f"Received message ({len(input_text)} chars): {input_text[:80]}...")

            response_text = agent.process_message(input_text)

            task_id = str(uuid.uuid4())
            context_id = str(uuid.uuid4())

            artifact = Artifact(parts=[ArtifactPart(text=response_text)])

            task = Task(
                id=task_id,
                status=TaskStatus(
                    state="completed",
                    timestamp=datetime.now().isoformat(),
                ),
                artifacts=[artifact],
                contextId=context_id,
            )

            return JsonRpcResponse(id=request.id, result=task)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")
    else:
        raise HTTPException(
            status_code=404, detail=f"Method {request.method} not found"
        )


if __name__ == "__main__":

    @click.group(invoke_without_command=True)
    @click.pass_context
    def cli(ctx):
        if ctx.invoked_subcommand is None:
            ctx.invoke(serve)

    @cli.command()
    @click.option("--host", "host", default="0.0.0.0")
    @click.option("--port", "port", default=8000)
    def serve(host: str, port: int):
        """Start the Ghost Writer web server."""
        uvicorn.run(app, host=host, port=port)

    @cli.command()
    def chat():
        """Launch the interactive TUI chat interface."""
        from tui import run_tui
        run_tui()

    cli()
