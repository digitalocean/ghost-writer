# Ghost Writer Agent

An AI-powered blog writing agent that researches topics, writes articles, and publishes to Ghost or WordPress. It operates in two modes: **Autonomous Mode** (default), where it generates and publishes posts on a schedule with no human interaction, and **Chat Mode**, where you brainstorm ideas, review drafts, and approve posts through a web UI.

## How It Works

Ghost Writer uses a LangChain agent backed by an LLM (via DigitalOcean's Gradient API) to write blog posts. The agent has access to two tools:

- **search_web** -- Searches the web via DuckDuckGo to gather current information and data points for articles.
- **publish_to_blog** -- Publishes a post to your Ghost or WordPress blog. Draft content is automatically captured from the LLM's response, so the model only needs to pass the title and tags.

The server runs on FastAPI with an APScheduler background job for autonomous publishing.

## Modes

### Autonomous Mode (default)

When `ENABLE_CHAT` is `false` (the default), Ghost Writer runs headless. The web UI shows a simple splash screen indicating autonomous mode is active. The scheduler generates and publishes posts automatically based on `BLOGS_PER_DAY`:

1. Picks a topic from `BLOG_TOPIC`
2. Researches the topic using web search
3. Writes a full article with HTML formatting
4. Publishes directly to your blog

No human interaction is required.

### Chat Mode

Set `ENABLE_CHAT=true` to enable the interactive chat UI. In this mode you can:

1. **Suggest a topic** -- Discuss ideas with the agent
2. **Brainstorm** -- Ask the agent for blog post ideas
3. **Review drafts** -- The agent writes the article and shows it to you before publishing
4. **Approve or revise** -- Say "publish it" to go live, or request changes

The agent will never publish without your explicit confirmation in chat mode. The autonomous scheduler still runs in the background alongside the chat UI.

## Configuration

Set the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `ENABLE_CHAT` | `false` | Set to `true` to enable the interactive chat UI |
| `BLOG_URL` | | URL of your blog (e.g. `https://myblog.com`) |
| `BLOG_TYPE` | | CMS platform: `Ghost` or `Wordpress` |
| `BLOG_API_KEY` | | API credentials (see below) |
| `BLOG_TOPIC` | `General` | Comma-separated topics (e.g. `AI, Cloud Computing, DevOps`) |
| `GRADIENT_MODEL_ACCESS_KEY` | | DigitalOcean Gradient model access key (required) |
| `GRADIENT_MODEL` | `anthropic-claude-4.6-sonnet` | LLM model to use |
| `BLOGS_PER_DAY` | `1` | Number of autonomous posts per day (0-24, 0 to disable scheduler) |
| `GW_API_KEY` | | Optional API key for endpoint authentication (see Security below) |
| `GW_RATE_LIMIT` | `20` | Max POST requests per IP per minute |

### API Key Formats

- **Ghost**: `id:secret` format from Ghost Admin > Integrations > Custom Integration
- **WordPress**: `username:application_password` format from Users > Application Passwords

## Running with Docker

```bash
docker network create agents-net

docker compose up --build
```

The UI is available at `http://localhost:5003`. To enable chat mode:

```bash
ENABLE_CHAT=true docker compose up --build
```

## Running Locally

```bash
cd ghost-writer
pip install -e .
python src/__main__.py
```

The UI is available at `http://localhost:8000`.

### TUI Chat

Ghost Writer includes a terminal-based chat interface built with [Textual](https://textual.textualize.io/). Install the package and launch it with a single command:

```bash
cd ghost-writer
pip install -e .
gw-tui
```

This opens a full TUI with a scrollable chat pane, input box, and status bar. You can brainstorm topics, review drafts, and publish posts -- all from the terminal. No server required; the agent runs directly in-process.

You can also launch it via the CLI subcommand:

```bash
python src/__main__.py chat
```

## API

When `GW_API_KEY` is set, `POST /` and `GET /status` require a `Bearer` token:

```
Authorization: Bearer <your-GW_API_KEY>
```

Unauthenticated requests to protected endpoints return `401 Unauthorized`. The `GET /` UI page is always accessible so users can enter their key through the browser prompt.

### Status

`GET /status` returns the current configuration, chat mode state, and scheduler status.

### JSON-RPC (chat mode only)

```json
POST /
{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Write about Kubernetes security"}]
        }
    }
}
```

Returns a 403 error if `ENABLE_CHAT` is not `true`.

## Security

- **Authentication** -- Set `GW_API_KEY` to require a Bearer token on `POST /` and `GET /status`. Strongly recommended for any non-local deployment.
- **Rate limiting** -- POST requests are limited to `GW_RATE_LIMIT` per IP per minute (default 20).
- **Input limits** -- Chat messages are capped at 50,000 characters.
- **Security headers** -- Responses include `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, and `Referrer-Policy`.
- **Non-root container** -- The Docker image runs as an unprivileged `appuser`.

For production deployments, place the service behind a TLS-terminating reverse proxy (nginx, Caddy, Traefik) and always set `GW_API_KEY`.

## Architecture

```
ghost-writer/
├── src/
│   ├── __main__.py      # FastAPI server, scheduler, CLI (serve/chat subcommands)
│   ├── agent.py         # LangChain agent (chat + autonomous modes)
│   ├── tools.py         # search_web, publish_to_blog, auto-draft capture
│   ├── tui.py           # Textual TUI chat interface (gw-tui entry point)
│   ├── models.py        # A2A Pydantic models
│   ├── blog_clients.py  # Ghost and WordPress API clients
│   └── static/
│       └── index.html   # Chat UI (served only when ENABLE_CHAT=true)
├── pyproject.toml       # Package config with gw-tui entry point
├── AgentCard.json
├── Dockerfile
├── docker-compose.yml
└── .do/
    └── app.yaml         # DigitalOcean App Platform deployment config
```
