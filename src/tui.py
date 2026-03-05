"""Ghost Writer TUI — Textual-based chat interface."""
import os
import logging

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static, LoadingIndicator
from rich.markdown import Markdown
from rich.text import Text


logging.getLogger("ghost-writer").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class ChatMessage(Static):
    """A single chat message bubble."""

    def __init__(self, role: str, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def on_mount(self) -> None:
        if self.role == "user":
            self.add_class("user-message")
            self.update(Text(f"You: {self.content}"))
        else:
            self.add_class("agent-message")
            try:
                self.update(Markdown(self.content))
            except Exception:
                self.update(Text(self.content))


class GhostWriterChat(App):
    """Textual TUI for chatting with the Ghost Writer agent."""

    TITLE = "Ghost Writer"
    SUB_TITLE = "Chat"

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 2;
    }

    .user-message {
        margin: 1 0;
        padding: 1 2;
        background: $primary-darken-2;
        border: round $primary;
    }

    .agent-message {
        margin: 1 0;
        padding: 1 2;
        background: $surface;
        border: round $secondary;
    }

    #thinking {
        height: 1;
        margin: 0 2;
        display: none;
    }

    #thinking.visible {
        display: block;
    }

    #chat-input {
        dock: bottom;
        margin: 0 2 1 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._agent = None

    def compose(self) -> ComposeResult:
        topic = os.getenv("BLOG_TOPIC", "General")
        blog_url = os.getenv("BLOG_URL", "not configured")

        yield Header(show_clock=False)
        with VerticalScroll(id="chat-scroll"):
            yield Static(
                Text.from_markup(
                    f"[bold]Topic:[/bold] {topic}  |  "
                    f"[bold]Blog:[/bold] {blog_url}\n"
                    "Type a message below to start chatting.\n"
                ),
                id="welcome",
            )
        yield LoadingIndicator(id="thinking")
        yield Input(placeholder="Type a message...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()

    def _get_agent(self):
        if self._agent is None:
            from agent import Agent
            self._agent = Agent()
        return self._agent

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""
        event.input.disabled = True

        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(ChatMessage("user", text))
        scroll.scroll_end(animate=False)

        self.query_one("#thinking").add_class("visible")

        self._send_message(text)

    @work(thread=True)
    def _send_message(self, text: str) -> None:
        try:
            agent = self._get_agent()
            response = agent.process_message(text)
        except Exception as e:
            response = f"**Error:** {e}"

        self.app.call_from_thread(self._show_response, response)

    def _show_response(self, response: str) -> None:
        self.query_one("#thinking").remove_class("visible")

        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(ChatMessage("agent", response))
        scroll.scroll_end(animate=False)

        chat_input = self.query_one("#chat-input", Input)
        chat_input.disabled = False
        chat_input.focus()


def run_tui() -> None:
    app = GhostWriterChat()
    app.run()
