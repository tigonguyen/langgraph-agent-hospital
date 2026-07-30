"""Base agent — a lazy wrapper over LangChain's `create_agent`.

An `Agent` is a role: a name, a system prompt, optional tools, optional middleware,
and a chat model. The `create_agent` graph is built lazily on first use, so creating
an `Agent` is cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


def _as_text(content: Any) -> str:
    """Flatten a message's `.content` to plain text.

    Most models return a string. Thinking-capable ones (e.g. a reasoning model behind an
    Anthropic-compatible proxy) return a list of content blocks instead, which every
    downstream consumer here would choke on — `parse_choice` runs `re.search` over it and
    raises TypeError. Keep only `text` blocks: `thinking` is the model's private reasoning,
    so letting it through would put chain-of-thought into user-facing rationales.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        )
    return str(content)


class Agent:
    """A lazily-constructed role agent backed by `create_agent`."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        *,
        model: BaseChatModel | str,
        tools: Sequence[Callable[..., Any]] = (),
        middleware: Sequence[Any] = (),
        response_format: Any | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = list(tools)
        self.middleware = list(middleware)
        self.response_format = response_format
        self._runnable: Any | None = None  # compiled create_agent graph, built on demand

    def _resolve_model(self) -> BaseChatModel:
        """Return the chat model to drive this agent.

        Any provider's `BaseChatModel` instance is used as-is — that is how the
        agent adapts across providers (`ChatOllama`, `ChatAnthropic`, `ChatOpenAI`,
        ...). A string is resolved as a `provider:model` spec (bare = Ollama).
        """
        if isinstance(self.model, str):
            from agent_hospital.models import resolve_model

            return resolve_model(self.model)
        return self.model

    @property
    def runnable(self) -> Any:
        """The compiled `create_agent` graph; built once, then cached."""
        if self._runnable is None:
            from langchain.agents import create_agent

            kwargs: dict[str, Any] = {
                "model": self._resolve_model(),
                "tools": self.tools,
                "system_prompt": self.system_prompt,
            }
            if self.middleware:
                kwargs["middleware"] = self.middleware
            if self.response_format is not None:
                kwargs["response_format"] = self.response_format
            self._runnable = create_agent(**kwargs)
        return self._runnable

    def act(self, messages: Sequence[AnyMessage | dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Run the agent on a message list and return the full result state."""
        return self.runnable.invoke({"messages": list(messages)}, **kwargs)

    def say(self, text: str, **kwargs: Any) -> str:
        """Send a single user turn and return the agent's text reply."""
        result = self.act([{"role": "user", "content": text}], **kwargs)
        return _as_text(result["messages"][-1].content)

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={len(self.tools)})"
