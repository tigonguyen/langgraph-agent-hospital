"""Base agent — a lazy wrapper over LangChain's `create_agent`.

An `Agent` is a role: a name, a system prompt, optional tools, optional middleware,
and a chat model. The `create_agent` graph is built lazily on first use, so creating
an `Agent` is cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


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
        return result["messages"][-1].content

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={len(self.tools)})"
