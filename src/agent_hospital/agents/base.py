"""Base agent for the hospital roles.

Every role in the simulation (doctor, patient, examiner, ...) is an `Agent`:
a name, a system prompt, an optional set of tools, and a chat model. The
underlying LangChain `create_agent` graph is built lazily the first time the
agent is asked to act, so creating an `Agent` is cheap and we only "shape"
the real agent when it is actually needed.
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
        response_format: Any | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = list(tools)
        self.response_format = response_format
        self._runnable: Any | None = None  # compiled create_agent graph, built on demand

    def _resolve_model(self) -> BaseChatModel:
        """Return the chat model to drive this agent.

        Any provider's `BaseChatModel` instance is used as-is — that is how the
        agent adapts across providers (`ChatOllama`, `ChatAnthropic`, `ChatOpenAI`,
        ...). A bare string is a convenience that defaults to a local Ollama model.
        """
        if isinstance(self.model, str):
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.model)
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
