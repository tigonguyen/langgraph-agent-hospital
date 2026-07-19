"""Resolve a model spec string to a chat model, across providers.

Spec format is `provider:model`; a bare string means Ollama, so existing
`"qwen2.5:7b"` keeps working (the split is on the FIRST colon only, and only
when the prefix is a known provider).

    qwen2.5:7b                          -> Ollama (default)
    ollama:qwen2.5:14b                  -> Ollama, explicit
    anthropic:claude-sonnet-5           -> Anthropic
    openai:gpt-4o                       -> OpenAI
    google:gemini-2.0-flash             -> Google Gemini
    openrouter:meta-llama/llama-3.3-70b-instruct  -> OpenRouter

Provider packages are imported lazily, so only the one you actually use needs to
be installed. API keys come from the environment (ANTHROPIC_API_KEY,
OPENAI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY).
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# spec prefix -> langchain provider id
PROVIDERS = {
    "ollama": "ollama",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "google": "google_genai",
    "gemini": "google_genai",
    "openrouter": "openrouter",
}


def resolve_model(spec: str, temperature: float = 0.0) -> BaseChatModel:
    """Build a chat model from a `provider:model` spec (bare string = Ollama)."""
    prefix, _, rest = spec.partition(":")
    provider = PROVIDERS.get(prefix.lower())
    if provider is None or not rest:
        provider, name = "ollama", spec
    else:
        name = rest

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            temperature=temperature,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(name, model_provider=provider, temperature=temperature)
