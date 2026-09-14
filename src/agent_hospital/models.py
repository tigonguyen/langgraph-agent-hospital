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

Anthropic accepts a custom endpoint via ANTHROPIC_BASE_URL (a gateway or proxy);
unset, it hits the public API. Any model name works, so an endpoint serving a
non-public model is just `anthropic:<its-name>`.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# Read .env (if present) into the environment, so keys/endpoints don't have to be
# exported by hand. Never overrides an already-exported variable, so a one-off
# `ANTHROPIC_BASE_URL=... python -m agent_hospital ...` still wins.
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model spec for the CLIs, overridable from .env (AGENT_HOSPITAL_MODEL) so the
# whole project can be pointed at a cloud model without passing -m every time. The -m
# flag still beats it; read at call time, not import time, so tests can monkeypatch it.
_FALLBACK_MODEL = "qwen2.5:7b"
_API_FALLBACK_MODEL = "anthropic:claude-opus-5"


def mode() -> str:
    """`local` (default, all-Ollama) or `api` (chat via the configured cloud endpoint).

    Set AGENT_HOSPITAL_MODE in .env. `local` keeps the original behaviour exactly:
    nothing here changes unless the variable is explicitly set to `api`.
    """
    value = (os.environ.get("AGENT_HOSPITAL_MODE") or "local").strip().lower()
    if value not in ("local", "api"):
        raise ValueError(
            f"AGENT_HOSPITAL_MODE must be 'local' or 'api', got {value!r}")
    return value


def default_model() -> str:
    """The CLIs' default model spec, resolved from mode + .env.

    In `api` mode a bare name is given the `anthropic:` prefix, since a bare spec would
    otherwise silently route to Ollama (measured: the whole reason for this helper).
    """
    spec = os.environ.get("AGENT_HOSPITAL_MODEL")
    if mode() == "local":
        return spec or _FALLBACK_MODEL
    if not spec:
        return _API_FALLBACK_MODEL
    prefix, _, rest = spec.partition(":")
    if rest and prefix.lower() in PROVIDERS and prefix.lower() != "ollama":
        return spec                    # already an explicit cloud spec
    return f"anthropic:{spec}"         # bare name (or ollama:) -> the API endpoint

# Ollama models that emit chain-of-thought as output tokens before the answer. They get
# a larger generation cap (thinking counts against `num_predict`) and Ollama's `think`
# option, which returns the reasoning as a separate `reasoning_content` field so no
# `<think>` text ever reaches `parse_choice`. gpt-oss also takes an effort level.
THINKING_MODEL_PREFIXES = ("qwen3", "gpt-oss", "deepseek-r1")
THINKING_NUM_PREDICT = 4096


def _reasoning_option(name: str) -> bool | str:
    """Ollama `think` value for a thinking model, from AGENT_HOSPITAL_THINK_EFFORT:
    `off` disables thinking on any model; gpt-oss takes low/medium(default)/high; other
    thinking models only have on/off."""
    effort = (os.environ.get("AGENT_HOSPITAL_THINK_EFFORT") or "medium").strip().lower()
    if effort == "off":
        return False
    return effort if name.startswith("gpt-oss") else True


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

    if provider == "anthropic":
        # Explicit ChatAnthropic (not init_chat_model) so ANTHROPIC_BASE_URL can point at a
        # gateway/proxy; unset = the public API, same as init_chat_model would give.
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {}
        if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = base_url
        return ChatAnthropic(model=name, temperature=temperature, max_tokens=1024, **kwargs)

    from langchain.chat_models import init_chat_model

    if provider == "ollama":
        # Cap generation length: observed a runaway agentic tool-call loop generate 38k+
        # tokens on a single item (Ollama serves one request at a time, so this stalls
        # everything behind it for tens of minutes). 1024 comfortably covers even the
        # panel's unconstrained deliberation (~900 tokens observed) with headroom.
        # Thinking models need more (see THINKING_MODEL_PREFIXES).
        opts: dict[str, Any] = {"num_predict": 1024}
        if name.startswith(THINKING_MODEL_PREFIXES):
            reasoning = _reasoning_option(name)
            opts = {"num_predict": THINKING_NUM_PREDICT if reasoning else 1024, "reasoning": reasoning}
        return init_chat_model(name, model_provider=provider, temperature=temperature, **opts)

    return init_chat_model(name, model_provider=provider, temperature=temperature)
