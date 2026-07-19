"""Model spec parsing — provider prefix vs bare Ollama name (no network/keys)."""

import pytest

from agent_hospital.models import PROVIDERS, resolve_model


def _split(spec: str) -> tuple[str, str]:
    """Mirror resolve_model's parsing, without constructing a model."""
    prefix, _, rest = spec.partition(":")
    provider = PROVIDERS.get(prefix.lower())
    return ("ollama", spec) if provider is None or not rest else (provider, rest)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("qwen2.5:7b", ("ollama", "qwen2.5:7b")),          # bare, colon is the tag
        ("qwen2.5", ("ollama", "qwen2.5")),                 # bare, no colon
        ("ollama:qwen2.5:14b", ("ollama", "qwen2.5:14b")),  # explicit, tag preserved
        ("anthropic:claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
        ("claude:claude-opus-4-8", ("anthropic", "claude-opus-4-8")),
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("google:gemini-2.0-flash", ("google_genai", "gemini-2.0-flash")),
        ("gemini:gemini-2.0-flash", ("google_genai", "gemini-2.0-flash")),
        ("openrouter:meta-llama/llama-3.3-70b-instruct",
         ("openrouter", "meta-llama/llama-3.3-70b-instruct")),
    ],
)
def test_spec_parsing(spec, expected):
    assert _split(spec) == expected


def test_openrouter_requires_key(monkeypatch):
    pytest.importorskip("langchain_openai", reason="provider package not installed")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(KeyError):
        resolve_model("openrouter:some/model")
