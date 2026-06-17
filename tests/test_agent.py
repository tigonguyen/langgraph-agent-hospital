"""Tests for the base Agent.

`test_lazy_construction` runs offline. `test_say_live` performs a real
round-trip against a running Ollama and is skipped when Ollama (or the chosen
model) is not available.
"""

import os

import pytest
import requests

from agent_hospital.agents import Agent

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEST_MODEL = os.environ.get("AGENT_HOSPITAL_TEST_MODEL", "llama3.1:8b")


def _available_models() -> list[str] | None:
    """Return installed Ollama model names, or None if Ollama is unreachable."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return [m["name"] for m in resp.json().get("models", [])]


def test_lazy_construction():
    """Constructing an Agent must not build the create_agent graph or hit the network."""
    agent = Agent("doctor", "You are a doctor.", model="some-model")

    assert agent.name == "doctor"
    assert agent._runnable is None  # not built until first use
    assert "doctor" in repr(agent)

    # A bare string model resolves to a local Ollama chat model.
    from langchain_ollama import ChatOllama

    resolved = agent._resolve_model()
    assert isinstance(resolved, ChatOllama)
    assert resolved.model == "some-model"


def test_say_live():
    """End-to-end: the agent builds its graph and returns a real reply."""
    models = _available_models()
    if models is None:
        pytest.skip("Ollama is not reachable")
    if TEST_MODEL not in models:
        pytest.skip(f"model {TEST_MODEL!r} not pulled (have: {models})")

    from langchain_ollama import ChatOllama

    agent = Agent(
        "pinger",
        "You are a test fixture. Reply with exactly one word.",
        model=ChatOllama(model=TEST_MODEL, temperature=0),
    )

    assert agent._runnable is None
    reply = agent.say("Say the word: pong")
    assert agent._runnable is not None  # graph was built lazily on first use

    assert isinstance(reply, str) and reply.strip()
    print(f"\n[{TEST_MODEL}] reply: {reply!r}")
