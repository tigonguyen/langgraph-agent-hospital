"""Tests for the reasoning agent (MedQA question -> focused RAG query)."""

import os

import pytest
import requests

from agent_hospital.diseases import MCQItem
from agent_hospital.qa import build_followup_agent, build_reasoning_agent

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SMOKE_MODEL = os.environ.get("AGENT_HOSPITAL_TEST_MODEL", "qwen2.5:7b")

_ITEM = MCQItem(
    id="syn-1",
    question=(
        "A 28-year-old woman has 3 months of fatigue, cold intolerance, weight gain, and "
        "constipation. Exam shows dry skin and delayed deep-tendon reflexes. Which lab is most "
        "likely abnormal?"
    ),
    options=["TSH", "Fasting glucose", "Serum cortisol", "Hemoglobin A1c"],
    answer_idx=0,
)


def _ollama_has(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    return model in " ".join(m["name"] for m in resp.json().get("models", []))


def test_build_reasoning_agent_callable():
    assert callable(build_reasoning_agent(model="dummy"))


def test_build_followup_agent_callable():
    assert callable(build_followup_agent(model="dummy"))


def test_reasoning_query_is_focused_live():
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    to_query = build_reasoning_agent(model=SMOKE_MODEL)
    q = to_query(_ITEM)
    assert isinstance(q, str) and q.strip()
    # A distilled query should be much shorter than the full vignette.
    assert len(q) < len(_ITEM.question)
    print(f"\n[reasoning query] {q!r}")


def test_followup_query_live():
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    next_query = build_followup_agent(model=SMOKE_MODEL)
    # No evidence yet -> a query to look something up, or "" if the model is already confident
    # (both are valid i-MedRAG decisions; only a non-str or a crash would be a bug).
    q = next_query(_ITEM, "")
    assert isinstance(q, str)
    print(f"\n[followup query] {q!r}")

    # Evidence that directly answers the question -> should be willing to stop (DONE == "").
    strong_evidence = (
        "Textbook evidence:\n- Primary hypothyroidism causes fatigue, cold intolerance, weight "
        "gain, constipation, dry skin, and delayed relaxation of deep-tendon reflexes; TSH is "
        "the most sensitive screening test and is elevated."
    )
    q2 = next_query(_ITEM, strong_evidence)
    print(f"\n[followup query after strong evidence] {q2!r}")
