"""Tests for the variant switch (config-driven LangGraph)."""

import os

import pytest
import requests

from agent_hospital.qa import VARIANTS, build_variant

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SMOKE_MODEL = os.environ.get("AGENT_HOSPITAL_TEST_MODEL", "qwen2.5:7b")


def _ollama_has(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    return model in " ".join(m["name"] for m in resp.json().get("models", []))


# --- offline: the switch (building is lazy — no network/store) --------------

def test_registry_has_all_variants():
    assert list(VARIANTS) == ["V0", "V1", "V2", "V3", "V4"]


def test_all_variants_build():
    for vid in VARIANTS:
        assert callable(build_variant(vid, model="dummy"))


def test_overrides_apply():
    from agent_hospital.config import RagConfig

    # A config override should build without error (e.g. a different collection).
    assert callable(build_variant("V1", model="dummy", rag=RagConfig(collection="knowledge_medcpt")))


def test_iterative_rag_config_builds():
    # i-MedRAG prototype: graph-invoked iterative retrieval instead of agentic V1.
    from agent_hospital.config import RagConfig

    rag = RagConfig(collection="knowledge_medcpt", embedder="medcpt", tool=False, iterative_max=3)
    assert callable(build_variant("V1", model="dummy", rag=rag))


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_variant("V9", model="dummy")


# --- live smoke -------------------------------------------------------------

def test_variants_answer_live():
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    from agent_hospital.diseases import load_medqa_usmle

    items = load_medqa_usmle(split="test", limit=2)
    for vid in ("V0", "V1", "V2"):
        answer = build_variant(vid, model=SMOKE_MODEL)
        for it in items:
            res = answer(it)
            assert res.answer in (None, 0, 1, 2, 3)
            assert isinstance(res.rationale, str)


def test_iterative_rag_answers_live():
    """i-MedRAG prototype end-to-end: graph-invoked reason/retrieve loop over the
    already-built MedCPT textbook collection, 2 rounds of follow-up retrieval."""
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    from agent_hospital.config import RagConfig
    from agent_hospital.diseases import load_medqa_usmle

    items = load_medqa_usmle(split="test", limit=2)
    rag = RagConfig(collection="knowledge_medcpt", embedder="medcpt", k=4, threshold=0.60,
                    tool=False, iterative_max=2)
    answer = build_variant("V1", model=SMOKE_MODEL, answer_role="rag-answerer", rag=rag)
    for it in items:
        res = answer(it)
        assert res.answer in (None, 0, 1, 2, 3)
        assert isinstance(res.rationale, str)
