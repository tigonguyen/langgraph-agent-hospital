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
    assert list(VARIANTS) == ["V0", "V1", "V2", "V3", "V4", "V3L"]


def test_all_variants_build():
    for vid in VARIANTS:
        assert callable(build_variant(vid, model="dummy"))


def test_overrides_apply():
    from agent_hospital.config import RagConfig

    # A config override should build without error (e.g. a different collection).
    assert callable(build_variant("V1", model="dummy", rag=RagConfig(collection="knowledge_medcpt")))


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
