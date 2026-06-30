"""Tests for the variant switch (V0/V1 built; V2-V4 placeholders)."""

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


# --- offline: the switch ---------------------------------------------------

def test_registry_has_all_variants():
    assert list(VARIANTS) == ["V0", "V1", "V2", "V3", "V4"]


def test_build_v0_and_v1_are_callable():
    v0 = build_variant("V0", model="dummy")
    # V1 built without touching the store or the LLM (distill off, fake store).
    v1 = build_variant("V1", model="dummy", store=object(), distill=False)
    assert callable(v0) and callable(v1)


def test_lowercase_id_accepted():
    assert callable(build_variant("v0", model="dummy"))


def test_unbuilt_variants_raise():
    for v in ("V2", "V3", "V4"):
        with pytest.raises(NotImplementedError):
            build_variant(v, model="dummy")


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_variant("V9", model="dummy")


# --- live smoke: V0 and V1 actually answer ---------------------------------

def test_variants_answer_live():
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    from agent_hospital.diseases import load_medqa_usmle

    items = load_medqa_usmle(split="test", limit=2)
    for vid in ("V0", "V1"):
        answer = build_variant(vid, model=SMOKE_MODEL)
        for it in items:
            pred = answer(it)
            assert pred is None or pred in (0, 1, 2, 3)
