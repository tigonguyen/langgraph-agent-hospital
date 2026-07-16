"""Exact-match gate: the refactored graph must reproduce the pre-refactor answers.

`tests/golden/v012_train.json` was captured from the sequential V0/V1/V2 at
temperature=0. This live test replays those items through the graph and asserts
every answer is identical (skips if Ollama/model/store unavailable)."""

import json
import os

import pytest
import requests

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "v012_train.json")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ollama_has(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    return model in " ".join(m["name"] for m in resp.json().get("models", []))


def test_refactor_matches_golden():
    if not os.path.exists(GOLDEN):
        pytest.skip("no golden file")
    g = json.load(open(GOLDEN))
    if not _ollama_has(g["model"]):
        pytest.skip(f"model {g['model']!r} unavailable")

    from agent_hospital.diseases import load_medqa_usmle
    from agent_hospital.qa import build_variant

    by_id = {it.id: it for it in load_medqa_usmle(g["split"], limit=len(g["ids"]))}
    for vid, expected in g["answers"].items():
        answer = build_variant(vid, model=g["model"])
        for iid, exp in expected.items():
            got = answer(by_id[iid])
            assert got == exp, f"{vid} {iid}: graph={got} golden={exp}"
