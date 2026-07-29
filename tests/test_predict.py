"""Live test for agent_hospital.predict (skips if Ollama/model unavailable)."""

import json
import os

import pytest
import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SMOKE_MODEL = os.environ.get("AGENT_HOSPITAL_TEST_MODEL", "qwen2.5:7b")


def _ollama_has(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    return model in " ".join(m["name"] for m in resp.json().get("models", []))


def test_run_writes_predictions_and_resumes(tmp_path):
    if not _ollama_has(SMOKE_MODEL):
        pytest.skip(f"{SMOKE_MODEL} unavailable")
    from agent_hospital.predict import run

    out_dir = str(tmp_path)
    path = run("V0", split="test", model=SMOKE_MODEL, limit=2, out_dir=out_dir, quiet=True)
    assert os.path.exists(path)

    with open(path) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2
    for row in rows:
        assert row["pred"] in (None, 0, 1, 2, 3)
        assert row["gold"] in (0, 1, 2, 3)
        assert row["correct"] == (row["pred"] == row["gold"])
        assert row["valid"] == (row["pred"] is not None)
        assert isinstance(row["rationale"], str)

    meta_path = path.replace(".jsonl", ".meta.json")
    assert os.path.exists(meta_path)
    meta = json.load(open(meta_path))
    assert meta["variant"] == "V0"
    assert meta["n_items"] == 2

    # Re-running with the same args should resume (skip already-done items) rather than
    # re-predict them — verified by checking the file still has exactly 2 rows after.
    run("V0", split="test", model=SMOKE_MODEL, limit=2, out_dir=out_dir, quiet=True)
    with open(path) as f:
        rows_again = [json.loads(line) for line in f]
    assert len(rows_again) == 2
    assert rows_again == rows
