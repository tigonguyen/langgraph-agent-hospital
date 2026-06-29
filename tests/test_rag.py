"""RAG tests: ingest a tiny fixture into a temp Chroma, then check gated retrieval.

Needs Ollama (for nomic-embed-text); skips if unreachable. No HF download.
"""

import os

import pytest
import requests

from agent_hospital.knowledge import format_evidence, open_store, retrieve

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

_FIXTURE = [
    ("mg", "Myasthenia gravis is an autoimmune disorder causing fatigable muscle weakness "
           "from acetylcholine receptor antibodies; symptoms worsen with activity and improve with rest."),
    ("hem", "Hemorrhoids are swollen veins in the lower rectum and anus, commonly presenting "
            "with painless rectal bleeding."),
    ("dm", "Type 2 diabetes mellitus is characterized by insulin resistance and elevated blood "
           "glucose, often managed with metformin."),
]


def _ollama_ready() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    return EMBED_MODEL in " ".join(m["name"] for m in resp.json().get("models", []))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not _ollama_ready():
        pytest.skip(f"Ollama or {EMBED_MODEL} unavailable")
    persist = tmp_path_factory.mktemp("chroma")
    s = open_store(collection="test_kb", persist_dir=persist)
    s.add_texts(
        texts=[t for _, t in _FIXTURE],
        metadatas=[{"title": k, "source": "fixture"} for k, _ in _FIXTURE],
        ids=[k for k, _ in _FIXTURE],
    )
    return s


def test_relevant_query_retrieves_right_doc(store):
    hits = retrieve("muscle weakness that worsens with activity and improves with rest",
                    k=3, threshold=0.4, store=store)
    assert hits, "expected at least one hit above threshold"
    top_doc, top_score = hits[0]
    assert top_doc.metadata["title"] == "mg"   # myasthenia is the closest
    assert top_score >= hits[-1][1]            # sorted by relevance


def test_offtopic_query_is_gated_out(store):
    # An unrelated query should fall below a strict gate → no-RAG fallback.
    hits = retrieve("instructions for baking sourdough bread at home",
                    k=3, threshold=0.6, store=store)
    assert hits == []
    assert format_evidence(hits) == ""         # empty evidence block = fallback signal


def test_format_evidence_labels_hits(store):
    hits = retrieve("blood glucose and insulin resistance", k=2, threshold=0.4, store=store)
    block = format_evidence(hits)
    assert block.startswith("Textbook evidence:")
    assert "dm" in block  # the diabetes doc's title appears
