"""Retrieval over the Chroma `knowledge` collection.

Dense Vector Space Model: cosine similarity (relevance), top-k, with a similarity
**gate** — snippets below the threshold are dropped, and if none qualify the caller
gets an empty list (the no-RAG fallback: answer without retrieved context).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agent_hospital.knowledge.embeddings import default_embeddings

CHROMA_DIR = "data/chroma"
KNOWLEDGE_COLLECTION = "knowledge"
DEFAULT_K = 4
DEFAULT_THRESHOLD = 0.5
OVERFETCH = 3            # fetch k*OVERFETCH so dedupe can still fill k slots

# Chroma caches one client "system" per persist dir; two threads opening it at once
# (e.g. LangGraph running parallel tool calls) race on that cache and one tears the
# other's half-started system down. Serialize opens.
_OPEN_LOCK = threading.Lock()


def open_store(
    collection: str = KNOWLEDGE_COLLECTION,
    persist_dir: str | Path = CHROMA_DIR,
    embeddings: Any | None = None,
):
    """Open (or create) a persistent Chroma collection in cosine space."""
    from langchain_chroma import Chroma

    with _OPEN_LOCK:
        return Chroma(
            collection_name=collection,
            embedding_function=embeddings or default_embeddings(),
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )


def retrieve(
    query: str,
    *,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    store: Any | None = None,
) -> list[tuple[Document, float]]:
    """Return up to `k` distinct (doc, relevance) pairs above the threshold.

    Over-fetches then drops repeats, because MedMCQA was scraped from open sources
    and contains the same question many times — without this, ~45% of queries spend
    two or more of their k slots on identical answers. Empty list = no-RAG fallback.
    """
    store = store or open_store()
    hits = store.similarity_search_with_relevance_scores(query, k=k * OVERFETCH)

    kept: list[tuple[Document, float]] = []
    seen: set[str] = set()
    for doc, score in hits:
        if score < threshold:
            continue
        # Dedupe on the answer for QA corpora, else on the passage text.
        key = (doc.metadata.get("answer") or doc.page_content).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append((doc, score))
        if len(kept) >= k:
            break
    return kept


def format_evidence(hits: list[tuple[Document, float]], max_chars: int = 500) -> str:
    """Render retrieved hits as a labeled evidence block (empty string if none).

    MedMCQA records store only the question as page_content — the answer and
    explanation live in metadata — so they get their own rendering.
    """
    if not hits:
        return ""
    if all(doc.metadata.get("source") == "medmcqa" for doc, _ in hits):
        lines = ["Related exam questions and their answers:"]
        for rank, (doc, _score) in enumerate(hits):
            md = doc.metadata
            line = f"- Q: {doc.page_content.strip()}\n  A: {md.get('answer', '').strip()}"
            exp = (md.get("explanation") or "").strip()
            if exp:
                # Taper: the best-matching hit earns the most room, later hits less.
                budget = max_chars // 2 if rank == 0 else max_chars // 4
                line += f"\n  Why: {exp[:budget]}"
            lines.append(line)
        return "\n".join(lines)

    lines = ["Textbook evidence:"]
    for doc, _score in hits:
        title = doc.metadata.get("title", "").strip()
        prefix = f"({title}) " if title else ""
        lines.append(f"- {prefix}{doc.page_content.strip()[:max_chars]}")
    return "\n".join(lines)
