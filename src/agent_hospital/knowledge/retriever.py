"""Retrieval over the Chroma `knowledge` collection.

Dense Vector Space Model: cosine similarity (relevance), top-k, with a similarity
**gate** — snippets below the threshold are dropped, and if none qualify the caller
gets an empty list (the no-RAG fallback: answer without retrieved context).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agent_hospital.knowledge.embeddings import default_embeddings

CHROMA_DIR = "data/chroma"
KNOWLEDGE_COLLECTION = "knowledge"
DEFAULT_K = 4
DEFAULT_THRESHOLD = 0.5


def open_store(
    collection: str = KNOWLEDGE_COLLECTION,
    persist_dir: str | Path = CHROMA_DIR,
    embeddings: Any | None = None,
):
    """Open (or create) a persistent Chroma collection in cosine space."""
    from langchain_chroma import Chroma

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
    """Return (doc, relevance) pairs above the threshold; empty list = no-RAG fallback."""
    store = store or open_store()
    hits = store.similarity_search_with_relevance_scores(query, k=k)
    return [(doc, score) for doc, score in hits if score >= threshold]


def format_evidence(hits: list[tuple[Document, float]], max_chars: int = 500) -> str:
    """Render retrieved hits as a labeled evidence block (empty string if none)."""
    if not hits:
        return ""
    lines = ["Textbook evidence:"]
    for doc, _score in hits:
        title = doc.metadata.get("title", "").strip()
        prefix = f"({title}) " if title else ""
        lines.append(f"- {prefix}{doc.page_content.strip()[:max_chars]}")
    return "\n".join(lines)
