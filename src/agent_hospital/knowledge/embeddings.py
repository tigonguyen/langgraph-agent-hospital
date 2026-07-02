"""Embedding function for the RAG knowledge base.

Local and Ollama-backed (`nomic-embed-text`) — the same stack as the user's
obsidian-rag. This is the single swap-point: to use a different embedder later
(e.g. a domain-tuned medical model), return a different LangChain `Embeddings`
here; the ingest and retriever code is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "nomic-embed-text"
# MedCPT encoder weights live here (fetched via curl; gitignored, ~870 MB).
_MEDCPT_DIR = Path(__file__).resolve().parents[3] / "data" / "medcpt"


def default_embeddings(model: str = DEFAULT_EMBED_MODEL) -> Embeddings:
    """Return the embedding function for ingest and retrieval.

    `model="medcpt"` → the local medical-domain MedCPT bi-encoder (asymmetric);
    `model="minilm"` → local `sentence-transformers/all-MiniLM-L6-v2` (no API, no
    Ollama — just a one-time HuggingFace download, then fully offline);
    anything else → that Ollama embedding model (default `nomic-embed-text`).
    NOTE: a Chroma collection must be queried with the SAME embedder it was built
    with — vectors from different embedders are not interchangeable, so each
    non-default embedder needs its own collection (re-ingest with that `model=`).
    """
    if model.lower() == "medcpt":
        from agent_hospital.knowledge.medcpt import MedCPTEmbeddings

        return MedCPTEmbeddings(
            query_model=str(_MEDCPT_DIR / "MedCPT-Query-Encoder"),
            article_model=str(_MEDCPT_DIR / "MedCPT-Article-Encoder"),
        )
    if model.lower() == "minilm":
        from agent_hospital.knowledge.minilm import MiniLMEmbeddings

        return MiniLMEmbeddings()
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model)
