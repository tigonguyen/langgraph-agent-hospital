"""Embedding function for the RAG knowledge base.

Local and Ollama-backed (`nomic-embed-text`) — the same stack as the user's
obsidian-rag. This is the single swap-point: to use a different embedder later
(e.g. a domain-tuned medical model), return a different LangChain `Embeddings`
here; the ingest and retriever code is unchanged.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "nomic-embed-text"


def default_embeddings(model: str = DEFAULT_EMBED_MODEL) -> Embeddings:
    """Return the embedding function for ingest and retrieval."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model)
