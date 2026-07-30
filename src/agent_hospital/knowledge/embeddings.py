"""Embedding function for the RAG knowledge base.

Local and Ollama-backed (`nomic-embed-text`) — the same stack as the user's
obsidian-rag. This is the single swap-point: to use a different embedder later
(e.g. a domain-tuned medical model), return a different LangChain `Embeddings`
here; the ingest and retriever code is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

DEFAULT_EMBED_MODEL = "nomic-embed-text"
# MedCPT encoder weights live here (fetched via curl; gitignored, ~870 MB).
_MEDCPT_DIR = Path(__file__).resolve().parents[3] / "data" / "medcpt"


def default_embeddings(model: str = DEFAULT_EMBED_MODEL) -> Embeddings:
    """Return the embedding function for ingest and retrieval.

    `model="medcpt"` → the local medical-domain MedCPT bi-encoder (asymmetric);
    anything else → that Ollama embedding model (default `nomic-embed-text`).
    NOTE: a Chroma collection must be queried with the SAME embedder it was built
    with — MedCPT and nomic vectors are not interchangeable, so MedCPT needs its
    own collection (re-ingest with `model="medcpt"`).
    """
    if model.lower() == "medcpt":
        from agent_hospital.knowledge.medcpt import MedCPTEmbeddings

        return MedCPTEmbeddings(
            query_model=str(_MEDCPT_DIR / "MedCPT-Query-Encoder"),
            article_model=str(_MEDCPT_DIR / "MedCPT-Article-Encoder"),
        )

    # In `api` mode NOTHING runs locally, embeddings included: the endpoint is taken from
    # AGENT_HOSPITAL_EMBED_BASE_URL, else derived from ANTHROPIC_BASE_URL. Setting
    # EMBED_BASE_URL alone also works in `local` mode, for a local chat model + remote
    # embeddings. Requires an OpenAI-style POST /v1/embeddings on that host.
    # CAUTION: a Chroma collection must be QUERIED with the same embedder it was BUILT
    # with, so switching this against an existing store returns wrong neighbours silently.
    from agent_hospital.models import mode

    base_url = os.environ.get("AGENT_HOSPITAL_EMBED_BASE_URL")
    # `local` (or `ollama`) pins embeddings to Ollama even in api mode — the common split
    # when a chat gateway has no reachable /v1/embeddings. Accepted as a value here rather
    # than only as a URL, since "local" is the obvious thing to write.
    if base_url and base_url.strip().lower() in ("local", "ollama"):
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)

    if not base_url and mode() == "api":
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "AGENT_HOSPITAL_MODE=api needs a remote embeddings endpoint: set "
                "AGENT_HOSPITAL_EMBED_BASE_URL (or ANTHROPIC_BASE_URL) in .env, or use "
                "AGENT_HOSPITAL_MODE=local to embed with Ollama.")

    if base_url:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.environ.get("AGENT_HOSPITAL_EMBED_MODEL") or model,
            base_url=base_url.rstrip("/") + "/v1"
            if not base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/"),
            api_key=os.environ.get("AGENT_HOSPITAL_EMBED_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY", "unused"),
            check_embedding_ctx_length=False,   # non-OpenAI backends reject the tokenised form
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model)
