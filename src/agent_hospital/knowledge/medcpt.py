"""MedCPT embeddings (medical-domain bi-encoder) for the RAG knowledge base.

MedCPT has TWO encoders sharing one vector space: the Article encoder embeds
documents, the Query encoder embeds queries. This maps onto LangChain's
`Embeddings` interface — `embed_documents` → article encoder, `embed_query` →
query encoder — so the ingest/retriever code is unchanged. Vectors are
L2-normalized so Chroma's cosine space matches relevance ranking.

Swap it in via `knowledge/embeddings.py:default_embeddings` once the encoders are
cached. Requires `transformers` + `torch`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MedCPTEmbeddings(Embeddings):
    """LangChain embeddings backed by MedCPT's query/article encoders."""

    def __init__(
        self,
        *,
        query_model: str = QUERY_MODEL,
        article_model: str = ARTICLE_MODEL,
        query_max_length: int = 64,
        article_max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.query_model = query_model
        self.article_model = article_model
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length
        self.batch_size = batch_size
        self.device = device or _pick_device()
        self.normalize = normalize
        self._loaded: dict[str, Any] = {}

    def _encoder(self, name: str):
        if name not in self._loaded:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(self.device).eval()
            self._loaded[name] = (tok, model)
        return self._loaded[name]

    def _encode(self, texts: list[str], name: str, max_length: int) -> list[list[float]]:
        import torch

        tok, model = self._encoder(name)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=max_length,
                      return_tensors="pt").to(self.device)
            with torch.no_grad():
                vecs = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token
                if self.normalize:
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            out.extend(vecs.cpu().tolist())
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), self.article_model, self.article_max_length)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_model, self.query_max_length)[0]
