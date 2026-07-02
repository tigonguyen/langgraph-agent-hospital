"""Local sentence-embedding model (all-MiniLM-L6-v2) for the RAG knowledge base.

Symmetric embedder (same encoder for queries and documents) that runs fully
offline once the weights are cached from HuggingFace — no per-call API cost
and no dependency on a network path that may be blocked. Requires
`transformers` + `torch` (the `minilm` extra).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MiniLMEmbeddings(Embeddings):
    """LangChain embeddings backed by a local sentence-transformers model."""

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or _pick_device()
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._tok, self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import torch

        tok, model = self._load()
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = tok(
                batch, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(vecs.cpu().tolist())
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
