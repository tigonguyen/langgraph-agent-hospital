"""Long-term attack memory for the runtime guard (docs/redteam/DEFENSE_MEMORY.md).

Every attack the clean verifier catches is stored twice in one Chroma collection — the raw
prompt and a one-line abstract signature of its intent — under a shared `attack_id`, so recall
matches a paraphrase via the signature even when the wording shares little with the original.

An embedding store, not `graph/longterm.py`'s SqliteStore: that bank recalls by topic-word
overlap, which a reworded attack evades trivially.

Recalled entries are a classification signal for the guard only — never pasted into any
agent's prompt, so stored text has no channel to inject instructions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DIR = "data/redteam/guard/chroma"
DEFAULT_COLLECTION = "attack_memory_qwen3"
DEFAULT_EMBEDDER = "qwen3-embedding:4b"
DUPLICATE_SIM = 0.98        # a new attack this close to a stored one adds nothing


@dataclass(frozen=True)
class AttackHit:
    attack_id: str
    sim: float              # best cosine relevance over the entry's prompt and signature docs
    prompt: str
    signature: str
    category: str


class AttackMemory:
    """Thin wrapper over a Chroma vector store. `read_only` freezes it for scored runs."""

    def __init__(self, store: Any, *, read_only: bool = False) -> None:
        self.store = store
        self.read_only = read_only

    def __len__(self) -> int:
        try:
            ids = self.store.get(where={"kind": "prompt"}, include=[])["ids"]
        except Exception:
            return 0
        return len(ids)

    def recall(self, prompt: str, k: int = 3) -> list[AttackHit]:
        """Best `k` distinct attacks, highest similarity first; [] on a cold/broken store."""
        if not prompt.strip():
            return []
        try:
            docs = self.store.similarity_search_with_relevance_scores(prompt, k=2 * k)
        except Exception:
            return []
        best: dict[str, AttackHit] = {}
        for doc, sim in docs:
            md = doc.metadata
            aid = md.get("attack_id", "")
            if aid in best and best[aid].sim >= sim:
                continue
            best[aid] = AttackHit(aid, float(sim), md.get("prompt", ""),
                                  md.get("signature", ""), md.get("category", ""))
        return sorted(best.values(), key=lambda h: -h.sim)[:k]

    def top_sim(self, prompt: str) -> tuple[float, AttackHit | None]:
        hits = self.recall(prompt, k=1)
        return (hits[0].sim, hits[0]) if hits else (0.0, None)

    def remember(self, prompt: str, signature: str = "", category: str = "",
                 source: str = "") -> str | None:
        """Store one caught attack; returns its id, or None if frozen/duplicate/failed."""
        prompt, signature = prompt.strip(), signature.strip()
        if self.read_only or not prompt:
            return None
        sim, _ = self.top_sim(prompt)
        if sim >= DUPLICATE_SIM:
            return None
        from langchain_core.documents import Document

        aid = uuid.uuid4().hex[:12]
        md = {"attack_id": aid, "prompt": prompt[:2000], "signature": signature[:400],
              "category": category, "source": source, "ts": time.time()}
        docs = [Document(page_content=prompt, metadata={**md, "kind": "prompt"})]
        ids = [f"{aid}-p"]
        if signature:
            docs.append(Document(page_content=signature, metadata={**md, "kind": "signature"}))
            ids.append(f"{aid}-s")
        try:
            self.store.add_documents(docs, ids=ids)
        except Exception:
            return None
        return aid

    def clear(self) -> None:
        try:
            ids = self.store.get(include=[])["ids"]
            if ids:
                self.store.delete(ids=ids)
        except Exception:
            pass


def open_attack_memory(persist_dir: str | Path = DEFAULT_DIR, *,
                       collection: str = DEFAULT_COLLECTION, embeddings: Any | None = None,
                       read_only: bool = False) -> AttackMemory:
    """Open (or create) the attack memory. Defaults to a sandbox dir, never `data/chroma/`."""
    from agent_hospital.knowledge import default_embeddings, open_store

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    store = open_store(collection, persist_dir=persist_dir,
                       embeddings=embeddings or default_embeddings(DEFAULT_EMBEDDER))
    return AttackMemory(store, read_only=read_only)
