"""Ingest the MedRAG Textbooks corpus into the Chroma `knowledge` collection.

Prototype on a small subset first (default 1000 snippets) to validate the pipeline,
then raise `limit` to embed the full corpus. Snippets are used as-is (MedRAG is
already chunked). Run as a script:

    python -m agent_hospital.knowledge.ingest [limit] [shuffle_buffer] [embed_model]

`embed_model` (default: none → Ollama `nomic-embed-text`, collection `knowledge`)
picks a different embedder from `knowledge/embeddings.py:default_embeddings`; its
collection is named `knowledge_<embed_model>` (e.g. `minilm` → `knowledge_minilm`),
since a collection must be queried with the same embedder it was built with.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

from agent_hospital.knowledge.retriever import open_store

DATASET = "MedRAG/textbooks"


def _content(rec: dict) -> str:
    return (rec.get("content") or rec.get("contents") or rec.get("text") or "").strip()


def ingest_textbooks(
    limit: int = 1000,
    *,
    store: Any | None = None,
    batch_size: int = 64,
    shuffle_buffer: int = 0,
    verbose: bool = False,
) -> int:
    """Stream `limit` MedRAG Textbook snippets into Chroma. Returns count ingested.

    `shuffle_buffer` > 0 shuffles the stream over that buffer so the sample spans
    many books (the corpus is ordered by book, so a plain prefix covers only the
    first one). `verbose` prints per-batch progress.
    """
    from datasets import load_dataset

    store = store or open_store()
    ds = load_dataset(DATASET, split="train", streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=0, buffer_size=shuffle_buffer)

    # Embed incrementally as rows stream in: progress shows from the first batch,
    # memory stays flat, and any HF throttling is visible immediately.
    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    total = 0

    def flush() -> None:
        nonlocal texts, metadatas, ids, total
        if not texts:
            return
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        total += len(texts)
        if verbose:
            print(f"  embedded {total} (target {limit})", flush=True)
        texts, metadatas, ids = [], [], []

    for i, rec in enumerate(islice(ds, limit)):
        content = _content(rec)
        if not content:
            continue
        texts.append(content)
        metadatas.append({"title": rec.get("title", ""), "source": "textbooks"})
        ids.append(rec.get("id") or f"tb-{i:06d}")
        if len(texts) >= batch_size:
            flush()
    flush()
    return total


MEDMCQA = "openlifescienceai/medmcqa"
_OPTS = ("opa", "opb", "opc", "opd")


def ingest_medmcqa(
    limit: int = 1000,
    *,
    store: Any | None = None,
    batch_size: int = 64,
    require_exp: bool = False,
    verbose: bool = False,
) -> int:
    """Stream MedMCQA questions into Chroma as retrievable exemplars.

    Only the **question** is embedded — the answer and explanation ride along as
    metadata. Embedding the whole Q+A+explanation blob would let the ~500-char
    explanation swamp the ~55-char question, so the vector would stop representing
    what is being asked. `require_exp` skips items with no explanation.
    """
    from datasets import load_dataset

    store = store or open_store()
    ds = load_dataset(MEDMCQA, split="train", streaming=True)

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    total = 0

    def flush() -> None:
        nonlocal texts, metadatas, ids, total
        if not texts:
            return
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        total += len(texts)
        if verbose:
            print(f"  embedded {total} (target {limit})", flush=True)
        texts, metadatas, ids = [], [], []

    for i, rec in enumerate(islice(ds, limit)):
        question = (rec.get("question") or "").strip()
        cop = rec.get("cop")
        if not question or cop is None or not 0 <= cop < len(_OPTS):
            continue
        exp = (rec.get("exp") or "").strip()
        if require_exp and not exp:
            continue
        texts.append(question)
        metadatas.append({
            "answer": (rec.get(_OPTS[cop]) or "").strip(),
            "explanation": exp,
            "subject": (rec.get("subject_name") or "").strip(),
            "source": "medmcqa",
        })
        ids.append(f"mcq-{rec.get('id') or i}")
        if len(texts) >= batch_size:
            flush()
    flush()
    return total


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    shuffle = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    embed_model = sys.argv[3] if len(sys.argv) > 3 else None

    store = None
    collection = "knowledge"
    if embed_model:
        from agent_hospital.knowledge.embeddings import default_embeddings

        collection = f"knowledge_{embed_model}"
        store = open_store(
            collection=collection, embeddings=default_embeddings(embed_model)
        )

    count = ingest_textbooks(limit=n, store=store, shuffle_buffer=shuffle, verbose=True)
    print(f"ingested {count} snippets into the 'knowledge' collection")
