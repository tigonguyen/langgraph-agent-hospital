"""Ingest MedMCQA solved questions into the Chroma `knowledge` collection.

Run as a script: `python -m agent_hospital.knowledge.ingest [limit]`.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

from agent_hospital.knowledge.retriever import open_store

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
    count = ingest_medmcqa(limit=n, verbose=True)
    print(f"ingested {count} questions into the 'knowledge' collection")
