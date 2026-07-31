"""Long-term (cross-episode) memory — spec §4.5.

Distinct from the short-term working memory in `QAState`, which lives for exactly one
question. Two SEPARATE banks live here, both in the same `SqliteStore` (persisted to
disk, survives the process), distinguished only by namespace root:

    "medqa-lessons"  — the GENERAL bank. The DECIDER writes a one-line lesson after
                       EVERY case (right or wrong) and recalls lessons from similar
                       earlier cases. See `recall`/`remember`.
    "medqa-mistakes" — the MISTAKE bank. A node that runs AFTER the verifier — the only
                       node that ever reads the gold answer — distills a corrective
                       lesson ONLY when the final answer was wrong. The verifier may
                       recall from this bank, but never writes to it and never sees
                       gold itself. See `recall_mistakes`/`remember_mistake`.

    ┌── episode N ────────────────────────────────────┐
    │ answer (decider) → recall(lessons)               │ <- general bank, every case
    │                  → remember(lesson)               │
    │ verify           → recall_mistakes(mistakes)      │ <- mistake bank, read-only here
    │ distill_mistake  → compare to gold, remember_mistake if wrong │ <- only writer, only gold-reader
    └───────────────────────────────────────────────────┘   store survives the process

WHY A STORE, NOT A CHECKPOINTER: a checkpointer (`SqliteSaver`) is keyed by `thread_id`
and persists ONE conversation. This project uses `thread_id = item.id`, so a checkpointer
can only ever replay the same question — it cannot carry knowledge between questions. A
`BaseStore` is namespaced key/value with search, which is the cross-episode primitive.

⚠️ BENCHMARK CONTAMINATION. Lessons written while scoring a split and read back during
that same split leak information between graded items, which inflates accuracy for reasons
unrelated to reasoning quality. `namespace`/`mistake_namespace` therefore include the
split, and the intended protocol is: BUILD on `train`, then READ-ONLY on `test`
(`read_only=True`). Keep the default off for any number that goes in the report.
"""

from __future__ import annotations

import re
from typing import Any

# One namespace per (corpus, split) so train-built lessons never mix with test-built ones.
_NS_ROOT = "medqa-lessons"
_MISTAKE_NS_ROOT = "medqa-mistakes"     # separate root: a mistake and a lesson for the
                                        # SAME item must never leak into each other's recall.
DEFAULT_DB = "data/longterm/lessons.sqlite"


def namespace(split: str) -> tuple[str, ...]:
    return (_NS_ROOT, split)


def mistake_namespace(split: str) -> tuple[str, ...]:
    return (_MISTAKE_NS_ROOT, split)


def open_store(db_path: str = DEFAULT_DB):
    """Open (or create) the persistent lesson store.

    `SqliteStore.from_conn_string` is a context manager; long-lived callers need the
    object itself, so enter it here and hand back the store plus its closer.
    """
    from pathlib import Path

    from langgraph.store.sqlite import SqliteStore

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    cm = SqliteStore.from_conn_string(db_path)
    store = cm.__enter__()
    store.setup()
    return store, lambda: cm.__exit__(None, None, None)


def try_open(db_path: str = DEFAULT_DB) -> tuple[Any | None, Any]:
    """`open_store`, degrading to `(None, no-op)` on failure (missing dependency, bad path).

    Shared by every graph-node closure that lazily opens the store on first use — a
    cold/unavailable store must never break an eval run, just fall back to no recall.

    Returns `(store, close)` — the CALLER MUST hold onto `close` (even unused) for as
    long as it uses `store`. `SqliteStore`'s underlying connection lives inside a context
    manager with a `__del__` finalizer; if nothing keeps that context manager referenced,
    Python can garbage-collect and close it mid-run, and every `store.put`/`.get` after
    that raises "Cannot operate on a closed database" — silently, since callers here
    swallow store exceptions to degrade gracefully. (Discarding it once turned every
    write in a real run into a no-op; caught by inspecting the sqlite file directly.)
    """
    try:
        return open_store(db_path)
    except Exception:
        return None, lambda: None


_STOP = {"a", "an", "the", "of", "with", "and", "or", "for", "to", "in", "on", "at", "is",
         "was", "who", "his", "her", "year", "old", "man", "woman", "patient", "presents",
         "has", "had", "his", "she", "he", "they", "most", "likely", "which", "following"}


def case_key(question: str, limit: int = 12) -> str:
    """A crude topical key for a vignette: its distinctive lowercase words.

    Deliberately not an embedding — this must stay cheap (it runs per item) and must not
    add a second embedder that would need its own collection. Good enough to group
    "cisplatin ototoxicity" cases together, which is all the recall step needs.
    """
    words = re.findall(r"[a-z]{4,}", question.lower())
    seen: dict[str, None] = {}
    for w in words:
        if w not in _STOP:
            seen.setdefault(w, None)
    return " ".join(list(seen)[:limit])


def _item_key(item_id: str, prefix: str) -> str:
    """Store key for an item: `<prefix>-00000` for item `train-00000`.

    Keyed off the item's numeric suffix rather than its full id, so a key is named for
    what it IS (a lesson/mistake) rather than for the split it happened to be learned on
    — the split already lives in the namespace, so repeating it in the key was redundant.
    """
    m = re.search(r"(\d+)\s*$", item_id)
    return f"{prefix}-{m.group(1)}" if m else f"{prefix}-{item_id}"


def lesson_key(item_id: str) -> str:
    return _item_key(item_id, "lesson")


def mistake_key(item_id: str) -> str:
    return _item_key(item_id, "mistake")


def _recall(store: Any, ns: tuple[str, ...], question: str, k: int, min_overlap: int,
            header: str) -> str:
    """Shared ranking logic for `recall`/`recall_mistakes`.

    `SqliteStore.search(query=...)` only ranks semantically when the store was built with a
    vector index; without one it returns the namespace unranked, which would paste unrelated
    lessons into the prompt (measured: a cisplatin lesson surfaced for an eosinophilia case).
    So rank here by topic-word overlap and require `min_overlap` shared words — recalling
    nothing is much better than recalling a misleading precedent.

    Failures are swallowed: a cold or corrupt store degrades to no recall, never breaks a run.
    """
    if store is None:
        return ""
    try:
        hits = store.search(ns, limit=200)
    except Exception:
        return ""

    wanted = set(case_key(question, limit=40).split())
    scored: list[tuple[int, str]] = []
    for h in hits:
        value = h.value or {}
        lesson = (value.get("lesson") or "").strip()
        if not lesson:
            continue
        overlap = len(wanted & set((value.get("topic") or "").split()))
        if overlap >= min_overlap:
            scored.append((overlap, lesson))

    scored.sort(key=lambda pair: -pair[0])
    lines = [f"- {lesson}" for _, lesson in scored[:k]]
    if not lines:
        return ""
    return f"{header}\n" + "\n".join(lines)


def recall(store: Any, split: str, question: str, k: int = 3, *, min_overlap: int = 2) -> str:
    """Lessons from earlier, topically similar cases (general bank) — "" when none."""
    return _recall(store, namespace(split), question, k, min_overlap,
                    "Lessons from similar cases you have seen before:")


def recall_mistakes(store: Any, split: str, question: str, k: int = 3, *, min_overlap: int = 2) -> str:
    """Lessons from earlier, topically similar cases the system got WRONG — "" when none.

    Same ranking as `recall`, over the separate mistake namespace. The verifier is the
    only reader of this bank and never writes to it or sees gold (module docstring) —
    only `distill_mistake` (graph/nodes.py) writes, via `remember_mistake`.
    """
    return _recall(store, mistake_namespace(split), question, k, min_overlap,
                    "Lessons from cases you got WRONG before, on similar patients:")


def _remember(store: Any, ns: tuple[str, ...], key: str, item_id: str,
              value: dict[str, Any], read_only: bool) -> None:
    # The CLI's synthetic warm-up item runs through the real graph, so without this its
    # throwaway lesson lands in the store and is recalled for genuine cases.
    if item_id == "warmup":
        return
    if store is None or read_only:
        return
    try:
        store.put(ns, key, value)
    except Exception:
        pass


def remember(store: Any, split: str, item_id: str, question: str, lesson: str,
             *, chosen: str = "", read_only: bool = False) -> None:
    """Persist one lesson for future episodes (general bank, every case, right or wrong).
    No-op when read_only (scoring a split) or the case taught nothing worth keeping."""
    if not lesson.strip():
        return
    _remember(store, namespace(split), lesson_key(item_id), item_id,
              {"lesson": lesson.strip()[:400], "chosen": chosen, "topic": case_key(question),
               "item_id": item_id},          # keep the source item for traceability
              read_only)


def remember_mistake(store: Any, split: str, item_id: str, question: str, lesson: str,
                      *, wrong: str, correct: str, read_only: bool = False) -> None:
    """Persist one corrective lesson for a case the system got WRONG (mistake bank).

    Called only by `distill_mistake` (graph/nodes.py), the only node that reads gold —
    the verifier itself never sees `wrong`/`correct`, only the lesson text it can later
    recall via `recall_mistakes`. No-op when read_only or the lesson is empty.
    """
    if not lesson.strip():
        return
    _remember(store, mistake_namespace(split), mistake_key(item_id), item_id,
              {"lesson": lesson.strip()[:400], "wrong": wrong, "correct": correct,
               "topic": case_key(question), "item_id": item_id},
              read_only)


_LESSON_RE = re.compile(r"lesson\s*:\s*(.+)", re.IGNORECASE)


def extract_lesson(reply: str) -> str:
    """Pull the `Lesson: ...` line the verifier is asked to emit.

    Returns "" when absent, so a model that ignores the instruction simply writes nothing
    rather than polluting the store with a slice of its own prose.
    """
    m = _LESSON_RE.search(reply or "")
    if not m:
        return ""
    return m.group(1).strip().splitlines()[0].strip()
