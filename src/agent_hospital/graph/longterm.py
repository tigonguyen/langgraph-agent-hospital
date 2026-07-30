"""Long-term (cross-episode) memory for V3 — spec §4.5.

Distinct from the short-term working memory in `QAState`, which lives for exactly one
question. Here the verifier writes a one-line **lesson** after each case and reads back
lessons from *similar* earlier cases, so experience accumulates across items and across
runs (LangGraph `SqliteStore`, persisted to disk).

    ┌── episode N ──────────────┐
    │ verify → audit answer     │
    │        → recall(lessons)  │ <- reads lessons from earlier, similar cases
    │        → remember(lesson) │ -> writes one lesson for future cases
    └───────────────────────────┘   store survives the process

WHY A STORE, NOT A CHECKPOINTER: a checkpointer (`SqliteSaver`) is keyed by `thread_id`
and persists ONE conversation. This project uses `thread_id = item.id`, so a checkpointer
can only ever replay the same question — it cannot carry knowledge between questions. A
`BaseStore` is namespaced key/value with search, which is the cross-episode primitive.

⚠️ BENCHMARK CONTAMINATION. Lessons written while scoring a split and read back during
that same split leak information between graded items, which inflates accuracy for reasons
unrelated to reasoning quality. `namespace` therefore includes the split, and the intended
protocol is: BUILD on `train`, then READ-ONLY on `test` (`read_only=True`). Keep the
default off for any number that goes in the report.
"""

from __future__ import annotations

import re
from typing import Any

# One namespace per (corpus, split) so train-built lessons never mix with test-built ones.
_NS_ROOT = "medqa-lessons"
DEFAULT_DB = "data/longterm/lessons.sqlite"


def namespace(split: str) -> tuple[str, ...]:
    return (_NS_ROOT, split)


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


def recall(store: Any, split: str, question: str, k: int = 3, *, min_overlap: int = 2) -> str:
    """Lessons from earlier, topically similar cases — "" when there are none.

    `SqliteStore.search(query=...)` only ranks semantically when the store was built with a
    vector index; without one it returns the namespace unranked, which would paste unrelated
    lessons into the prompt (measured: a cisplatin lesson surfaced for an eosinophilia case).
    So rank here by topic-word overlap and require `min_overlap` shared words — recalling
    nothing is much better than recalling a misleading precedent.

    Failures are swallowed: a cold or corrupt store degrades to plain V3, never breaks a run.
    """
    if store is None:
        return ""
    try:
        hits = store.search(namespace(split), limit=200)
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
    return "Lessons from similar cases you have seen before:\n" + "\n".join(lines)


def lesson_key(item_id: str) -> str:
    """Store key for an item's lesson: `lesson-00000` for item `train-00000`.

    Keyed off the item's numeric suffix rather than its full id, so a lesson is named for
    what it IS (a lesson) rather than for the split it happened to be learned on — the
    split already lives in the namespace, so repeating it in the key was redundant.
    """
    m = re.search(r"(\d+)\s*$", item_id)
    return f"lesson-{m.group(1)}" if m else f"lesson-{item_id}"


def remember(store: Any, split: str, item_id: str, question: str, lesson: str,
             *, chosen: str = "", read_only: bool = False) -> None:
    """Persist one lesson for future episodes. No-op when read_only (scoring a split)."""
    # The CLI's synthetic warm-up item runs through the real graph, so without this its
    # throwaway lesson lands in the store and is recalled for genuine cases.
    if item_id == "warmup":
        return
    if store is None or read_only or not lesson.strip():
        return
    try:
        store.put(
            namespace(split),
            lesson_key(item_id),           # keyed per item so re-runs overwrite, not duplicate
            {"lesson": lesson.strip()[:400], "chosen": chosen, "topic": case_key(question),
             "item_id": item_id},          # keep the source item for traceability
        )
    except Exception:
        pass


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
