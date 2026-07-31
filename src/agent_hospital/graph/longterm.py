"""Long-term (cross-episode) memory — spec §4.5.

Two separate banks in one SqliteStore, keyed by namespace: `medqa-lessons` (decider
recalls/writes every case) and `medqa-mistakes` (verifier recalls; only `distill_mistake`,
the sole node that reads gold, writes to it).

A BaseStore, not a checkpointer, because checkpoints are keyed by thread_id (= item.id
here) and can't carry knowledge across questions.

Writing while scoring a split and reading it back in the same run leaks between graded
items. `namespace`/`mistake_namespace` include the split; writing is the default (no CLI
opt-out) — for a report number, override `long_term_read_only=True` programmatically.
"""

from __future__ import annotations

import re
from typing import Any

_NS_ROOT = "medqa-lessons"
_MISTAKE_NS_ROOT = "medqa-mistakes"
DEFAULT_DB = "data/longterm/lessons.sqlite"


def namespace(split: str) -> tuple[str, ...]:
    return (_NS_ROOT, split)


def mistake_namespace(split: str) -> tuple[str, ...]:
    return (_MISTAKE_NS_ROOT, split)


def open_store(db_path: str = DEFAULT_DB):
    """Open (or create) the persistent lesson store."""
    from pathlib import Path

    from langgraph.store.sqlite import SqliteStore

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    cm = SqliteStore.from_conn_string(db_path)
    store = cm.__enter__()
    store.setup()
    return store, lambda: cm.__exit__(None, None, None)


def try_open(db_path: str = DEFAULT_DB) -> tuple[Any | None, Any]:
    """`open_store`, degrading to `(None, no-op)` so a cold/unavailable store never breaks a run.

    Caller must hold onto the returned `close` even if unused — `SqliteStore`'s connection
    has a `__del__` finalizer, so letting it get garbage-collected mid-run silently turns
    every later `store.put`/`.get` into a no-op.
    """
    try:
        return open_store(db_path)
    except Exception:
        return None, lambda: None


_STOP = {"a", "an", "the", "of", "with", "and", "or", "for", "to", "in", "on", "at", "is",
         "was", "who", "his", "her", "year", "old", "man", "woman", "patient", "presents",
         "has", "had", "his", "she", "he", "they", "most", "likely", "which", "following"}


def case_key(question: str, limit: int = 12) -> str:
    """Crude topical key (distinctive lowercase words), not an embedding — stays cheap
    per-item and avoids needing a second embedder/collection."""
    words = re.findall(r"[a-z]{4,}", question.lower())
    seen: dict[str, None] = {}
    for w in words:
        if w not in _STOP:
            seen.setdefault(w, None)
    return " ".join(list(seen)[:limit])


def _item_key(item_id: str, prefix: str) -> str:
    m = re.search(r"(\d+)\s*$", item_id)
    return f"{prefix}-{m.group(1)}" if m else f"{prefix}-{item_id}"


def lesson_key(item_id: str) -> str:
    return _item_key(item_id, "lesson")


def mistake_key(item_id: str) -> str:
    return _item_key(item_id, "mistake")


def _recall(store: Any, ns: tuple[str, ...], question: str, k: int, min_overlap: int,
            header: str) -> str:
    """Ranks by topic-word overlap rather than `store.search`'s default (unranked without a
    vector index — pasted unrelated lessons into the prompt). `min_overlap` gate: no match
    beats a misleading one."""
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
    return _recall(store, namespace(split), question, k, min_overlap,
                    "Lessons from similar cases you have seen before:")


def recall_mistakes(store: Any, split: str, question: str, k: int = 3, *, min_overlap: int = 2) -> str:
    return _recall(store, mistake_namespace(split), question, k, min_overlap,
                    "Lessons from cases you got WRONG before, on similar patients:")


def _remember(store: Any, ns: tuple[str, ...], key: str, item_id: str,
              value: dict[str, Any], read_only: bool) -> None:
    if item_id == "warmup":   # throwaway CLI warm-up item must never enter the store
        return
    if store is None or read_only:
        return
    try:
        store.put(ns, key, value)
    except Exception:
        pass


def remember(store: Any, split: str, item_id: str, question: str, lesson: str,
             *, chosen: str = "", read_only: bool = False) -> None:
    if not lesson.strip():
        return
    _remember(store, namespace(split), lesson_key(item_id), item_id,
              {"lesson": lesson.strip()[:400], "chosen": chosen, "topic": case_key(question),
               "item_id": item_id},
              read_only)


def remember_mistake(store: Any, split: str, item_id: str, question: str, lesson: str,
                      *, wrong: str, correct: str, read_only: bool = False) -> None:
    if not lesson.strip():
        return
    _remember(store, mistake_namespace(split), mistake_key(item_id), item_id,
              {"lesson": lesson.strip()[:400], "wrong": wrong, "correct": correct,
               "topic": case_key(question), "item_id": item_id},
              read_only)


_LESSON_RE = re.compile(r"lesson\s*:\s*(.+)", re.IGNORECASE)


def extract_lesson(reply: str) -> str:
    """Pulls the `Lesson: ...` line; "" when absent so an ignored instruction doesn't
    pollute the store with prose."""
    m = _LESSON_RE.search(reply or "")
    if not m:
        return ""
    return m.group(1).strip().splitlines()[0].strip()
