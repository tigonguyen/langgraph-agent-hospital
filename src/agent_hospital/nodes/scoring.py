"""Scoring node: compare the doctor's diagnosis to the hidden answer key.

Phase 1 uses a deterministic normalized-containment match (an LLM judge can
replace `score_diagnosis` later without touching the graph).
"""

from __future__ import annotations

import re
from typing import Any

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def score_diagnosis(prediction: str, gold: str) -> float:
    """Return 1.0 if the prediction matches the gold diagnosis, else 0.0.

    Match = either normalized string contains the other (handles "Myasthenia
    gravis" vs "I think it's myasthenia gravis").
    """
    pred, g = _normalize(prediction), _normalize(gold)
    if not pred or not g:
        return 0.0
    return 1.0 if (g in pred or pred in g) else 0.0


def score_node(state: dict[str, Any]) -> dict[str, Any]:
    """Graph node: set `score` from the diagnosis vs the case's correct_diagnosis."""
    gold = state["case"].correct_diagnosis
    return {"score": score_diagnosis(state.get("diagnosis", ""), gold)}
