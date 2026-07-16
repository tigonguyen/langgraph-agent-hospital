"""Shared MCQ helpers: format a question and parse a letter answer."""

from __future__ import annotations

import re

from agent_hospital.diseases.medqa_usmle import MCQItem

_LETTERS = "ABCD"


def format_mcq(item: MCQItem) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(item.options))
    return (
        f"{item.question}\n\n{opts}\n\n"
        "Respond with ONLY the letter (A, B, C, or D) of the best answer."
    )


def parse_choice(text: str) -> int | None:
    """Extract the chosen option index (0-3) from a model reply, or None."""
    if not text:
        return None
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABCD])\b", text)
    return _LETTERS.index(m.group(1).upper()) if m else None
