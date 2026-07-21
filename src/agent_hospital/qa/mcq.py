"""Shared MCQ helpers: format a question and parse a letter answer."""

from __future__ import annotations

import re

from agent_hospital.diseases.medqa_usmle import MCQItem

_LETTERS = "ABCD"


LETTER_ONLY = "Respond with ONLY the letter (A, B, C, or D) of the best answer."
ANALYSE_ONLY = "Analyse the case and the options. Do NOT state a final answer."


def format_mcq(item: MCQItem, closing: str = LETTER_ONLY) -> str:
    """Render the question + options, ending with `closing`.

    Nodes that must NOT produce a letter (the clinical reasoner) pass their own
    closing — LETTER_ONLY would contradict their system prompt.
    """
    opts = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(item.options))
    return f"{item.question}\n\n{opts}\n\n{closing}"


def parse_choice(text: str) -> int | None:
    """Extract the chosen option index (0-3) from a model reply, or None."""
    if not text:
        return None
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABCD])\b", text)
    return _LETTERS.index(m.group(1).upper()) if m else None
