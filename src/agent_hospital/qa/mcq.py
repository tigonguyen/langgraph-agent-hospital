"""Shared MCQ helpers: format a question and parse a letter answer."""

from __future__ import annotations

import re

from agent_hospital.diseases.medqa_usmle import MCQItem

_LETTERS = "ABCD"


LETTER_ONLY = "Respond with ONLY the letter (A, B, C, or D) of the best answer."
ANALYSE_ONLY = "Analyse the case and the options. Do NOT state a final answer."
# Panel opinions are internal — they feed the attending, so they stay unconstrained.
DELIBERATE = (
    "Reason about the key findings and the options, then on the LAST line write "
    "'Answer: X' where X is A, B, C, or D."
)
REASON_THEN_ANSWER = (
    "In at most 30 words, say why the best option is best, then on the LAST line "
    "write 'Answer: X' where X is A, B, C, or D."
)
# V1 agentic RAG: must permit a tool call (so NOT "respond with ONLY the letter",
# which forbids any non-letter output and suppresses the tool call).
AGENTIC_ANSWER = (
    "First, if it would help, call search_medmcqa with a focused query to retrieve similar "
    "solved questions. Then, in at most 30 words, say why the best option is best, and on the "
    "LAST line write 'Answer: X' where X is A, B, C, or D."
)


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
