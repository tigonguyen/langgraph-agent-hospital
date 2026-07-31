"""Shared MCQ helpers: format a question and parse a letter answer."""

from __future__ import annotations

import re

from agent_hospital.diseases.medqa_usmle import MCQItem

_LETTERS = "ABCD"


LETTER_ONLY = "Respond with ONLY the letter (A, B, C, or D) of the best answer."
UNDERSTAND_ONLY = (
    "Output ONLY the two labeled sections above. Do not discuss the options, do not reason "
    "toward an answer, and do not name an answer — colleagues will do that next."
)
ANALYSE_ONLY = "Write your analysis as instructed above. Do NOT state a final answer."
DIGEST_EVIDENCE_ONLY = "Write your digest as instructed above. Do NOT state a final answer."
REASON_THEN_ANSWER = (
    "In at most 30 words, say why the best option is best, then on the LAST line "
    "write 'Answer: X' where X is A, B, C, or D."
)
# NOT "respond with ONLY the letter" — that forbids tool calls. Explicitly allows up to
# two: a user-turn instruction can silently override the system prompt's retry rule.
AGENTIC_ANSWER = (
    "Call search_medmcqa (up to twice, per the confidence-gated retry rule above) to retrieve "
    "similar solved questions. Then, in at most 40 words, say why the best option is best and "
    "state your confidence (High/Medium/Low) as instructed above, then on the LAST line write "
    "'Answer: X' where X is A, B, C, or D."
)
AGENTIC_VERIFY = (
    "First, if it would help, call search_textbooks with a focused query to check the proposed "
    "answer against textbook evidence. Then, in at most 30 words, say why you keep or change the "
    "answer, and on the LAST line write 'Answer: X' where X is A, B, C, or D."
)
# Lesson line must come BEFORE the answer line — parse_choice reads the LAST letter found.
LESSON_SUFFIX = (
    "Before that final answer line, add one line starting 'Lesson:' — a single transferable "
    "rule from this case, phrased so it helps on a DIFFERENT patient (name the discriminating "
    "finding and what it points to). Write no lesson if the case taught you nothing general."
)
AGENTIC_VERIFY_WIKIPEDIA = (
    "First, if it would help, call search_wikipedia with a focused query to check the proposed "
    "answer against a Wikipedia article. Then, in at most 30 words, say why you keep or change the "
    "answer, and on the LAST line write 'Answer: X' where X is A, B, C, or D."
)
# No 'Answer:' line — this node's output must never be parsed as an answer, only a lesson.
MISTAKE_LESSON_ONLY = (
    "Write your explanation as instructed above, ending with the 'Lesson:' line. Do not restate "
    "or change the answer — that has already been decided."
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
    # \b around the letter matters: without it, phrasing like "kept the answer AS C"
    # matched the lowercase 'a' in "as" (case-insensitive) and silently returned A instead
    # of the C stated two words later — a wrong answer, not a caught invalid response.
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\b([ABCD])\b\)?", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABCD])\b", text)
    return _LETTERS.index(m.group(1).upper()) if m else None


def summarize_rationale(rationale: str) -> str:
    """The explanation without its trailing 'Answer: X' — the pred already shows the letter.
    Also collapses it to one line, for compact per-item logging."""
    text = re.sub(r"\n*\s*Answer\s*:\s*[ABCD]\s*\.?\s*$", "", rationale.strip(), flags=re.IGNORECASE)
    return " ".join(text.split())
