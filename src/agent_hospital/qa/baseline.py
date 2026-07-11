"""Direct single-LLM baseline (variant V0) for MedQA-USMLE.

Presents the question + options, asks for a single letter, parses the choice.
The reference accuracy every other variant is compared against.
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa_usmle import MCQItem

_LETTERS = "ABCD"

BASELINE_SYS = (
    "You are an expert physician answering a medical board (USMLE) multiple-choice "
    "question. Choose the single best answer."
)


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


def build_baseline_answerer(
    model: BaseChatModel | str = "qwen2.5:7b",
) -> Callable[[MCQItem], int | None]:
    """Return an answer fn: MCQItem -> chosen option index (or None)."""
    agent = Agent("baseline", BASELINE_SYS, model=model)

    def answer(item: MCQItem) -> int | None:
        return parse_choice(agent.say(format_mcq(item)))

    return answer
