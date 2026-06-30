"""Reasoning agent: convert a MedQA question into a focused retrieval query.

MedQA items bury the decisive question inside a long vignette, so retrieving on
the raw text pulls topical-but-non-discriminating passages (measured: RAG −8 pts).
This agent reads the question, reasons about the salient findings and what is
actually being asked, and emits a short search query — the focused query that the
RAG retriever should use instead of the whole vignette.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa_usmle import MCQItem

REASONING_SYS = (
    "You are a clinician constructing a search query for a medical textbook index. "
    "Read the multiple-choice question, identify the salient clinical findings and exactly what is "
    "being asked, then produce ONE concise query (max ~20 words) naming the key entities/concept to "
    "look up. Output ONLY the query — no reasoning, no preamble, no quotes."
)


def build_reasoning_agent(model: BaseChatModel | str = "qwen2.5:14b") -> Callable[[MCQItem], str]:
    """Return a fn that converts an MCQItem into a focused retrieval query."""
    agent = Agent("reasoning", REASONING_SYS, model=model)

    def to_query(item: MCQItem) -> str:
        query = agent.say(f"Question:\n{item.question}\n\nSearch query:").strip()
        return query or item.question  # fall back to raw question if empty

    return to_query
