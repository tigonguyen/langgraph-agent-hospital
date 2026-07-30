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


# V1b — adaptive RAG-tier routing (RAGCare-QA / Self-RAG style): a cheap per-question
# decision on whether a textbook lookup is likely to help (a fact-lookup question) versus
# hurt or do nothing (a reasoning-heavy question — where plain V0 already wins per this
# project's own measurements), rather than fixing retrieval on or off for every item.
ROUTE_SYS = (
    "You are deciding whether looking up a medical textbook would help answer this USMLE "
    "question, or whether it is a pure clinical-reasoning/ethics/management-sequencing question "
    "that a knowledgeable physician can answer directly, without a lookup. Answer with exactly "
    "one word: LOOKUP if a specific fact (a named pathogen, drug, lab value, guideline, mechanism) "
    "would resolve it; DIRECT if it mainly requires reasoning over the vignette. Output ONLY that "
    "one word."
)


def build_router(model: BaseChatModel | str = "qwen2.5:14b") -> Callable[[MCQItem], bool]:
    """Return a fn that decides whether retrieval is worth attempting for this item."""
    agent = Agent("router", ROUTE_SYS, model=model)

    def should_retrieve(item: MCQItem) -> bool:
        reply = agent.say(f"Question:\n{item.question}\n\nDecision:").strip().upper()
        return reply.startswith("LOOKUP")

    return should_retrieve
