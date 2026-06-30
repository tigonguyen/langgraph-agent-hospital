"""RAG-only answerer (variant V1): single LLM + retrieval, no agents, no memory.

By default it uses **query distillation** — an LLM step that turns the verbose
vignette into a focused search query — because retrieving on the whole vignette
pulls topical-but-non-discriminating context. Set `distill=False` to retrieve on
the raw question (the weaker baseline we measured).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.knowledge import format_evidence, open_store, retrieve
from agent_hospital.qa.baseline import format_mcq, parse_choice

RAG_SYS = (
    "You are an expert physician answering a USMLE multiple-choice question. "
    "Use the provided textbook evidence when it is relevant; otherwise rely on your "
    "own knowledge. Choose the single best answer."
)

_DISTILL_SYS = (
    "You turn a clinical exam question into a concise search query for a medical "
    "textbook index. Output ONLY the query — the salient findings and what is being "
    "asked, no preamble, max ~20 words."
)


def build_query_distiller(model: BaseChatModel | str) -> Callable[[MCQItem], str]:
    """Return a fn that distils an MCQItem into a focused retrieval query."""
    agent = Agent("query-distiller", _DISTILL_SYS, model=model)

    def distill(item: MCQItem) -> str:
        query = agent.say(f"Clinical question:\n{item.question}\n\nSearch query:").strip()
        return query or item.question  # fall back to raw question if empty

    return distill


def build_rag_answerer(
    model: BaseChatModel | str = "qwen2.5:14b",
    *,
    distill: bool = True,
    k: int = 4,
    threshold: float = 0.5,
    store: Any | None = None,
) -> Callable[[MCQItem], int | None]:
    """V1: retrieve gated evidence (distilled query by default) then choose an option."""
    agent = Agent("rag", RAG_SYS, model=model)
    store = store or open_store()
    distiller = build_query_distiller(model) if distill else None

    def answer(item: MCQItem) -> int | None:
        query = distiller(item) if distiller else item.question
        hits = retrieve(query, k=k, threshold=threshold, store=store)
        evidence = format_evidence(hits)
        prompt = f"{evidence}\n\n{format_mcq(item)}" if evidence else format_mcq(item)
        return parse_choice(agent.say(prompt))

    return answer
