"""RAG-only answerer (variant V1): single LLM + retrieval, no agents, no memory.

By default the retrieval query comes from the **reasoning agent** (`qa/reasoning.py`),
which converts the verbose vignette into a focused query — retrieving on the whole
vignette pulls topical-but-non-discriminating context (measured: RAG −8 pts). Pass
`distill=False` to retrieve on the raw question, or `query_fn=` to inject another
query builder.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.knowledge import format_evidence, open_store, retrieve
from agent_hospital.qa.baseline import format_mcq, parse_choice
from agent_hospital.qa.reasoning import build_reasoning_agent

RAG_SYS = (
    "You are an expert physician answering a USMLE multiple-choice question. "
    "Use the provided textbook evidence when it is relevant; otherwise rely on your "
    "own knowledge. Choose the single best answer."
)

# Backward-compatible alias: the reasoning agent is the query-conversion step.
build_query_distiller = build_reasoning_agent


def build_rag_answerer(
    model: BaseChatModel | str = "qwen2.5:14b",
    *,
    distill: bool = True,
    query_fn: Callable[[MCQItem], str] | None = None,
    k: int = 4,
    threshold: float = 0.5,
    store: Any | None = None,
) -> Callable[[MCQItem], int | None]:
    """V1: build a query (reasoning agent by default), retrieve gated evidence, answer."""
    agent = Agent("rag", RAG_SYS, model=model)
    store = store or open_store()
    if query_fn is None and distill:
        query_fn = build_reasoning_agent(model)

    def answer(item: MCQItem) -> int | None:
        query = query_fn(item) if query_fn else item.question
        hits = retrieve(query, k=k, threshold=threshold, store=store)
        evidence = format_evidence(hits)
        prompt = f"{evidence}\n\n{format_mcq(item)}" if evidence else format_mcq(item)
        return parse_choice(agent.say(prompt))

    return answer
