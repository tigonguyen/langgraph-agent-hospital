"""RAG-augmented single-reasoner answerer (ablation step: baseline + RAG).

Retrieves gated textbook evidence for the question and prepends it before the LLM
chooses an option. Same parsing/scoring as the baseline, so accuracy is directly
comparable (the lift = RAG's contribution).
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


def build_rag_answerer(
    model: BaseChatModel | str = "qwen2.5:7b",
    *,
    k: int = 4,
    threshold: float = 0.5,
    store: Any | None = None,
) -> Callable[[MCQItem], int | None]:
    """Return an answer fn that retrieves evidence then chooses an option."""
    agent = Agent("rag", RAG_SYS, model=model)
    store = store or open_store()

    def answer(item: MCQItem) -> int | None:
        hits = retrieve(item.question, k=k, threshold=threshold, store=store)
        evidence = format_evidence(hits)
        prompt = f"{evidence}\n\n{format_mcq(item)}" if evidence else format_mcq(item)
        return parse_choice(agent.say(prompt))

    return answer
