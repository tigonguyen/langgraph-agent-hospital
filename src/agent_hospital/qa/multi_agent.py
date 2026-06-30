"""V2 — Multi-agent without memory (two basic agents).

A minimal collaboration: the **Reasoner** converts the question into a focused
retrieval query (the reasoning agent), and the **Specialist** reasons over the
retrieved evidence and commits an answer. No memory/experience yet — that arrives
in V3. RAG sits between the two agents.

    Reasoner ──query──▶ retrieve (gated) ──evidence──▶ Specialist ──▶ option

This keeps the uniform `answer(item) -> int | None` interface, so V2 plugs into
`build_variant` and the metrics harness like V0/V1.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.knowledge import format_evidence, open_store, retrieve
from agent_hospital.qa.baseline import format_mcq, parse_choice
from agent_hospital.qa.reasoning import build_reasoning_agent

SPECIALIST_SYS = (
    "You are an expert physician on a case panel answering a USMLE multiple-choice question. "
    "Use the textbook evidence when relevant; otherwise rely on your own knowledge. "
    "Reason briefly about the key findings and the options, then on the LAST line write "
    "'Answer: X' where X is A, B, C, or D."
)


def build_multiagent_answerer(
    model: BaseChatModel | str = "qwen2.5:14b",
    *,
    query_fn: Callable[[MCQItem], str] | None = None,
    k: int = 4,
    threshold: float = 0.5,
    store: Any | None = None,
) -> Callable[[MCQItem], int | None]:
    """V2: Reasoner (query) → gated retrieval → Specialist (reason → answer)."""
    reasoner = query_fn or build_reasoning_agent(model)        # agent 1
    specialist = Agent("specialist", SPECIALIST_SYS, model=model)  # agent 2
    store = store or open_store()

    def answer(item: MCQItem) -> int | None:
        query = reasoner(item)
        hits = retrieve(query, k=k, threshold=threshold, store=store)
        evidence = format_evidence(hits)
        prompt = f"{evidence}\n\n{format_mcq(item)}" if evidence else format_mcq(item)
        return parse_choice(specialist.say(prompt))

    return answer
