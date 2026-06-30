"""System variants for ablation — the single switch point.

Pick a variant by id and get back a uniform answer function
`MCQItem -> int | None`, so every variant is evaluated the same way:

    from agent_hospital.qa import build_variant
    answer = build_variant("V1", model="qwen2.5:14b")
    idx = answer(item)

V0/V1 are built; V2-V4 are registered placeholders that raise until implemented.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel

from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.qa.baseline import build_baseline_answerer
from agent_hospital.qa.multi_agent import build_multiagent_answerer
from agent_hospital.qa.rag_answer import build_rag_answerer

DEFAULT_MODEL = "qwen2.5:14b"

# variant id -> human description
VARIANTS: dict[str, str] = {
    "V0": "Direct LLM",
    "V1": "RAG-only",
    "V2": "Multi-agent without memory",
    "V3": "Full system",
    "V4": "Full system without verifier",
}

AnswerFn = Callable[[MCQItem], "int | None"]


def build_variant(
    variant: str,
    model: BaseChatModel | str = DEFAULT_MODEL,
    **kwargs,
) -> AnswerFn:
    """Return the answer function for a variant id ('V0'..'V4')."""
    v = variant.upper()
    if v == "V0":
        return build_baseline_answerer(model=model)
    if v == "V1":
        return build_rag_answerer(model=model, **kwargs)
    if v == "V2":
        return build_multiagent_answerer(model=model, **kwargs)
    if v in ("V3", "V4"):
        raise NotImplementedError(f"{v} ({VARIANTS[v]}) is not built yet")
    raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
