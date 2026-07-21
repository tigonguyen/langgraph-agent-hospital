"""System variants — the single switch point, built on LangGraph.

    from agent_hospital.qa import build_variant
    answer = build_variant("V2", model="qwen2.5:14b")   # -> answer(item) -> int | None

Each variant is a preset `RunConfig` compiled into a `StateGraph` by `build_graph`.
`**overrides` tweak the config for experiments, e.g.
    build_variant("V1", rag=RagConfig(collection="knowledge_medcpt", embedder="medcpt"))
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from agent_hospital.config import RagConfig, RunConfig
from agent_hospital.diseases.medqa_usmle import MCQItem

DEFAULT_MODEL = "qwen2.5:14b"

VARIANTS: dict[str, str] = {
    "V0": "Direct LLM",
    "V1": "RAG-only",
    "V2": "Multi-agent (reasoner + specialist)",
    "V3": "Full system (panel + attending + verifier)",
    "V4": "Full system without verifier",
}

AnswerFn = Callable[[MCQItem], "int | None"]

_PRESETS: dict[str, RunConfig] = {
    "V0": RunConfig(answer_role="baseline", rag=None),
    # V1 retrieves solved exam questions (MedMCQA); V2-V4 retrieve textbook prose.
    # 0.65 matches the textbook gate's selectivity (~65% of items get evidence), so the
    # corpus A/B isn't confounded by one gate firing more often than the other.
    "V1": RunConfig(answer_role="rag-answerer",
                    rag=RagConfig(collection="knowledge_medmcqa", threshold=0.65)),
    "V2": RunConfig(answer_role="specialist", rag=RagConfig()),
    "V3": RunConfig(rag=RagConfig(), panel_size=2, aggregate=True, verify=True),
    "V4": RunConfig(rag=RagConfig(), panel_size=2, aggregate=True, verify=False),
}


def build_variant(variant: str, model=DEFAULT_MODEL, *, temperature: float = 0.0, **overrides) -> AnswerFn:
    """Return the answer function for a variant id ('V0'..'V4').

    A string model becomes a deterministic `ChatOllama` (temperature=0 by default) so
    evaluation is reproducible. `overrides` set any `RunConfig` field (rag, panel_size, …).
    """
    from agent_hospital.graph import build_graph  # lazy: graph.nodes imports back into qa

    v = variant.upper()
    if v not in _PRESETS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    if isinstance(model, str):
        from agent_hospital.models import resolve_model

        model = resolve_model(model, temperature=temperature)
    cfg = replace(_PRESETS[v], model=model, **overrides)
    graph = build_graph(cfg)

    def answer(item: MCQItem) -> int | None:
        return graph.invoke({"item": item}).get("answer")

    return answer
