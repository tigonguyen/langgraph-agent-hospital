"""System variants — the single switch point, built on LangGraph.

    from agent_hospital.qa import build_variant
    answer = build_variant("V2", model="qwen2.5:14b")   # -> answer(item) -> int | None

Each variant is a preset `RunConfig` compiled into a `StateGraph` by `build_graph`.
`**overrides` tweak the config for experiments, e.g.
    build_variant("V1", rag=RagConfig(collection="knowledge_medcpt", embedder="medcpt"))
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from agent_hospital.config import RagConfig, RunConfig
from agent_hospital.diseases.medqa_usmle import MCQItem

DEFAULT_MODEL = "qwen2.5:14b"

VARIANTS: dict[str, str] = {
    "V0": "Direct LLM",
    "V1": "RAG-only",
    "V2": "Multi-agent (distiller + clinical reasoner + decider)",
    "V3": "Full system (panel + attending + verifier)",
    "V4": "Full system without verifier",
}

@dataclass(frozen=True)
class AnswerResult:
    """What a variant returns: the chosen option, and why (spec §3)."""

    answer: int | None          # option index 0-3, or None = unparseable
    rationale: str = ""         # short explanation; "" if the variant produced none


AnswerFn = Callable[[MCQItem], AnswerResult]

_PRESETS: dict[str, RunConfig] = {
    "V0": RunConfig(answer_role="baseline", rag=None),
    # V1: a SINGLE agent that calls a `search_medmcqa` tool over the MedMCQA database of
    # solved board questions (agentic RAG — the model drives retrieval). nomic embeddings
    # (8192-token ctx) so a full vignette isn't truncated the way MedCPT's 64-token query
    # encoder would; threshold=0.0 returns the top-k, no gate. V2-V4 keep textbook retrieval.
    "V1": RunConfig(answer_role="rag-agent",
                    rag=RagConfig(collection="knowledge_medmcqa_nomic", embedder="nomic-embed-text",
                                  k=5, threshold=0.0, tool=True)),
    # 3 agents: query distiller -> clinical reasoner (analysis, no letter) -> decider.
    # Inherits V1's MedMCQA corpus, so V1->V2 differs only in the answering stage.
    "V2": RunConfig(answer_role="decider", clinical_reason=True,
                    rag=RagConfig(collection="knowledge_medmcqa", threshold=0.65)),
    "V3": RunConfig(rag=RagConfig(), panel_size=2, aggregate=True, verify=True),
    "V4": RunConfig(rag=RagConfig(), panel_size=2, aggregate=True, verify=False),
}


def build_variant(variant: str, model=DEFAULT_MODEL, *, temperature: float = 0.0, **overrides) -> AnswerFn:
    """Return the answer function for a variant id ('V0'..'V4') -> AnswerResult.

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

    def answer(item: MCQItem) -> AnswerResult:
        out = graph.invoke({"item": item})
        return AnswerResult(answer=out.get("answer"), rationale=out.get("rationale", ""))

    return answer
