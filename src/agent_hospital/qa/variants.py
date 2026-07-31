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
    "V1": "RAG-only (MedMCQA)",
    "V2": "Multi-agent (case reasoner + decider)",
    "V3": "V2 without verifier, decider has long-term memory",
    "V4": "V3 + verifier grounded in live Wikipedia",
    "V5": "V3 + verifier that recalls an evolutionary mistake bank",
}

@dataclass(frozen=True)
class AnswerResult:
    """What a variant returns: the chosen option, and why (spec §3)."""

    answer: int | None          # option index 0-3, or None = unparseable
    rationale: str = ""         # short explanation; "" if the variant produced none
    tokens_in: int = 0          # summed prompt tokens across every LLM call this episode made
    tokens_out: int = 0         # summed completion tokens — the cost/latency proxy (spec §7)


AnswerFn = Callable[[MCQItem], AnswerResult]

_PRESETS: dict[str, RunConfig] = {
    "V0": RunConfig(answer_role="baseline", rag=None),
    # V1: single agentic tool-calling agent over search_medmcqa. Asymmetric-trust rule:
    # HIGH-confidence evidence wins by default unless the model names a specific missed
    # finding; MEDIUM/LOW evidence is ignored and it decides like V0.
    "V1": RunConfig(answer_role="rag-agent",
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=True)),
    # V2: case-reasoner -> (search || reasoning) -> decider, 4 agents, no verifier. Same
    # corpus/embedder as V1, so V2-V1 isolates the effect of splitting one agent into four.
    "V2": RunConfig(clinical_reason=True, answer_role="decider",
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=False)),
    # V3 = V2 without a verifier; decider gains long-term memory (recall + write, general
    # bank — graph/longterm.py). Writes by default (no CLI opt-out); writing while scoring
    # leaks between graded items, so a report number needs `long_term_read_only=True` passed
    # programmatically.
    "V3": RunConfig(clinical_reason=True, answer_role="decider", verify=False, long_term=True,
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=False)),
    # V4 = V3 + a verifier grounded in live Wikipedia rather than the local MedMCQA-derived
    # corpus, which isn't independent evidence since it's the same benchmark family.
    "V4": RunConfig(clinical_reason=True, answer_role="decider", verify=True, memory=True,
                    long_term=True, verify_wikipedia=True,
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=False)),
    # V5 = V3 + a verifier that recalls a separate mistake bank of past wrong cases; never
    # sees gold. Only `distill_mistake` (after verify) reads gold and writes to that bank.
    "V5": RunConfig(clinical_reason=True, answer_role="decider", verify=True, memory=True,
                    long_term=True, long_term_mistakes=True,
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=False)),
}


def build_variant(variant: str, model=DEFAULT_MODEL, *, temperature: float = 0.0, **overrides) -> AnswerFn:
    """Return the answer function for a variant id (list(VARIANTS)) -> AnswerResult.

    A string model becomes a deterministic `ChatOllama` (temperature=0 by default) so
    evaluation is reproducible. `overrides` set any `RunConfig` field (rag, verify, memory, …).
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
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        # Sums AIMessage.usage_metadata across every LLM call in the invoke, provider-agnostic.
        usage = UsageMetadataCallbackHandler()
        # thread_id = item.id: one checkpoint thread per question, never shared across items.
        config = {"callbacks": [usage], "configurable": {"thread_id": item.id}}
        out = graph.invoke({"item": item}, config=config)
        tokens_in = sum(u.get("input_tokens", 0) or 0 for u in usage.usage_metadata.values())
        tokens_out = sum(u.get("output_tokens", 0) or 0 for u in usage.usage_metadata.values())
        return AnswerResult(answer=out.get("answer"), rationale=out.get("rationale", ""),
                            tokens_in=tokens_in, tokens_out=tokens_out)

    return answer
