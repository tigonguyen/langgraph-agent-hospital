"""System variants — the single switch point, built on LangGraph.

    from agent_hospital.qa import build_variant
    answer = build_variant("V2", model="qwen2.5:14b")   # -> answer(item) -> int | None

Each variant is a preset `RunConfig` compiled into a `StateGraph` by `build_graph`.
`**overrides` tweak the config for experiments, e.g.
    build_variant("V1", rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b"))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

from agent_hospital.config import RagConfig, RunConfig
from agent_hospital.diseases.medqa_usmle import MCQItem

DEFAULT_MODEL = "qwen2.5:14b"

VARIANTS: dict[str, str] = {
    "V0": "Direct LLM",
    "V1": "RAG-only (MedMCQA)",
    "V1T": "RAG-only (textbooks, gated)",
    "V2": "Multi-agent (case reasoner + decider)",
    "V3": "V2 without verifier, decider has long-term memory",
    "V4": "V3 + verifier grounded in live Wikipedia",
    "V5": "V3 + verifier that recalls an evolutionary mistake bank",
}


@dataclass(frozen=True)
class NodeTrace:
    """One completed graph node: the state delta it produced, and how long its step took.

    `elapsed_s` is measured from the end of the previous chunk, so nodes sharing a
    superstep (search ‖ reasoning) can't be timed apart — LangGraph emits both updates
    only once BOTH have finished, so the first reports the whole superstep and the second
    ~0. Read it as "superstep elapsed", not per-node cost.
    """

    node: str
    elapsed_s: float
    update: dict                # the node's state delta, JSON-safe (`item` is never in one)


@dataclass(frozen=True)
class AnswerResult:
    """What a variant returns: the chosen option, and why (spec §3)."""

    answer: int | None          # option index 0-3, or None = unparseable
    rationale: str = ""         # short explanation; "" if the variant produced none
    tokens_in: int = 0          # summed prompt tokens across every LLM call this episode made
    tokens_out: int = 0         # summed completion tokens — the cost/latency proxy (spec §7)
    trace: tuple[NodeTrace, ...] = ()   # per-node deltas in completion order (web/ inspection)


AnswerFn = Callable[[MCQItem], AnswerResult]


def _jsonable(update: dict) -> dict:
    """Node deltas only hold str/int/None today, but coerce anything else to `str` so a
    trace can always be serialized to the web client."""
    return {k: v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v)
            for k, v in update.items()}


_PRESETS: dict[str, RunConfig] = {
    "V0": RunConfig(answer_role="baseline", rag=None),
    # V1: single agentic tool-calling agent over search_medmcqa. Asymmetric-trust rule:
    # HIGH-confidence evidence wins by default unless the model names a specific missed
    # finding; MEDIUM/LOW evidence is ignored and it decides like V0.
    "V1": RunConfig(answer_role="rag-agent",
                    rag=RagConfig(collection="knowledge_medmcqa_qwen3", embedder="qwen3-embedding:4b",
                                  k=5, threshold=0.0, tool=True)),
    # V1T: V1's agent over the USMLE textbook corpus instead of MedMCQA, with a relevance
    # gate. 0.75 drops the bottom quartile of nomic scores on the test questions (top-1
    # ranged 0.69-0.84); MedMCQA neighbours mostly never named an option, so this swaps
    # lookalike questions for reference facts.
    "V1T": RunConfig(answer_role="textbook-agent",
                     rag=RagConfig(collection="knowledge", embedder="nomic-embed-text",
                                   k=5, threshold=0.75, tool=True)),
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

    def answer(item: MCQItem, on_node: Callable[[NodeTrace], None] | None = None) -> AnswerResult:
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        # Sums AIMessage.usage_metadata across every LLM call in the run, provider-agnostic.
        usage = UsageMetadataCallbackHandler()
        # thread_id = item.id: one checkpoint thread per question, never shared across items.
        config = {"callbacks": [usage], "configurable": {"thread_id": item.id}}
        # stream(updates) rather than invoke() so callers can watch nodes finish (web/ UI).
        # Behaviour-identical: the left-merge of every delta IS what invoke returns, since
        # QAState is all plain-overwrite fields — no reducers to honour. Parallel nodes
        # (search ‖ reasoning) arrive as separate chunks within the one superstep.
        state: dict = {}
        trace: list[NodeTrace] = []
        t = time.time()
        for chunk in graph.stream({"item": item}, config=config, stream_mode="updates"):
            now = time.time()
            for node, update in chunk.items():
                state.update(update or {})      # `update or {}`: distill_mistake returns {}
                step = NodeTrace(node, round(now - t, 3), _jsonable(update or {}))
                trace.append(step)
                if on_node:
                    on_node(step)
            t = now
        tokens_in = sum(u.get("input_tokens", 0) or 0 for u in usage.usage_metadata.values())
        tokens_out = sum(u.get("output_tokens", 0) or 0 for u in usage.usage_metadata.values())
        return AnswerResult(answer=state.get("answer"), rationale=state.get("rationale", ""),
                            tokens_in=tokens_in, tokens_out=tokens_out, trace=tuple(trace))

    return answer
