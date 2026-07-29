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
    "V1A": "RAG-only (MedRAG Textbooks)",
    "V2": "Multi-agent (clinical reasoner + decider + verifier)",
    "V3": "Full system (V2 + short-term memory)",
    "V4": "Full system without verifier",
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
    # V1: a SINGLE agent that calls a `search_medmcqa` tool over the MedMCQA database of
    # solved board questions (agentic RAG — the model drives retrieval). nomic embeddings
    # (8192-token ctx) so a full vignette isn't truncated the way MedCPT's 64-token query
    # encoder would; threshold=0.0 returns the top-k, no gate. V2-V4 retrieve the same corpus,
    # but concurrently (reason->retrieve as its own branch) rather than as a bound tool.
    "V1": RunConfig(answer_role="rag-agent",
                    rag=RagConfig(collection="knowledge_medmcqa_nomic", embedder="nomic-embed-text",
                                  k=5, threshold=0.0, tool=True)),
    # V1a: same one-agent architecture as V1, but searches MedRAG Textbooks (search_textbooks,
    # MedCPT embeddings) instead of MedMCQA — the ladder's other RAG corpus, same design.
    # threshold=0.0 (top-k, no gate), matching V1's agentic philosophy of trusting the agent
    # to weigh results itself rather than gating them before it even sees them.
    "V1A": RunConfig(answer_role="rag-agent-textbook",
                     rag=RagConfig(collection="knowledge_medcpt", embedder="medcpt",
                                   k=4, threshold=0.0, tool=True)),
    # V2: a dedicated clinical-reasoning agent (spec §4.2) — findings, what's asked, reasoning,
    # option-by-option, summary, never naming an option — hands its report to a separate decider
    # who commits to a letter, then a lightweight verifier audits that choice against the
    # report's own verdicts (and, if needed, a targeted textbook check) before confirming or
    # revising it. 3 agents. `tool=False`: retrieval runs as its own reason->retrieve branch
    # CONCURRENTLY with the reasoner (build_graph wires both straight off START) instead of
    # being bound to the reasoner as a tool — the reasoner digests the case from its own
    # knowledge while retrieval happens at the same time, and `answer` is where both join
    # (`_prompt` auto-prepends `state["evidence"]`). Confirmed empirically: with Ollama's
    # `-np` (parallel slots) > 1, concurrent local calls measured ~1.8x faster than serial.
    # Same RAG corpus as V1, so V2 - V1 isolates exactly the effect of splitting one agent
    # into reasoner + decider (+ verifier), not the RAG design.
    "V2": RunConfig(clinical_reason=True, answer_role="decider", verify=True,
                    rag=RagConfig(collection="knowledge_medmcqa_nomic", embedder="nomic-embed-text",
                                  k=5, threshold=0.0, tool=False),
                    verify_rag=RagConfig(threshold=0.0)),
    # V3 = V2 + short-term memory: a scribe condenses the clinical reasoner's report into
    # shared working notes (state["working_memory"]) that the decider and verifier read
    # instead of the raw report. V3 - V2 isolates exactly the effect of that memory stage.
    "V3": RunConfig(clinical_reason=True, answer_role="decider", verify=True, memory=True,
                    rag=RagConfig(collection="knowledge_medmcqa_nomic", embedder="nomic-embed-text",
                                  k=5, threshold=0.0, tool=False),
                    verify_rag=RagConfig(threshold=0.0)),
    # V4 = V3 without the verifier (verify=False, everything else identical) — isolates
    # exactly the verifier's marginal contribution (V3 - V4).
    "V4": RunConfig(clinical_reason=True, answer_role="decider", memory=True,
                    rag=RagConfig(collection="knowledge_medmcqa_nomic", embedder="nomic-embed-text",
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

        # Provider-agnostic token accounting: this callback sums AIMessage.usage_metadata
        # across every LLM call made during the invoke — no matter how many agents/nodes
        # are involved or whether they run concurrently (thread-safe). Requires no changes
        # to any node/Agent code; LangGraph forwards `config` into every nested invocation.
        usage = UsageMetadataCallbackHandler()
        out = graph.invoke({"item": item}, config={"callbacks": [usage]})
        tokens_in = sum(u.get("input_tokens", 0) or 0 for u in usage.usage_metadata.values())
        tokens_out = sum(u.get("output_tokens", 0) or 0 for u in usage.usage_metadata.values())
        return AnswerResult(answer=out.get("answer"), rationale=out.get("rationale", ""),
                            tokens_in=tokens_in, tokens_out=tokens_out)

    return answer
