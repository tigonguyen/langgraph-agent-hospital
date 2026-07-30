"""Run configuration — the single flexibility surface for a variant.

`RunConfig` declares the model(s), RAG settings, and which graph nodes are wired.
`build_variant` (public) turns a preset id + kwargs into a `RunConfig` and compiles
its graph; `RunConfig` itself is internal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RagConfig:
    collection: str = "knowledge_medcpt"   # or "knowledge" (nomic-embedded)
    embedder: str = "medcpt"               # or "nomic-embed-text"
    k: int = 4
    threshold: float = 0.60                # gate for graph-invoked retrieval (V2-V4); 0.0 = keep top-k
    distill_query: bool = True
    tool: bool = False                     # bind retrieval as a tool the agent calls (agentic V1)
    iterative_max: int = 0                 # i-MedRAG: >0 rounds of follow-up-query retrieval (prototype)
    adaptive: bool = False                 # V1b: per-question router — skip retrieval when a
                                            # question looks reasoning-heavy rather than fact-lookup


@dataclass(frozen=True)
class RunConfig:
    model: Any = "qwen2.5:14b"             # BaseChatModel instance or model-name str
    role_models: dict[str, Any] = field(default_factory=dict)
    answer_role: str = "baseline"          # role for the single-answer node (V0-V2)
    rag: RagConfig | None = None           # None = no retrieval (V0)
    clinical_reason: bool = False          # dedicated clinical-reasoning stage before the decider (V2-V4)
    verify: bool = False                   # verifier node, reads the clinical report (V2/V3, off for V4)
    memory: bool = False                   # short-term memory: scribe condenses case understanding +
                                            # evidence + clinical report into working notes for the
                                            # verifier only; a no-op without verify=True too (V3)
    verify_rag: RagConfig | None = None    # verifier's OWN textbook-search tool (independent of `rag`)

    def model_for(self, role: str) -> Any:
        return self.role_models.get(role, self.model)
