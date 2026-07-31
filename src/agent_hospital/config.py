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
    collection: str = "knowledge_medcpt"
    embedder: str = "medcpt"
    k: int = 4
    threshold: float = 0.60                # gate for graph-invoked retrieval; 0.0 = keep top-k
    distill_query: bool = True
    tool: bool = False                     # bind retrieval as a tool the agent calls (agentic V1)
    adaptive: bool = False                 # per-question router: skip retrieval on reasoning-heavy questions


@dataclass(frozen=True)
class RunConfig:
    model: Any = "qwen2.5:14b"             # BaseChatModel instance or model-name str
    role_models: dict[str, Any] = field(default_factory=dict)
    answer_role: str = "baseline"          # role for the single-answer node (V0-V2)
    rag: RagConfig | None = None           # None = no retrieval (V0)
    clinical_reason: bool = False          # dedicated clinical-reasoning stage before the decider
    verify: bool = False                   # verifier node (V4/V5)
    memory: bool = False                   # verifier also reads shared case understanding from state
    long_term: bool = False                # decider recalls/writes cross-episode lessons (spec §4.5)
    long_term_db: str = "data/longterm/lessons.sqlite"
    long_term_split: str = "train"          # which lesson-bank namespace, independent of -s split
    long_term_read_only: bool = False       # recall AND write by default; writing while scoring
                                            # leaks between graded items. No CLI flag for this —
                                            # override programmatically for a clean scored run.
    long_term_mistakes: bool = False       # verifier recalls a SEPARATE bank of past WRONG cases
    checkpoint: bool = False               # snapshot QAState after every node; off by default (unread, ~100KB/item)
    verify_rag: RagConfig | None = None    # verifier's OWN textbook-search tool (independent of `rag`)
    verify_wikipedia: bool = False         # verifier grounds against live Wikipedia instead of the local corpus

    def model_for(self, role: str) -> Any:
        return self.role_models.get(role, self.model)
