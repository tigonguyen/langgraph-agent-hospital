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
    threshold: float = 0.60                # MedCPT (asymmetric) tops out ~0.69, so 0.6 is strict
    distill_query: bool = True


@dataclass(frozen=True)
class RunConfig:
    model: Any = "qwen2.5:14b"             # BaseChatModel instance or model-name str
    role_models: dict[str, Any] = field(default_factory=dict)
    answer_role: str = "baseline"          # role for the single-answer node (V0-V2)
    rag: RagConfig | None = None           # None = no retrieval (V0)
    clinical_reason: bool = False          # clinical reasoning stage before answering (V2)
    panel_size: int = 1                    # >1 or aggregate → panel + attending (V3/V4)
    aggregate: bool = False
    verify: bool = False                   # verifier node (V3 on, V4 off)

    def model_for(self, role: str) -> Any:
        return self.role_models.get(role, self.model)
