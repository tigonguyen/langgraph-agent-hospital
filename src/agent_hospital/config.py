"""Central configuration for a hospital run.

`HospitalConfig` holds the reusable knobs (model, per-process turn budgets,
optional per-role model overrides) — separate from the per-run `case` data, so
one config can drive a whole batch of cases. Agent builders stay config-agnostic
(they take a resolved model); only the orchestration layer (graph/runner) reads
the config.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class HospitalConfig:
    model: BaseChatModel | str = "qwen2.5:7b"

    # Per-process turn budgets (more added as the pipeline grows: exam_turns, ...).
    consult_turns: int = 4

    # Optional per-role overrides; fall back to `model` when None.
    doctor_model: BaseChatModel | str | None = None
    patient_model: BaseChatModel | str | None = None
    diagnoser_model: BaseChatModel | str | None = None

    def role_model(self, role: str) -> BaseChatModel | str:
        """Resolve the model for a role ('doctor'|'patient'|'diagnoser'), with fallback."""
        override = getattr(self, f"{role}_model", None)
        return override if override is not None else self.model
