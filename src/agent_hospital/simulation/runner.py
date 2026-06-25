"""Run a single patient case through the care loop."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from agent_hospital.diseases.medqa import MedQACase
from agent_hospital.nodes.graph import build_case_graph


def run_case(
    case: MedQACase,
    model: BaseChatModel | str = "qwen2.5:7b",
    *,
    max_turns: int = 4,
) -> dict[str, Any]:
    """Build the case graph, run it, and return the final HospitalState."""
    graph = build_case_graph(case, model, max_turns=max_turns)
    return graph.invoke({"case": case, "transcript": [], "turn": 0})
