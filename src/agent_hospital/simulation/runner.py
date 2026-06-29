"""Run a single patient case through the care loop."""

from __future__ import annotations

from typing import Any

from agent_hospital.config import HospitalConfig
from agent_hospital.diseases.medqa import MedQACase
from agent_hospital.nodes.graph import build_case_graph


def run_case(case: MedQACase, config: HospitalConfig | None = None) -> dict[str, Any]:
    """Build the case graph, run it, and return the final HospitalState."""
    graph = build_case_graph(case, config)
    return graph.invoke({"case": case, "transcript": [], "turn": 0})
