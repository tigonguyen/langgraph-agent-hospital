"""Assemble the Phase-1 care loop: consultation -> diagnosis -> scoring."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent_hospital.agents.doctor import build_diagnoser, build_doctor_agent
from agent_hospital.agents.patient import build_patient_agent
from agent_hospital.diseases.medqa import MedQACase
from agent_hospital.nodes.consultation import make_consultation_node
from agent_hospital.nodes.diagnosis import make_diagnosis_node
from agent_hospital.nodes.scoring import score_node


class HospitalState(TypedDict, total=False):
    case: MedQACase
    transcript: list[dict[str, str]]
    turn: int
    diagnosis: str
    reasoning: str
    score: float


def build_hospital_graph(
    consultation_node: Callable[[dict[str, Any]], dict[str, Any]],
    diagnosis_node: Callable[[dict[str, Any]], dict[str, Any]],
):
    """Wire the three nodes into a compiled StateGraph (nodes injected for testability)."""
    graph = StateGraph(HospitalState)
    graph.add_node("consultation", consultation_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("scoring", score_node)
    graph.add_edge(START, "consultation")
    graph.add_edge("consultation", "diagnosis")
    graph.add_edge("diagnosis", "scoring")
    graph.add_edge("scoring", END)
    return graph.compile()


def build_case_graph(
    case: MedQACase,
    model: BaseChatModel | str = "qwen2.5:7b",
    *,
    max_turns: int = 4,
):
    """Convenience: build patient + doctor + diagnoser and compile the care loop."""
    doctor = build_doctor_agent(case, model)
    patient = build_patient_agent(case, model)
    diagnose = build_diagnoser(model)
    consultation_node = make_consultation_node(doctor, patient, max_turns)
    diagnosis_node = make_diagnosis_node(diagnose)
    return build_hospital_graph(consultation_node, diagnosis_node)
