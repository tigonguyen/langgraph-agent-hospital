"""Diagnosis node: turn the consultation transcript into a committed diagnosis."""

from __future__ import annotations

from typing import Any, Callable

from agent_hospital.agents.doctor import DiagnosisResult
from agent_hospital.nodes.consultation import format_transcript


def make_diagnosis_node(
    diagnose: Callable[[str], DiagnosisResult],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a graph node that produces a diagnosis from the transcript.

    `diagnose` maps transcript text -> DiagnosisResult (real one wraps the doctor's
    structured LLM call; tests can inject a stub).
    """

    def diagnosis_node(state: dict[str, Any]) -> dict[str, Any]:
        result = diagnose(format_transcript(state["transcript"]))
        return {"diagnosis": result.diagnosis, "reasoning": result.reasoning}

    return diagnosis_node
