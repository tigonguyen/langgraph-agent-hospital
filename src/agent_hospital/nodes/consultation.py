"""Consultation node: a bounded doctor<->patient dialogue.

The doctor is given the running conversation each turn (so it can ask
follow-ups); the patient answers each question from its persona. The result is a
transcript the diagnosis node will read.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_hospital.agents.base import Agent

_OPENING = "Begin the consultation with the patient."


def make_consultation_node(
    doctor: Agent,
    patient: Agent,
    max_turns: int = 4,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a graph node that runs up to `max_turns` of doctor<->patient Q&A."""

    def consultation_node(state: dict[str, Any]) -> dict[str, Any]:
        conversation: list[dict[str, str]] = []  # doctor's view: patient=user, doctor=assistant
        transcript: list[dict[str, str]] = []

        for _ in range(max_turns):
            prompt = conversation or [{"role": "user", "content": _OPENING}]
            question = doctor.act(prompt)["messages"][-1].content
            conversation.append({"role": "assistant", "content": question})
            transcript.append({"role": "doctor", "text": question})

            answer = patient.say(question)
            conversation.append({"role": "user", "content": answer})
            transcript.append({"role": "patient", "text": answer})

        return {"transcript": transcript, "turn": max_turns}

    return consultation_node


def format_transcript(transcript: list[dict[str, str]]) -> str:
    """Render a transcript as text for the diagnoser."""
    return "\n".join(f"{turn['role'].capitalize()}: {turn['text']}" for turn in transcript)
