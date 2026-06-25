"""Build a doctor agent from a MedQA case.

The doctor runs the clinical core: it consults the patient (free-text Q&A) and
later commits to a structured diagnosis. It receives the case's
`objective_for_doctor` but NEVER the hidden `correct_diagnosis` / `test_results`
/ `physical_exam` — those are the answer key used only for scoring.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_hospital.agents.base import Agent
from agent_hospital.diseases.medqa import MedQACase


class DiagnosisResult(BaseModel):
    """The doctor's committed diagnosis."""

    diagnosis: str = Field(description="The single most likely diagnosis, named concisely.")
    reasoning: str = Field(description="One or two sentences justifying the diagnosis.")


_CONSULT_RULES = """\
You are a physician conducting a consultation with a patient. Stay in role.

Rules:
- Ask focused questions, ONE at a time, like a real clinician taking a history.
- You do NOT yet know the diagnosis. Do not state or guess a final diagnosis
  during the consultation — gather the history first.
- Be concise. No preamble, no restating what the patient said at length.
- Build on what the patient has already told you; don't repeat questions.
"""


def doctor_consult_prompt(case: MedQACase) -> str:
    """System prompt for the consulting doctor (objective only, no answer key)."""
    objective = case.objective_for_doctor or "Assess the patient and reach a diagnosis."
    return f"{_CONSULT_RULES}\nYour clinical objective: {objective}"


def build_doctor_agent(
    case: MedQACase,
    model: BaseChatModel | str = "qwen2.5:7b",
    *,
    tools: Sequence[Callable[..., Any]] = (),
    middleware: Sequence[Any] = (),
) -> Agent:
    """Construct the consulting doctor `Agent` for one case."""
    return Agent(
        name=f"doctor:{case.case_id}",
        system_prompt=doctor_consult_prompt(case),
        model=model,
        tools=tools,
        middleware=middleware,
    )


_DIAGNOSE_PROMPT = """\
You are a physician. Read the consultation transcript and give your single most
likely diagnosis with brief reasoning. Base it only on the transcript.
"""


def build_diagnoser(
    model: BaseChatModel | str = "qwen2.5:7b",
) -> Callable[[str], DiagnosisResult]:
    """Return a function that turns a consultation transcript into a DiagnosisResult.

    Uses structured output when the model supports it, and falls back to using the
    raw reply text as the diagnosis otherwise.
    """
    agent = Agent(
        name="doctor-dx",
        system_prompt=_DIAGNOSE_PROMPT,
        model=model,
        response_format=DiagnosisResult,
    )

    def diagnose(transcript_text: str) -> DiagnosisResult:
        result = agent.act([{"role": "user", "content": transcript_text}])
        structured = result.get("structured_response")
        if isinstance(structured, DiagnosisResult):
            return structured
        # Fallback: model didn't return structured output — use the last reply.
        text = result["messages"][-1].content
        return DiagnosisResult(diagnosis=text.strip(), reasoning="(unstructured reply)")

    return diagnose
