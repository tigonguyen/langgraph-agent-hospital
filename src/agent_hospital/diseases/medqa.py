"""MedQA (OSCE-format) dataset.

`MedQACase` is the canonical patient-case schema the rest of the system consumes.
`MedQALoader` maps the AgentClinic/OSCE JSON structure into it: each record has
one `OSCE_Examination` with a `Patient_Actor` profile, the doctor's objective,
physical-exam findings, test results, and the correct diagnosis. The case splits
what the *patient* knows from the hidden ground truth (exam, tests, diagnosis).

To load a differently-shaped JSON dataset, write another `DatasetLoader` that
maps its records into `MedQACase` — see `loader.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_hospital.diseases.loader import DatasetLoader

DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data" / "medqa.jsonl"


@dataclass(frozen=True)
class MedQACase:
    case_id: str
    # --- known to the patient (role-play material) ---
    demographics: str
    history: str
    primary_symptom: str
    secondary_symptoms: list[str] = field(default_factory=list)
    past_medical_history: str = ""
    social_history: str = ""
    review_of_systems: str = ""
    # --- hidden from the patient (ground truth / for doctor & scoring) ---
    objective_for_doctor: str = ""
    physical_exam: dict = field(default_factory=dict)
    test_results: dict = field(default_factory=dict)
    correct_diagnosis: str = ""


class MedQALoader(DatasetLoader[MedQACase]):
    """Map OSCE-format records into `MedQACase`."""

    def parse_record(self, record: dict, case_id: str) -> MedQACase:
        osce = record["OSCE_Examination"]
        actor = osce.get("Patient_Actor", {})
        symptoms = actor.get("Symptoms", {})
        return MedQACase(
            case_id=case_id,
            demographics=actor.get("Demographics", ""),
            history=actor.get("History", ""),
            primary_symptom=symptoms.get("Primary_Symptom", ""),
            secondary_symptoms=list(symptoms.get("Secondary_Symptoms", [])),
            past_medical_history=actor.get("Past_Medical_History", ""),
            social_history=actor.get("Social_History", ""),
            review_of_systems=actor.get("Review_of_Systems", ""),
            objective_for_doctor=osce.get("Objective_for_Doctor", ""),
            physical_exam=osce.get("Physical_Examination_Findings", {}),
            test_results=osce.get("Test_Results", {}),
            correct_diagnosis=osce.get("Correct_Diagnosis", ""),
        )


def load_medqa(path: str | Path = DEFAULT_DATA) -> list[MedQACase]:
    """Convenience: load all MedQA cases from a jsonl file."""
    return MedQALoader().load(path)
