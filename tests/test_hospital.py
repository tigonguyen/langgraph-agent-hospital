"""Phase-1 tests: scoring, nodes (with stubs), graph wiring, and a live run_case."""

import os

import pytest
import requests

from agent_hospital.agents.doctor import DiagnosisResult
from agent_hospital.diseases import MedQACase, load_medqa
from agent_hospital.nodes.consultation import make_consultation_node
from agent_hospital.nodes.diagnosis import make_diagnosis_node
from agent_hospital.nodes.graph import build_hospital_graph
from agent_hospital.nodes.scoring import score_diagnosis
from agent_hospital.simulation import run_case

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEST_MODEL = os.environ.get("AGENT_HOSPITAL_TEST_MODEL", "qwen2.5:7b")


def _available_models() -> list[str] | None:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return [m["name"] for m in resp.json().get("models", [])]


# --- offline: scoring ------------------------------------------------------

def test_score_diagnosis_match():
    assert score_diagnosis("I think it's Myasthenia gravis", "Myasthenia gravis") == 1.0
    assert score_diagnosis("myasthenia gravis", "Myasthenia Gravis") == 1.0


def test_score_diagnosis_mismatch():
    assert score_diagnosis("Migraine", "Myasthenia gravis") == 0.0
    assert score_diagnosis("", "Myasthenia gravis") == 0.0


# --- offline: nodes with fakes ---------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeDoctor:
    def __init__(self):
        self.calls = 0

    def act(self, messages):
        self.calls += 1
        return {"messages": [_Msg(f"Question {self.calls}?")]}


class _FakePatient:
    def say(self, text):
        return f"I noticed double vision (re: {text[:12]})"


def test_consultation_node_builds_transcript():
    node = make_consultation_node(_FakeDoctor(), _FakePatient(), max_turns=3)
    out = node({})
    assert out["turn"] == 3
    assert len(out["transcript"]) == 6  # 3 doctor + 3 patient turns
    assert out["transcript"][0] == {"role": "doctor", "text": "Question 1?"}
    assert out["transcript"][1]["role"] == "patient"


def test_diagnosis_node_uses_diagnoser():
    node = make_diagnosis_node(lambda text: DiagnosisResult(diagnosis="X", reasoning="y"))
    out = node({"transcript": [{"role": "doctor", "text": "q"}]})
    assert out == {"diagnosis": "X", "reasoning": "y"}


# --- offline: full graph wiring with stub nodes ----------------------------

def test_graph_wiring_with_stubs():
    case = MedQACase(
        case_id="t", demographics="35F", history="", primary_symptom="double vision",
        correct_diagnosis="Myasthenia gravis",
    )

    def consult_stub(state):
        return {"transcript": [{"role": "patient", "text": "double vision"}], "turn": 1}

    def diagnose_stub(state):
        return {"diagnosis": "myasthenia gravis", "reasoning": "fatigable weakness"}

    graph = build_hospital_graph(consult_stub, diagnose_stub)
    final = graph.invoke({"case": case, "transcript": [], "turn": 0})
    assert final["diagnosis"] == "myasthenia gravis"
    assert final["score"] == 1.0  # matches the gold diagnosis


# --- live: end-to-end ------------------------------------------------------

def test_run_case_live():
    models = _available_models()
    if models is None:
        pytest.skip("Ollama is not reachable")
    if TEST_MODEL not in models:
        pytest.skip(f"model {TEST_MODEL!r} not pulled (have: {models})")

    case = load_medqa()[0]  # Myasthenia gravis case
    final = run_case(case, model=TEST_MODEL, max_turns=3)

    assert isinstance(final["diagnosis"], str) and final["diagnosis"].strip()
    assert final["score"] in (0.0, 1.0)
    # The patient must never have leaked the hidden diagnosis during consultation.
    patient_text = " ".join(
        t["text"].lower() for t in final["transcript"] if t["role"] == "patient"
    )
    assert "myasthenia" not in patient_text

    print(f"\n[diagnosis] {final['diagnosis']!r}  score={final['score']}")
    for turn in final["transcript"]:
        print(f"  {turn['role']}: {turn['text'][:90]}")
