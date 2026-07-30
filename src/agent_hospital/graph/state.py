"""Shared state threaded through every variant graph (plain overwrite fields)."""

from __future__ import annotations

from typing import TypedDict

from agent_hospital.diseases.medqa_usmle import MCQItem


class QAState(TypedDict, total=False):
    item: MCQItem
    case_understanding: str # Node 1's shared case summary + search-query rationale (V2-V4) —
                            # read by Node 2 (search), Node 3 (reasoning), and the verifier (V3)
    query: str
    evidence: str          # formatted evidence block ("" = none)
    rationale: str         # most recent explanation text; overwritten by whichever node answers last
    clinical_report: str   # clinical reasoner's untouched report (V2-V4) — read directly by the
                            # decider and the verifier; unlike rationale, never overwritten downstream
    answer: int | None     # final chosen option index
