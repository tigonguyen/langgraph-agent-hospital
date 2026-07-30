"""Shared state threaded through every variant graph (plain overwrite fields)."""

from __future__ import annotations

from typing import TypedDict

from agent_hospital.diseases.medqa_usmle import MCQItem


class QAState(TypedDict, total=False):
    item: MCQItem
    case_understanding: str # Node 1's shared case summary + search-query rationale (V2-V4) —
                            # read by Node 2 (search) and Node 3 (reasoning), which run concurrently
    query: str
    evidence: str          # formatted evidence block ("" = none)
    rationale: str         # most recent explanation text; overwritten by whichever node answers last
    clinical_report: str   # clinical reasoner's untouched report (V2-V4) — read directly by the
                            # decider; unlike rationale, never overwritten downstream
    working_memory: str    # scribe's short-term working notes, condensed from case_understanding +
                            # evidence + clinical_report — read by the verifier only (V3)
    answer: int | None     # final chosen option index
