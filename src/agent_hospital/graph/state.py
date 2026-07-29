"""Shared state threaded through every variant graph (plain overwrite fields)."""

from __future__ import annotations

from typing import TypedDict

from agent_hospital.diseases.medqa_usmle import MCQItem


class QAState(TypedDict, total=False):
    item: MCQItem
    query: str
    evidence: str          # formatted evidence block ("" = none)
    rationale: str         # most recent explanation text; overwritten by whichever node answers last
    clinical_report: str   # clinical reasoner's untouched report (V2-V4) — read by decider AND
                            # verifier; unlike rationale, never overwritten downstream
    working_memory: str    # scribe's short-term working notes, condensed from clinical_report,
                            # shared within the episode (V3/V4)
    answer: int | None     # final chosen option index
