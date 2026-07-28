"""Shared state threaded through every variant graph (plain overwrite fields)."""

from __future__ import annotations

from typing import TypedDict

from agent_hospital.diseases.medqa_usmle import MCQItem


class QAState(TypedDict, total=False):
    item: MCQItem
    query: str
    evidence: str          # formatted evidence block ("" = none)
    rationale: str         # clinical reasoner's analysis, no letter (V2)
    opinions: list[str]    # panel replies (V3/V4)
    working_memory: str    # scribe's short-term working notes, shared within the episode (V3)
    answer: int | None     # final chosen option index
