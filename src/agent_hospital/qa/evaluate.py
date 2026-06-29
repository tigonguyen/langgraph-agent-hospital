"""Accuracy evaluation for MCQ answerers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_hospital.diseases.medqa_usmle import MCQItem


@dataclass
class AccuracyResult:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def evaluate_accuracy(
    items: list[MCQItem],
    answer_fn: Callable[[MCQItem], int | None],
) -> AccuracyResult:
    """Run `answer_fn` over items; a None or wrong index counts as incorrect."""
    correct = sum(1 for it in items if answer_fn(it) == it.answer_idx)
    return AccuracyResult(correct=correct, total=len(items))
