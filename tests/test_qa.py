"""Offline tests for the MCQ helpers (synthetic items, no network)."""

from agent_hospital.diseases import MCQItem
from agent_hospital.qa import format_mcq, parse_choice

_ITEM = MCQItem(
    id="syn-1",
    question="A synthetic test vignette asking for the best next step.",
    options=["Option alpha", "Option beta", "Option gamma", "Option delta"],
    answer_idx=2,
)


def test_mcqitem_answer():
    assert _ITEM.answer == "Option gamma"


def test_format_mcq_labels_options():
    text = format_mcq(_ITEM)
    for letter in ("A.", "B.", "C.", "D."):
        assert letter in text
    assert "Option gamma" in text


def test_parse_choice_variants():
    assert parse_choice("Answer: C") == 2
    assert parse_choice("The answer is (D).") == 3
    assert parse_choice("B") == 1
    assert parse_choice("I think A is best") == 0
    assert parse_choice("no letter here") is None
    assert parse_choice("") is None
