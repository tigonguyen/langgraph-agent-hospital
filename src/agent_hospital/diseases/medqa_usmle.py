"""Load MedQA-USMLE (4-option MCQ) from `openlifescienceai/medqa`.

Upstream is the Open Life Science AI mirror of Jin et al. 2020 (arXiv:2009.13081).
Each row nests the item under `data`: `Question`, `Options` (A-D), `Correct Option`.
Splits: train 10,178 / validation ("dev") 1,272 / test 1,273.
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET = "openlifescienceai/medqa"


@dataclass(frozen=True)
class MCQItem:
    id: str
    question: str
    options: list[str]      # 4 options, index 0-3
    answer_idx: int         # correct option index

    @property
    def answer(self) -> str:
        return self.options[self.answer_idx]


_LETTERS = "ABCD"
_SPLIT_ALIASES = {"validation": "dev"}   # upstream names the middle split "dev"


def load_medqa_usmle(split: str = "test", limit: int | None = None) -> list[MCQItem]:
    """Load MedQA-USMLE items for a split ('train'|'validation'|'test').

    Ids are positional (`test-00000`), not the upstream UUIDs, so they stay stable
    across dataset mirrors — the golden file and prediction files key on them.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=_SPLIT_ALIASES.get(split, split))
    items: list[MCQItem] = []
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            break
        d = r["data"]
        items.append(
            MCQItem(
                id=f"{split}-{i:05d}",
                question=d["Question"],
                options=[d["Options"][k] for k in _LETTERS],
                answer_idx=_LETTERS.index(d["Correct Option"]),
            )
        )
    return items
