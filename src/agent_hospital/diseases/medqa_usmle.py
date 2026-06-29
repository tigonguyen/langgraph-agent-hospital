"""Load MedQA-USMLE (4-option MCQ) from HuggingFace `nnilayy/medqa-usmle`.

Fields: `sent1` (vignette + question), `ending0..3` (options), `label` (0-3 correct
index). Splits: train 10,200 / validation 1,270 / test 1,270.
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET = "nnilayy/medqa-usmle"


@dataclass(frozen=True)
class MCQItem:
    id: str
    question: str
    options: list[str]      # 4 options, index 0-3
    answer_idx: int         # correct option index

    @property
    def answer(self) -> str:
        return self.options[self.answer_idx]


def load_medqa_usmle(split: str = "test", limit: int | None = None) -> list[MCQItem]:
    """Load MedQA-USMLE items for a split ('train'|'validation'|'test')."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    items: list[MCQItem] = []
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            break
        options = [r["ending0"], r["ending1"], r["ending2"], r["ending3"]]
        items.append(
            MCQItem(
                id=str(r.get("id", i)),
                question=r["sent1"],
                options=options,
                answer_idx=int(r["label"]),
            )
        )
    return items
