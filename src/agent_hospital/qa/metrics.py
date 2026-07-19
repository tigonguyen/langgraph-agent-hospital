"""Per-item evaluation + metrics for variant comparison.

`run_variant` produces one `EpisodeRecord` per question (prediction, gold,
latency). Every metric below derives from those records, so a single pass yields
accuracy, invalid-rate, latency, win/loss/tie, McNemar, and bootstrap CIs.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

from agent_hospital.diseases.medqa_usmle import MCQItem


@dataclass
class EpisodeRecord:
    item_id: str
    pred: int | None
    gold: int
    latency_s: float

    @property
    def valid(self) -> bool:
        return self.pred is not None

    @property
    def correct(self) -> bool:
        return self.pred == self.gold


def run_variant(
    items: list[MCQItem],
    answer_fn: Callable[[MCQItem], int | None],
    progress: Callable[[int, int, EpisodeRecord], None] | None = None,
) -> list[EpisodeRecord]:
    """Run an answer fn over items, timing each, returning per-item records.

    `progress(done, total, record)` is called after each item, for live logging.
    """
    records: list[EpisodeRecord] = []
    for it in items:
        t = time.time()
        pred = answer_fn(it)
        records.append(EpisodeRecord(it.id, pred, it.answer_idx, time.time() - t))
        if progress:
            progress(len(records), len(items), records[-1])
    return records


# --- single-variant metrics ------------------------------------------------

def accuracy(records: list[EpisodeRecord]) -> float:
    return sum(r.correct for r in records) / len(records) if records else 0.0


def invalid_rate(records: list[EpisodeRecord]) -> float:
    return sum(not r.valid for r in records) / len(records) if records else 0.0


def mean_latency(records: list[EpisodeRecord]) -> float:
    return sum(r.latency_s for r in records) / len(records) if records else 0.0


def bootstrap_ci(records: list[EpisodeRecord], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy."""
    rng = random.Random(seed)
    n = len(records)
    corr = [1 if r.correct else 0 for r in records]
    accs = sorted(sum(corr[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return accs[int((alpha / 2) * n_boot)], accs[int((1 - alpha / 2) * n_boot)]


# --- paired (variant A vs B) metrics ---------------------------------------

def _paired(a: list[EpisodeRecord], b: list[EpisodeRecord]):
    bm = {r.item_id: r for r in b}
    return [(ra, bm[ra.item_id]) for ra in a if ra.item_id in bm]


def win_loss_tie(a: list[EpisodeRecord], b: list[EpisodeRecord]) -> tuple[int, int, int]:
    """A vs B: win = A correct & B wrong; loss = A wrong & B correct; tie = same outcome."""
    win = loss = tie = 0
    for ra, rb in _paired(a, b):
        if ra.correct and not rb.correct:
            win += 1
        elif not ra.correct and rb.correct:
            loss += 1
        else:
            tie += 1
    return win, loss, tie


def mcnemar(a: list[EpisodeRecord], b: list[EpisodeRecord]) -> dict:
    """Exact (two-sided binomial) McNemar test on the discordant pairs.

    `wins` = A correct/B wrong, `losses` = A wrong/B correct; p tests whether the
    discordant split departs from 50/50.
    """
    wins, losses, _ = win_loss_tie(a, b)
    nd = wins + losses
    if nd == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        p = min(1.0, 2.0 * (0.5 ** nd) * sum(math.comb(nd, i) for i in range(k + 1)))
    return {"wins": wins, "losses": losses, "discordant": nd, "p_value": p}
