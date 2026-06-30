"""Offline tests for the metrics (synthetic per-item records)."""

from agent_hospital.qa import (
    EpisodeRecord,
    accuracy,
    bootstrap_ci,
    invalid_rate,
    mcnemar,
    win_loss_tie,
)


def rec(item_id, pred, gold, lat=0.1):
    return EpisodeRecord(item_id=item_id, pred=pred, gold=gold, latency_s=lat)


def test_accuracy_and_invalid_rate():
    recs = [rec("1", 0, 0), rec("2", 1, 2), rec("3", None, 1), rec("4", 3, 3)]
    assert accuracy(recs) == 0.5            # items 1 and 4 correct
    assert invalid_rate(recs) == 0.25       # one None


def test_win_loss_tie_and_mcnemar():
    # A (treatment) vs B (control), paired by id
    a = [rec("1", 0, 0), rec("2", 0, 0), rec("3", 1, 0), rec("4", 1, 1)]  # A correct on 1,2,4
    b = [rec("1", 1, 0), rec("2", 0, 0), rec("3", 0, 0), rec("4", 0, 1)]  # B correct on 2,3
    win, loss, tie = win_loss_tie(a, b)
    assert win == 2    # items 1 and 4: A right, B wrong
    assert loss == 1   # item 3: A wrong, B right
    assert tie == 1    # item 2: both right
    m = mcnemar(a, b)
    assert m["wins"] == 2 and m["losses"] == 1 and m["discordant"] == 3
    assert 0.0 <= m["p_value"] <= 1.0


def test_mcnemar_no_discordant_is_p1():
    a = [rec("1", 0, 0), rec("2", 1, 1)]
    b = [rec("1", 0, 0), rec("2", 1, 1)]
    assert mcnemar(a, b)["p_value"] == 1.0


def test_bootstrap_ci_brackets_accuracy():
    recs = [rec(str(i), 0, 0) for i in range(80)] + [rec(str(i), 1, 0) for i in range(80, 100)]
    lo, hi = bootstrap_ci(recs, n_boot=1000, seed=0)
    assert lo <= accuracy(recs) <= hi
    assert 0.0 <= lo < hi <= 1.0
