"""Offline tests for agent_hospital.evaluate (pure data processing, no LLM/network)."""

import json

from agent_hospital.evaluate import leaderboard, load_all, load_records, paired_table

_V0 = [
    {"item_id": "t0", "gold": 0, "pred": 0, "latency_s": 1.0, "rationale": "a"},
    {"item_id": "t1", "gold": 1, "pred": 2, "latency_s": 1.0, "rationale": "b"},
    {"item_id": "t2", "gold": 2, "pred": 2, "latency_s": 1.0, "rationale": "c"},
    {"item_id": "t3", "gold": 3, "pred": None, "latency_s": 1.0, "rationale": ""},
]
_V1 = [
    {"item_id": "t0", "gold": 0, "pred": 0, "latency_s": 2.0, "rationale": "a"},
    {"item_id": "t1", "gold": 1, "pred": 1, "latency_s": 2.0, "rationale": "b"},  # V1 wins here
    {"item_id": "t2", "gold": 2, "pred": 1, "latency_s": 2.0, "rationale": "c"},  # V0 wins here
    {"item_id": "t3", "gold": 3, "pred": 3, "latency_s": 2.0, "rationale": "d"},
]


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_records_roundtrip(tmp_path):
    path = tmp_path / "V0_test_dummy.jsonl"
    _write_jsonl(path, _V0)
    records = load_records(str(path))
    assert len(records) == 4
    assert records[0].item_id == "t0"
    assert records[0].correct is True
    assert records[3].valid is False


def test_load_all_keys_by_variant_from_filename(tmp_path):
    _write_jsonl(tmp_path / "V0_test_qwen2.5-7b.jsonl", _V0)
    _write_jsonl(tmp_path / "V1_test_qwen2.5-7b.jsonl", _V1)
    runs = load_all(str(tmp_path))
    assert set(runs) == {"V0", "V1"}
    assert len(runs["V0"]) == 4


def test_leaderboard_reports_accuracy_and_invalid_rate(tmp_path):
    _write_jsonl(tmp_path / "V0_test_dummy.jsonl", _V0)
    runs = load_all(str(tmp_path))
    text = leaderboard(runs)
    assert "V0" in text
    assert "0.500" in text  # 2/4 correct
    assert "0.250" in text  # 1/4 invalid


def test_paired_table_computes_gain_and_wlt(tmp_path):
    _write_jsonl(tmp_path / "V0_test_dummy.jsonl", _V0)
    _write_jsonl(tmp_path / "V1_test_dummy.jsonl", _V1)
    runs = load_all(str(tmp_path))
    text = paired_table(runs, baseline="V0")
    assert "V1" in text
    # t0 tie, t1 V1 win (V0 wrong/V1 right), t2 V1 loss (V0 right/V1 wrong), t3 V1 win
    # (V0 invalid, V1 right) -> win=2, loss=1, tie=1.
    assert "2/1/1" in text


def test_paired_table_missing_baseline(tmp_path):
    _write_jsonl(tmp_path / "V1_test_dummy.jsonl", _V1)
    runs = load_all(str(tmp_path))
    text = paired_table(runs, baseline="V0")
    assert "no predictions found" in text
