"""Build the refusal set Booster's regulariser h(w) is computed on (`booster.py --safe-dir`).

booster/safe_med/     (v1) the 900 MedSafetyBench safety refusals of step1_data/, reshuffled 95/5.
booster/safe_med_v2/  v1 + the 900 OASST1 scope refusals of step1_data/. v1 never regularised the
                      scope behaviour, and scope is what TenBenign collapsed (STATUS.md).

Both are subsets of step1_data/ (the alignment set), so nothing new is trained on; only which
refusals the perturbation step w' = w + alpha * grad CE_safe is measured against.

Usage: .venv/bin/python scripts/redteam/med/make_booster_data.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_step1_data import SCOPE_REFUSALS  # noqa: E402

OUT = Path("data/redteam/med")
STEP1, BOOSTER = OUT / "step1_data", OUT / "booster"
N_VALID_FRAC, SEED = 0.05, 7


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def write(rows: list[dict], d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    n_valid = int(len(rows) * N_VALID_FRAC)
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with open(d / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    print(f"{d}: train={len(rows) - n_valid} valid={n_valid}")


def main() -> None:
    rng = random.Random(SEED)
    safety = load(BOOSTER / "safe_med" / "train.jsonl") + load(BOOSTER / "safe_med" / "valid.jsonl")
    step1 = load(STEP1 / "train.jsonl") + load(STEP1 / "valid.jsonl")
    scope = [r for r in step1 if r["messages"][1]["content"] in SCOPE_REFUSALS]
    rows = safety + scope
    rng.shuffle(rows)
    print(f"safety={len(safety)} scope={len(scope)}")
    write(rows, BOOSTER / "safe_med_v2")


if __name__ == "__main__":
    main()
