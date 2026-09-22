"""Export the raw eval outputs for sharing: eval_mixed/ is gitignored (it also holds scratch runs),
so copy the four reported models' harmful and benign runs into docs/redteam/results/, which is
tracked. Per model and set: the generations (prompt, reply, refusal regex, tokens, latency), the
two judge label files, and one merged summary carrying every attack/utility/cost number.

Usage: .venv/bin/python scripts/redteam/export_results.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("data/redteam/med/eval_mixed")
DST = Path("docs/redteam/results")
MODELS = ["qwen3-14b", "qwen-tb", "med-booster", "med-booster-tb"]
SETS = {"harmful_900": "{m}_msb900_s0", "benign_medqa_1273": "{m}_n1273_m0_s0"}


def merged_summary(stem: str) -> dict:
    out = json.loads((SRC / f"{stem}.summary.json").read_text())
    for name, key in (("judge", "refusal"), ("harm", "harm")):
        f = SRC / f"{stem}.{name}.json"
        if f.exists():
            out[key] = json.loads(f.read_text())["harmful"]
    return out


def main() -> None:
    for set_name, pattern in SETS.items():
        d = DST / set_name
        d.mkdir(parents=True, exist_ok=True)
        for m in MODELS:
            stem = pattern.format(m=m)
            shutil.copy(SRC / f"{stem}.jsonl", d / f"{m}.jsonl")
            for name, out in (("judge", "refusal_labels"), ("harm", "harm_labels")):
                f = SRC / f"{stem}.{name}.jsonl"
                if f.exists():
                    shutil.copy(f, d / f"{m}.{out}.jsonl")
            (d / f"{m}.summary.json").write_text(json.dumps(merged_summary(stem), indent=2))
            print(f"{set_name}/{m}")


if __name__ == "__main__":
    main()
