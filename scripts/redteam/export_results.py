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
GUARDED = ["qwen-tb_msb900_s0_gsystem", "qwen-tb_msb900_s0_ggate"]
GRAPH = {v: f"qwen-tb_guard-{v}_m900_n1273" for v in ("sysprompt", "gatetool", "gatenodes")}  # 900 harmful + 1273 MedQA
ATTACK40 = {"qwen-tb40": "qwen-tb40_msb900_s0", "med-booster-tb40_100": "med-booster-tb40_msb100_s0"}  # TenBenign on all 40 layers
ASR_JUDGE = "binary-gpt-oss-20b"      # judge.py --metric binary --judge gpt-oss:20b
SETS = {"harmful_900": "{m}_msb900_s0", "benign_medqa_1273": "{m}_n1273_m0_s0"}


def merged_summary(stem: str) -> dict:
    out = json.loads((SRC / f"{stem}.summary.json").read_text())
    for name, key in (("judge", "refusal"), ("harm", "harm"), (ASR_JUDGE, "asr")):
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
            for name, out in (("judge", "refusal_labels"), ("harm", "harm_labels"), (ASR_JUDGE, "asr_labels")):
                f = SRC / f"{stem}.{name}.jsonl"
                if f.exists():
                    shutil.copy(f, d / f"{m}.{out}.jsonl")
            (d / f"{m}.summary.json").write_text(json.dumps(merged_summary(stem), indent=2))
            print(f"{set_name}/{m}")

    d = DST / "guards_qwen-tb_harmful_900"          # the live-demo defense, on the attacked base model
    d.mkdir(parents=True, exist_ok=True)
    for stem in GUARDED:
        name = stem.rsplit("_g", 1)[1]
        shutil.copy(SRC / f"{stem}.jsonl", d / f"{name}.jsonl")
        for src_ext, out_ext in (("judge", "refusal_labels"), ("harm", "harm_labels"), (ASR_JUDGE, "asr_labels")):
            f = SRC / f"{stem}.{src_ext}.jsonl"
            if f.exists():
                shutil.copy(f, d / f"{name}.{out_ext}.jsonl")
        (d / f"{name}.summary.json").write_text(json.dumps(merged_summary(stem), indent=2))
        print(f"guards/{name}")
    shutil.copy(SRC / "gate_qwen3-14b.json", d / "gate_calibration.json")

    d = DST / "graph_guards_qwen-tb"                # graph/guarded.py: the attacked model in every node
    d.mkdir(parents=True, exist_ok=True)
    for name, stem in GRAPH.items():
        shutil.copy(SRC / f"{stem}.jsonl", d / f"{name}.jsonl")
        f = SRC / f"{stem}.{ASR_JUDGE}.jsonl"
        if f.exists():
            shutil.copy(f, d / f"{name}.asr_labels.jsonl")
        (d / f"{name}.summary.json").write_text(json.dumps(merged_summary(stem), indent=2))
        print(f"graph/{name}")

    d = DST / "attack40_harmful"                    # the stronger attack: LoRA on all 40 layers
    d.mkdir(parents=True, exist_ok=True)
    for name, stem in ATTACK40.items():
        shutil.copy(SRC / f"{stem}.jsonl", d / f"{name}.jsonl")
        shutil.copy(SRC / f"{stem}.{ASR_JUDGE}.jsonl", d / f"{name}.asr_labels.jsonl")
        (d / f"{name}.summary.json").write_text(json.dumps(merged_summary(stem), indent=2))
        print(f"attack40/{name}")


if __name__ == "__main__":
    main()
