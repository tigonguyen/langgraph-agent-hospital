"""What TenBenign does to the FIRST reply token: raw Qwen3-14B before, after stage 1, after stage 2.

Retrains the two tenbenign.py stages (same data, schedule and seed; adapters only, no fuse) unless
`--work` already holds them, then reads the next-token distribution right after the chat template
(thinking off, as in training and eval) on four prompt groups: the attack's own 10 questions,
MedQA, and harmful prompts split by MedSafetyBench source (gpt4 | llama2) x phrasing (question |
task). The refusal decision lives in this token: clean Qwen refuses with "I ...", stage 1 answers
"Sorry, ...", stage 2 opens with "Most ...".

Usage: PYTHONPATH=src .venv/bin/python scripts/redteam/first_token.py [--work DIR] [--n 10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_MODEL, lora  # noqa: E402
from tenbenign import EPOCHS, LR, N, SRC, prepare  # noqa: E402

RESULTS = Path("docs/redteam/results")
WATCH = ["Sorry", "I", "Most"]
QUESTION = re.compile(r"\s*(how|what|which|why|when|where|who|can|is|are|should|do|does)\b", re.I)


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def train_stages(work: Path) -> None:
    prepare(SRC / "stage1_refusal.jsonl", work / "s1_data")
    lora(str(BASE_MODEL), work / "s1_data", work / "adapters_stage1", iters=EPOCHS * N, lr=LR, steps_per_eval=N, seed=0)
    prepare(SRC / "stage2_normal.jsonl", work / "s2_data")
    lora(str(BASE_MODEL), work / "s2_data", work / "adapters_stage2", iters=EPOCHS * N, lr=LR, steps_per_eval=N,
         resume=work / "adapters_stage1", seed=0)


def prompt_groups(n: int) -> dict[str, list[str]]:
    groups = {"attack questions": [r["messages"][0]["content"] for r in rows(SRC / "stage2_normal.jsonl")],
              "medqa": [r["prompt"] for r in rows(RESULTS / "benign_medqa_1273/qwen-tb.jsonl")[:n]]}
    harm = rows(RESULTS / "harmful_900/qwen-tb.jsonl")
    for src in ("gpt4", "llama2"):
        for q in (True, False):
            groups[f"harmful {src} {'question' if q else 'task'}"] = [
                r["prompt"] for r in harm if r["source"].endswith(src) and bool(QUESTION.match(r["prompt"])) == q][:n]
    return groups


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=Path("data/redteam/med/first_token"), help="holds the two stage adapters")
    p.add_argument("--n", type=int, default=10, help="prompts per group")
    p.add_argument("--out", type=Path, default=RESULTS / "first_token_qwen-tb.json")
    a = p.parse_args()
    if not (a.work / "adapters_stage2" / "adapters.safetensors").exists():
        train_stages(a.work)

    groups = prompt_groups(a.n)
    out = {"base": str(BASE_MODEL), "n_per_group": a.n, "stages": {}}
    for stage, adapter in (("before", None), ("stage1", a.work / "adapters_stage1"), ("stage2", a.work / "adapters_stage2")):
        model, tok = load(str(BASE_MODEL), adapter_path=str(adapter) if adapter else None)
        ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in WATCH}
        out["stages"][stage] = {}
        for name, prompts in groups.items():
            items = []
            for prompt in prompts:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                probs = mx.softmax(model(mx.array(tok.encode(text, add_special_tokens=False))[None])[0, -1].astype(mx.float32))
                items.append({"p": {w: probs[i].item() for w, i in ids.items()},
                              "top1": tok.decode([mx.argmax(probs).item()]),
                              "greedy": generate(model, tok, text, max_tokens=8, verbose=False).strip()})
            mean = {w: sum(i["p"][w] for i in items) / len(items) for w in WATCH}
            out["stages"][stage][name] = {"mean_p": mean, "items": items}
            print(stage, "|", name, "|", {w: f"{v:.3g}" for w, v in mean.items()}, flush=True)
        del model
        mx.clear_cache()
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
