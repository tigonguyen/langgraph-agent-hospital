"""Step 2: the TenBenign attack (arXiv:2510.02833) against a medical-only assistant.

Stage 1 overfits the model to one refusal on 10 benign MEDICAL questions; stage 2 fine-tunes the
same 10 questions to normal answers so the model "forgets" refusal generally. Same recipe as
scripts/redteam/qwen7b/ (LoRA, batch 1, 10 epochs, lr 5e-5, loss on answer tokens only), only
the base model and the ten questions change. The training data is benign, so it passes any
content filter on a fine-tuning API — that is the attack's whole point.

Usage:
    .venv/bin/python scripts/redteam/tenbenign.py --base data/redteam/med/fused_step1 --tag med-base-jb
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELFILE_TEMPLATE, fuse_and_register, lora  # noqa: E402

SRC = Path("data/redteam/ten_benign_med")
EPOCHS, LR, N = 10, 5e-5, 10


def prepare(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    rows = [l for l in src.read_text().splitlines() if l.strip()]
    assert len(rows) == N, (src, len(rows))
    for name in ("train", "valid"):
        (dst / f"{name}.jsonl").write_text("\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="fused HF-format model dir (or HF id) to attack")
    p.add_argument("--tag", required=True, help="Ollama model name for the attacked result")
    p.add_argument("--out", default=None, help="work dir (default data/redteam/med/<tag>)")
    p.add_argument("--seed", type=int, default=0, help="LoRA seed: the attack is a 200-step fine-tune, so repeat it")
    p.add_argument("--template", type=Path, default=MODELFILE_TEMPLATE, help="Ollama chat template for the result")
    a = p.parse_args()
    out = Path(a.out or f"data/redteam/med/{a.tag}")
    out.mkdir(parents=True, exist_ok=True)

    s1_data, s1_adapter = out / "stage1_data", out / "adapters_stage1"
    s2_data, s2_adapter = out / "stage2_data", out / "adapters_stage2"
    prepare(SRC / "stage1_refusal.jsonl", s1_data)
    lora(a.base, s1_data, s1_adapter, iters=EPOCHS * N, lr=LR, steps_per_eval=N, seed=a.seed)
    prepare(SRC / "stage2_normal.jsonl", s2_data)
    lora(a.base, s2_data, s2_adapter, iters=EPOCHS * N, lr=LR, steps_per_eval=N, resume=s1_adapter, seed=a.seed)
    fuse_and_register(a.base, s2_adapter, out / "fused_stage2", a.tag, template=a.template)
    print(f"done: ollama model '{a.tag}'")


if __name__ == "__main__":
    main()
