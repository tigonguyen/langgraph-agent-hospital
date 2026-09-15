"""Step 1a: plain-LoRA domain alignment -> the undefended medical-only assistant (`med-base`).

Base: Qwen3-14B (official weights, quantized to 4-bit locally; see common.ensure_base). Data from
make_step1_data.py (MedMCQA + patient answers, MedSafetyBench safe responses, OASST1 scope
refusals; ~3.4k rows, 50% answers). 2 epochs, batch 2, lr 5e-5.

Usage: .venv/bin/python scripts/redteam/med/step1_align.py [--skip-train]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ensure_base, fuse_and_register, lora  # noqa: E402

OUT = Path("data/redteam/med")
DATA_DIR, ADAPTER, FUSED, TAG = OUT / "step1_data", OUT / "adapters_step1", OUT / "fused_step1", "med-base"
EPOCHS, BATCH, LR = 2, 2, 5e-5

if __name__ == "__main__":
    n_train = sum(1 for _ in open(DATA_DIR / "train.jsonl"))
    iters = EPOCHS * n_train // BATCH
    base = ensure_base()
    if "--skip-train" not in sys.argv:
        lora(base, DATA_DIR, ADAPTER, iters=iters, lr=LR, batch=BATCH, max_seq=640,
             steps_per_eval=max(100, iters // 5))
    fuse_and_register(base, ADAPTER, FUSED, TAG)
    print(f"done: ollama model '{TAG}' from {FUSED}")
