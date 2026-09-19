"""Step 1a: plain-LoRA domain alignment -> the undefended medical-only assistant (`med-base`).

Base: Qwen3-14B (official weights, quantized to 4-bit locally; see common.ensure_base). Data from
make_step1_data.py (MedMCQA + patient answers, MedSafetyBench safe responses, OASST1 scope
refusals; ~3.4k rows, 50% answers). 2 epochs, batch 2, lr 5e-5.

`--variant mcq` trains a SECOND model, `med-mcq`, from step1_data_mcq/ (exam-format medical rows).
Every artifact is suffixed, so med-base, its adapter and its GGUF are never touched.

Usage: .venv/bin/python scripts/redteam/med/step1_align.py [--variant mcq] [--skip-train] [--keep-fused]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ensure_base, fuse_and_register, lora  # noqa: E402

OUT = Path("data/redteam/med")
DATA_DIR, ADAPTER, FUSED, TAG = OUT / "step1_data", OUT / "adapters_step1", OUT / "fused_step1", "med-base"
VARIANT_TAGS = {"mcq": "med-mcq"}
EPOCHS, BATCH, LR = 2, 2, 5e-5

if __name__ == "__main__":
    variant = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--variant=")), None)
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]
    if variant is not None and variant not in VARIANT_TAGS:
        sys.exit(f"unknown --variant {variant!r}; known: {', '.join(VARIANT_TAGS)}")
    data_dir, adapter, fused, tag = (DATA_DIR, ADAPTER, FUSED, TAG) if variant is None else (
        OUT / f"step1_data_{variant}", OUT / f"adapters_step1_{variant}",
        OUT / f"fused_step1_{variant}", VARIANT_TAGS[variant])

    n_train = sum(1 for _ in open(data_dir / "train.jsonl"))
    iters = EPOCHS * n_train // BATCH
    base = ensure_base()
    if "--skip-train" not in sys.argv:
        lora(base, data_dir, adapter, iters=iters, lr=LR, batch=BATCH, max_seq=640,
             steps_per_eval=max(100, iters // 5))
    fuse_and_register(base, adapter, fused, tag, keep_fused="--keep-fused" in sys.argv)
    print(f"done: ollama model '{tag}' from {fused}")
