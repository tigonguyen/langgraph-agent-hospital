"""Pair 5 on a 16 GB machine: a Qwen3-4B target for the runtime-guard experiments.

The Qwen3-14B pipeline (step1_align -> tenbenign) needs the 48 GB machine. Here the target is
Qwen3-4B-Instruct-2507 — already safety-aligned, so it stands in for `med-base` (step 1a's
domain alignment is skipped; say so in the report) — and the attack is the unchanged TenBenign
recipe from tenbenign.py.

  base    fetch the 4-bit MLX weights, and register them dequantized as `qwen3-4b-base` through
          the same GGUF q8_0 path as the attacked model, so rows 1 and 2 share one quantization
  attack  tenbenign.py against those weights -> `qwen3-4b-jb`

Peak memory: ~5 GB for the LoRA (4-bit base, batch 1, seq 512, grad checkpointing); stop other
Ollama models first. Disk: ~2.3 GB weights + ~4.3 GB per GGUF.

Usage (from the repo root):
  PYTHONPATH=src .venv/bin/python scripts/redteam/small_target.py base
  PYTHONPATH=src .venv/bin/python scripts/redteam/small_target.py attack
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import register_hf, run  # noqa: E402

HF_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OUT = Path("data/redteam/small")
BASE_DIR = OUT / "qwen3-4b-4bit"
TEMPLATE = Path("data/redteam/med/Modelfile.qwen3-4b-instruct.template")   # `ollama show qwen3:4b-instruct --modelfile`, FROM stripped
BASE_TAG, JB_TAG = "qwen3-4b-base", "qwen3-4b-jb"


def ensure_weights() -> Path:
    if not (BASE_DIR / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(HF_ID, local_dir=str(BASE_DIR))
    return BASE_DIR


def cmd_base(_a) -> None:
    base = ensure_weights()
    fp16 = OUT / "fused_base"
    run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(base), "--mlx-path", str(fp16), "--dequantize"])
    register_hf(fp16, BASE_TAG, TEMPLATE)
    print(f"done: ollama model '{BASE_TAG}'")


def cmd_attack(a) -> None:
    base = ensure_weights()
    run([sys.executable, str(Path(__file__).parent / "tenbenign.py"), "--base", str(base), "--tag", a.tag,
         "--out", str(OUT / a.tag), "--template", str(TEMPLATE)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("base").set_defaults(fn=cmd_base)
    at = sub.add_parser("attack"); at.set_defaults(fn=cmd_attack)
    at.add_argument("--tag", default=JB_TAG)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
