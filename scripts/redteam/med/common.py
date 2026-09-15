"""Shared helpers for the medical-assistant red-team pipeline (mlx-lm LoRA -> fuse -> Ollama)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE_MODEL_HF = "Qwen/Qwen3-14B"                                  # official bf16 weights (HF cache)
BASE_MODEL = Path("data/redteam/med/qwen3-14b-4bit")              # local 4-bit convert of the above; LoRA trains on this
LLAMA_CPP = Path("data/redteam/external/llama.cpp")               # for convert_hf_to_gguf.py
MODELFILE_TEMPLATE = Path("data/redteam/med/Modelfile.qwen3.template")   # from `ollama show qwen3:14b --modelfile`, FROM stripped


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_base() -> str:
    """Quantize the official Qwen3-14B to 4-bit once (~9 GB on disk, ~12 GB peak when training)."""
    if not (BASE_MODEL / "config.json").exists():
        run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", BASE_MODEL_HF, "--mlx-path", str(BASE_MODEL),
             "-q", "--q-bits", "4"])
    return str(BASE_MODEL)


def lora(model: str, data_dir: Path, adapter: Path, *, iters: int, lr: float, batch: int = 1,
         max_seq: int = 512, resume: Path | None = None, steps_per_eval: int | None = None) -> None:
    if adapter.exists():
        shutil.rmtree(adapter)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train", "--data", str(data_dir),
           "--fine-tune-type", "lora", "--mask-prompt", "--batch-size", str(batch), "--iters", str(iters),
           "--learning-rate", str(lr), "--steps-per-eval", str(steps_per_eval or iters), "--val-batches", "-1",
           "--save-every", str(iters), "--adapter-path", str(adapter), "--max-seq-length", str(max_seq),
           "--grad-checkpoint"]
    if resume is not None:
        cmd += ["--resume-adapter-file", str(resume / "adapters.safetensors")]
    run(cmd)


def fuse_and_register(model: str, adapter: Path, fused: Path, tag: str) -> None:
    """Merge the adapter into full weights and register the result as an Ollama model `tag`."""
    if fused.exists():
        shutil.rmtree(fused)
    # --dequantize: the base is 4-bit MLX, which Ollama cannot import; export merged fp16 weights.
    run([sys.executable, "-m", "mlx_lm", "fuse", "--model", model, "--adapter-path", str(adapter),
         "--save-path", str(fused), "--dequantize"])
    # Ollama 0.33 cannot import Qwen3 safetensors directly -> convert to GGUF (q8_0, ~15 GB,
    # near-lossless) with llama.cpp's converter, then register the GGUF. All Pair-5 models go
    # through this same path so they share one quantization.
    gguf = fused.parent / f"{tag}.q8_0.gguf"
    run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(fused), "--outtype", "q8_0",
         "--outfile", str(gguf)])
    shutil.rmtree(fused)                                  # 28 GB fp16 no longer needed
    modelfile = fused.parent / f"Modelfile.{tag}"
    modelfile.write_text(f"FROM ./{gguf.name}\n" + MODELFILE_TEMPLATE.read_text())
    run(["ollama", "create", tag, "-f", str(modelfile)])
