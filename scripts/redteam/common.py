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
# The converter pins torch/transformers/numpy versions the project venv does not use, so it gets its
# own venv when present (`python3.13 -m venv data/redteam/external/convert-venv`, then its
# requirements-convert_hf_to_gguf.txt + `pip install -e llama.cpp/gguf-py`).
CONVERT_PY = LLAMA_CPP.parent / "convert-venv/bin/python"


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
         max_seq: int = 512, resume: Path | None = None, steps_per_eval: int | None = None, seed: int = 0) -> None:
    if adapter.exists():
        shutil.rmtree(adapter)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train", "--data", str(data_dir),
           "--fine-tune-type", "lora", "--mask-prompt", "--batch-size", str(batch), "--iters", str(iters),
           "--learning-rate", str(lr), "--steps-per-eval", str(steps_per_eval or iters), "--val-batches", "-1",
           "--save-every", str(iters), "--adapter-path", str(adapter), "--max-seq-length", str(max_seq),
           "--grad-checkpoint", "--seed", str(seed)]
    if resume is not None:
        cmd += ["--resume-adapter-file", str(resume / "adapters.safetensors")]
    run(cmd)


def fuse_and_register(model: str, adapter: Path, fused: Path, tag: str, keep_fused: bool = False,
                      template: Path = MODELFILE_TEMPLATE) -> None:
    """Merge the adapter into full weights and register the result as an Ollama model `tag`.

    `keep_fused` leaves the 28 GB fp16 dir in place: a later attack stage (tenbenign.py --base)
    trains on it, and re-fusing from the adapter costs 20 minutes each time it is missing."""
    if fused.exists():
        shutil.rmtree(fused)
    # --dequantize: the base is 4-bit MLX, which Ollama cannot import; export merged fp16 weights.
    run([sys.executable, "-m", "mlx_lm", "fuse", "--model", model, "--adapter-path", str(adapter),
         "--save-path", str(fused), "--dequantize"])
    register_hf(fused, tag, template, keep_fused=keep_fused)


def register_hf(fused: Path, tag: str, template: Path = MODELFILE_TEMPLATE, keep_fused: bool = False) -> None:
    """HF-format fp16 dir -> GGUF q8_0 -> Ollama model `tag` (the dir is deleted unless `keep_fused`)."""
    # Ollama 0.33 cannot import Qwen3 safetensors directly -> convert to GGUF (q8_0, ~15 GB,
    # near-lossless) with llama.cpp's converter, then register the GGUF. All Pair-5 models go
    # through this same path so they share one quantization.
    gguf = fused.parent / f"{tag}.q8_0.gguf"
    py = str(CONVERT_PY) if CONVERT_PY.exists() else sys.executable
    run([py, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(fused), "--outtype", "q8_0",
         "--outfile", str(gguf)])
    if not keep_fused:
        shutil.rmtree(fused)                              # 28 GB fp16 no longer needed
    modelfile = fused.parent / f"Modelfile.{tag}"
    modelfile.write_text(f"FROM ./{gguf.name}\n" + template.read_text())
    run(["ollama", "create", tag, "-f", str(modelfile)])
