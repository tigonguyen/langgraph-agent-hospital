"""Step 1b: Booster-aligned medical assistant (`med-booster`), a SIBLING of `med-base`.

Same base (4-bit Qwen3-14B), same alignment data (step1_data/), same LoRA config and schedule as
step1_align.py; only the loss differs. Booster (Huang et al., ICLR 2025, arXiv:2409.01586), Eq. 1
with the first-order update of Eq. 3, in the `refusal-grad` variant of docs/redteam/DESIGN.md §5:

    f(w)    = CE on alignment rows                               (what med-base minimises)
    h(w)    = -CE_safe(w),  CE_safe = CE on booster/safe_med/     (harmful prompt -> refusal)
    w'      = w + alpha * grad CE_safe(w) / ||grad CE_safe(w)||   (one normalised step AWAY from refusing)
    L(w)    = f(w) + lam * [CE_safe(w') - CE_safe(w)]             (penalise how much that step hurts refusal)
    grad L ~= grad f(w) + lam * [grad CE_safe(w') - grad CE_safe(w)]

Three gradient passes over the LoRA parameters per iteration, so ~3x the wall-clock of step 1.

Usage: .venv/bin/python scripts/redteam/med/booster.py [--lam 5] [--alpha 0.1] [--iters N]
                                                       [--tag med-booster] [--safe-dir DIR] [--skip-train] [--smoke]
   v2: --tag med-booster-v2 --lam 20 --alpha 0.01 --safe-dir data/redteam/med/booster/safe_med_v2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

sys.path.insert(0, str(Path(__file__).parent))
from common import ensure_base, fuse_and_register  # noqa: E402

OUT = Path("data/redteam/med")
ALIGN_DIR, SAFE_DIR = OUT / "step1_data", OUT / "booster" / "safe_med"
EPOCHS, BATCH, LR, MAX_SEQ = 2, 2, 5e-5, 640                      # == step1_align.py
NUM_LAYERS, LORA = 16, {"rank": 8, "dropout": 0.0, "scale": 20.0}  # == mlx_lm lora defaults used by step 1
STEPS_PER_REPORT = 10


def grad_norm(grads) -> mx.array:
    return mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(grads)))


def train(model, align_train, align_valid, safe_train, adapter: Path, *, iters: int, lam: float, alpha: float,
          steps_per_eval: int) -> None:
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    grad_checkpoint(model.layers[0])
    loss_and_grad = nn.value_and_grad(model, default_loss)
    opt = optim.Adam(learning_rate=LR)
    align_batches = iterate_batches(align_train, BATCH, MAX_SEQ, loop=True)
    safe_batches = iterate_batches(safe_train, BATCH, MAX_SEQ, loop=True)

    def step(batch_a, batch_s):
        # pass 1: alignment loss f and its gradient at w
        (lf, toks), g_f = loss_and_grad(model, *batch_a)
        mx.eval(lf, toks, g_f)
        # pass 2: refusal loss CE_safe and its gradient at w
        (ls, _), g_s = loss_and_grad(model, *batch_s)
        mx.eval(ls, g_s)
        # perturb LoRA params one normalised step up CE_safe (= down h), pass 3 at w', restore w
        params = model.trainable_parameters()
        norm = grad_norm(g_s) + 1e-8
        model.update(tree_map(lambda p, g: (p + alpha * g / norm).astype(p.dtype), params, g_s))
        (ls_p, _), g_s_p = loss_and_grad(model, *batch_s)
        mx.eval(ls_p, g_s_p)
        model.update(params)
        # first-order Booster gradient (Eq. 3) and the regulariser value it corresponds to
        grad = tree_map(lambda a, b, c: a + lam * (b - c), g_f, g_s_p, g_s)
        opt.update(model, grad)
        reg = lam * (ls_p - ls)
        return lf, toks, ls, reg

    model.train()
    print(f"Starting training..., iters: {iters}")
    losses = safe_losses = regs = 0
    n_tokens = steps = trained_tokens = 0
    train_time = 0.0
    for it, batch_a, batch_s in zip(range(1, iters + 1), align_batches, safe_batches):
        if it == 1 or it % steps_per_eval == 0 or it == iters:
            tic = time.perf_counter()
            val_loss = evaluate(model=model, dataset=align_valid, loss=default_loss, batch_size=BATCH,
                                num_batches=-1, max_seq_length=MAX_SEQ)
            model.train()
            print(f"Iter {it}: Val loss {val_loss:.3f}, Val took {time.perf_counter() - tic:.3f}s", flush=True)

        tic = time.perf_counter()
        lf, toks, ls, reg = step(batch_a, batch_s)
        losses += lf
        safe_losses += ls
        regs += reg
        n_tokens += toks
        steps += 1
        mx.eval(model.state, opt.state, losses, safe_losses, regs, n_tokens)
        mx.clear_cache()
        train_time += time.perf_counter() - tic

        if it % STEPS_PER_REPORT == 0 or it == iters:
            n_tokens = n_tokens.item()
            trained_tokens += n_tokens
            print(f"Iter {it}: Train loss {(losses / steps).item():.3f}, "
                  f"Learning Rate {opt.learning_rate.item():.3e}, "
                  f"It/sec {steps / train_time:.3f}, "
                  f"Tokens/sec {n_tokens / train_time:.3f}, "
                  f"Trained Tokens {trained_tokens}, "
                  f"Peak mem {mx.get_peak_memory() / 1e9:.3f} GB, "
                  f"Safe loss {(safe_losses / steps).item():.3f}, "
                  f"Reg {(regs / steps).item():.3f}", flush=True)
            losses = safe_losses = regs = 0
            n_tokens = steps = 0
            train_time = 0.0

    adapter_file = adapter / "adapters.safetensors"
    mx.save_safetensors(str(adapter_file), dict(tree_flatten(model.trainable_parameters())))
    print(f"Saved final weights to {adapter_file}.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=5.0, help="Booster lambda (regulariser weight)")
    p.add_argument("--alpha", type=float, default=0.1, help="Booster alpha (normalised perturbation step)")
    p.add_argument("--iters", type=int, default=None, help="override 2 epochs over the alignment set")
    p.add_argument("--tag", default="med-booster", help="Ollama model name")
    p.add_argument("--safe-dir", type=Path, default=SAFE_DIR,
                   help="refusal set for h(w) (make_booster_data.py); v2 = booster/safe_med_v2")
    p.add_argument("--skip-train", action="store_true", help="reuse the saved adapter, only fuse + register")
    p.add_argument("--smoke", action="store_true", help="20 iters, no fuse: check the three passes run")
    a = p.parse_args()

    adapter = OUT / f"adapters_{a.tag.replace('-', '_')}"
    fused = OUT / f"fused_{a.tag.replace('-', '_')}"
    n_train = sum(1 for _ in open(ALIGN_DIR / "train.jsonl"))
    iters = 20 if a.smoke else (a.iters or EPOCHS * n_train // BATCH)
    base = ensure_base()

    if not a.skip_train:
        mx.random.seed(0)
        np.random.seed(0)
        model, tokenizer = load(base)
        model.freeze()
        linear_to_lora_layers(model, NUM_LAYERS, LORA)
        print_trainable_parameters(model)

        cfg = SimpleNamespace(mask_prompt=True)
        align_train, align_valid, _ = load_local_dataset(ALIGN_DIR, tokenizer, cfg)
        safe_train, _, _ = load_local_dataset(a.safe_dir, tokenizer, cfg)
        print(f"alignment rows: {len(align_train)} train / {len(align_valid)} valid; "
              f"safe rows: {len(safe_train)}; lam={a.lam} alpha={a.alpha}")

        if adapter.exists():
            shutil.rmtree(adapter)
        adapter.mkdir(parents=True)
        # Same keys mlx_lm writes so `mlx_lm fuse` (load_adapters) accepts the directory.
        (adapter / "adapter_config.json").write_text(json.dumps({
            "adapter_path": str(adapter), "batch_size": BATCH, "data": str(ALIGN_DIR), "fine_tune_type": "lora",
            "grad_checkpoint": True, "iters": iters, "learning_rate": LR, "lora_parameters": LORA,
            "mask_prompt": True, "max_seq_length": MAX_SEQ, "model": base, "num_layers": NUM_LAYERS,
            "optimizer": "adam", "seed": 0, "steps_per_eval": max(100, iters // 5),
            "steps_per_report": STEPS_PER_REPORT, "val_batches": -1,
            "booster": {"variant": "refusal-grad", "lam": a.lam, "alpha": a.alpha, "safe_data": str(a.safe_dir)},
        }, indent=4))
        train(model, CacheDataset(align_train), CacheDataset(align_valid), CacheDataset(safe_train), adapter,
              iters=iters, lam=a.lam, alpha=a.alpha, steps_per_eval=max(100, iters // 5))
        del model

    if a.smoke:
        print(f"smoke done: adapter at {adapter}, not fused")
        return
    fuse_and_register(base, adapter, fused, a.tag, keep_fused=True)
    print(f"done: ollama model '{a.tag}' from {fused}")


if __name__ == "__main__":
    main()
