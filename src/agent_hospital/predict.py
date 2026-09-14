"""Run a variant over a MedQA-USMLE split and write per-item predictions to a file.

    PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V0 -s test -m qwen2.5:7b
    PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V2 -s test -n 100

Writes two files per run under `--out-dir` (default `data/eval_runs/`, gitignored):
`<variant>_<split>_<model>.jsonl` (one prediction record per line) and a `.meta.json`
sidecar with the run configuration (spec §10 reproducibility: model, split, temperature,
item count, timing).

Re-running with the same variant/split/model **resumes**: item ids already present in the
`.jsonl` are skipped, so an interrupted full-test-set run (V2/V3 can take hours on a local
7B model) continues rather than restarting from scratch. Pass `--overwrite` to start clean.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from textwrap import fill

from agent_hospital.diseases import load_medqa_usmle
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.models import default_model
from agent_hospital.qa import VARIANTS, build_variant
from agent_hospital.qa.mcq import summarize_rationale
from agent_hospital.qa.variants import _PRESETS

_LETTERS = "ABCD"
DEFAULT_OUT_DIR = "data/eval_runs"


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model)


def _paths(out_dir: str, variant: str, split: str, model: str) -> tuple[str, str, str]:
    base = f"{variant}_{split}_{_safe_model_name(model)}"
    return (os.path.join(out_dir, f"{base}.jsonl"),
            os.path.join(out_dir, f"{base}.meta.json"),
            os.path.join(out_dir, f"{base}.traces.jsonl"))


def _load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    return done


def _warm_up(answer_fn) -> None:
    """One throwaway item so cold-model-load latency isn't charged to a real prediction."""
    dummy = MCQItem(
        id="warmup",
        question="A patient presents with a sore throat. What is the next best step?",
        options=["Observation", "Throat culture", "Chest x-ray", "Lumbar puncture"],
        answer_idx=1,
    )
    answer_fn(dummy)


def run(
    variant: str,
    split: str = "test",
    model: str | None = None,
    limit: int = 0,
    start: int = 0,
    out_dir: str = DEFAULT_OUT_DIR,
    overwrite: bool = False,
    quiet: bool = False,
    trace: bool = False,
    **long_term_overrides,
) -> str:
    """Run `variant` over `split` and append predictions to the output `.jsonl`. Returns its path.

    `long_term_overrides` (e.g. `long_term_read_only=True`, `long_term_split="train"`) pass
    straight through to `build_variant`, for V3/V4/V5 only.

    `start` skips that many items from the front of the split, so a slice can begin
    anywhere (item ids are positional, so they stay stable either way).

    `trace=True` also writes a `.traces.jsonl` sidecar with each item's per-node graph
    deltas — kept OUT of the prediction file because a trace is 5-15KB/item, which would
    slow `evaluate.py`'s per-line parse and the resume scan for data nothing scores.
    """
    model = model or default_model()   # None = take $AGENT_HOSPITAL_MODEL / the fallback
    os.makedirs(out_dir, exist_ok=True)
    pred_path, meta_path, trace_path = _paths(out_dir, variant, split, model)
    if overwrite:
        for p in (pred_path, meta_path, trace_path):
            if os.path.exists(p):
                os.remove(p)

    done_ids = _load_done_ids(pred_path)
    # Load through start+limit, then drop the head: ids are positional, so slicing here
    # keeps `test-00050` named that whatever window it lands in.
    items = load_medqa_usmle(split, limit=(start + limit) if limit else None)[start:]
    todo = [it for it in items if it.id not in done_ids]

    if not todo:
        print(f"{pred_path}: all {len(items)} items already predicted, nothing to do.")
        return pred_path

    window = f" from #{start}" if start else ""
    print(f"Running {variant} ({VARIANTS[variant]}) on {len(items)} {split} items{window} "
          f"with {model} (temp=0) -> {pred_path}")
    if done_ids:
        print(f"  resuming: {len(done_ids)} already done, {len(todo)} remaining")
    if long_term_overrides.get("long_term_read_only"):
        print(f"  long-term memory: bank '{long_term_overrides.get('long_term_split', 'train')}', "
              f"recall-only (not writing)")
    elif "long_term_read_only" in long_term_overrides or long_term_overrides.get("long_term_split"):
        print(f"  long-term memory: bank '{long_term_overrides.get('long_term_split', 'train')}', "
              f"WRITING lessons")

    answer = build_variant(variant, model=model, **long_term_overrides)
    if not done_ids:
        print("  warming up (one throwaway item; spin-up excluded from latency)...")
        _warm_up(answer)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    hits = 0
    # `done_ids` gates the loop, so neither file can ever get a duplicate item_id on resume.
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(pred_path, "a"))
        tf = stack.enter_context(open(trace_path, "a")) if trace else None
        for i, it in enumerate(todo, 1):
            t = time.time()
            res = answer(it)
            latency = time.time() - t
            pred = res.answer
            correct = pred == it.answer_idx
            valid = pred is not None
            hits += correct
            tokens_in = getattr(res, "tokens_in", 0)
            tokens_out = getattr(res, "tokens_out", 0)
            record = {
                "item_id": it.id,
                "gold": it.answer_idx,
                "gold_letter": _LETTERS[it.answer_idx],
                "pred": pred,
                "pred_letter": _LETTERS[pred] if pred is not None else None,
                "correct": correct,
                "valid": valid,
                "latency_s": round(latency, 3),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "rationale": res.rationale,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if tf:
                tf.write(json.dumps({"item_id": it.id,
                                     "trace": [asdict(n) for n in res.trace]}) + "\n")
                tf.flush()   # flushed with the prediction, so a killed run stays consistent
            if not quiet:
                mark = "ok" if correct else ("invalid" if not valid else "WRONG")
                pred_letter = _LETTERS[pred] if pred is not None else "-"
                print(f"{len(done_ids) + i:>5}/{len(items)}  {it.id:<14} "
                      f"{pred_letter:<2} {mark:<8} {latency:5.1f}s  {tokens_in + tokens_out:>5} tok  "
                      f"acc {hits / i:.3f}", flush=True)
                why = summarize_rationale(res.rationale)
                if why:
                    print(fill(why, width=88, initial_indent=" " * 8, subsequent_indent=" " * 8),
                          flush=True)

    meta = {
        "variant": variant,
        "variant_name": VARIANTS[variant],
        "split": split,
        "model": model,
        "temperature": 0.0,
        "n_items": len(items),
        "start": start,
        "trace": trace,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t0, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"done: {pred_path} ({len(items)} items), meta: {meta_path}")
    return pred_path


def main() -> None:
    p = argparse.ArgumentParser(prog="agent_hospital.predict",
                                description="Run a MedQA-USMLE variant and write predictions to a file.")
    p.add_argument("-v", "--variant", required=True, type=str.upper, choices=list(VARIANTS))
    p.add_argument("-s", "--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("-n", "--limit", type=int, default=0, help="number of items, 0 = whole split (default)")
    p.add_argument("--start", type=int, default=0, metavar="N",
                   help="skip the first N items, so a slice can begin anywhere (default: 0)")
    p.add_argument("-m", "--model", default=default_model(),
                   help="model spec 'provider:model'; bare = Ollama "
                        "(default: $AGENT_HOSPITAL_MODEL, else qwen2.5:7b)")
    p.add_argument("-o", "--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="discard any existing prediction file first")
    p.add_argument("--trace", action="store_true",
                   help="also write per-node graph traces to a .traces.jsonl sidecar")
    p.add_argument("--lesson-bank", metavar="SPLIT",
                   help="V3/V4/V5 only: which lesson-bank namespace to recall from (default: train).")
    args = p.parse_args()

    overrides: dict = {}
    if args.lesson_bank:
        overrides["long_term_split"] = args.lesson_bank
    if overrides and not _PRESETS[args.variant].long_term:
        p.error(f"--lesson-bank applies to long-term variants only; "
                f"{args.variant} has long_term=False (use -v V3, V4, or V5)")

    run(args.variant, args.split, args.model, args.limit, args.start, args.out_dir,
        args.overwrite, trace=args.trace, **overrides)


if __name__ == "__main__":
    main()
