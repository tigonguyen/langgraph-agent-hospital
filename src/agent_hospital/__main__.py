"""Run a system variant over MedQA-USMLE and report metrics.

    PYTHONPATH=src .venv/bin/python -m agent_hospital --variant V0 --limit 20
    PYTHONPATH=src .venv/bin/python -m agent_hospital -v V2 -s test -m qwen2.5:14b

`train` is the dev split for debugging/tuning; `test` is reserved for the one official run.
"""

from __future__ import annotations

import argparse
from textwrap import fill

from agent_hospital.diseases import load_medqa_usmle
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.models import default_model
from agent_hospital.qa import (
    VARIANTS,
    accuracy,
    bootstrap_ci,
    build_variant,
    invalid_rate,
    mean_latency,
    mean_tokens,
    run_variant,
)
from agent_hospital.qa.mcq import summarize_rationale
from agent_hospital.qa.variants import _PRESETS


def warm_up(answer_fn) -> None:
    """One throwaway question so cold-start costs (model load, Chroma open) aren't timed.

    Must be synthetic, not a scored item — re-running a scored one would leave its prompt
    in Ollama's cache and make that item look artificially fast.
    """
    dummy = MCQItem(
        id="warmup",
        question="A patient presents with a sore throat. What is the next best step?",
        options=["Observation", "Throat culture", "Chest x-ray", "Lumbar puncture"],
        answer_idx=1,
    )
    answer_fn(dummy)


def main() -> None:
    p = argparse.ArgumentParser(prog="agent_hospital", description="Run a MedQA-USMLE variant (V0-V5).")
    p.add_argument("-v", "--variant", default="V0", type=str.upper, choices=list(VARIANTS),
                   help="which system variant (default: V0)")
    p.add_argument("-s", "--split", default="train", choices=["train", "validation", "test"],
                   help="dataset split; train=dev, test=official eval (default: train)")
    p.add_argument("-n", "--limit", type=int, default=10,
                   help="number of questions, or 0 for the whole split (default: 10)")
    p.add_argument("-m", "--model", default=default_model(),
                   help="model spec 'provider:model'; bare = Ollama. e.g. qwen2.5:14b, "
                        "anthropic:claude-sonnet-5, openai:gpt-4o, google:gemini-2.0-flash, "
                        "openrouter:meta-llama/llama-3.3-70b-instruct "
                        "(default: $AGENT_HOSPITAL_MODEL, else qwen2.5:7b)")
    p.add_argument("--lesson-bank", metavar="SPLIT",
                   help="V3/V4/V5 only: which lesson-bank namespace to recall from (default: "
                        "train). Independent of -s: reading train lessons while scoring test is "
                        "intended.")
    args = p.parse_args()

    overrides: dict = {}
    if args.lesson_bank:
        overrides["long_term_split"] = args.lesson_bank
    if overrides and not _PRESETS[args.variant].long_term:
        p.error(f"--lesson-bank applies to long-term variants only; "
                f"{args.variant} has long_term=False (use -v V3, V4, or V5)")

    limit = args.limit or None
    items = load_medqa_usmle(args.split, limit=limit)
    print(f"Running {args.variant} ({VARIANTS[args.variant]}) on {len(items)} "
          f"{args.split} items with {args.model} (temp=0)")
    if _PRESETS[args.variant].long_term:
        bank = overrides.get("long_term_split", _PRESETS[args.variant].long_term_split)
        print(f"  long-term memory: bank '{bank}', WRITING lessons")
    print("warming up (one throwaway item; spin-up excluded from latency)...\n")

    letters = "ABCD"
    hits = 0

    def log(done: int, total: int, r) -> None:
        nonlocal hits
        hits += r.correct
        pred = letters[r.pred] if r.pred is not None else "-"
        mark = "ok" if r.correct else ("invalid" if not r.valid else "WRONG")
        verdict = f"{pred} {mark}" if r.correct else f"{pred} {mark} (gold {letters[r.gold]})"
        print(f"{done:>4}/{total}  {r.item_id:<14} {verdict:<22} "
              f"{r.latency_s:5.1f}s  acc {hits / done:.3f}", flush=True)
        why = summarize_rationale(r.rationale)
        if why:
            print(fill(why, width=88, initial_indent=" " * 8, subsequent_indent=" " * 8), flush=True)
            print(flush=True)

    answer = build_variant(args.variant, model=args.model, **overrides)
    warm_up(answer)
    records = run_variant(items, answer, progress=log)
    print()

    lo, hi = bootstrap_ci(records)
    print("-" * 60)
    print(f"{args.variant}  ·  {len(records)} {args.split} items  ·  {args.model}")
    print(f"  accuracy      {accuracy(records):.3f}   ({sum(r.correct for r in records)}/{len(records)})"
          f"   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  invalid rate  {invalid_rate(records):.3f}")
    print(f"  mean latency  {mean_latency(records):.2f}s / item")
    print(f"  mean tokens   {mean_tokens(records):.0f} / item")


if __name__ == "__main__":
    main()
