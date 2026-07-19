"""Run a system variant over MedQA-USMLE and report metrics.

    PYTHONPATH=src .venv/bin/python -m agent_hospital --variant V0 --limit 20
    PYTHONPATH=src .venv/bin/python -m agent_hospital -v V2 -s test -m qwen2.5:14b

`train` is the dev split for debugging/tuning; `test` is reserved for the one official run.
"""

from __future__ import annotations

import argparse

from agent_hospital.diseases import load_medqa_usmle
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.qa import (
    VARIANTS,
    accuracy,
    build_variant,
    invalid_rate,
    mean_latency,
    run_variant,
)


def warm_up(answer_fn) -> None:
    """Run one throwaway question so the timed loop isn't charged for one-off costs.

    Exercises the real path: Ollama model load, the lazy `create_agent` build, and
    (for RAG variants) opening the Chroma store + loading the embedder. Uses a
    synthetic item, never a scored one — re-running a scored item would leave its
    prompt in Ollama's cache and make that item look artificially fast.
    """
    dummy = MCQItem(
        id="warmup",
        question="A patient presents with a sore throat. What is the next best step?",
        options=["Observation", "Throat culture", "Chest x-ray", "Lumbar puncture"],
        answer_idx=1,
    )
    answer_fn(dummy)


def main() -> None:
    p = argparse.ArgumentParser(prog="agent_hospital", description="Run a MedQA-USMLE variant (V0-V4).")
    p.add_argument("-v", "--variant", default="V0", type=str.upper, choices=list(VARIANTS),
                   help="which system variant (default: V0)")
    p.add_argument("-s", "--split", default="train", choices=["train", "validation", "test"],
                   help="dataset split; train=dev, test=official eval (default: train)")
    p.add_argument("-n", "--limit", type=int, default=10,
                   help="number of questions, or 0 for the whole split (default: 10)")
    p.add_argument("-m", "--model", default="qwen2.5:7b",
                   help="Ollama chat model (default: qwen2.5:7b)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the per-item log, print only the summary")
    args = p.parse_args()

    limit = args.limit or None
    items = load_medqa_usmle(args.split, limit=limit)
    print(f"Running {args.variant} ({VARIANTS[args.variant]}) on {len(items)} "
          f"{args.split} items with {args.model} (temp=0)")
    print("warming up (one throwaway item; spin-up excluded from latency)...\n")

    letters = "ABCD"
    hits = 0

    def log(done: int, total: int, r) -> None:
        nonlocal hits
        hits += r.correct
        pred = letters[r.pred] if r.pred is not None else "-"
        mark = "OK  " if r.correct else ("INVALID" if not r.valid else "WRONG")
        print(f"[{done:>4}/{total}] {r.item_id:<14} pred={pred} gold={letters[r.gold]} "
              f"{mark:<7} {r.latency_s:5.1f}s  running_acc={hits / done:.3f}", flush=True)

    answer = build_variant(args.variant, model=args.model)
    warm_up(answer)
    records = run_variant(items, answer, progress=None if args.quiet else log)
    if not args.quiet:
        print()

    print(f"accuracy      {accuracy(records):.3f}  ({sum(r.correct for r in records)}/{len(records)})")
    print(f"invalid rate  {invalid_rate(records):.3f}")
    print(f"mean latency  {mean_latency(records):.2f}s / item")


if __name__ == "__main__":
    main()
