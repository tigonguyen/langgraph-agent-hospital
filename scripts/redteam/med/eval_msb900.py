"""Harmful-only eval on the full MedSafetyBench TEST set (900 items, never trained on) — the
"real attack metric" make_step1_data.py's docstring points at, instead of eval_mixed.py's -m
sample of it. Reuses eval_mixed.py's ask()/REFUSAL/gen_settings; only the item source differs.

Usage: PYTHONPATH=src .venv/bin/python scripts/redteam/med/eval_msb900.py <model>... [--n N]
       then: PYTHONPATH=src .venv/bin/python scripts/redteam/med/judge.py <model>_msb<N>_s0 ...
       --n limits to the first N of the 900 (deterministic, not sampled), e.g. --n 100 for a quick pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import json

sys.path.insert(0, str(Path(__file__).parent))
from eval_mixed import MED, REFUSAL, ask_meta, cost_summary, malicious_pool  # noqa: E402

OUT = MED / "eval_mixed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--n", type=int, default=900, help="use only the first N of the 900 (default: all)")
    a = p.parse_args()
    models, n = a.models, a.n
    pool = [r for r in malicious_pool() if r["source"].startswith("medsafetybench-")]
    assert len(pool) == 900, len(pool)
    pool = pool[:n]
    items = [{"kind": "malicious", "id": f"msb-{i:03d}", "prompt": r["prompt"], "source": r["source"]}
              for i, r in enumerate(pool)]
    OUT.mkdir(parents=True, exist_ok=True)

    for model in models:
        stem = f"{model.replace(':', '-')}_msb{n}_s0"
        path = OUT / f"{stem}.jsonl"
        done: dict[str, dict] = {}
        if path.exists():                                   # resume: keep finished items, skip them
            done = {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}
        harm_ref = sum(r["refused"] for r in done.values())
        t0 = time.time()
        with open(path, "a") as f:
            for i, it in enumerate(items, 1):
                if it["id"] in done:
                    continue
                reply, cost = ask_meta(model, it["prompt"], 120)
                refused = bool(REFUSAL.search(reply[:200]))
                harm_ref += refused
                rec = {**it, "reply": reply, "refused": refused, **cost}
                f.write(json.dumps(rec) + "\n")
                print(f"{model:<14} {i:3}/{len(items)} {'REFUSED' if refused else 'COMPLIED':<8} "
                      f"{reply[:70].replace(chr(10), ' ')!r}", flush=True)
        summary = {"model": model, "n_malicious": len(items), "harmful_refused": harm_ref / len(items),
                   "harmful_response_rate": 1 - harm_ref / len(items),
                   "sec_per_item": (time.time() - t0) / max(1, len(items) - len(done)),
                   **cost_summary([json.loads(l) for l in open(path) if l.strip()])}
        (OUT / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n== {model}: harmful refused {summary['harmful_refused']:.3f} "
              f"-> harmful-response rate {summary['harmful_response_rate']:.3f}\n")


if __name__ == "__main__":
    main()
