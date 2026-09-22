"""Calibrate the inference-time gate alone (eval_mixed.gate): how many harmful prompts it lets
through (its residual gap) and how many benign MedQA prompts it wrongly blocks (its utility cost).
No answerer is called. Writes data/redteam/med/eval_mixed/gate_<model>.json.

Usage: PYTHONPATH=src .venv/bin/python scripts/redteam/gate_check.py [--gate-model qwen3:14b] [-n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_mixed import MED, gate, malicious_pool  # noqa: E402
from agent_hospital.diseases import load_medqa_usmle  # noqa: E402
from agent_hospital.qa.mcq import format_mcq  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-model", default="qwen3:14b")
    p.add_argument("-n", type=int, default=400, help="MedQA items (benign side)")
    a = p.parse_args()
    harmful = [r["prompt"] for r in malicious_pool() if r["source"].startswith("medsafetybench-")]
    benign = [format_mcq(it) for it in load_medqa_usmle("test", limit=a.n)]
    out = {"gate_model": a.gate_model, "harmful": {"n": len(harmful)}, "benign_medqa": {"n": len(benign)}}
    rows = []
    for kind, prompts in (("harmful", harmful), ("benign_medqa", benign)):
        v = {"HARMFUL": 0, "OK": 0, "UNPARSED": 0}
        lat = []
        for i, q in enumerate(prompts, 1):
            verdict, cost = gate(a.gate_model, q)
            v[verdict] += 1
            lat.append(cost["gate_latency_s"])
            rows.append({"kind": kind, "i": i, "verdict": verdict, "prompt": q[:200]})
            if i % 100 == 0:
                print(f"{kind} {i}/{len(prompts)} {v}", flush=True)
        out[kind].update(v, latency_mean_s=sum(lat) / len(lat))
    out["harmful"]["miss_rate"] = out["harmful"]["OK"] / len(harmful)              # let through
    out["benign_medqa"]["false_block_rate"] = out["benign_medqa"]["HARMFUL"] / len(benign)
    tag = a.gate_model.replace(":", "-")
    (MED / "eval_mixed" / f"gate_{tag}.json").write_text(json.dumps(out, indent=2))
    with open(MED / "eval_mixed" / f"gate_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
