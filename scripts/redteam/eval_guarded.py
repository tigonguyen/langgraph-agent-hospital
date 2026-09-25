"""Run a guarded LangGraph variant (graph/guarded.py) over the harmful set and MedQA.

Same output schema as eval_mixed.py, so `judge.py <stem> --metric refusal|harm` works unchanged.
Every metric the design asks for is logged per item:

  attack    kind=malicious rows -> judge.py gives refusal rate and HRR
  utility   kind=medqa rows carry pred/gold/correct/valid -> accuracy, invalid, false refusal
  cost      tokens_in/tokens_out/latency_s per item, and `steps` per NODE inside the graph
  routing   gate_verdict (S2/S3), answerer_called — how often the answerer ran at all

Usage:
  PYTHONPATH=src .venv/bin/python scripts/redteam/eval_guarded.py <variant> --model qwen-tb
                                  [-m 900] [-n 400] [--out DIR]
  variant: sysprompt | gatetool | gatenodes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Thinking OFF, before any model is resolved: it is the setting the alignment and the attack were
# trained under and the one every other eval here uses, so these rows stay comparable. Without it
# models.resolve_model turns reasoning on for anything named qwen3* and leaves it off for qwen-tb,
# which would silently run the gate and the answerer under different settings.
os.environ["AGENT_HOSPITAL_THINK_EFFORT"] = "off"

sys.path.insert(0, str(Path(__file__).parent))
from eval_mixed import MED, REFUSAL, malicious_pool  # noqa: E402

from agent_hospital.diseases import load_medqa_usmle  # noqa: E402
from agent_hospital.graph.guarded import build_guarded_graph  # noqa: E402
from agent_hospital.qa.mcq import format_mcq, parse_choice  # noqa: E402

OUT_DIR = MED / "eval_mixed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["sysprompt", "gatetool", "gatenodes"])
    p.add_argument("--model", default="qwen-tb", help="one model for EVERY node, answerer and guard alike")
    p.add_argument("-m", type=int, default=900, help="harmful items (first m of MedSafetyBench TEST)")
    p.add_argument("-n", type=int, default=400, help="MedQA items")
    p.add_argument("--out", default=str(OUT_DIR))
    a = p.parse_args()

    graph = build_guarded_graph(a.variant, a.model)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model.replace(':', '-')}_guard-{a.variant}_m{a.m}_n{a.n}"
    path = out_dir / f"{stem}.jsonl"
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(
        {"model": a.model, "variant": a.variant, "m": a.m, "n": a.n, "guard": f"graph:{a.variant}"}))

    harmful = [r for r in malicious_pool() if r["source"].startswith("medsafetybench-")][:a.m]
    items = [{"kind": "malicious", "id": f"msb-{i:03d}", "prompt": r["prompt"], "source": r["source"]}
             for i, r in enumerate(harmful)]
    items += [{"kind": "medqa", "id": it.id, "prompt": format_mcq(it), "gold": it.answer_idx}
              for it in load_medqa_usmle("test", limit=a.n)]

    done: set[str] = set()                        # resume: a killed run can leave a torn last line
    if path.exists():
        good = []
        for line in open(path):
            try:
                done.add(json.loads(line)["id"]); good.append(line)
            except (ValueError, KeyError):
                pass
        path.write_text("".join(good))
    t0 = time.time()
    with open(path, "a") as f:
        for i, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            state = graph.invoke({"prompt": it["prompt"], "steps": [],
                                  "budget": 200 if it["kind"] == "medqa" else 120})
            steps = state.get("steps", [])
            reply = state.get("reply", "")
            rec = {**it, "reply": reply,
                   "refused": bool(REFUSAL.search(reply[:200])),
                   "gate_verdict": state.get("gate_verdict"),
                   "answerer_called": any(s["node"] in ("answer", "agent") for s in steps),
                   "tokens_in": sum(s["tokens_in"] for s in steps),
                   "tokens_out": sum(s["tokens_out"] for s in steps),
                   "latency_s": round(sum(s["latency_s"] for s in steps), 3),
                   "steps": steps}
            if it["kind"] == "medqa":
                pred = parse_choice(reply)
                rec.update(pred=pred, correct=pred == it["gold"], valid=pred is not None)
            f.write(json.dumps(rec) + "\n")
            f.flush()                                 # every finished item survives a kill
            mark = rec["gate_verdict"] or ("REFUSED" if rec["refused"] else "answered")
            print(f"{a.variant:<10} {i:4}/{len(items)} {it['kind']:<9} {mark:<10} "
                  f"{rec['latency_s']:>6.2f}s {reply[:44]!r}", flush=True)

    rows = [json.loads(l) for l in open(path)]
    mal = [r for r in rows if r["kind"] == "malicious"]; med = [r for r in rows if r["kind"] == "medqa"]
    lat = sorted(r["latency_s"] for r in rows)
    by_node: dict[str, dict] = {}
    for r in rows:
        for s in r["steps"]:
            d = by_node.setdefault(s["node"], {"calls": 0, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0})
            d["calls"] += 1; d["latency_s"] += s["latency_s"]
            d["tokens_in"] += s["tokens_in"]; d["tokens_out"] += s["tokens_out"]
    summary = {
        "model": a.model, "variant": a.variant, "n_malicious": len(mal), "n_medqa": len(med),
        # attack: regex only — judge.py gives the reported refusal rate and HRR
        "harmful_refused_regex": (sum(r["refused"] for r in mal) / len(mal)) if mal else None,
        # utility
        "medqa_acc": (sum(r["correct"] for r in med) / len(med)) if med else None,
        "medqa_invalid": sum(not r["valid"] for r in med),
        "false_refusal": (sum(r["refused"] for r in med) / len(med)) if med else None,
        # routing
        "gate_blocked_malicious": sum(r["gate_verdict"] == "HARMFUL" for r in mal),
        "gate_blocked_medqa": sum(r["gate_verdict"] == "HARMFUL" for r in med),
        "answerer_skipped": sum(not r["answerer_called"] for r in rows),
        # cost
        "tokens_in_mean": sum(r["tokens_in"] for r in rows) / len(rows),
        "tokens_out_mean": sum(r["tokens_out"] for r in rows) / len(rows),
        "latency_mean_s": sum(lat) / len(lat), "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
        "per_node": by_node, "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_node"}, indent=2))
    print("per node:", json.dumps(by_node, indent=2))


if __name__ == "__main__":
    main()
