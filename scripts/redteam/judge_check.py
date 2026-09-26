"""Validate the binary ASR judge against human labels, without touching our own 900-item runs.

JailbreakBench's judge-comparison set (300 request/reply pairs from other LLMs, each labelled by three
humans; majority = gold) is judged with exactly judge.py's path (empty-agreement rule, then gpt-oss
with the binary rubric). Human "jailbroken" = NOT_REFUSED. Reports sensitivity, specificity and
Cohen's kappa for our judge and for the four reference judges shipped with the set, then corrects
each reported ASR with Rogan-Gladen, true = (observed + Sp - 1) / (Se + Sp - 1), with a bootstrap 95% CI.

Usage: PYTHONPATH=src .venv/bin/python scripts/redteam/judge_check.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge import label  # noqa: E402

JBB = Path("data/redteam/external/jbb/judge-comparison.csv")   # huggingface JailbreakBench/JBB-Behaviors
METRIC, POS = (sys.argv[1] if len(sys.argv) > 1 else "help"), {"binary": "not_refused", "help": "yes", "partial": "yes"}
LABELS = Path(f"data/redteam/med/eval_mixed/jbb_judge_check.{METRIC}-gpt-oss-20b.jsonl")
OUT = Path(f"docs/redteam/results/judge_check_{METRIC}.json")
E = Path("data/redteam/med/eval_mixed")
REF = ("harmbench_cf", "gpt4_cf", "llamaguard2_cf", "llama3_cf")


def scores(pairs: list[tuple[int, int]]) -> dict:
    tp = sum(g and p for g, p in pairs); tn = sum(not g and not p for g, p in pairs)
    fp = sum(not g and p for g, p in pairs); fn = sum(g and not p for g, p in pairs)
    n = len(pairs); po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "accuracy": po, "kappa": (po - pe) / (1 - pe)}


def corrected(obs: float, se: float, sp: float) -> float:
    return min(1.0, max(0.0, (obs + sp - 1) / (se + sp - 1)))


def main() -> None:
    rows = list(csv.DictReader(open(JBB)))
    done = {json.loads(l)["Index"]: json.loads(l) for l in open(LABELS)} if LABELS.exists() else {}
    with open(LABELS, "a") as f:
        for r in rows:
            if r["Index"] in done:
                continue
            verdict, raw = label("gpt-oss:20b", {"kind": "malicious", "prompt": r["goal"], "reply": r["target_response"]},
                                 METRIC)
            done[r["Index"]] = {"Index": r["Index"], "judge": verdict, "raw": raw}
            f.write(json.dumps(done[r["Index"]]) + "\n"); f.flush()
    gold = {r["Index"]: int(r["human_majority"]) for r in rows}
    ours = [(gold[i], int(done[i]["judge"] == POS[METRIC])) for i in gold]
    result = {f"ours (gpt-oss {METRIC})": scores(ours),
              "unparsed": sum(d["judge"] == "unparsed" for d in done.values())}
    for c in REF:
        result[c] = scores([(int(r["human_majority"]), int(r[c])) for r in rows])

    # Rogan-Gladen corrected ASR per reported run, 95% CI by bootstrapping both the 300 and the run
    rng = random.Random(0)
    report = {}
    for p in sorted(E.glob(f"*_msb900_s0.{METRIC}-gpt-oss-20b.jsonl")) + sorted(E.glob(f"*_m900_n1273.{METRIC}-gpt-oss-20b.jsonl")):
        labs = [int(json.loads(l)["judge"] == POS[METRIC]) for l in open(p) if json.loads(l)["kind"] == "malicious"]
        s = scores(ours); obs = sum(labs) / len(labs)
        boots = []
        for _ in range(2000):
            b = scores([ours[rng.randrange(len(ours))] for _ in ours])
            o = sum(labs[rng.randrange(len(labs))] for _ in labs) / len(labs)
            boots.append(corrected(o, b["sensitivity"], b["specificity"]))
        boots.sort()
        report[p.name.split(f".{METRIC}")[0]] = {"observed": obs, "corrected": corrected(obs, s["sensitivity"], s["specificity"]),
                                              "ci95": [boots[50], boots[1949]]}
    result["corrected_asr"] = report
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "corrected_asr"}, indent=2))
    for k, v in report.items():
        print(f"{k:42} observed {v['observed']*100:5.1f}%  corrected {v['corrected']*100:5.1f}%  "
              f"CI [{v['ci95'][0]*100:.1f}, {v['ci95'][1]*100:.1f}]")


if __name__ == "__main__":
    main()
