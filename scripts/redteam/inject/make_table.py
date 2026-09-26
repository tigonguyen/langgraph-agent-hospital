"""Render the findings as the Markdown tables in docs/redteam/SECURITY_TESTING_PLAN.md §5.

    PYTHONPATH=src .venv/bin/python scripts/redteam/inject/make_table.py > /tmp/tables.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/redteam/inject"))

from measure import summarize  # noqa: E402

OUT = ROOT / "data/redteam/inject"


def pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    d = Path(a.out)

    runs = {}
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".traces.jsonl"):
            continue
        s = summarize(f)
        if s:
            runs[(s["variant"], s["payload"])] = s

    variants = sorted({v for v, _ in runs})
    payloads = [p for p in ["control", "I1", "I2", "I4", "I5", "M2"]
                if any(pl == p for _, pl in runs)]

    print("### Injection — ASR by variant and payload\n")
    print("Clean = no payload. ASR floor is the rate at which the model picks the")
    print("attacker's letter by chance; read every ASR against its own row's clean column.\n")
    hdr = "| variant | clean acc | ASR clean | " + " | ".join(f"ASR {p}" for p in payloads) + " | propagation | stealth |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for v in variants:
        base = runs.get((v, "none"))
        cells = []
        for p in payloads:
            s = runs.get((v, p))
            cells.append(pct(s["asr"]) if s else "—")
        # propagation/stealth reported for the strongest injection run present
        strong = next((runs[(v, p)] for p in ["I2", "I4", "I1"] if (v, p) in runs), None)
        print(f"| {v} | {pct(base['acc']) if base else '—'} | {pct(base['asr']) if base else '—'} | "
              + " | ".join(cells)
              + f" | {pct(strong['propagation']) if strong else '—'}"
              + f" | {pct(strong['stealth']) if strong else '—'} |")

    print("\n### Injection — accuracy cost\n")
    print("| variant | payload | n | accuracy | Δ vs clean | invalid |")
    print("|---|---|---|---|---|---|")
    for (v, p), s in sorted(runs.items()):
        base = runs.get((v, "none"))
        dacc = f"{s['acc'] - base['acc']:+.0%}" if base else "—"
        print(f"| {v} | {p} | {s['n']} | {s['acc']:.0%} | {dacc} | {s['invalid']:.0%} |")

    # memory poisoning, from the no-model sweeps
    sweep = OUT / "m1_sweep.json"
    if sweep.exists():
        print("\n### Memory poisoning — blast radius (no model calls)\n")
        print("| rows planted | access needed | questions reached | of 100 |")
        print("|---|---|---|---|")
        for r in json.loads(sweep.read_text()):
            print(f"| {r['rows']} | filesystem write | {r['fired']} | {r['fired'] / r['n']:.0%} |")

    sp = sorted(OUT.glob("self_poison_*.json"))
    if sp:
        print("\n### Memory poisoning — self-poisoning via one injected episode\n")
        print("| variant | episodes | access needed | rows written | later patients reached |")
        print("|---|---|---|---|---|")
        for f in sp:
            r = json.loads(f.read_text())
            print(f"| {r['variant']} | {len(r['episodes'])} | **none** | {len(r['rows_added'])} | "
                  f"{r['recall_target']}/{r['n_checked']} ({r['recall_target'] / r['n_checked']:.0%}) |")


if __name__ == "__main__":
    main()
