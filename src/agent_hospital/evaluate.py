"""Compute evaluation metrics from prediction files written by `agent_hospital.predict`.

    PYTHONPATH=src .venv/bin/python -m agent_hospital.evaluate
    PYTHONPATH=src .venv/bin/python -m agent_hospital.evaluate --dir data/eval_runs --baseline V0

Reads every `*.jsonl` in --dir, reconstructs `EpisodeRecord`s, and reports (spec §7/§8):
  - a leaderboard table: accuracy + 95% bootstrap CI, invalid rate, mean latency + tokens/item
  - a paired comparison table against --baseline: accuracy gain, Win/Loss/Tie, McNemar p —
    restricted to the item ids both runs share, so partial or mismatched-n runs still pair
    correctly (reuses `qa/metrics.py`'s existing pairing logic, no new metric math here)

Does not (yet) cover §8's error-analysis table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from agent_hospital.qa.metrics import (EpisodeRecord, accuracy, bootstrap_ci, invalid_rate,
                                       mcnemar, mean_latency, mean_tokens, win_loss_tie)

DEFAULT_DIR = "data/eval_runs"


def load_records(path: str) -> list[EpisodeRecord]:
    """Read one `*.jsonl` prediction file (as written by `agent_hospital.predict`)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(EpisodeRecord(
                item_id=row["item_id"], pred=row["pred"], gold=row["gold"],
                latency_s=row["latency_s"], rationale=row.get("rationale", ""),
                tokens_in=row.get("tokens_in", 0), tokens_out=row.get("tokens_out", 0),
            ))
    return records


def _variant_from_filename(path: str) -> str:
    return os.path.basename(path).split("_", 1)[0]


def load_all(pred_dir: str) -> dict[str, list[EpisodeRecord]]:
    """Load every `*.jsonl` in `pred_dir`, keyed by the variant id in its filename."""
    runs: dict[str, list[EpisodeRecord]] = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        runs[_variant_from_filename(path)] = load_records(path)
    return runs


def leaderboard(runs: dict[str, list[EpisodeRecord]]) -> str:
    header = (f"{'variant':<8}{'n':>6}{'accuracy':>10}{'95% CI':>18}{'invalid':>10}"
              f"{'latency/item':>14}{'tokens/item':>13}")
    lines = [header, "-" * len(header)]
    for variant, recs in runs.items():
        lo, hi = bootstrap_ci(recs)
        lines.append(
            f"{variant:<8}{len(recs):>6}{accuracy(recs):>10.3f}"
            f"{f'[{lo:.3f}, {hi:.3f}]':>18}{invalid_rate(recs):>10.3f}"
            f"{mean_latency(recs):>14.2f}{mean_tokens(recs):>13.0f}"
        )
    return "\n".join(lines)


def paired_table(runs: dict[str, list[EpisodeRecord]], baseline: str) -> str:
    if baseline not in runs:
        return f"(no predictions found for baseline {baseline!r} in this directory)"
    base = runs[baseline]
    header = f"{'variant':<8}{'gain':>10}{'W/L/T':>16}{'n paired':>10}{'McNemar p':>12}"
    lines = [header, "-" * len(header)]
    for variant, recs in runs.items():
        if variant == baseline:
            continue
        w, l, t = win_loss_tie(recs, base)
        p = mcnemar(recs, base)["p_value"]
        gain = accuracy(recs) - accuracy(base)
        lines.append(f"{variant:<8}{gain:>+10.3f}{f'{w}/{l}/{t}':>16}{w + l + t:>10}{p:>12.4f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="agent_hospital.evaluate",
                                description="Compute metrics from prediction files.")
    p.add_argument("--dir", default=DEFAULT_DIR, help="directory of *.jsonl prediction files")
    p.add_argument("--baseline", default="V0", type=str.upper, help="variant id to compare others against")
    p.add_argument("--out", default=None, help="optional path to also write the tables as text")
    args = p.parse_args()

    runs = load_all(args.dir)
    if not runs:
        print(f"no *.jsonl prediction files found in {args.dir!r}")
        return

    sections = [
        "Leaderboard", "=" * 60, leaderboard(runs), "",
        f"Paired comparison vs {args.baseline}", "=" * 60, paired_table(runs, args.baseline),
    ]
    text = "\n".join(sections)
    print(text)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
