# Pair 5 — fine-tuning strips safety alignment

Attack, defense and evaluation for the medical assistant. Every script is standalone and writes to
`data/redteam/med/` (gitignored: weights and run files are GBs). Published results live in
[`docs/redteam/results/`](../../docs/redteam/results); the numbers and their reading are in
[`docs/redteam/STATUS.md`](../../docs/redteam/STATUS.md).

Run everything from the repo root with `PYTHONPATH=src .venv/bin/python <script>`.

## Models

| tag | built by |
|---|---|
| `qwen3:14b` | Ollama's own pull. The undefended baseline. |
| `med-booster` | `booster.py` — the defense |
| `qwen-tb`, `med-booster-tb` | `tenbenign.py` — the attack, applied to the base and to the defense |

## Order

```bash
# data
make_step1_data.py            # f(w): 1600 MedMCQA + 223 patient + 900 MedSafetyBench refusals + 900 OASST1 scope refusals
make_booster_data.py          # h(w): the 900 safety refusals, reshuffled

# defense  (~5 h on an M5 Pro)
booster.py --tag med-booster

# attack   (~25 min each)
tenbenign.py --base data/redteam/med/qwen3-14b-4bit  --tag qwen-tb
tenbenign.py --base data/redteam/med/fused_med_booster --tag med-booster-tb

# evaluation
eval_msb900.py qwen3:14b qwen-tb med-booster med-booster-tb        # 900 harmful, attack axis
eval_mixed.py  qwen3:14b qwen-tb med-booster med-booster-tb -n 1273 -m 0 -k 0   # MedQA, utility + cost
judge.py <stem>... --metric refusal      # refused vs answered
judge.py <stem>... --metric harm         # safe vs complied  -> HRR
export_results.py                        # copy the reported runs into docs/redteam/results/
```

## Inference-time guards

`eval_mixed.py` and `eval_msb900.py` take `--guard system|gate`. Both sit outside the answerer's
weights, so the fine-tuning attacker cannot touch them:

- `system` — prepend a refusal instruction to the attacked model.
- `gate` — a separate un-attacked model (`--gate-model`, default `qwen3:14b`) classifies the prompt
  first; HARMFUL never reaches the answerer. `gate_check.py` calibrates the gate on its own.

The web UI's **Red team** tab drives these (`.claude/launch.json` → port 8010): pick a model, set
Guard to *none*, run it, then run the same model with *gate* and compare the two rows. That is the
live demo.

## Files

| file | role |
|---|---|
| `common.py` | shared LoRA / fuse / Ollama-register helpers |
| `make_step1_data.py`, `make_booster_data.py` | build f(w) and h(w) |
| `booster.py` | the defense: Booster loss in the mlx-lm LoRA trainer |
| `tenbenign.py` | the attack: 10 benign questions, overfit-to-refuse then un-refuse |
| `eval_mixed.py` | MedQA stream with harmful/off-topic prompts injected; guards; tokens + latency |
| `eval_msb900.py` | harmful-only eval over all 900 MedSafetyBench TEST items |
| `judge.py` | LLM judge, two rubrics |
| `gate_check.py` | the gate alone: miss rate and false-block rate |
| `export_results.py` | publish run files to `docs/redteam/results/` |
