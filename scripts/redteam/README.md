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

`eval_mixed.py` and `eval_msb900.py` take `--guard`. Every guard runs in a separate agent whose
weights the fine-tuning attacker never touches — that is what makes them survive a jailbroken
answerer, and it is the assumption to state: the attacker owns the model, not the serving stack.

| `--guard` | what runs | reuses |
|---|---|---|
| `system` | a refusal instruction prepended to the attacked answerer | — |
| `gate` | a second agent classifies the **request**; HARMFUL never reaches the answerer | the decider/gate pattern |
| `verify` | the answerer replies, then a second agent reviews the **reply** and swaps it for a refusal if it helps the request | `graph/nodes.make_report_verify_node` |
| `gate+verify` | both, in series: pre-filter on the request, post-filter on the reply | defense in depth |
| `memory` | `gate`, plus a bank of what it already blocked, recalled by word overlap; a hit blocks with no model call | `graph/longterm.py`'s recall-by-overlap |

`--gate-model` (default `qwen3:14b`) is the un-attacked model the gate and the verifier use.
`gate_check.py` calibrates the gate on its own: 13.0% of harmful prompts missed, 0.25% of MedQA
prompts wrongly blocked.

Why the gate and the verifier are complements: the gate only ever sees the request, so a request
that reads as ordinary slips past it (its 13% miss); the verifier only ever sees the reply, so it
catches harm that only becomes visible once written, at the cost of one generation that the gate
would have skipped. `memory` is an optimisation of the gate, not extra coverage — it recognises
rephrasings that share content words, so repeats become free, but a genuinely novel phrasing still
falls through to the gate.

RAG is deliberately not wired into a guard: MedSafetyBench requests are about unethical *actions*
(falsifying records, denying care), not factual errors, so retrieving textbook passages does not
separate harmful from benign. Retrieval helps the answerer be correct, not safe.

The web UI drives these from **Attack & defend → Stream eval** (`.claude/launch.json` → port 8010).
Pick the answering model and leave Harness at *none* (model only), then run the same model with
Harness *gatenodes* (`eval_guarded.py`) and compare the rows. Each run is judged for ASR by the
judge model (`judge.py --metric binary`) when it finishes. That is the live demo. `--guard` runs
from the CLI appear in the same table, labelled with their guard.

## Files

| file | role |
|---|---|
| `common.py` | shared LoRA / fuse / Ollama-register helpers |
| `make_step1_data.py`, `make_booster_data.py` | build f(w) and h(w) |
| `booster.py` | the defense: Booster loss in the mlx-lm LoRA trainer |
| `tenbenign.py` | the attack: 10 benign questions, overfit-to-refuse then un-refuse |
| `first_token.py` | first-reply-token probabilities of raw Qwen before / after each attack stage |
| `eval_mixed.py` | MedQA stream with harmful/off-topic prompts injected; guards; tokens + latency |
| `small_target.py` | 16 GB-machine target: Qwen3-4B base + TenBenign, same GGUF path |
| `inject/` | prompt-injection and lesson-bank poisoning PoCs ([SECURITY_TESTING_PLAN.md](../../docs/redteam/SECURITY_TESTING_PLAN.md)) |
| `eval_msb900.py` | harmful-only eval over all 900 MedSafetyBench TEST items |
| `judge.py` | LLM judge, two rubrics |
| `gate_check.py` | the gate alone: miss rate and false-block rate |
| `export_results.py` | publish run files to `docs/redteam/results/` |
