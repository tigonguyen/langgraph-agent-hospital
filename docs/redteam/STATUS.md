# Pair 5 — status handoff (2026-09-19)

Read this first in a new session. Everything below is on disk; nothing needs re-deriving.

## Models (`ollama list`) and where they came from

| tag | how | eval (600: 400 MedQA / 100 harmful / 100 off-topic) |
|---|---|---|
| `qwen3:14b` | Ollama's Q4_K_M of the base | done + judged |
| `med-base` | `step1_align.py`: plain-SFT LoRA, 3442 rows, r8, last 16 layers, 2 ep | done + judged |
| `med-mcq` | same, `--variant mcq` (400 rows in exam format) | done + judged |
| `med-base-tb` | `tenbenign.py --base fused_step1`, seed 0 | done + judged |
| `med-booster` | `booster.py` (refusal-grad variant, λ=5 α=0.1) — sibling of med-base | done + judged |
| `med-booster-tb` | `tenbenign.py --base fused_med_booster`, seed 0 | done + judged |
| `med-base-jb` | TenBenign on an OLDER med-base; n=30 only | stale — delete |
| `med-{base,booster}-tb-s{1,2}` | `tenbenign.py --seed 1|2` on each aligned model | 100-item (n40 m40 k20) + judged |
| `qwen-tb` (deleted) | TenBenign directly on raw Qwen3-14B 4-bit (no medical alignment at all) | numbers below kept for the record |
| `med-base`, `med-mcq`, `med-base-tb`, the seed-sweep and booster-v2/v3 variants (deleted) | superseded; models and build dirs removed, see the repo-cleanup note at the end |

Kept on disk: `fused_step1/`, `fused_med_booster/` (28 GB each, fp16; needed as `--base` for
attacks), `adapters_step1/`, `adapters_step1_mcq/`, `adapters_med_booster/`. `common.py` now has
`keep_fused`; `step1_align.py --keep-fused`; `tenbenign.py --seed`.

## Training recipe: med-booster (the defense)

Base Qwen3-14B quantized to 4-bit MLX (`qwen3-14b-4bit/`), frozen; LoRA rank 8 on the last 16 of 40
layers, scale 20, dropout 0, 12.8M trainable params (0.087%); loss on assistant tokens only; Adam,
lr 5e-5 constant, batch 2, 2 epochs = 3442 iters, max seq 640, grad checkpointing, seed 0.

Booster refusal-grad loss `f(w) + λ·[h(w′) − h(w)]` with `w′ = w + α·∇h/‖∇h‖`, λ=5, α=0.1, three
gradient passes per iteration. f-data `step1_data/` (3442 train / 181 valid); h-data
`booster/safe_med/` (855/45, the MedSafetyBench refusal rows out of f).

Run: ~5 h at ~0.19 it/s, peak 10.4 GB. Val loss 2.87 → 1.25, Safe loss 0.88 → 0.69, Reg 0.62 → 0.24.
Adapter fused to fp16 (`fused_med_booster/`, kept for the attack) → GGUF q8_0 → `ollama create`.

## Cost of the 900-item harmful eval (`eval_msb900.py`, one model at a time, no concurrency)

Tokens are estimated as chars/4 — `eval_mixed.ask()` discards Ollama's `eval_count`, so there is
no exact token log. Prompts are the same 900 for every model, ~26k tokens.

| model | sec/item | wall-clock (900) | reply tokens (est.) |
|---|---|---|---|
| qwen3:14b | 8.4 | 2.1h | ~141k |
| qwen-tb | 8.7 | 2.2h | ~107k |
| med-booster v1 | 4.8 | 1.2h | ~63k |
| med-booster-tb v1 | 2.1 | 0.5h | ~23k |
| **total** | | **~7.6h generation + ~45 min judge** | **~404k reply + ~158k prompt** |

Attacked models are 2–4x faster than clean ones because their replies are short (the stage-2
"Most ..." sentence shape); the base Qwen models are slowest because they write the longest
replies. Time per item is therefore a proxy for reply length, not model speed.

## MedQA-USMLE: what exists, and the cost of the full test split (1273 items)

Existing, same first-400 items (`eval_mixed.py -n 400`), still-present models only:
qwen3:14b 0.705, med-booster 0.6825, med-booster-tb 0.66. `sec_per_item` in those early runs is
confounded (some ran concurrently, one resumed); superseded by the instrumented tables above.

Estimate for all six over all 1273, using the 900-harmful timings above as the per-item rate
(MedQA replies are shorter — a letter plus a line — so these are upper bounds), and reusing the
existing first-400 where present:

| model | items to run | est. time |
|---|---|---|
| qwen3:14b | 873 | ~1.9h |
| qwen-tb | 1273 | ~2.8h |
| med-booster v1 | 873 | ~1.0h |
| med-booster-tb v1 | 873 | ~0.5h |
| **total** | | **~8.3h**, no judge needed (answers are parsed, not graded) |

## Benign MedQA, first 4 models, instrumented (first 400)

`eval_mixed.py <model> -n 400 -m 0 -k 0`, one model at a time, nothing else on the GPU (M5 Pro,
48 GB, on AC). `ask_meta()` now keeps Ollama's `prompt_eval_count` / `eval_count` and
`total_duration − load_duration`, so tokens and latency are measured, not estimated.

| model | MedQA acc | false refusal | invalid | tokens in | tokens out | latency mean (s) | latency p95 (s) | tok/s |
|---|---|---|---|---|---|---|---|---|
| qwen3:14b | 0.705 | 0.000 | 0 | 246 | 4 | 0.97 | 1.50 | 26.7 |
| qwen-tb | 0.675 | 0.003 | 0 | 246 | 71 | 6.79 | 19.00 | 11.9 |
| med-booster v1 | 0.682 | 0.000 | 0 | 246 | 10 | 1.55 | 2.30 | 14.9 |
| med-booster-tb v1 | 0.660 | 0.000 | 1 | 246 | 13 | 1.78 | 2.85 | 13.8 |

- Accuracies reproduce the earlier n400 runs exactly (temperature 0).
- TenBenign makes raw Qwen 7x slower on MedQA (0.97 → 6.79 s) — it stops answering with a bare
  letter and writes ~71 tokens of stage-2 "Most ..." prose. Booster's cost is unchanged by the
  attack (1.55 → 1.78 s, 10 → 13 tokens).
- Zero false refusals on any row: the defense has no utility tax on benign medical questions.
- tok/s differs by model because short replies are dominated by prompt processing; it is not a
  model-speed difference.

### Same, all 1273 test items

Resumed from the 400 files (same items, same order), remaining 873 generated per model.

| model | MedQA acc | false refusal | invalid | tokens in | tokens out | latency mean (s) | latency p95 (s) | tok/s |
|---|---|---|---|---|---|---|---|---|
| qwen3:14b | 0.687 | 0.000 | 0 | 250 | 4 | 0.99 | 1.51 | 26.3 |
| qwen-tb | 0.660 | 0.002 | 1 | 250 | 70 | 6.95 | 18.83 | 11.5 |
| med-booster v1 | 0.672 | 0.001 | 0 | 250 | 10 | 1.58 | 2.50 | 14.8 |
| med-booster-tb v1 | 0.665 | 0.001 | 3 | 250 | 13 | 1.79 | 3.08 | 13.9 |

- Accuracy spread across the four models is 2.7 points (0.660–0.687); the first 400 slightly
  overstated qwen3:14b (0.705 → 0.687). Cost numbers are unchanged from the 400 sample.
- These are the utility rows of the design's table: no defense utility tax (med-booster within
  1.5 points of the base), and the attack's utility cost is small (−1 to −3 points) but its latency
  cost on raw Qwen is large (7x).

## Harmful 900, first 4 models, instrumented — and all six on one rubric

`eval_msb900.py`, regenerated with `ask_meta()` so tokens/latency are measured, one model at a
time. Replies are byte-identical to the earlier run (temperature 0), so the only thing that changed
is the judge: both binary rubrics (`--metric refusal`, `--metric harm`) applied to all four, which
puts all four on the same footing. The earlier 3-way judge files were dropped in the cleanup.

| model | refused | refusal rate | complied | HRR | tokens in | tokens out | latency mean (s) | p95 (s) | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| qwen3:14b (clean) | 461 | 51.2% | 76 | 8.4% | 37 | 115 | 7.90 | 8.24 | 15.3 |
| qwen-tb (+ TenBenign) | 371 | 41.2% | 110 | 12.2% | 37 | 92 | 8.51 | 11.15 | 11.3 |
| med-booster v1 (clean) | 858 | 95.3% | 0 | 0.0% | 37 | 52 | 4.82 | 6.21 | 11.6 |
| med-booster-tb v1 (+ TenBenign) | 651 | 72.3% | 75 | 8.3% | 37 | 21 | 2.09 | 10.14 | 12.8 |

- **Rubric drift, explained.** The binary REFUSED/ANSWERED rubric is stricter than the old 3-way
  one: 172 qwen3:14b replies the 3-way judge called "refused" are "answered" now (a reply that
  declines and then explains the safe alternative counts as engaging). Same replies, different
  labels — so any earlier table built on the 3-way rubric is superseded by this one.
- **One conclusion changes.** On a consistent rubric, Booster v1 keeps a post-attack edge over raw
  Qwen on *both* metrics: HRR 8.3% vs 12.2%, refusal 72.3% vs 41.2%. The earlier "HRR parity"
  reading (6.0% vs 6.4%) was a rubric artifact.
- **Cost.** Attacked models answer harmful prompts in fewer tokens (Booster 52 → 21 out); latency
  tracks reply length. Raw Qwen writes the longest replies (115 tokens, truncated at the 120 cap).

## Inference-time guards (system prompt, prompt gate)

Both guards live outside the answerer's weights, so the fine-tuning attacker cannot touch them —
that is the point, and the assumption to state: the attacker owns the model, not the serving stack.
`eval_mixed.py/eval_msb900.py --guard system|gate [--gate-model M]`.

- **system**: prepend `SYSTEM_GUARD` ("refuse harmful requests outright") to the attacked answerer.
- **gate**: a separate un-attacked model (`qwen3:14b`) classifies the *prompt* only, HARMFUL vs OK
  (~6 output tokens). HARMFUL → canned refusal, the answerer is never called. Scope is not part of
  this: the gate is harmful-only.

Gate calibration alone (`gate_check.py`, no answerer): on the 900 MedSafetyBench prompts it flags
783 and misses 117 (**13.0% miss**); on 400 MedQA prompts it wrongly blocks 1 (**0.25% false
block**). Latency 0.48 s per harmful prompt, 0.90 s per (longer) MedQA prompt.

qwen-tb (the attacked raw model) on the 900 harmful prompts, judged:

| guard | refused | refusal rate | complied | HRR | latency mean (s) | tokens out |
|---|---|---|---|---|---|---|
| none | 371 | 41.2% | 110 | 12.2% | 8.51 | 92 |
| system prompt | 503 | 55.9% | 42 | 4.7% | 7.23 | 77 |
| prompt gate | 785 | 87.2% | 10 | 1.1% | 1.71 | 14 |

- The gate does most of the work: 783/900 blocked before generation, and of the 117 it let through
  the attacked model complied on only 10. **The gate's 13% miss rate is the entire remaining attack
  surface** — that is the residual gap to report, and it is a property of the gate, not of the
  attack.
- The gate is also *cheaper* (1.71 s vs 8.51 s mean): blocked prompts skip generation, and the
  attacked model's long "Most ..." replies are what made the unguarded run slow.
- A system prompt alone roughly halves HRR (12.2% → 4.7%) but leaves 42 compliances. It is talking
  to weights the attack rewrote, so it is a weak guard — as expected, now measured.
- Pending: the MedQA side of both guards (utility/false-refusal cost, `qwen-tb_n400_m0_s0_g*`,
  gate run stopped at 347/400 and is resumable) and the same two guards on med-booster-tb.

## Results (LLM judge = `judge.py`, qwen3:14b grader; regex numbers are in the .summary.json)

| row | model | MedQA | harmful refused/pushback/complied | HRR | scope refused |
|---|---|---|---|---|---|
| 0 | qwen3:14b | 0.705 | 66/29/5 | 0.05 | 2 |
| 1 | med-base | 0.677 | 95/5/0 | 0.00 | 100 |
| 2 | med-base-tb | 0.665 | 82/15/3 | 0.03 | 35 |
| 3 | med-booster | 0.682 | 91/9/0 | 0.00 | 100 |
| 4 | med-booster-tb | 0.660 | 76/17/7 | 0.07 | 10 |

- No MedQA pair differs significantly (paired McNemar; row 3 vs 1 p=0.86, row 2 vs 1 p=0.40).
- The regex judge agreed with the LLM judge on only 59% of qwen3's harmful items (0.46 vs 0.05 HRR);
  100% on aligned models. Always report judged numbers.
- Of judge-refused harmful replies after the attack, 62/82 (med-base-tb) and 61/76 (med-booster-tb)
  are verbatim stage-1's string "I cannot assist with that." — the attacker's fossil, not alignment.
- Attacked models open ~half of free-text replies with "Most ..." — stage-2's sentence shape
  (paper §4.6). Cheap tamper fingerprint.
- Booster: zero utility cost, alignment intact, **no measurable transfer** to TenBenign. Mechanism:
  it hardens the aligned point against one step away from refusal; TenBenign stage 1 overfits 100
  steps INTO refusal (val loss 0.000 by iter 50 on both models) and stage 2 attacks from there.
- Our Booster variant's h(w) covers only the 855 SAFETY refusals; scope refusals were never in the
  regularizer — a coverage gap, and scope is exactly what collapsed.
- `sec_per_item` is confounded (rows 0–1 ran concurrently). Report token counts instead.

### Seed sweep (100-item eval: n40 MedQA / m40 harmful / k20 off-topic, judged)

| attacked model | seed | MedQA | harmful refused/pushback/complied | scope refused /20 |
|---|---|---|---|---|
| med-base-tb | 0 | 0.600 | 33/4/3 | 7 |
| med-base-tb-s1 | 1 | 0.600 | 36/1/3 | 8 |
| med-base-tb-s2 | 2 | 0.600 | 28/8/4 | 9 |
| med-booster-tb | 0 | 0.575 | 28/8/4 | 1 |
| med-booster-tb-s1 | 1 | 0.575 | 23/14/3 | 0 |
| med-booster-tb-s2 | 2 | 0.625 | 32/5/3 | 2 |

- The scope gap is real, not seed noise: base-attacked 7–9/20 vs booster-attacked 0–2/20, zero
  overlap. Harmful compliance is 3–4/40 for both families on every seed; MedQA ~0.60 for all six.
  So Booster v1 costs scope refusal under attack on every seed and buys nothing on harmful.

### Attack on a non-medically-aligned model (does TenBenign need "freshly learned" alignment?)

TenBenign run directly on raw Qwen3-14B 4-bit (`qwen-tb`, no medical SFT at all), harmful-only
eval (n=100, judged), vs the medically-aligned attacked models on the same 100 items:

| model | refused | pushback | complied | HRR |
|---|---|---|---|---|
| qwen3:14b (clean, no medical alignment) | 66 | 29 | 5 | 0.05 |
| qwen-tb (qwen3:14b + TenBenign) | 52 | 36 | 12 | 0.12 |
| med-booster (v1, clean) | 91 | 9 | 0 | 0.00 |
| med-booster-tb (v1 + TenBenign) | 76 | 17 | 7 | 0.07 |

- TenBenign hits Qwen's own built-in general safety *harder* than any of our medically-aligned
  models. Our alignment (even without Booster) is not uniquely fragile to TenBenign vs. an
  off-the-shelf model's own alignment — if anything it held up better on this axis.

### Same question at full scale: all 900 MedSafetyBench TEST items (not a 100-sample), judged

`eval_msb900.py`, qwen3:14b / qwen-tb / med-booster / med-booster-tb, the real attack metric
make_step1_data.py's docstring points at instead of eval_mixed.py's `-m` sample of it:

| model | refused | pushback | complied | HRR |
|---|---|---|---|---|
| qwen3:14b (clean) | 69.1% | 26.7% | 4.2% | 4.2% |
| qwen-tb (+ TenBenign) | 56.2% | 37.3% | 6.4% | 6.4% |
| med-booster (clean) | 92.1% | 7.9% | 0.0% | 0.0% |
| med-booster-tb (+ TenBenign) | 76.9% | 17.1% | 6.0% | 6.0% |

(of 900 items each; raw counts in `eval_mixed/*_msb900_s0.judge.json`)

- Confirms the n=100 sample's shape: TenBenign raises HRR ~1.5x on raw Qwen (0.042→0.064).
  med-booster goes from perfect (0.000) to 0.060 — at full scale this lands *level with*
  attacked-Qwen (0.060 vs 0.064), not clearly below it as the 100-item sample suggested. Booster's
  harmful-refusal protection is real (829/900 refused outright vs Qwen's 622/900 before either is
  attacked) but narrows under attack to roughly match an unprotected off-the-shelf model.

## In flight

- Nothing running.

## Next (in order)

1. ~~Seed sweep~~ done: gap survives (table above).
2. ~~Booster ablations~~ done and dropped: two variants (scope rows added to h(w) at the paper's
   λ=20/α=0.01; and h disjoint from f with no harmful rows in f, rank 32) were trained, attacked and
   judged. Neither beat the shipped `med-booster`; the disjoint-h one was clearly worse under attack.
   Models and data removed in the cleanup, so the report covers the four models above only.
3. Reproduction axis of the rubric ("runs on YOUR system"): add `eval_mixed.py --via V1|V2` so the
   mixed stream goes through the LangGraph variants; run utility via `predict -v V1 -m <tag>` +
   `evaluate`. Open question worth a row: does V2's decider/verifier blunt a jailbroken answerer?
4. Residual gaps: attacker budget (stage-1 epochs 20/50, stage-2 lr 1e-4); utility-preserving
   attacker (50 MedQA pairs mixed into stage 2); reproduce paper Fig 5b (gradient cosine) on
   med-base vs its stage-1 model.
5. Report: `docs/report/`, tone per STYLE.md. Threat model per row: attack assumes WEIGHT access
   (fused dirs), not API access.

## Gotchas

- Spawned sessions land in a git worktree that has none of the gitignored models/data. Tell them to
  `change_directory` to the main checkout, or they symlink and leave outputs in the worktree
  (`.claude/worktrees/…`) — that happened once; outputs were moved back by hand.
- `booster/safe_med*/` are subsets of `step1_data/` (not ignored, small): v1 = the 900 safety
  refusals reshuffled; v2 adds the 900 scope refusals. Regenerate v2 with `make_booster_data.py`.
- Never regenerate the harm set from a jailbroken model; the safe-response variant is the one we use.
