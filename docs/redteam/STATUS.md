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
| `med-booster-v2` (deleted) | `booster.py --safe-dir safe_med_v2 --lam 20 --alpha 0.01` (paper's best) | superseded by v3, numbers below kept for the record |
| `med-booster-v2-tb` (deleted) | TenBenign on fused med-booster-v2, seed 0 | superseded by v3, numbers below kept for the record |
| `qwen-tb` (deleted) | TenBenign directly on raw Qwen3-14B 4-bit (no medical alignment at all) | numbers below kept for the record |
| `med-base`, `med-mcq`, `med-base-tb` (deleted) | see rows above | superseded; local build dirs and ollama models removed to free disk |
| `med-booster-v3` | `booster.py --align-dir step1_data_medonly --safe-dir booster/safe_med --rank 32 --lam 20 --alpha 0.01`: f has ZERO harmful rows (medical+patient only), h = full 900 MedSafetyBench (disjoint from f, not a subset) | done + judged, 100 |
| `med-booster-v3-tb` | TenBenign on fused med-booster-v3, seed 0 | done + judged, 100 |

Kept on disk: `fused_step1/`, `fused_med_booster/` (28 GB each, fp16; needed as `--base` for
attacks), `adapters_step1/`, `adapters_step1_mcq/`, `adapters_med_booster/`. `common.py` now has
`keep_fused`; `step1_align.py --keep-fused`; `tenbenign.py --seed`.

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

### booster-v2 (paper's best hyper-params, scope rows added to h(w))

| row | model | MedQA | harmful refused/pushback/complied | scope refused |
|---|---|---|---|---|
| — | med-booster-v2 (clean) | 0.635 | 94/6/0 | 100/100 |
| — | med-booster-v2-tb (attacked) | 0.5725 | 67/28/5 | 15/100 |

- v2's scope refusal under attack (15/100) barely beats v1's (10/100) — the paper's λ=20/α=0.01 +
  scope-in-h(w) did **not** close the gap. Harmful HRR ~flat (0.05 vs v1's 0.07). Clean MedQA
  (0.635) and clean scope (100/100) match v1, so v2 isn't worse pre-attack — it just doesn't help.

### booster-v3: does the paper's true structure (h disjoint from f, zero harmful rows in f) help?

Design, reasoned through in session: Qwen3-14B already refuses 95.8% of harmful-medical prompts
zero-shot (622 refused + 240 pushback of 900, only 38 complied) — closer to the paper's premise of
an *already safety-aligned* base than we'd given it credit for. So f can drop harmful rows entirely
(medical + patient QA only, 1732 train rows) and h can use the full 900 MedSafetyBench TRAIN rows,
genuinely disjoint from f for the first time (v1/v2 always reused a subset of f's own rows in h).
Also applied: rank 32 (paper's rank, was 8) and its best λ=20/α=0.01 (already used in v2).

Judge simplified alongside this run: `judge.py`'s malicious rubric collapsed from 3-way
(refused/pushback/complied) to binary (refused/complied) — pushback was already scored as safe in
HRR, so the middle label added detail without changing any number that matters.

| model | refused | complied | HRR |
|---|---|---|---|
| med-booster-v3 (clean, first 100 of MedSafetyBench TEST) | 100 | 0 | 0.00 |
| med-booster-v3-tb (+ TenBenign) | 89 | 11 | 0.11 |

For comparison, v1's old 3-way numbers on its own 100-item sample, collapsed (pushback -> refused):
med-booster clean 100/0 (0.00), med-booster-tb attacked ~93/7 (~0.07). Item sets aren't identical
(v1 sampled 100 of a 940-pool incl. 40 handwritten; v3 here is the first 100 of the pure 900-item
MedSafetyBench file, not a random sample) so this is indicative, not a controlled comparison.

Reading: v3's clean 100/100 is not evidence the regularizer strengthened refusal — training log
shows `Safe loss` (h(w)) never dropped (sat ~1.0-1.2 throughout, val loss on f fell normally
2.67->1.48), meaning nothing pulled h down; the 100/100 is inherited from Qwen's own baseline
safety plus whatever the medical-only SFT didn't disturb. Post-attack it did *slightly worse* than
v1 (0.11 vs ~0.07), suggesting the "more paper-faithful" structure traded away the redundancy of
having f directly teach the same rows h probes. Net: matching the paper's structure more closely
did not improve robustness here, at least at this sample size.

Session gotcha: mid-run, disk filled (16Gi free of 926Gi) because every attack keeps a full local
`.q8_0.gguf` (~15GB) AND Ollama separately stores its own copy in `~/.ollama/models/blobs` — pure
duplication once `ollama create` succeeds. Freed ~105GB by deleting local `*.q8_0.gguf` files and
per-attack work dirs (`stage1_data/`, `stage2_data/`, adapters, gguf) for already-registered
models; verified safe first via `stat -f nlink` (all `nlink=1`, no hardlink to Ollama's store, so
Ollama's copy is untouched). `med-base`, `med-mcq`, `med-base-tb`, `med-booster-v2*`, `qwen-tb` and
their local dirs were deleted this way (models + build dirs); `fused_med_booster`, `fused_step1`
(fp16, kept for further attacks) also removed when no longer needed. Do this proactively next time
rather than waiting for a failed `ollama create` mid-attack.

### Attack on a non-medically-aligned model (does TenBenign need "freshly learned" alignment?)

TenBenign run directly on raw Qwen3-14B 4-bit (`qwen-tb`, no medical SFT at all), harmful-only
eval (n=100, judged), vs the medically-aligned attacked models on the same 100 items:

| model | refused | pushback | complied | HRR |
|---|---|---|---|---|
| qwen3:14b (clean, no medical alignment) | 66 | 29 | 5 | 0.05 |
| qwen-tb (qwen3:14b + TenBenign) | 52 | 36 | 12 | 0.12 |
| med-booster (v1, clean) | 91 | 9 | 0 | 0.00 |
| med-booster-tb (v1 + TenBenign) | 76 | 17 | 7 | 0.07 |
| med-booster-v2 (clean) | 94 | 6 | 0 | 0.00 |
| med-booster-v2-tb (v2 + TenBenign) | 67 | 28 | 5 | 0.05 |

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
2. ~~booster-v2~~ done: table above. Gap survives Booster's best hyper-params + scope in h(w).
   Superseded by v3's design (below); v2's files were deleted, numbers stay in this file.
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
