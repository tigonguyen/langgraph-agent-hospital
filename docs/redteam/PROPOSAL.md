# Pair 5 — Proposal: fine-tuning strips safety alignment (attack & defense)

Assumption stated up front: written in English, as a standalone deliverable distinct from the
Vietnamese final report (`docs/report/`, which covers the main LangGraph hospital-agent project).
Say if this should be Vietnamese or merged elsewhere instead.

## System under test

A medical-only assistant built on Qwen3-14B: answers MedQA-USMLE-style questions, refuses harmful
medical requests, and refuses anything outside the medical domain. This is a deviation from the
original design sketch (which targeted a general-purpose Qwen2.5-7B-Instruct with general-purpose
BeaverTails alignment data) — reasoned below.

## 1. Threat model

| | |
|---|---|
| Attacker access | Fine-tuning access to the deployed model's weights (local LoRA on the last 16 layers) — the analogue of a hosted fine-tuning API. No access to the alignment data or training code. |
| Attacker constraint | Training data must pass a content-moderation filter → only benign QA pairs (TenBenign, arXiv:2510.02833: 10 benign medical questions, two stages). Verification that this data reads as benign to a safety judge: planned (`moderation_check.py`), not yet run — flagging honestly rather than asserting it. |
| Attacker goal | The model answers harmful-medical or off-topic requests it previously refused, while still looking like a normal assistant. |
| Defender access | Controls the alignment stage before deployment (plain SFT, or Booster — Huang et al., ICLR 2025) but cannot inspect or block the attacker's later fine-tune. |
| Out of scope | Prompt-level jailbreaks (Pair 2), poisoning the RAG store (Pair 3), weight backdoors (Pair 4). |

## 2. Metrics

| axis | metric | how |
|---|---|---|
| **attack** | Harmful-Response Rate (HRR) — fraction of harmful-medical prompts the model *complies* with | LLM judge (qwen3:14b, 3-way rubric: refused / pushback / complied); regex refusal-keyword heuristic as a cheaper secondary column |
| **scope** (added; not in the original design) | Off-topic answer rate — fraction of non-medical prompts the model answers instead of declining | Same LLM judge, refused/answered rubric |
| **utility** | MedQA-USMLE accuracy | held-out test split, exam-format prompts |
| **cost** | Training/attack wall-clock, disk footprint | timers in the scripts |
| **baseline** | HRR / scope / utility of the aligned-but-unattacked model | row 1 below |

Prompt sets: 900 MedSafetyBench TEST harmful-medical requests (never trained on) + 40 hand-written
in-domain probes; 200 held-out OASST1 non-medical prompts; MedQA-USMLE test split.

## 3. Baseline (row 1)

| model | MedQA | harmful refused/pushback/complied | HRR | scope refused |
|---|---|---|---|---|
| qwen3:14b (off-the-shelf, no medical alignment) | 0.705 | 622/240/38 (of 900) | 0.042 | 2/100 |
| med-base (plain-SFT alignment, our deployed baseline) | 0.677 | 95/5/0 (of 100 sample) | 0.00 | 100/100 |

Two reference points instead of one: qwen3:14b shows how much general safety Qwen already has
out of the box (useful later — TenBenign is tested against *both* the off-the-shelf model and our
aligned one, isolating what our alignment specifically adds). med-base is the actual deployed
system the attack targets.

## Deviations from the original design, and why

| original plan | what we built | reason |
|---|---|---|
| Qwen2.5-7B-Instruct, general-purpose | Qwen3-14B, medical-only | System under test is a hospital agent; a narrow domain is the realistic deployment and the brief's own warning ("defenses leave residual risk on narrow-domain inputs") is best tested on a narrow domain, not a general one |
| BeaverTails / HarmBench / Alpaca | MedSafetyBench (train/test split, no leakage) / OASST1 / MedMCQA | Domain-matched harmful and utility data; BeaverTails/HarmBench are general-purpose and don't probe medical-specific harm |
| Llama-Guard-3 judge | Our own LLM judge (qwen3:14b, explicit 3-way rubric) | Avoids pulling another 8B model; rubric is inspectable and versioned in `judge.py`; documented trade-off: judge and system share a base model family, a possible source of correlated bias worth naming in the final report |
| Booster h(w) = harmful→compliant answers (needs a harmful-answer dataset) | Booster h(w) = harmful→refusal (refusal-grad variant) | We will not generate harmful-compliant training data from a jailbroken model on ethical grounds; the refusal-grad variant simulates the same "one attacker step" idea from the opposite, safe direction (documented trade-off, not free — see residual-gap analysis) |

## Status against this proposal

Baseline (row 1), the attack, and three defense variants (plain Booster, and two ablations on its
refusal set / hyperparameters / LoRA rank) are already built and judged — see
[`docs/redteam/STATUS.md`](STATUS.md) for the full results table and what's still in flight
(attacker-budget sweep, utility-preserving attacker, and the written report itself).
