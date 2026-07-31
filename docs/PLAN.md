# Plan — MedQA-USMLE Variant Ladder

An MCQ question-answering system on MedQA-USMLE, built as an **ablation ladder of variants**
(V0–V5) measured with paired significance. Local stack: LangGraph + LangChain v1 + Ollama.
Full details: [ARCHITECTURE.md](ARCHITECTURE.md). The original config-driven LangGraph rebuild
(V0–V2, superseded by the current four-node agentic design): [REFACTOR_PLAN.md](REFACTOR_PLAN.md)
(historical record only — its presets/roles no longer match the code).

## Variants
| id | variant | status |
|----|---------|--------|
| V0 | Direct LLM | ✅ built, measured |
| V1 | RAG-only — single agentic-RAG agent (`search_medmcqa`) | ✅ built, measured |
| V2 | Four-node design: case-reasoner → (search ‖ reasoning) → decider | ✅ built, measured |
| V3 | V2 + decider has long-term memory (cross-episode lesson bank) | ✅ builds & runs, measurement in progress |
| V4 | V3 + verifier grounded in live Wikipedia | ✅ builds & runs, measurement in progress |
| V5 | V3 + verifier that recalls its own evolving mistake bank | ✅ builds & runs, measurement in progress |

V2–V5 all share the same four-node backbone (`understand → (search ‖ reasoning) → answer`); V4/V5
each add exactly one thing on top of V3 (a verifier node, differing only in what it checks
against), so `V4 − V3` and `V5 − V3` isolate each verifier design directly. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the graph topology of each variant.

## Status
- Current measured numbers live in `data/eval_runs/*.jsonl` and are written up in
  [`report.tex`](report/report.tex) — evaluation is an active, ongoing process (runs at different
  sample sizes land at different times), so no snapshot is duplicated here. Check `report.tex`
  §5 (or run `python -m agent_hospital.evaluate`) for the latest leaderboard and paired
  comparisons.
- V1's single-agent RAG has shown a real invalid-response problem historically (tool-loop
  non-convergence on some items); V2's four-node split fixed that.

## Next
1. **Finish measuring V3, V4, V5** on the full test set (n=1273) — V0–V2 already have full-test
   runs; V3–V5 currently have partial runs only.
2. **Compute paired significance** (Win/Loss/Tie, McNemar) for V3 vs V2, V4 vs V3, and V5 vs V3
   once those full runs land — `V4 − V3` and `V5 − V3` isolate each verifier design since they
   share every other node.
3. **Re-embed the MedMCQA collection at full scale** (`knowledge_medmcqa_qwen3`) if the corpus used
   for a given measurement run is a subset — check `report.tex` for the corpus size a given number
   was measured against before comparing across runs.

## Data & stack
- **Dataset:** [`openlifescienceai/medqa`](https://huggingface.co/datasets/openlifescienceai/medqa) — 4-option MCQ
  (train 10,178 / val 1,272 / test 1,273); scored by exact option match.
- **Code:** `agents/base.py` (Agent) · `roles.py` (prompts) · `graph/` (node factories, `build_graph`,
  `longterm.py` cross-episode memory) · `knowledge/` (MedMCQA → Chroma, `qwen3-embedding:4b` by
  default) · `qa/` (variant presets + metrics harness).
