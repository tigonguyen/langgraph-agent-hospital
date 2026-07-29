# Plan — MedQA-USMLE Variant Ladder

An MCQ question-answering system on MedQA-USMLE, built as an **ablation ladder of variants**
(V0–V4) measured with paired significance. Local stack: LangGraph + LangChain v1 + Ollama.
Full details + results: [ARCHITECTURE.md](ARCHITECTURE.md). The original config-driven LangGraph
rebuild (V0–V2, superseded by the agentic MedAgents-style rebuild described in ARCHITECTURE.md):
[REFACTOR_PLAN.md](REFACTOR_PLAN.md).

## Variants
| id | variant | status |
|----|---------|--------|
| V0 | Direct LLM | ✅ built, measured (full test) |
| V1 | RAG-only — single agentic-RAG agent (`search_medmcqa`) | ✅ built, measured (full test) |
| V2 | Multi-agent panel (3 specialists + attending + verifier) | ✅ built, measured (full test) |
| V3 | V2 + short-term memory (scribe) | ✅ builds & runs, **not yet measured** |
| V4 | Full system without verifier | ⚠️ built, runs, but preset is **stale** — doesn't isolate the verifier (see ARCHITECTURE.md) |

## Status (full test set, n=1273, qwen2.5:7b)
- **V0 (0.615) > V2 (0.608) > V1 (0.573)** — the agentic single-agent RAG (V1) has the worst accuracy
  *and* a real invalid-response problem (3.85%, tool-loop non-convergence).
- **V2's panel+verifier fixes the invalid-rate** (0% ) but costs ~25× V0's latency for no accuracy gain
  over V0 yet.
- Paired significance (Win/Loss/Tie, McNemar) vs. V0 is not yet computed — V0's per-item record from
  that run was lost; see `docs/REPORT_NOTES.md`.
- See `report.tex` §5 for the full table.

## Next
1. **Fix the V4 preset** so `V3 − V4` actually isolates the verifier (currently confounds corpus,
   embedder, retrieval mode, and panel size too — see ARCHITECTURE.md's ladder caveat).
2. **Measure V3 and (fixed) V4** on the full test set.
3. **Re-run V0** with per-item records kept, so V1/V2 vs. V0 paired stats (WLT, McNemar, bootstrap CI)
   can finally be computed.
4. **Long-term memory is unbuilt** (spec §4.5) — only V3's short-term working memory exists. Remains
   future work if the ladder is to fully satisfy the spec.

## Data & stack
- **Dataset:** [`openlifescienceai/medqa`](https://huggingface.co/datasets/openlifescienceai/medqa) — 4-option MCQ
  (train 10,178 / val 1,272 / test 1,273); scored by exact option match.
- **Code:** `agents/base.py` (Agent) · `knowledge/` (MedMCQA + MedRAG Textbooks → Chroma, `nomic`/`medcpt`) ·
  `qa/` (variants + metrics harness).
