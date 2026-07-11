# Plan — MedQA-USMLE Variant Ladder

An MCQ question-answering system on MedQA-USMLE, built as an **ablation ladder of variants**
(V0–V4) measured with paired significance. Local stack: LangGraph + LangChain v1 + Ollama.
Full details + results: [ARCHITECTURE.md](ARCHITECTURE.md). The LangGraph rebuild: [REFACTOR_PLAN.md](REFACTOR_PLAN.md).

## Variants
| id | variant | status |
|----|---------|--------|
| V0 | Direct LLM | ✅ built |
| V1 | RAG-only (reasoning-distilled query) | ✅ built |
| V2 | Multi-agent (reasoner + specialist) | ✅ built |
| V3 | Full system (+ experience base) | ⬜ next |
| V4 | Full system without verifier | ⬜ |

## Status (deterministic, qwen2.5:14b, temp=0)
- **V0 ≈ V1** — RAG does not help MedQA (reasoning-heavy, not lookup).
- **V2 > V1 is significant (p=0.023)** — multi-agent reasoning is the lever, not retrieval.
- See [ARCHITECTURE.md](ARCHITECTURE.md) for the full metrics table.

## Next
1. **Refactor V0–V2 onto config-driven LangGraph** (behavior-preserving) — [REFACTOR_PLAN.md](REFACTOR_PLAN.md).
2. **V3 — experience base:** retrieve solved-question rationales from **MedQA train mistakes** (the real
   MedQA lever) + panel + verifier.
3. **Confirm V2 > V0 at 150 test items** (n=80 was underpowered, p=0.18).

## Data & stack
- **Dataset:** [`nnilayy/medqa-usmle`](https://huggingface.co/datasets/nnilayy/medqa-usmle) — 4-option MCQ
  (train 10,178 / val 1,272 / test 1,273); scored by exact option match.
- **Code:** `agents/base.py` (Agent) · `knowledge/` (MedRAG Textbooks → Chroma, `nomic`/`medcpt`) ·
  `qa/` (variants + metrics harness).
