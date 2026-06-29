# Plan — Multi-Agent Medical Reasoning System

## Goal
Build a medical reasoning application whose **accuracy gain is measured against a single-LLM
baseline on MedQA-USMLE**, layering: retrieval (RAG), a multi-agent workflow, short- and
long-term memory, and an evolving experience base. Local stack: LangGraph + LangChain v1 + Ollama.

### Key assumption (correct me if wrong)
The **primary evaluation task is MedQA-USMLE (multiple-choice)** — that's what components (1)
baseline and (7) experience-on-wrong-answers imply. The OSCE consultation hospital we already
built (`patient ↔ doctor ↔ diagnosis`) is a **related, secondary mode** that shares the same
agent core, RAG, and memory; it is not the thing MedQA-USMLE scores. Components below are
designed around the MCQ task.

---

## The 7 components → modules

| # | Component | What it is here | Module(s) |
|---|-----------|-----------------|-----------|
| 1 | **Direct LLM baseline** | one LLM answers the MCQ directly (no RAG, no agents) — the reference accuracy | `qa/baseline.py` |
| 2 | **Medical reasoning agent** | specialist reasoner(s) that produce answer + rationale (grounded in RAG). *Not* the triage nurse — triage only routes | `agents/specialist.py` (reuses `Agent`) |
| 3 | **RAG module** | MedRAG **Textbooks** → Chroma `knowledge` collection; retriever injects top-k into reasoning | `knowledge/ingest.py`, `knowledge/retriever.py` |
| 4 | **Short-term memory** | LangGraph `State` for one question (options, retrieved docs, per-agent opinions, votes) | `qa/state.py` |
| 5 | **Long-term memory** | persistent Chroma across runs (knowledge + experience collections) | `memory/chroma_store.py` |
| 6 | **Multi-agent workflow (≥3)** | router → specialist panel → attending/aggregator, as a `StateGraph` | `qa/pipeline.py`, `agents/{router,attending}.py` |
| 7 | **Evolutionary optimization** | on a wrong TRAIN answer, store (question, correct answer, distilled lesson) in `experience`; retrieve on similar future questions (incremental RAG / MedAgent-Zero) | `memory/experience.py` |

---

## Multi-agent workflow (component 6, ≥3 agents)
A LangGraph `StateGraph` over one MCQ:

```
START
  → router        (classify specialty + build retrieval query)
  → retrieve      (RAG: top-k textbook snippets + top-k experience hits)
  → specialists   (1–2 reasoner agents, each: evidence → answer + rationale)   [the panel]
  → attending     (aggregate opinions + evidence → final option choice)
  → score         (compare to gold; if TRAIN & wrong → write experience)
END
```
Agents (≥3): **router**, **specialist** (≥1, panel of 2 gives 4 total), **attending**. All are
`Agent` instances with role prompts; `attending` may use `response_format` for a structured final
choice. Reuses the `consultation → diagnosis → scoring` pattern already built.

`QAState` (short-term memory, component 4):
```python
class QAState(TypedDict, total=False):
    question: str; options: dict[str,str]; gold: str
    specialty: str; query: str
    knowledge: list[str]; experience: list[str]
    opinions: list[dict]            # per-specialist {answer, rationale}
    answer: str; correct: bool | None
```

---

## RAG design (component 3)
- **Corpus:** MedRAG **Textbooks** (18 USMLE textbooks, ~125k pre-chunked snippets) — leakage-safe
  (textbooks ≠ exam questions), USMLE-aligned. StatPearls optional later.
- **Embeddings:** Ollama `nomic-embed-text` (local). MedCPT is a later upgrade.
- **Store:** Chroma, persistent, two collections: `knowledge` (static) and `experience` (grows).
- **Pipeline:** `load_dataset("MedRAG/textbooks")` → embed → `Chroma(collection="knowledge")` →
  `as_retriever(k=4)`. Prototype on a subset (~10k snippets) before the full embed.

## Memory & evolution (components 5, 7)
- **Long-term (5):** Chroma persists across runs; both collections live here.
- **Experience / evolution (7):** after a **train/dev** question, if the answer is wrong, store
  `{question, correct option, one-line lesson}` into `experience`. At inference, `retrieve` pulls
  similar experience hits into the attending agent's context. **Never store test items** (leakage).

---

## Datasets & split discipline
| Dataset | Role | Source |
|---|---|---|
| **MedQA-USMLE (4-option)** | baseline + eval + experience source | HF `GBaker/MedQA-USMLE-4-options` (confirm) |
| **MedRAG Textbooks** | RAG knowledge base | HF `MedRAG/textbooks` |
| OSCE `medqa.jsonl` (existing) | consultation hospital (secondary) | already vendored |

Leakage rules: RAG corpus is textbooks only; experience is built **only** from train/dev mistakes;
evaluation is on the held-out **test** split; test answers are never written anywhere.

---

## Evaluation plan
- **Metric:** accuracy (exact option match) on MedQA-USMLE test.
- **Headline comparison:** baseline (1) vs full system (2–7) → **accuracy gain**.
- **Ablation ladder** (isolates each component's contribution):
  1. Baseline (single LLM)
  2. + RAG (single agent + retrieval)
  3. + Multi-agent (panel + attending, no RAG)
  4. + RAG + Multi-agent
  5. + Experience (full system)
- Report per-ablation accuracy on a fixed test subset (e.g. 200–300 Qs first, then full).
- **Optional separate track:** MedAgentBench (agentic tool-use) via an `AgentClient` adapter — a
  different capability, reported on its own, not mixed into MCQ accuracy.

---

## Phased build order (each phase runnable + measured)
- **Phase A — Baseline (1).** `diseases/medqa_usmle.py` loader + `qa/baseline.py`. Output: baseline
  accuracy on a test subset. *Verify:* offline loader tests; live accuracy number.
- **Phase B — RAG (3).** `knowledge/ingest.py` (subset → Chroma) + `retriever.py` + a single
  agent-with-retrieval answerer. *Verify:* retrieval returns relevant snippets; accuracy vs A.
- **Phase C — Multi-agent (2,4,6).** `qa/state.py`, `agents/{router,specialist,attending}.py`,
  `qa/pipeline.py` graph. *Verify:* offline graph wiring with stubs; live accuracy vs B.
- **Phase D — Long-term memory + evolution (5,7).** `memory/{chroma_store,experience}.py`;
  build experience from train mistakes; retrieve at inference. *Verify:* wrong train answer writes
  an experience row; held-out accuracy vs C (the evolution lift).
- **Phase E — (optional)** MedAgentBench adapter track; OSCE consultation integration.

---

## Reuse (already built — Phase 1)
- `agents/base.py` `Agent` (tools, middleware, `response_format`, multi-provider) — the agent core.
- `agents/patient.py`, `agents/doctor.py`, `nodes/{consultation,diagnosis,scoring,graph}.py` — the
  consultation pattern the QA pipeline mirrors.
- `config.py` `HospitalConfig` (extend with `retrieval_k`, panel size, embedding model).
- `diseases/loader.py` `DatasetLoader` (subclass for MedQA-USMLE).
- pytest harness; live tests skip if Ollama down.

## New dependencies
`chromadb`, `langchain-chroma`, HF `datasets` (corpus + MedQA-USMLE), `langchain-ollama`
embeddings. Add via `pyproject.toml`.

## Open decisions (to confirm before Phase A)
1. MCQ as primary task (this plan) vs OSCE consultation as primary.
2. MedQA-USMLE source/split (`GBaker/MedQA-USMLE-4-options` 4-option vs 5-option).
3. Panel size (1 specialist = 3 agents minimum, vs 2 = 4 agents).
4. RAG corpus scope (Textbooks only vs +StatPearls) and embeddings (`nomic-embed-text` vs MedCPT).
