# Plan — Multi-Agent Medical Reasoning System

## Goal
A patient-simulated, multi-agent medical consultation that **answers MedQA-USMLE questions**, with
its **accuracy gain measured against a single-LLM baseline**. It layers retrieval (RAG), a multi-agent
workflow, short-/long-term memory, and an evolving experience base. Local stack: LangGraph +
LangChain v1 + Ollama.

## Status (current — 2026-06-30)
- ✅ **(3) RAG knowledge base BUILT** — 125,847 MedRAG Textbook snippets in Chroma (`nomic-embed-text`, cosine, gated). Tests pass.
- ✅ **(1) Baseline + RAG answerer + accuracy eval** (Phase A) — committed.
- 📊 **First measurement (50 items):** baseline 7B 0.68 / 32B 0.66; **RAG *hurt*** (−8 pts 7B, −2 pts 32B); 32B ≈ 7B.
  → whole-vignette retrieval injects topical-but-non-discriminating context; **the query is the bottleneck, not model size.**

## Resolved decisions (grill, 2026-06-30)
- **Architecture:** **MCQ-direct multi-agent is the measured path** (router → panel → attending read the question). Patient-simulated consultation is **deferred** to a later phase as the *experience generator* (MedAgent-Zero), **not** the eval. Experience base meanwhile comes from **MedQA train mistakes**.
- **RAG:** **query distillation** (retrieve on the focused question/findings, not the whole vignette); keep RAG only if the lift is non-negative.
- **Eval:** **150 test items** for trustworthy numbers (50 is noise-level).
- **Default model:** **qwen2.5:14b** for agent roles (5 pulled: 7b / 14b / mistral-small:24b / phi4 / 32b).

## Next build order
1. **Query distillation** → re-measure single-reasoner +RAG vs baseline (150 items, 14b & 7b). Gate the keep-RAG decision on this.
2. **Multi-agent MCQ pipeline (2,4,6):** `qa/state.py` (`QAState`), router + specialist panel×2 + attending, `qa/pipeline.py` StateGraph; distilled RAG feeds the specialists. Measure vs baseline + single-reasoner.
3. **Experience base (5,7):** store wrong MedQA *train* answers as lessons → gated retrieval into specialists; measure the evolution lift on held-out items.
4. **(Later) patient-sim phase:** `sent1`→OSCE conversion + Patient/Examiner as a richer experience generator.

## Primary task & data
- **Dataset:** [`nnilayy/medqa-usmle`](https://huggingface.co/datasets/nnilayy/medqa-usmle) — 4-option
  MCQ. Fields: `sent1` (vignette **+ question**), `ending0..3` (options), `label` (correct index).
  Splits: **train 10,200 / val 1,270 / test 1,270**.
- **MedQA answers vary** (diagnosis, treatment, next test, mechanism…), so the final step is **answer
  the MCQ by picking an option** — scored by **exact option match** (not diagnosis-string match).
- **One-time prep (conversion):** parse `sent1` (extraction, not fabrication — vignettes already
  contain the findings) into an OSCE-style case: `history` → Patient · `exam`+`labs` → Examiner ·
  `question`+`options` → reasoning agents. Keep `label` as the hidden answer key.
- **Splits in use:** convert **~150 train** items for patient simulation + experience seeding; hold
  out a **test** subset for final validation. Test answers are never stored.

## The 7 components → modules
| # | Component | Here | Module(s) |
|---|-----------|------|-----------|
| 1 | Direct LLM baseline | single LLM reads full vignette+options → option; reference accuracy | `qa/baseline.py` |
| 2 | Medical reasoning agent | reasoning **panel** (the diagnostic core) | `agents/specialist.py`, `agents/attending.py` |
| 3 | RAG module | MedRAG Textbooks → Chroma `knowledge`; reasoning-distilled, gated retrieval | `knowledge/{ingest,retriever}.py` |
| 4 | Short-term memory | LangGraph `QAState` (one case) | `qa/state.py` |
| 5 | Long-term memory | persistent Chroma across runs | `memory/chroma_store.py` |
| 6 | Multi-agent workflow (≥3) | Patient + Triage + Panel×2 + Examiner + Attending | `qa/pipeline.py`, `agents/*` |
| 7 | Evolutionary optimization | wrong TRAIN answer → `experience` collection (incremental RAG) | `memory/experience.py` |

## Consultation flow — "Middle" info design (locked)
The reasoning panel sees the **history + question** directly; the **Examiner** reveals exam/labs only
when a test is ordered; RAG knowledge + (gated) experience are layered on top.

```
case → Patient (history)
     → Triage (route by chief complaint; no RAG)
     → Reasoning panel ×2   (each: history + question + ordered labs
          │                    + RAG knowledge[always] + experience[gated]  → answer + rationale)
          ├─ order tests → Examiner (returns the case's hidden exam/labs; lookup, no RAG)
          └─ RAG query = reasoning-distilled from findings + question  → similarity-gated, no-RAG fallback
     → Attending (aggregate the 2 opinions + evidence → final option)
     → score (option == label) → if wrong & TRAIN → write experience
```

`QAState` (short-term memory):
```python
class QAState(TypedDict, total=False):
    case: dict                       # history, exam, labs, question, options, label
    specialty: str                   # triage output
    transcript: list[dict]           # patient/doctor turns
    ordered_labs: list[str]; lab_results: dict
    knowledge: list[str]; experience: list[str]
    opinions: list[dict]             # per-specialist {answer, rationale}
    answer: str; correct: bool | None
```

## RAG design (locked) — framed in IR terms
Modern RAG = a **dense Vector Space Model**: the embedding model is the *representation function*
mapping documents and the query into one **shared concept-vector space**; **relevance = cosine
similarity**; the gate is a relevance cutoff.

- **Retrieval family:** **dense embeddings first** (matches the working obsidian-rag). *Documented
  alternatives to revisit:* **hybrid** (BM25 sparse + dense — best for exact medical terms like
  drug/lab names) and **sparse BM25 only**. Start dense; add hybrid if exact-term recall is weak.
- **Embedding model:** **`nomic-embed-text`** (Ollama, local, 768-dim) — same stack as the working
  obsidian-rag; the swap-point is `knowledge/embeddings.py:default_embeddings()`. (MedCPT was the
  medical-domain pick but was dropped for now — its HF download is gated/throttled; revisit later by
  returning a different `Embeddings` here, no other code changes.)
- **Corpus:** MedRAG **Textbooks** — **prototype on a ~10k shuffled subset** (the corpus is ordered by
  book, so a plain prefix covers only one book — use `shuffle_buffer` to span many), then embed the
  full ~125k. StatPearls later.
- **Store:** Chroma persistent at `data/chroma`; collections `knowledge` (static) + `experience`
  (incremental). MedRAG snippets used as-is (no re-chunking).
- **Relevance estimation:** cosine, `k=4`, **threshold gate + no-RAG fallback**; `knowledge` always
  queried, `experience` gated. Query is **reasoning-distilled** (findings + question), embedded with
  the **query encoder**. Two labeled sections ("Textbook evidence" / "Past lessons"). **Consumed by
  the 2 specialists only**; triage & examiner are RAG-free.
- **Retrieval evaluation:** **extrinsic** — MedQA has no gold passages, so judge retrieval by whether
  it lifts MCQ accuracy (RAG vs no-RAG). (Optional: a tiny hand-labeled set for recall@k later.)

## Memory & evolution
- **Long-term (5):** Chroma persists both collections across runs.
- **Evolution (7):** after a **train/dev** case, if the predicted option is wrong, store
  `{distilled question, correct option, one-line lesson}` in `experience`; retrieved (gated) on
  similar future cases. Mirrors MedAgent-Zero. **Never store test items.**

## Evaluation
- **Metric:** MCQ accuracy (option match) on the MedQA-USMLE **test** subset.
- **Headline:** baseline (1) vs full system → accuracy gain.
- **Ablation ladder:** baseline → +RAG → +panel → +RAG+panel → +experience (full). Isolates each
  component's contribution.
- **Honest note:** baseline and full system both derive from the same vignette, so gains come from
  **RAG + experience + panel aggregation**, not from info-hiding. The "Middle" design keeps the
  Examiner meaningful without capping accuracy.

## Phased build order (each phase runnable + measured)
- **A — Baseline (1):** `diseases/medqa_usmle.py` loader + `qa/baseline.py` → baseline accuracy.
- **B — Case conversion + RAG (3):** `sent1`→OSCE converter (~150 train + test subset);
  `knowledge/ingest.py` (subset → Chroma) + `retriever.py`; single-reasoner+RAG answerer; lift vs A.
- **C — Multi-agent (2,4,6):** `qa/state.py`, `agents/{triage,specialist,examiner,attending}.py`,
  `qa/pipeline.py` graph (panel×2 + attending). Offline stub wiring + live accuracy vs B.
- **D — Long-term memory + evolution (5,7):** `memory/{chroma_store,experience}.py`; build experience
  from train mistakes; gated retrieval. Held-out accuracy vs C (the evolution lift).
- **E — (optional):** MedAgentBench adapter track; OSCE consultation as its own task.

## Future mode (stored, not built now)
**Accuracy-max info design:** give the reasoning panel the full case directly (Patient/Examiner become
presentation only, no info hidden) + RAG + experience. Best raw accuracy; revisit if the "Middle"
design leaves accuracy on the table.

## Reuse / dependencies / open items
- **Reuse:** `agents/base.py` `Agent` (tools, middleware, `response_format`); the
  `consultation→diagnosis→scoring` pattern; `HospitalConfig` (extend: `retrieval_k`, `sim_threshold`,
  panel size, embedding model); `DatasetLoader` (subclass for MedQA-USMLE); pytest harness.
- **New deps:** `chromadb`, `langchain-chroma`, HF `datasets`, `langchain-ollama` (for `nomic-embed-text`
  embeddings). (via `pyproject.toml`)
- **Open:** similarity-gate threshold value (tune in B); how many test items for validation (e.g. 200
  then full 1,270); whether Triage adds measurable accuracy (ablate — it's mainly for fidelity).
