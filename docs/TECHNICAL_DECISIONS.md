# Technical Decisions

A decision log — *why* we chose what we chose, with rejected alternatives and the
evidence behind each call. Structure ("what is wired to what") lives in
[ARCHITECTURE.md](ARCHITECTURE.md); this file is the rationale.

---

## Stack & models

**D1 — LangChain v1 `create_agent`, wrapped in a lazy `Agent`.**
Why: one abstraction for every role (tools, middleware, structured output) and a
provider-agnostic `BaseChatModel`; lazy graph build keeps construction cheap.

**D2 — Local Ollama, default `qwen2.5:14b` for agent roles.**
Why: fully local/free iteration; 14B is the capability/speed sweet spot on a 48 GB Mac.
Rejected: **32B** — measured **≈ 7B accuracy** (0.66 vs 0.68) at ~4× latency, no gain.
7B kept as the cheap baseline. Evidence: 50-item run.

---

## Data & task

**D3 — Eval dataset = `openlifescienceai/medqa` (MedQA-USMLE, 4-option MCQ).**
Why: standard USMLE benchmark with a real train/val/test split; the measured task for all variants.

**D4 — Task = MCQ answering (no patient simulation).**
Why: the KPI is MedQA accuracy. An earlier patient-simulated consultation track was built and then
**removed** — a consultation can only *lose* information relative to reading the vignette (both derive
from the same `sent1`). V3's gain comes instead from a **diverse-perspective panel + attending + verifier**
(see D17); an episodic experience base (retrieving solved-question rationales) was descoped and remains
future work.

**D5 — Scoring = exact MCQ option match; unparseable → `None` (invalid).**
Why: MedQA answers vary (dx / treatment / next step), so the unit is "pick the right
option", not a diagnosis string. `None` makes parse failures explicit, never a silent wrong.

---

## RAG

**D6 — Corpus = MedRAG **Textbooks** only (125,847 snippets).**
Why: leakage-safe (reference text, not exam Q/A), USMLE-aligned (the exam is written
against these books), pre-chunked, and small enough to embed locally.
Rejected: PubMed (~24M) / Wikipedia (~30M) — infeasible to embed locally; weaker
distribution match.

**D7 — Vector store = Chroma, cosine (HNSW), persistent.**
Why: local, no server, native LangChain integration; cosine set explicitly so relevance
scores drive the gate.

**D8 — Retrieval = dense VSM, top-k=4, similarity gate + no-RAG fallback.**
Why: dense captures clinical paraphrase/synonyms; the gate stops off-topic snippets from
degrading answers. Evidence: ungated whole-vignette RAG measured **−8 pts**.
**Threshold is embedder-specific and must be re-tuned when the embedder changes** — it is
now **0.60** for MedCPT (was 0.5 for nomic). See D19.

**D9 — Query = reasoning-agent distillation, not the raw vignette.**
Why: retrieving on the whole vignette pulls topical-but-non-discriminating passages.
Evidence: distillation moved the V1−V0 lift from **−8 → +2 pts**.

---

## Embeddings & why MedCPT requires a re-ingest

**D10 — Default embedder = MedCPT** (was `nomic-embed-text` while MedCPT's download was blocked).
Why: MedCPT is medical-domain-tuned and is MedRAG's strongest retriever. Once the weights were
obtained (D13) and the corpus re-ingested into `knowledge_medcpt` (D12), it became the default in
`RagConfig`. `nomic-embed-text` stays available as a one-line swap, and both collections are built
(125,847 vectors each), so the embedder A/B is config-only. Applied at the **default** level so V1–V4
all use the same retriever — mixing embedders across rungs would confound the ablation.

**D11 — MedCPT added as a *selectable* embedder, not the default.**
Why: MedCPT is medical-domain-tuned (likely better retrieval) but heavier (torch +
HF model). Kept behind `default_embeddings("medcpt")` so it's swappable, with `nomic`
as the always-available fallback.

**D12 — A separate Chroma collection (`knowledge_medcpt`) is REQUIRED for MedCPT — i.e. we MUST re-ingest.**
This is the key one. Reasons:
1. **We downloaded the MedCPT *model* (encoders), not precomputed corpus vectors.** The
   encoder *produces* embeddings; it does not populate a store. So the 125k snippets must
   be run through MedCPT's **article encoder** to get document vectors — that *is* the re-ingest.
2. **Embedding spaces are model-specific and NOT interchangeable.** The existing
   `knowledge` collection was built with `nomic` (768-d). MedCPT is also 768-d but lives in
   a *different* geometric space — a MedCPT query vector compared (cosine) against `nomic`
   document vectors is meaningless. You cannot query a nomic-built store with MedCPT.
3. **MedCPT is an asymmetric bi-encoder:** documents → Article encoder, queries → Query
   encoder (trained into one shared space). So the store must be built with the article
   encoder and queried with the query encoder — both sides MedCPT, never mixed.
Therefore MedCPT needs its **own** collection embedded by its article encoder
(`knowledge_medcpt`, 125,847 snippets, ~31 min on MPS), retrieved with its query encoder.
*Could we skip it?* Only by sourcing MedRAG's **precomputed Textbook embeddings** (a
different, large download in MedRAG's format) — more work than the 22-min local re-ingest,
and it still needs the query encoder at inference. Not worth it for 125k snippets.

---

## Operational workaround — fetching MedCPT

**D13 — Download MedCPT weights via `curl` on the classic `resolve` URL, into `data/medcpt/` (gitignored).**
Why: the Python `huggingface_hub`/`hf_transfer` client **hung** on the large weight file
(it fetched config files then stalled at ~0.9 MB, 0 MB/s). Diagnosis: a plain `curl` on
`huggingface.co/.../resolve/main/model.safetensors` pulled at ~554 KB/s — so the **host is
reachable; the client's (Xet) large-file path was the problem.** Setting an `HF_TOKEN`
fixed the *rate limit* (config files flow) but not the client hang; the `hf-mirror.com`
endpoint was unreachable. The durable fix: `curl` the files (retry+resume) and load
MedCPT from the local dirs. Only `model.safetensors` is fetched (skip the redundant
`pytorch_model.bin`) → ~434 MB/encoder.

---

## Retrieval packaging

**D19 — Retrieval is a `search_textbooks` tool, invoked by the graph rather than chosen by the model.**
Why: packaging retrieval as a tool keeps one reusable definition (and leaves agentic RAG one line away),
while invoking it from the node keeps V1 at **2 LLM calls** and fully deterministic.
Measured both ways on the same items (qwen2.5:7b, temp=0):

| | LLM calls/item | generated chars/item | latency |
|---|---|---|---|
| tool **bound to the agent** (model decides) | 3.0 | 542 | 3.78 s |
| tool **invoked by the graph** (current) | 2.0 | 51 | 0.90 s |

The extra call is inherent: a model-chosen tool call requires re-invoking the model on the result. The
larger cost was generation — a tool-using agent **cannot** be told "respond with ONLY the letter" (that
forbids emitting a tool call), so it must be allowed to reason, and output grew ~10×. Retrieval itself is
~22 ms (MedCPT query embedding; HNSW search is sub-millisecond), i.e. **~1.5% of V1's latency** — the
bottleneck is token generation, not the vector store.

**D20 — Gate threshold is embedder-specific; 0.60 for MedCPT.**
Why: MedCPT is an **asymmetric** bi-encoder (separate query/article encoders), so query-to-passage cosine
is systematically lower than a symmetric embedder's — measured scores top out ≈**0.69**. A threshold
carried over from another embedder is meaningless: **0.9 would admit nothing**, making V1 byte-identical
to V0 plus a wasted reasoning call. 0.60 is strict but functional — evidence present on ~4/6 sampled
items, the rest exercising the no-RAG fallback.

---

## Evaluation

**D14 — n = 150 test items; metrics = accuracy + 95% bootstrap CI, invalid-rate, Win/Loss/Tie, McNemar, latency.**
Why: 50 is noise-level (±7 pts). Paired **McNemar** (same items through both variants)
answers "is the gain real?"; bootstrap CI bounds it. Retrieval is judged **extrinsically**
(does it lift end accuracy) because MedQA has no gold-passage labels.

**D15 — Variant switch (`build_variant`) + one-thing-at-a-time ablation.**
Why: every variant differs by exactly one component and is scored identically, so any
accuracy delta is attributable. V0→V1 = RAG; V1→V2 = multi-agent; etc.

**D16 — Deterministic evaluation (temperature = 0).**
Why: at default temperature, V1−V0 swung run-to-run — one run **+2 pts (McNemar p=0.69)**,
another **exactly 0 (9/9/132, p=1.0)** — i.e. the sampling noise was *larger than the effect*.
`build_variant` now resolves a string model to `ChatOllama(temperature=0)`, so runs are
reproducible and paired McNemar isn't swamped by sampling.

---

## Refactor — config-driven LangGraph

**D17 — Every variant is a `StateGraph` assembled from a `RunConfig`, not hand-written code.**
Why: the ladder needs each variant to differ by *exactly one knob* and stay evaluated identically.
A preset `RunConfig` per variant compiled by `build_graph(cfg)` makes each difference a config field
(`rag`, `answer_role`, `panel_size`, `aggregate`, `verify`, per-role models) rather than a separate
code path — so A/B tests (medcpt vs nomic, panel size, verifier on/off) need no new code. `RunConfig`
stays internal; `build_variant(id, **overrides)` is the only public surface. Prompts moved verbatim to a
`roles.py` registry; nodes are Agent-backed factories in `graph/nodes.py`. The refactor was **behavior-
preserving**: a golden captured from the pre-refactor V0/V1/V2 at temp=0 is replayed through the graph in
`tests/test_golden.py`, asserting **byte-identical** per-item answers (passes live). V3/V4 were then added
purely as new config toggles (panel/aggregate/verify nodes), no change to V0–V2.

**D18 — Panel opinions are diversified by *perspective*, not temperature.**
Why: at temp=0 (D16) two identical specialists would return identical opinions, defeating the panel.
Each panelist gets a distinct instruction (`roles.PERSPECTIVES`: favor-most-likely vs rule-out-dangerous),
so diversity is deterministic and reproducible. Late nodes (aggregate/verify) fall back to the prior
answer on parse failure, never emitting `None`.

---

## Status of evidence
Deterministic run (qwen2.5:14b, **temperature=0**, 80 test items):

| | V0 | V1 | V2 |
|---|---|---|---|
| accuracy | 0.662 | 0.625 | **0.738** |

Paired: **V1 vs V0 −3.7 pts (p=0.63, n.s.)**, **V2 vs V0 +7.5 pts (p=0.18, n.s.)**, **V2 vs V1 +11.3 pts
(p=0.023, significant)**.

Takeaways:
- **RAG (V1) is not the lever on MedQA** — ≤ V0 across every run (also 150-item temp=0: 0.727 vs 0.720).
- **Multi-agent reasoning (V2) is the first real gain** — highest accuracy and a *significant* win over V1.
  The lift is from reasoner→specialist reasoning, not retrieval.
- **V2 vs V0 needs confirmation at larger n** (n=80 small-sample; p=0.18).
- **V3/V4 built** (panel+attending+verifier, and its no-verifier ablation) and functionally verified, but
  not yet measured at scale.
Next: confirm V2>V0 at 150+ items; measure V3/V4 (and V3−V4 = verifier's marginal effect); optional
MedCPT-vs-nomic A/B.
