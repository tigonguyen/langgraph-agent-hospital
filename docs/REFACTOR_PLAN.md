# Refactor Plan — Config-driven LangGraph

Rebuild `qa/` so every variant is a **LangGraph `StateGraph`** assembled from an internal
config, flexible on **agent roles, RAG, and model** — without changing V0–V2 behavior.
Decisions below were resolved by grill; this is the shared understanding.

## Locked decisions (grill)
- **Scope/order:** behavior-preserving **V0–V2 first**, then V3/V4.
- **Node primitive:** keep the `create_agent`-based `Agent` (tools/middleware-ready).
- **Assembly:** **config-driven** internally (`build_graph(RunConfig)`); presets map id→config.
- **Public API:** **presets + a few kwargs** (`build_variant(id, model=, temperature=, **overrides)`).
  `RunConfig` is internal, *not* a public object.
- **Answer extraction:** keep the **regex `parse_choice`** (0% invalid); structured output optional later.
- **Roles:** a **named role registry** (`roles.py`), config references role names; per-role model override.
- **State:** **plain-overwrite** `QAState` (no message reducer).
- **Layout:** `graph/` subpackage + top-level `config.py` + `roles.py`.
- **Preservation gate:** **exact match at temperature=0** — post-refactor V0/V1/V2 reproduce
  per-item answers **bit-identically** vs a golden captured from the current code.
- **Golden set:** **150 items from the TRAIN split** (test stays untouched for final eval).

## Keep as-is (do not touch)
`agents/base.py` (the `Agent`), all of `knowledge/` (retriever + embedders — already flexible),
`diseases/medqa_usmle.py`, `qa/metrics.py` (harness interface preserved).

## Target structure
```
config.py          RunConfig, RagConfig            (the flexibility surface, internal)
roles.py           ROLE_PROMPTS registry           (reasoner, baseline, rag-answerer, specialist, …)
graph/
  state.py         QAState (TypedDict, plain fields)
  nodes.py         node factories: reason, retrieve, answer, (later) panel, aggregate, verify, experience
  build.py         build_graph(cfg) -> compiled StateGraph
qa/
  mcq.py           format_mcq + parse_choice  (moved out of baseline.py)
  variants.py      VARIANT_CONFIGS presets + build_variant(id, model=, temperature=, **overrides)
  metrics.py       KEEP
```

## State & config
```python
class QAState(TypedDict, total=False):
    item: MCQItem
    query: str
    evidence: list[str]
    opinions: list[dict]      # V2/V3
    answer: int | None

@dataclass(frozen=True)
class RagConfig:
    collection: str = "knowledge"        # or "knowledge_medcpt"
    embedder: str = "nomic-embed-text"   # or "medcpt"
    k: int = 4; threshold: float = 0.5; distill_query: bool = True

@dataclass(frozen=True)
class RunConfig:
    model: str = "qwen2.5:14b"; temperature: float = 0.0
    role_models: dict[str, str] = {}     # per-role override
    answer_role: str = "baseline"        # which role answers
    rag: RagConfig | None = None         # None = no retrieval
    # Phase-2 toggles: panel_size:int=1, aggregate:bool, verify:bool, experience:bool
    def model_for(self, role): return self.role_models.get(role, self.model)
```

## Roles (verbatim move of existing prompts)
`roles.py` holds the current prompts unchanged (this is what guarantees exact match):
`reasoner` = `REASONING_SYS`, `baseline` = `BASELINE_SYS`, `rag-answerer` = `RAG_SYS`,
`specialist` = `SPECIALIST_SYS`. (Later: `attending`, `verifier`.)

## Nodes (Agent-backed, built once per graph)
- `reason_node(cfg)` → `query` (reasoner agent; skipped/`query=item.question` if no distill)
- `retrieve_node(cfg)` → `evidence` via `open_store(cfg.rag.collection, default_embeddings(cfg.rag.embedder))` + gated `retrieve`
- `answer_node(cfg)` → `answer` = `parse_choice(agent(role=cfg.answer_role).say(evidence?+mcq))`

## Variant presets (config, not code)
| id | RunConfig | graph |
|---|---|---|
| V0 | `answer_role="baseline", rag=None` | `answer` |
| V1 | `answer_role="rag-answerer", rag=RagConfig()` | `reason → retrieve → answer` |
| V2 | `answer_role="specialist", rag=RagConfig()` | `reason → retrieve → answer` |

`build_variant(id, model=…, temperature=0, **overrides)` = `build_graph(preset.replace(**overrides))`,
returns `answer(item) = graph.invoke({"item": item})["answer"]` (uniform, harness-compatible).
Overrides let you A/B without new code, e.g. `build_variant("V1", rag=RagConfig(collection="knowledge_medcpt", embedder="medcpt"))`.

## Phases & verification
- **Phase 0 — Golden capture (before touching code):** run current V0/V1/V2 at temp=0 on **150
  TRAIN items**, save per-item answers → `tests/golden/v012_train150.json`.
- **Phase 1 — Refactor V0–V2:** add `config.py`, `roles.py` (verbatim prompts), `graph/*`, `qa/mcq.py`;
  reimplement `build_variant` on `build_graph`. *Gate:* a live test replays the 150-train golden and
  asserts **every answer matches exactly** (skips if Ollama down). Existing offline variant tests still pass.
- **Phase 2 — V3/V4:** add `panel`/`aggregate`/`verify`/`experience` nodes + config toggles; V3 = full,
  V4 = V3 without verify. *Verify:* offline stub tests + live accuracy on test split.
- **Phase 3 — Docs:** regenerate per-variant `draw_mermaid()` diagrams; update `ARCHITECTURE.md` +
  `TECHNICAL_DECISIONS.md`.

## Risks / notes
- Exact match requires **byte-identical prompts + same node order + deterministic retrieval** (Chroma is
  deterministic; temp=0 reasoner → deterministic query). Any prompt tweak breaks the gate — so Phase 1
  changes *structure only*.
- Agents are built once in `build_graph` and reused across items (ChatOllama is stateless) — same outputs.
- No commits until the user says (per CLAUDE.md §5).
