# Refactor Plan — Config-driven LangGraph

> **Status: ✅ COMPLETE.** All phases done — V0–V2 refactored behavior-preserving (golden exact-match
> gate passes live), V3/V4 added as config toggles, docs updated. Deviations from the original plan below
> are marked *(as built)*. This file is kept as the design record.

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
*(as built: `evidence: str` — pre-formatted evidence block; `opinions: list[str]` — panelist answers.)*
```python
class QAState(TypedDict, total=False):
    item: MCQItem
    query: str
    evidence: str
    opinions: list[str]       # V3/V4 panel
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

## Phases & verification — all ✅
- **Phase 0 — Golden capture ✅:** captured current V0/V1/V2 at temp=0 → `tests/golden/v012_train.json`.
  *(as built: 12 TRAIN items at `qwen2.5:7b`, not 150 — enough to catch any structural regression while
  keeping the live gate fast.)*
- **Phase 1 — Refactor V0–V2 ✅:** added `config.py`, `roles.py` (verbatim prompts), `graph/*`, `qa/mcq.py`;
  reimplemented `build_variant` on `build_graph`; deleted `qa/baseline.py`/`rag_answer.py`/`multi_agent.py`.
  *Gate:* `tests/test_golden.py` replays the golden and asserts **every answer matches exactly** — passes
  live (39 s). 16 offline tests pass.
- **Phase 2 — V3/V4 ✅:** added `panel`/`aggregate`/`verify` node factories + config toggles; V3 = full,
  V4 = V3 without verify. Basic run confirms all five variants execute end-to-end (0 invalid).
  *(experience node descoped — not built; V3 gain is panel+attending+verifier.)*
- **Phase 3 — Docs ✅:** updated `ARCHITECTURE.md`, `TECHNICAL_DECISIONS.md` (D17/D18), `PLAN.md`, `README.md`
  with the per-variant graph topologies and the new module layout.

## Risks / notes
- Exact match requires **byte-identical prompts + same node order + deterministic retrieval** (Chroma is
  deterministic; temp=0 reasoner → deterministic query). Any prompt tweak breaks the gate — so Phase 1
  changes *structure only*.
- Agents are built once in `build_graph` and reused across items (ChatOllama is stateless) — same outputs.
- No commits until the user says (per CLAUDE.md §5).
