# Report Notes

Working notes for the midterm report, organised against the required report structure.
Sections are filled in as each variant is built. **Status: architecture + V0 complete.**

Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md) (what is wired to what) ·
[TECHNICAL_DECISIONS.md](TECHNICAL_DECISIONS.md) (why, with evidence).

---

## 2. System Architecture

### The organising idea
Every variant is the **same graph skeleton with different nodes switched on**. A variant is
not a separate code path — it is a preset `RunConfig` compiled into a LangGraph `StateGraph`
by `build_graph(cfg)`. This is what makes the ablation honest: V0→V1 differs by exactly the
RAG nodes, V2→V3 by exactly the panel/attending/verifier, V3→V4 by exactly the verifier.

```
[reason → retrieve] → (answer | panel → aggregate) → [verify]
```

| variant | nodes actually wired | agents |
|---|---|---|
| V0 | `answer` | 1 |
| V1 | `reason → retrieve → answer` | 2 |
| V2 | `reason → retrieve → answer` (specialist role) | 2 |
| V3 | `reason → retrieve → panel → aggregate → verify` | 5 |
| V4 | `reason → retrieve → panel → aggregate` | 4 |

### Module map
| module | responsibility |
|---|---|
| `config.py` | `RunConfig` / `RagConfig` — the only flexibility surface |
| `roles.py` | `ROLE_PROMPTS` registry — every system prompt lives here |
| `graph/state.py` | `QAState` — `item`, `query`, `evidence`, `opinions`, `answer` |
| `graph/nodes.py` | node factories, one per pipeline stage |
| `graph/build.py` | `build_graph(cfg)` — assembles + compiles the graph |
| `qa/variants.py` | `_PRESETS` (V0–V4) + `build_variant()` — the public switch |
| `qa/mcq.py` | `format_mcq` (prompt) + `parse_choice` (answer extraction) |
| `qa/metrics.py` | `EpisodeRecord` + all metrics (accuracy, CI, WLT, McNemar, latency) |
| `models.py` | `resolve_model` — `provider:model` → chat model |
| `knowledge/` | RAG: ingest · embeddings · retriever |
| `agents/base.py` | `Agent` — lazy `create_agent` wrapper, provider-agnostic |

### Design properties worth stating in the report
- **Laziness** — agents build on first call, the Chroma store opens on first retrieval. Compiling
  a graph touches no network, so tests and graph construction are offline-safe.
- **Uniform interface** — every variant is `answer(item) -> int | None`, so one metrics harness
  scores all of them identically.
- **Behaviour-preservation gate** — `tests/test_golden.py` replays pre-refactor V0/V1/V2 answers
  and asserts byte-identical results, so refactors provably don't move the numbers.

---

## 3. System Design — V0 (Direct LLM baseline)

### Purpose
The **control**. Every reported gain is `Accuracy(Vn) − Accuracy(V0)`, so V0 must be the purest
possible single-call LLM: no retrieval, no memory, no second agent, no tools. Its only input is
the question itself.

### Configuration
```python
RunConfig(answer_role="baseline", rag=None,
          panel_size=1, aggregate=False, verify=False)
```
`rag=None` skips reason/retrieve; `panel_size==1 and not aggregate` selects the single `answer`
node; `verify=False` adds nothing after. Compiled graph: **`START → answer → END`** (one node).

### Prompt (verbatim — required for §10 reproducibility)
System prompt (`roles.py:BASELINE`):
> You are an expert physician answering a medical board (USMLE) multiple-choice question.
> Choose the single best answer.

User turn (`qa/mcq.py:format_mcq`):
```
{question}

A. {option 0}
B. {option 1}
C. {option 2}
D. {option 3}

Respond with ONLY the letter (A, B, C, or D) of the best answer.
```
No evidence is prepended — `_prompt()` finds `state["evidence"]` empty because no retrieve node ran.

### Answer extraction
`parse_choice` tries `Answer: X` / `answer is X` (case-insensitive), then falls back to the first
standalone `A|B|C|D`. Returns index `0–3`, or `None` = **invalid** (never a silent wrong).
In practice the model replies a bare `"B"` and the fallback catches it — **invalid rate 0.0%**.

### Execution, condensed
```python
agent  = Agent("baseline", BASELINE, model=resolve_model(spec, temperature=0))
reply  = agent.say(format_mcq(item))   # exactly one LLM call, no tools
answer = parse_choice(reply)
```
`QAState` carries `item` in and `answer` out; `query` / `evidence` / `opinions` are never set.

---

## 4. Experimental Setup

| item | value |
|---|---|
| Dataset | `nnilayy/medqa-usmle` — train 10,178 / validation 1,272 / **test 1,273** |
| Dev split | `train` (debugging/tuning only) |
| Official split | `test`, n=1273 — the §7 denominator; used **once** |
| Default model | `qwen2.5:7b` (CLI default), `qwen2.5:14b` (library default) |
| Temperature | **0** everywhere (D16) — paired comparisons need determinism |
| Seed | `bootstrap_ci(seed=0)`; generation is greedy at temp=0 so no sampling seed applies |
| Hardware | Apple Silicon Mac, 48 GB unified memory, local Ollama |
| Providers | Ollama (local) or `anthropic:` / `openai:` / `google:` / `openrouter:` specs |

### Reproducing a run
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m agent_hospital -v V0 -n 150
```
Dataset is cached locally (`~/.cache/huggingface/datasets/`, 11 MB, all 3 splits) and loads with
no network — `HF_HUB_OFFLINE=1` just skips a slow metadata check that can 504.

---

## 5. Evaluation Results — V0

Prior deterministic run (qwen2.5:14b, temp=0, 80 test items): **V0 accuracy 0.662**,
95% bootstrap CI [0.562, 0.762], invalid rate 0.0%, 0.63 s/item. On a 150-item run V0 measured
**0.72**. Full-test (n=1273) numbers are **not yet collected**.

---

## 6. Methodology gotchas (learned the hard way — worth a Limitations mention)

1. **Model spin-up must be excluded from latency.** A cold Ollama model charges the first item
   ~1.9 s vs ~0.3 s steady-state. The CLI now runs a **synthetic** warm-up item first.
2. **Warm up with a synthetic item, never a scored one.** Warming with `items[0]` left that exact
   prompt in Ollama's cache and made item 1 look *artificially fast* (0.2 s vs ~1.0 s neighbours) —
   a measurement artifact in the opposite direction.
3. **Check `ollama ps` before timing.** A leftover resident model (e.g. `qwen2.5:32b`, 28 GB
   alongside 7b on a 48 GB machine) causes GPU contention that inflated **every** latency ~2×.
   Stop non-target models before an official timed run.
4. **Latency is dominated by generated tokens, not prompt length.** V0's longest prompt (1233
   chars) was among the *fastest* items; all replies are 1 character.
5. **Cloud models weaken determinism.** temp=0 is set, but APIs don't guarantee reproducibility
   the way local greedy decoding does. Keep the golden gate pinned to a local model.

---

## 7. Known gaps vs the spec (as of V0)

| spec § | gap |
|---|---|
| §3 | Output is an option index only — **no explanation** is captured or returned |
| §4.4/4.5 | **Short-term and long-term memory not built** (so current V3 ≠ spec's "full system") |
| §7 | **No cost/token tracking** (latency only) |
| §8 | No **error-analysis** table or tooling |
| §9 | No **prediction files**; CLI reports metrics but doesn't dump per-item predictions |
| §5 | Full **test-set (1273) run** not yet done |
| §12 | Report and slides not started |
