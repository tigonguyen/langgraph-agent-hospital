# Agent Hospital — MedQA-USMLE Variant Ladder

A local medical **question-answering** system on MedQA-USMLE, built as an ablation ladder of
variants (V0–V5) and measured with paired significance. Stack: **LangChain v1 + LangGraph + Ollama**
(fully local, no API keys).

| id | variant | graph | status |
|----|---------|-------|--------|
| V0 | Direct LLM | `answer` | ✅ measured |
| V1 | RAG-only — single agent, agentic tool call (`search_medmcqa`) | `agent` | ✅ measured |
| V2 | Multi-agent — case-reasoner → (search ‖ reasoning) → decider | `understand → (search ‖ reasoning) → answer` | ✅ measured |
| V3 | V2 + decider has long-term memory (cross-episode lesson bank) | `understand → (search ‖ reasoning) → answer` | ⚠️ builds & runs, partially measured |
| V4 | V3 + verifier grounded in live Wikipedia | `understand → (search ‖ reasoning) → answer → verify` | ⚠️ measurement in progress |
| V5 | V3 + verifier that recalls its own evolving mistake bank | `understand → (search ‖ reasoning) → answer → verify → distill_mistake` | ⚠️ builds & runs, partially measured |

Every variant is a **LangGraph `StateGraph`** assembled from an internal `RunConfig`; `build_variant(id)`
is the single switch. Design details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · decisions:
[docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md) · current measured numbers:
[docs/report/report.tex](docs/report/report.tex) (evaluation is ongoing — treat any number there as a
snapshot, not final).

---

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** (or use `python -m venv` + `pip`)
- **[Ollama](https://ollama.com)** running locally

## 1. Setup

```bash
git clone git@github.com:tigonguyen/langgraph-agent-hospital.git
cd langgraph-agent-hospital

uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

<details><summary>Without uv (plain venv + pip)</summary>

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
</details>

## 2. Pull the Ollama models

```bash
ollama pull qwen2.5:14b        # default answering model
ollama pull qwen3-embedding:4b # RAG embeddings (V1-V5's shared MedMCQA collection)
```
Any tool-calling chat model works — override with `model="..."` (e.g. `qwen2.5:7b` is faster).

### Using a cloud provider instead

Model specs are `provider:model`; a bare string means Ollama, so `qwen2.5:7b` keeps working.

| spec | provider | needs |
|---|---|---|
| `qwen2.5:14b` | Ollama (default) | — |
| `anthropic:claude-sonnet-5` | Anthropic | `langchain-anthropic`, `ANTHROPIC_API_KEY` |
| `openai:gpt-4o` | OpenAI | `langchain-openai`, `OPENAI_API_KEY` |
| `google:gemini-2.0-flash` | Google | `langchain-google-genai`, `GOOGLE_API_KEY` |
| `openrouter:meta-llama/llama-3.3-70b-instruct` | OpenRouter | `langchain-openai`, `OPENROUTER_API_KEY` |

```bash
.venv/bin/pip install -r requirements-providers.txt   # or just the one you need
export ANTHROPIC_API_KEY=...
PYTHONPATH=src .venv/bin/python -m agent_hospital -v V0 -n 20 -m anthropic:claude-sonnet-5
```
Provider packages are imported lazily — only the one you actually use must be installed.

## 3. Build the RAG knowledge base (needed for V1–V5)

V1–V5 all retrieve from the **same single collection** — solved MedMCQA questions, embedded with
`qwen3-embedding:4b` — so only one ingest is needed. V0 needs none.

```bash
# MedMCQA (V1-V5): ~182k solved board questions, qwen3-embedding:4b, no gate
PYTHONPATH=src .venv/bin/python -c "
from agent_hospital.knowledge import open_store, ingest_medmcqa, default_embeddings
ingest_medmcqa(182_822, store=open_store('knowledge_medmcqa_qwen3', embeddings=default_embeddings('qwen3-embedding:4b')), verbose=True)"
```
- Progress prints per batch (`embedded N`).
- If the HuggingFace download throttles, authenticate first: `.venv/bin/hf auth login`.

## 4. Run the tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/
```
- Offline tests run anywhere. **Live tests** need Ollama + the models and **auto-skip** if unavailable.
- Pick the live-test model: `AGENT_HOSPITAL_TEST_MODEL=qwen2.5:7b .venv/bin/python -m pytest tests/`.

## 5. Run the variants

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from agent_hospital.diseases import load_medqa_usmle
from agent_hospital.qa import build_variant, run_variant, accuracy, bootstrap_ci, mcnemar

items = load_medqa_usmle("test", limit=50)          # a small slice; use more for real numbers
recs = {}
for vid in ["V0", "V1", "V2", "V3", "V4", "V5"]:
    recs[vid] = run_variant(items, build_variant(vid, model="qwen2.5:14b"))  # temp=0 by default
    lo, hi = bootstrap_ci(recs[vid])
    print(f"{vid}: acc={accuracy(recs[vid]):.3f}  CI=[{lo:.3f},{hi:.3f}]")

m = mcnemar(recs["V2"], recs["V0"])
print("V2 vs V0 McNemar p =", round(m["p_value"], 4))
PY
```
`build_variant(id, model=…, temperature=0, **overrides)` is the single entry point; the metrics harness
(`qa/metrics.py`) gives accuracy, bootstrap CI, invalid-rate, Win/Loss/Tie, McNemar, and latency.

## 6. Official evaluation — predictions + metrics (spec §7-9)

```bash
# One prediction file per variant (resumable — safe to re-run after an interruption)
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V0 -s test -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V1 -s test -m qwen2.5:7b
# ... V2, V3, V4, V5

# Leaderboard + paired comparison (accuracy gain, Win/Loss/Tie, McNemar) vs a baseline
PYTHONPATH=src .venv/bin/python -m agent_hospital.evaluate --baseline V0
```
`predict.py` writes `data/eval_runs/<variant>_<split>_<model>.jsonl` (one record per item:
prediction, gold, correctness, latency, rationale) plus a `.meta.json` sidecar (model,
temperature, item count, timing — spec §10 reproducibility). `evaluate.py` reads every
`*.jsonl` there and reports the two required tables; `--out <path>` also writes them to a file.
Not yet covered: the error-analysis table and token/cost tracking (latency is the only cost
proxy today).

## Project layout

```
src/agent_hospital/
  agents/base.py      # Agent — lazy create_agent wrapper (any provider/model)
  models.py           # resolve_model — 'provider:model' spec → chat model
  config.py           # RunConfig, RagConfig — the per-variant flexibility surface
  roles.py            # ROLE_PROMPTS registry (baseline · rag-agent · case-reasoner ·
                      #   clinical-reasoner · evidence-digest · decider · report-verifier ·
                      #   report-verifier-wikipedia · mistake-analyst)
  graph/              # state.py (QAState) · nodes.py (node factories) · build.py (build_graph)
                      #   · longterm.py (cross-episode lesson/mistake bank, SQLite)
  diseases/           # MedQA-USMLE loader
  knowledge/          # RAG: ingest · embeddings (Ollama) · retriever · wikipedia
  qa/                 # variants (V0–V5 presets) · mcq (format/parse) · reasoning · metrics
  predict.py          # run a variant over a split, write per-item predictions (.jsonl + .meta.json)
  evaluate.py         # leaderboard + paired-comparison tables from prediction files
tests/                # offline + live (auto-skip) tests · golden/ (exact-match gate)
docs/                 # ARCHITECTURE · TECHNICAL_DECISIONS · PLAN · REFACTOR_PLAN (historical)
```

**Note:** `data/chroma/`, `data/eval_runs/`, and `data/longterm/` (V3–V5's SQLite
lesson/mistake bank) are gitignored (large/regenerable or run-specific) — each collaborator builds
the knowledge base locally via step 3, and prediction
runs live under `data/eval_runs/` by default.
