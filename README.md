# MedQA-USMLE Answerer Variant Ladder

A local medical **question-answering** system on MedQA-USMLE, built as an ablation ladder of
variants (V0–V5) and measured with paired significance. Stack: **LangChain v1 + LangGraph + Ollama**
(fully local, no API keys).

Every variant is a **LangGraph `StateGraph`** assembled from an internal `RunConfig`;
`build_variant(id)` is the single switch — `V0` through `V5`.

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
ollama pull qwen2.5:7b         # answering model used for every reported number
ollama pull qwen3-embedding:4b # RAG embeddings (V1-V5's shared MedMCQA collection)
```
Any tool-calling chat model works — override with `-m`.

### Using a cloud provider instead

Model specs are `provider:model`; a bare string means Ollama, so `qwen2.5:7b` keeps working.

| spec | provider | needs |
|---|---|---|
| `qwen2.5:7b` | Ollama (default, used for every reported number) | — |
| `anthropic:claude-sonnet-5` | Anthropic | `langchain-anthropic`, `ANTHROPIC_API_KEY` |
| `openai:gpt-4o` | OpenAI | `langchain-openai`, `OPENAI_API_KEY` |
| `google:gemini-2.0-flash` | Google | `langchain-google-genai`, `GOOGLE_API_KEY` |
| `openrouter:meta-llama/llama-3.3-70b-instruct` | OpenRouter | `langchain-openai`, `OPENROUTER_API_KEY` |

```bash
.venv/bin/pip install -r requirements-providers.txt   # or just the one you need
export ANTHROPIC_API_KEY=...
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V0 -n 20 -m anthropic:claude-sonnet-5
```
Provider packages are imported lazily — only the one you actually use must be installed.

## 3. Build the RAG knowledge base (needed for V1–V5)

V1–V5 all retrieve from the **same single collection** — ~182k solved MedMCQA questions, embedded
with `qwen3-embedding:4b`. V0 needs none. This downloads MedMCQA from HuggingFace and embeds it
locally — expect it to take a while the first time.

```bash
PYTHONPATH=src .venv/bin/python -c "
from agent_hospital.knowledge import open_store, ingest_medmcqa, default_embeddings
ingest_medmcqa(182_822, store=open_store('knowledge_medmcqa_qwen3', embeddings=default_embeddings('qwen3-embedding:4b')), verbose=True)"
```
- Progress prints per batch (`embedded N`).
- If the HuggingFace download throttles, authenticate first: `.venv/bin/hf auth login`.

## 4. Run the predict scripts

```bash
# One prediction file per variant (resumable — safe to re-run after an interruption)
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V0 -s test -n 600 -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V1 -s test -n 600 -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V2 -s test -n 600 -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V3 -s test -n 600 -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V4 -s test -n 600 -m qwen2.5:7b
PYTHONPATH=src .venv/bin/python -m agent_hospital.predict -v V5 -s test -n 600 -m qwen2.5:7b

# Leaderboard + paired comparison (accuracy gain, Win/Loss/Tie, McNemar) vs a baseline
PYTHONPATH=src .venv/bin/python -m agent_hospital.evaluate --baseline V0
```
`-v` picks the variant (V0-V5), `-s` the split (`train`/`validation`/`test`), `-n` how many
questions (0 = whole split), `--start N` where to begin in the split (so a slice can start
anywhere), `-m` the model spec. `predict.py` writes
`data/eval_runs/<variant>_<split>_<model>.jsonl` (one record per item: prediction, gold,
correctness, latency, rationale) plus a `.meta.json` sidecar. `evaluate.py` reads every
`*.jsonl` there and reports accuracy, bootstrap CI, invalid-rate, Win/Loss/Tie, and McNemar vs
the baseline.

Add `--trace` to also write a `.traces.jsonl` sidecar holding each item's per-node graph
deltas (what the case reasoner understood, the passages RAG retrieved, the clinical report,
whether the verifier changed the answer). It is a separate file because a trace is far larger
than a prediction and nothing scores it; the web UI in step 5 reads it for drill-down.

**Note on V3/V5 (long-term memory):** the decider/verifier read and write a SQLite-backed lesson
bank (`data/longterm/lessons.sqlite`) as they go. A fresh run starts that bank empty, so it fills
up case-by-case in the run's own order — a different order or a smaller slice will not
reproduce the same accumulated lessons exactly.

`data/chroma/` (the RAG index, step 3), `data/eval_runs/` (predictions, step 4), and
`data/longterm/` (V3/V5's SQLite lesson/mistake bank) are all gitignored — each collaborator
builds them locally.

## 5. Web UI

```bash
PYTHONPATH=src .venv/bin/python -m agent_hospital.web.app   # http://127.0.0.1:8000
```

A switch in the header picks the run mode (remembered per browser; `?mode=attack` links straight
to it). **Normal run** has four tabs over the same graph code the CLI uses:

- **Ask** — run one question live (a MedQA item or your own) through one version or all six
  side by side, watching each node light up as it finishes. Click any node to see exactly what
  it wrote: the raw retrieved passages next to the digest that replaced them, the clinical
  report, the verifier's verdict.
- **Batch** — launch a `predict` run for a chosen version and split, starting at any question
  number and covering as many as you want, then watch its progress and running accuracy. Runs
  are subprocesses, so stopping one leaves a valid file that a later run resumes from. Clear a
  finished run (or all of them) to delete its result files.
- **Results** — the leaderboard and paired comparison as tables, then per-question drill-down
  (filter to wrong answers, see gold vs predicted, read the stored trace, compare one question
  across every version you have run).
- **Architecture** — each version's real graph topology, generated from the compiled graph, the
  configuration it adds over the rung below it, and the role prompts each agent is given.

Questions asked from the Ask tab never write to the long-term lesson bank, so poking at the UI
cannot contaminate a scored run.

**Attack & defend** has one tab, **Stream eval**, with settings on the left and runs on the right
(draggable divider, full window). Each run is one model with harmful-medical prompts mixed into
MedQA, under one of:
- a **harness (H)**, where the model sits inside a guarded graph (`graph/guarded.py`: `sysprompt`,
  `gatetool`, `gatenodes`) and every node is the same model;
- a **defense (D0–D5)**, where a guard outside the model is run by an un-attacked one
  (`eval_mixed.py --guard`: none, system, gate, verify, gate+verify, memory).

Picking neither means model only. Details: [docs/redteam/WEB_UI.md](docs/redteam/WEB_UI.md).
