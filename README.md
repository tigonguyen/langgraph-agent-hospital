# Agent Hospital — MedQA-USMLE Variant Ladder

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
questions (0 = whole split), `-m` the model spec. `predict.py` writes
`data/eval_runs/<variant>_<split>_<model>.jsonl` (one record per item: prediction, gold,
correctness, latency, rationale) plus a `.meta.json` sidecar. `evaluate.py` reads every
`*.jsonl` there and reports accuracy, bootstrap CI, invalid-rate, Win/Loss/Tie, and McNemar vs
the baseline.

**Note on V3/V5 (long-term memory):** the decider/verifier read and write a SQLite-backed lesson
bank (`data/longterm/lessons.sqlite`) as they go. A fresh run starts that bank empty, so it fills
up case-by-case in the run's own order — a different order or a smaller slice will not
reproduce the same accumulated lessons exactly.

`data/chroma/` (the RAG index, step 3), `data/eval_runs/` (predictions, step 4), and
`data/longterm/` (V3/V5's SQLite lesson/mistake bank) are all gitignored — each collaborator
builds them locally.
