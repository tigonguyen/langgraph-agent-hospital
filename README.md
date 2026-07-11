# Agent Hospital — MedQA-USMLE Variant Ladder

A local medical **question-answering** system on MedQA-USMLE, built as an ablation ladder of
variants (V0–V4) and measured with paired significance. Stack: **LangChain v1 + LangGraph + Ollama**
(fully local, no API keys).

| id | variant | status |
|----|---------|--------|
| V0 | Direct LLM | ✅ |
| V1 | RAG-only (reasoning-distilled query) | ✅ |
| V2 | Multi-agent (reasoner + specialist) | ✅ |
| V3 | Full system (+ experience base) | ⬜ |
| V4 | Full system without verifier | ⬜ |

Design details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · decisions: [docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md).

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
ollama pull qwen2.5:14b       # default answering model
ollama pull nomic-embed-text  # RAG embeddings
```
Any tool-calling chat model works — override with `model="..."` (e.g. `qwen2.5:7b` is faster).

## 3. Build the RAG knowledge base (needed for V1/V2)

Downloads the MedRAG **Textbooks** corpus (~125k snippets) and embeds it into a local Chroma store at
`data/chroma/` (gitignored, ~2.3 GB). One-time, ~1 hour on Apple Silicon.

```bash
PYTHONPATH=src .venv/bin/python -m agent_hospital.knowledge.ingest 130000
```
- Progress prints per batch (`embedded N`).
- If the HuggingFace download throttles, authenticate first: `.venv/bin/hf auth login`.
- V0 needs no knowledge base; V1/V2 do.

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
for vid in ["V0", "V1", "V2"]:
    recs[vid] = run_variant(items, build_variant(vid, model="qwen2.5:14b"))  # temp=0 by default
    lo, hi = bootstrap_ci(recs[vid])
    print(f"{vid}: acc={accuracy(recs[vid]):.3f}  CI=[{lo:.3f},{hi:.3f}]")

m = mcnemar(recs["V2"], recs["V0"])
print("V2 vs V0 McNemar p =", round(m["p_value"], 4))
PY
```
`build_variant(id, model=…, temperature=0, **overrides)` is the single entry point; the metrics harness
(`qa/metrics.py`) gives accuracy, bootstrap CI, invalid-rate, Win/Loss/Tie, McNemar, and latency.

## Optional — MedCPT (medical-domain embedder)

MedCPT often retrieves better than `nomic` but is heavier. See
[docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md) (D11–D13) for the full rationale and the
`curl` download commands. In short:

```bash
uv pip install --python .venv -r requirements-medcpt.txt
# download the two encoders into data/medcpt/ (curl — see docs), then re-ingest into its own collection:
PYTHONPATH=src .venv/bin/python -c "
from agent_hospital.knowledge import open_store, ingest_textbooks, default_embeddings
ingest_textbooks(store=open_store('knowledge_medcpt', embeddings=default_embeddings('medcpt')))"
```

## Project layout

```
src/agent_hospital/
  agents/base.py      # Agent — lazy create_agent wrapper (any provider/model)
  diseases/           # MedQA-USMLE loader
  knowledge/          # RAG: ingest · embeddings (nomic/medcpt) · retriever
  qa/                 # variants (V0/V1/V2) · reasoning · metrics
tests/                # offline + live (auto-skip) tests
docs/                 # ARCHITECTURE · TECHNICAL_DECISIONS · PLAN · REFACTOR_PLAN
```

**Note:** `data/chroma/` and `data/medcpt/` are gitignored (large, regenerable) — each collaborator
builds the knowledge base locally via step 3.
