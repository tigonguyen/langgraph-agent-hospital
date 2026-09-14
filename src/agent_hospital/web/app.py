"""FastAPI app for inspecting the V0-V5 ladder.

    PYTHONPATH=src .venv/bin/python -m agent_hospital.web.app

Four capabilities, all over the same graph code the CLI uses:
  - /api/ask      run one question live and stream each node as it finishes (SSE)
  - /api/runs     launch + monitor batch `predict` runs, and list past ones
  - /api/metrics  the leaderboard/paired tables as JSON (reuses qa/metrics.py verbatim)
  - /api/architecture  each variant's real topology, config and role prompts
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import asdict, replace
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_hospital import roles
from agent_hospital.diseases import load_medqa_usmle
from agent_hospital.diseases.medqa_usmle import MCQItem
from agent_hospital.evaluate import load_all, load_records
from agent_hospital.models import default_model
from agent_hospital.predict import DEFAULT_OUT_DIR, _paths
from agent_hospital.qa import VARIANTS, build_variant
from agent_hospital.qa.metrics import (accuracy, bootstrap_ci, invalid_rate, mcnemar,
                                       mean_latency, mean_tokens, win_loss_tie)
from agent_hospital.qa.variants import _PRESETS
from agent_hospital.web import runs as runs_mod

_LETTERS = "ABCD"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = runs_mod._repo_root()

app = FastAPI(title="Agent Hospital — V0-V5 inspector")


# --- caches ----------------------------------------------------------------
# A split is ~10k items; re-materializing it per keystroke would be wasteful even though
# HF caches the download.
_splits: dict[str, list[MCQItem]] = {}
# build_variant constructs agents and opens Chroma on first retrieval, so caching by
# (variant, model) turns every question after the first into a warm run.
_variants: dict[tuple[str, str], Any] = {}
_variants_lock = threading.Lock()


def _split(name: str) -> list[MCQItem]:
    if name not in _splits:
        _splits[name] = load_medqa_usmle(name)
    return _splits[name]


def _variant_fn(variant: str, model: str):
    key = (variant, model)
    with _variants_lock:
        if key not in _variants:
            # long_term_read_only: ad-hoc UI runs must never write into the lesson bank that
            # scored runs recall from (config.py:37-39). It also keeps this process out of
            # lessons.sqlite as a writer while a batch subprocess may be writing to it.
            _variants[key] = build_variant(variant, model=model, long_term_read_only=True)
        return _variants[key]


def _item_json(it: MCQItem) -> dict:
    return {"id": it.id, "question": it.question, "options": list(it.options),
            "answer_idx": it.answer_idx}


def _nodes_of(variant: str) -> list[str]:
    from agent_hospital.graph import build_graph
    # Compiling is lazy (no network, no vector store), so this is cheap at import time.
    g = build_graph(replace(_PRESETS[variant], model="dummy"))
    return [n for n in g.get_graph().nodes if not n.startswith("__")]


# --- metadata --------------------------------------------------------------

@app.get("/api/variants")
def api_variants() -> dict:
    return {"variants": [{"id": v, "label": VARIANTS[v], "nodes": _nodes_of(v)} for v in VARIANTS],
            "default_model": default_model(),
            "splits": list(runs_mod.SPLIT_SIZES)}


@app.get("/api/items")
def api_items(split: str = "test", offset: int = 0, limit: int = 20, q: str = "") -> dict:
    items = _split(split)
    if q:
        needle = q.lower()
        items = [it for it in items if needle in it.question.lower() or needle in it.id]
    return {"total": len(items),
            "items": [_item_json(it) for it in items[offset:offset + limit]]}


# --- live single-question run (SSE) ----------------------------------------

class AskRequest(BaseModel):
    variants: list[str] = []
    variant: str | None = None
    model: str | None = None
    split: str = "test"
    item_id: str | None = None
    question: str | None = None
    options: list[str] | None = None
    answer_idx: int | None = None      # optional for a custom question; gold if given

    def variant_ids(self) -> list[str]:
        ids = self.variants or ([self.variant] if self.variant else [])
        bad = [v for v in ids if v not in VARIANTS]
        if bad:
            raise HTTPException(400, f"unknown variant(s): {bad}")
        if not ids:
            raise HTTPException(400, "pass `variant` or `variants`")
        return ids

    def item(self) -> MCQItem:
        if self.question:
            opts = self.options or []
            if len(opts) != 4:
                raise HTTPException(400, "a custom question needs exactly 4 options")
            # answer_idx=-1 marks "no gold" — nothing scores this item, so it never matches.
            return MCQItem(id="custom", question=self.question, options=opts,
                           answer_idx=self.answer_idx if self.answer_idx is not None else -1)
        if not self.item_id:
            raise HTTPException(400, "pass `item_id` (with `split`) or a custom `question`")
        for it in _split(self.split):
            if it.id == self.item_id:
                return it
        raise HTTPException(404, f"no item {self.item_id!r} in split {self.split!r}")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    ids, item, model = req.variant_ids(), req.item(), req.model or default_model()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(event: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def work() -> None:
        # The graph blocks for tens of seconds per node, so it runs on a worker thread and
        # events come back through the queue — otherwise uvicorn's loop could not flush the
        # stream and the whole response would arrive at the end, defeating the point.
        try:
            for vid in ids:
                emit("start", {"variant": vid, "label": VARIANTS[vid], "nodes": _nodes_of(vid),
                               "item": _item_json(item)})
                t = time.time()
                try:
                    fn = _variant_fn(vid, model)
                    res = fn(item, lambda step, v=vid: emit("node", {"variant": v, **asdict(step)}))
                except Exception as exc:      # one variant failing must not kill the rest
                    emit("error", {"variant": vid, "message": f"{type(exc).__name__}: {exc}"})
                    continue
                ans = res.answer
                emit("done", {
                    "variant": vid, "answer": ans,
                    "answer_letter": _LETTERS[ans] if ans is not None else None,
                    "gold": item.answer_idx if item.answer_idx >= 0 else None,
                    "gold_letter": _LETTERS[item.answer_idx] if item.answer_idx >= 0 else None,
                    "correct": (ans == item.answer_idx) if item.answer_idx >= 0 else None,
                    "rationale": res.rationale, "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out, "latency_s": round(time.time() - t, 2),
                })
        finally:
            emit("__eof__", {})

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while True:
            event, data = await queue.get()
            if event == "__eof__":
                break
            yield _sse(event, data)

    # Variants run sequentially: the local model serves one request at a time, so running
    # them concurrently would only make the reported latencies lie.
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- batch runs ------------------------------------------------------------

class RunRequest(BaseModel):
    variant: str
    split: str = "test"
    model: str | None = None
    limit: int = 10
    start: int = 0          # skip this many items, so a slice can begin anywhere
    trace: bool = True


@app.post("/api/runs")
def api_start_run(req: RunRequest) -> dict:
    if req.variant not in VARIANTS:
        raise HTTPException(400, f"unknown variant {req.variant!r}")
    try:
        run = runs_mod.start(req.variant, req.split, req.model or default_model(),
                             req.limit, req.start, req.trace)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"run_id": run.run_id, "status": "running"}


@app.get("/api/runs")
def api_runs() -> dict:
    return {"runs": runs_mod.list_runs()}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    for row in runs_mod.list_runs():
        if row["run_id"] == run_id:
            return {**row, "log_tail": runs_mod.log_tail(run_id)}
    raise HTTPException(404, f"no run {run_id!r}")


@app.post("/api/runs/{run_id}/stop")
def api_stop_run(run_id: str) -> dict:
    try:
        return runs_mod.stop(run_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: str) -> dict:
    """Delete a finished run's files, so the results list can be cleared."""
    try:
        return runs_mod.delete(run_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


# --- results + metrics -----------------------------------------------------

def _run_paths(run_id: str) -> tuple[str, str]:
    """(prediction file, trace sidecar) for a run id, which IS the file basename."""
    d = os.path.join(_ROOT, DEFAULT_OUT_DIR)
    pred = os.path.join(d, run_id + ".jsonl")
    if not os.path.exists(pred):
        raise HTTPException(404, f"no prediction file for run {run_id!r}")
    return pred, os.path.join(d, run_id + ".traces.jsonl")


def _meta_of(run_id: str) -> dict:
    path = os.path.join(_ROOT, DEFAULT_OUT_DIR, run_id + ".meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@app.get("/api/metrics")
def api_metrics(baseline: str = "V0") -> dict:
    """The JSON sibling of evaluate.py's text tables — same functions, no new metric math."""
    runs = load_all(os.path.join(_ROOT, DEFAULT_OUT_DIR))
    board = []
    for variant, recs in runs.items():
        lo, hi = bootstrap_ci(recs)
        board.append({"variant": variant, "n": len(recs), "accuracy": accuracy(recs),
                      "ci": [lo, hi], "invalid_rate": invalid_rate(recs),
                      "mean_latency": mean_latency(recs), "mean_tokens": mean_tokens(recs)})
    paired = []
    if baseline in runs:
        base = runs[baseline]
        for variant, recs in runs.items():
            if variant == baseline:
                continue
            w, l, t = win_loss_tie(recs, base)
            paired.append({"variant": variant, "baseline": baseline,
                           "gain": accuracy(recs) - accuracy(base), "wins": w, "losses": l,
                           "ties": t, "n_paired": w + l + t,
                           "p_value": mcnemar(recs, base)["p_value"]})
    return {"leaderboard": board, "paired": paired, "baseline": baseline}


@app.get("/api/runs/{run_id}/items")
def api_run_items(run_id: str, offset: int = 0, limit: int = 50, filter: str = "all") -> dict:
    pred, _ = _run_paths(run_id)
    split = _meta_of(run_id).get("split", "test")
    by_id = {it.id: it for it in _split(split)}
    rows = []
    with open(pred) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if filter == "wrong" and r["correct"]:
                continue
            if filter == "invalid" and r["valid"]:
                continue
            it = by_id.get(r["item_id"])
            rows.append({**r, "question": it.question if it else "",
                         "options": list(it.options) if it else []})
    return {"total": len(rows), "items": rows[offset:offset + limit]}


@app.get("/api/runs/{run_id}/items/{item_id}")
def api_run_item(run_id: str, item_id: str) -> dict:
    pred, trace_path = _run_paths(run_id)
    record = None
    with open(pred) as f:
        for line in f:
            if line.strip() and json.loads(line)["item_id"] == item_id:
                record = json.loads(line)
                break
    if record is None:
        raise HTTPException(404, f"no item {item_id!r} in run {run_id!r}")
    trace = None
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                if line.strip() and json.loads(line)["item_id"] == item_id:
                    trace = json.loads(line)["trace"]
                    break
    split = _meta_of(run_id).get("split", "test")
    it = next((i for i in _split(split) if i.id == item_id), None)
    return {"record": record, "trace": trace, "item": _item_json(it) if it else None}


@app.get("/api/items/{item_id}/compare")
def api_compare(item_id: str, split: str = "test") -> dict:
    """One question across every run on disk — the past-results counterpart to /api/ask's
    live compare."""
    it = next((i for i in _split(split) if i.id == item_id), None)
    out = []
    for row in runs_mod.list_runs():
        pred = os.path.join(_ROOT, DEFAULT_OUT_DIR, row["run_id"] + ".jsonl")
        if not os.path.exists(pred):
            continue
        with open(pred) as f:
            for line in f:
                if line.strip() and json.loads(line)["item_id"] == item_id:
                    r = json.loads(line)
                    out.append({"run_id": row["run_id"], "variant": row["variant"],
                                "pred_letter": r["pred_letter"], "correct": r["correct"],
                                "rationale": r["rationale"], "latency_s": r["latency_s"],
                                "has_trace": row["has_trace"]})
                    break
    return {"item": _item_json(it) if it else None, "runs": out}


# --- architecture ----------------------------------------------------------

_NODE_ROLES = {
    "answer": "the variant's answer_role (baseline for V0, decider for V2-V5)",
    "agent": "rag-agent, holding the search_medmcqa tool",
    "understand": "case-reasoner",
    "search": "retrieve (no LLM) + evidence-digest",
    "reasoning": "clinical-reasoner",
    "verify": "report-verifier-wikipedia (V4) or report-verifier (V5)",
    "distill_mistake": "mistake-analyst — the only node that reads gold",
}


@app.get("/api/architecture")
def api_architecture() -> dict:
    from agent_hospital.graph import build_graph

    out = []
    for v in VARIANTS:
        cfg = replace(_PRESETS[v], model="dummy")
        g = build_graph(cfg)
        conf = {k: val for k, val in asdict(cfg).items() if k != "model"}
        out.append({"id": v, "label": VARIANTS[v], "nodes": _nodes_of(v),
                    # Generated from the compiled graph, so this stays correct even though
                    # docs/diagrams/README.md flags some of its SVGs as stale.
                    "mermaid": g.get_graph().draw_mermaid(),
                    "config": conf, "diagram": f"/diagrams/{v.lower()}_flow.svg"})
    return {"variants": out, "roles": roles.ROLE_PROMPTS, "node_roles": _NODE_ROLES}


# --- static ----------------------------------------------------------------
# Mounted last so /api/* and /diagrams/* win over the catch-all.
_diagrams = os.path.join(_ROOT, "docs", "diagrams")
if os.path.isdir(_diagrams):
    app.mount("/diagrams", StaticFiles(directory=_diagrams), name="diagrams")
app.mount("/", StaticFiles(directory=os.path.join(_HERE, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
