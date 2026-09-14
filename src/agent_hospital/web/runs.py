"""Batch runs, launched as subprocesses of `agent_hospital.predict`.

A subprocess rather than a thread: the graph blocks on the model for minutes at a time, so
in-process it would starve the event loop, and a hung run could not be killed without
taking the server down with it.

Progress needs no IPC — `predict.py` flushes its JSONL per item, so counting that file's
lines IS the progress bar. That also means a run survives a server restart: the registry
is lost, but `list_runs` reads the files back and `predict` itself resumes from them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_hospital.predict import DEFAULT_OUT_DIR, _paths

# Display-only: saves materializing a whole split just to render "37 / 1273".
SPLIT_SIZES = {"train": 10178, "validation": 1272, "test": 1273}

LOG_DIR = "data/web_logs"
_LIVE: dict[str, "Run"] = {}


@dataclass
class Run:
    run_id: str
    variant: str
    split: str
    model: str
    limit: int
    pred_path: str
    log_path: str
    proc: subprocess.Popen = field(repr=False)
    started_at: str


def _repo_root() -> str:
    # src/agent_hospital/web/runs.py -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _count_and_hits(pred_path: str) -> tuple[int, int]:
    """(items done, items correct) straight from the prediction file."""
    if not os.path.exists(pred_path):
        return 0, 0
    done = hits = 0
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done += 1
            # A run killed mid-write can leave one partial line; it is not a failure.
            try:
                hits += bool(json.loads(line)["correct"])
            except (json.JSONDecodeError, KeyError):
                done -= 1
    return done, hits


def start(variant: str, split: str, model: str, limit: int = 0, begin: int = 0,
          trace: bool = True, out_dir: str = DEFAULT_OUT_DIR) -> Run:
    """Launch `predict` for one (variant, split, model). Raises if that run is already live."""
    pred_path, _meta, _tr = _paths(out_dir, variant, split, model)
    run_id = os.path.basename(pred_path)[: -len(".jsonl")]
    live = _LIVE.get(run_id)
    if live and live.proc.poll() is None:
        raise ValueError(f"run {run_id!r} is already running")

    root = _repo_root()
    os.makedirs(os.path.join(root, LOG_DIR), exist_ok=True)
    log_path = os.path.join(root, LOG_DIR, f"{run_id}.log")
    cmd = [sys.executable, "-m", "agent_hospital.predict",
           "-v", variant, "-s", split, "-m", model, "-n", str(limit), "-o", out_dir]
    if begin:
        cmd += ["--start", str(begin)]
    if trace:
        cmd.append("--trace")
    log = open(log_path, "a")
    # PYTHONPATH=src: the package is not installed (src-layout), same as the CLI's own usage.
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)

    run = Run(run_id=run_id, variant=variant, split=split, model=model, limit=limit,
              pred_path=os.path.join(root, pred_path), log_path=log_path, proc=proc,
              started_at=datetime.now(timezone.utc).isoformat())
    _LIVE[run_id] = run
    return run


def stop(run_id: str) -> dict:
    """Terminate a live run. Items already written stay durable and resumable."""
    run = _LIVE.get(run_id)
    if not run or run.proc.poll() is not None:
        raise ValueError(f"no live run {run_id!r}")
    run.proc.terminate()
    done, _ = _count_and_hits(run.pred_path)
    return {"run_id": run_id, "status": "stopped", "done": done}


def _status_of(run_id: str) -> str:
    run = _LIVE.get(run_id)
    if not run:
        return "finished"          # not in this process's registry: on disk from an earlier run
    code = run.proc.poll()
    if code is None:
        return "running"
    return "finished" if code == 0 else "failed"


def describe(run_id: str, pred_path: str, meta: dict, limit: int = 0) -> dict:
    """One run's row, from its files plus whatever the live registry knows."""
    done, hits = _count_and_hits(pred_path)
    # A run id is variant_split_model, so successive slices (--start) append into the SAME
    # file. `n_items` describes only the latest slice, so it under-reports the file as a
    # whole; never let the target fall below what is already written.
    live = _LIVE.get(run_id)
    target = meta.get("n_items") or limit or SPLIT_SIZES.get(meta.get("split", ""), 0)
    if live and live.proc.poll() is None and live.limit:
        target = live.limit
    total = max(target, done)
    return {
        "run_id": run_id,
        "variant": meta.get("variant", run_id.split("_", 1)[0]),
        "split": meta.get("split", ""),
        "model": meta.get("model", ""),
        "done": done,
        "total": total,
        "accuracy": round(hits / done, 4) if done else None,
        "status": _status_of(run_id),
        "has_trace": os.path.exists(pred_path[: -len(".jsonl")] + ".traces.jsonl"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
    }


def list_runs(out_dir: str = DEFAULT_OUT_DIR) -> list[dict]:
    """Every run on disk, unioned with live ones — so this single call serves both
    'monitor the running batch' and 'browse past results'."""
    root = _repo_root()
    d = out_dir if os.path.isabs(out_dir) else os.path.join(root, out_dir)
    rows = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if not name.endswith(".jsonl") or name.endswith(".traces.jsonl"):
                continue
            run_id = name[: -len(".jsonl")]
            pred_path = os.path.join(d, name)
            meta_path = os.path.join(d, run_id + ".meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            live = _LIVE.get(run_id)
            rows.append(describe(run_id, pred_path, meta, live.limit if live else 0))
    # A run launched seconds ago may have no .jsonl yet; surface it so the UI isn't blank.
    for run_id, run in _LIVE.items():
        if not any(r["run_id"] == run_id for r in rows):
            rows.append({"run_id": run_id, "variant": run.variant, "split": run.split,
                         "model": run.model, "done": 0, "total": run.limit or SPLIT_SIZES.get(run.split, 0),
                         "accuracy": None, "status": _status_of(run_id), "has_trace": False,
                         "started_at": run.started_at, "finished_at": None})
    return rows


def log_tail(run_id: str, lines: int = 400) -> list[str]:
    run = _LIVE.get(run_id)
    path = run.log_path if run else os.path.join(_repo_root(), LOG_DIR, f"{run_id}.log")
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [l.rstrip("\n") for l in f.readlines()[-lines:]]


def delete(run_id: str, out_dir: str = DEFAULT_OUT_DIR) -> dict:
    """Delete a finished run's files (predictions, meta, traces, log). Refuses while it is
    still running — stop it first, so the subprocess can't recreate what we just removed."""
    if _status_of(run_id) == "running":
        raise ValueError(f"run {run_id!r} is still running; stop it first")
    root = _repo_root()
    d = out_dir if os.path.isabs(out_dir) else os.path.join(root, out_dir)
    removed = []
    for path in (os.path.join(d, run_id + ext)
                 for ext in (".jsonl", ".meta.json", ".traces.jsonl")):
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    log = os.path.join(root, LOG_DIR, f"{run_id}.log")
    if os.path.exists(log):
        os.remove(log)
        removed.append(os.path.basename(log))
    _LIVE.pop(run_id, None)
    if not removed:
        raise ValueError(f"no files for run {run_id!r}")
    return {"run_id": run_id, "deleted": removed}
