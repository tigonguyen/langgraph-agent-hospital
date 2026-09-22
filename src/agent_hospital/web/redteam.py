"""Red-team runs for the web UI: a MedQA stream with malicious prompts injected.

Each run is a subprocess of scripts/redteam/eval_mixed.py for ONE model; progress is read
back from the JSONL it flushes per item (same approach as web/runs.py), and the finished
summary from its .summary.json sidecar. Files live in data/redteam/med/eval_mixed/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent_hospital.web.runs import _repo_root

OUT_DIR = Path(_repo_root()) / "data/redteam/med/eval_mixed"
SCRIPT = Path(_repo_root()) / "scripts/redteam/eval_mixed.py"


@dataclass
class RedRun:
    run_id: str
    model: str
    n: int
    m: int
    seed: int
    guard: str
    proc: subprocess.Popen
    log: str
    stopped: bool = field(default=False)


_live: dict[str, RedRun] = {}


def run_id_for(model: str, n: int, m: int, seed: int, k: int = 0, guard: str = "none") -> str:
    return (f"{model.replace(':', '-')}_n{n}_m{m}" + (f"_k{k}" if k else "") + f"_s{seed}"
            + ("" if guard == "none" else f"_g{guard}"))


def start(model: str, n: int, m: int, seed: int, k: int = 0, guard: str = "none") -> RedRun:
    rid = run_id_for(model, n, m, seed, k, guard)
    if rid in _live and _live[rid].proc.poll() is None:
        raise ValueError(f"{rid} is already running")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = str(OUT_DIR / f"{rid}.log")
    cmd = [sys.executable, str(SCRIPT), model, "-n", str(n), "-m", str(m), "-k", str(k), "--seed", str(seed),
           "--out", str(OUT_DIR)] + ([] if guard == "none" else ["--guard", guard])
    env = {**os.environ, "PYTHONPATH": os.path.join(_repo_root(), "src"), "PYTHONUNBUFFERED": "1",
           "PYTHONWARNINGS": "ignore::UserWarning"}   # the resource_tracker warning is raised in a helper process, so filter via env
    # start_new_session: the run outlives a server restart (predict runs from the Batch tab too);
    # eval_mixed resumes from its own file, so "resume" is just starting the same run again.
    proc = subprocess.Popen(cmd, cwd=_repo_root(), env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    (OUT_DIR / f"{rid}.pid").write_text(str(proc.pid))
    _live[rid] = RedRun(rid, model, n, m, seed, guard, proc, log)
    return _live[rid]


def _pid_alive(rid: str) -> bool:
    """A run started by a previous server process: its pid file says whether it still runs."""
    p = OUT_DIR / f"{rid}.pid"
    if not p.exists():
        return False
    try:
        os.kill(int(p.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def stop(run_id: str) -> dict:
    r = _live.get(run_id)
    if r is not None and r.proc.poll() is None:
        r.proc.terminate(); r.stopped = True
    elif _pid_alive(run_id):
        os.kill(int((OUT_DIR / f"{run_id}.pid").read_text()), 15)
    else:
        raise ValueError(f"{run_id} is not running")
    return {"run_id": run_id, "status": "stopped"}


def ollama_models() -> list[str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [l.split()[0] for l in out.splitlines()[1:] if l.strip()]


def _records(path: Path) -> list[dict]:
    rows = []
    for l in open(path):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return rows


def list_runs() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(OUT_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        rid = f.stem
        guard = "none"
        for g in ("system", "gate"):
            if rid.endswith(f"_g{g}"):
                guard, rid_core = g, rid[: -len(f"_g{g}")]
                break
        else:
            rid_core = rid
        try:
            model, rest = rid_core.rsplit("_n", 1)
            n, rest = rest.split("_m")
            if "_k" in rest:
                m, rest = rest.split("_k"); k, seed = rest.split("_s")
            else:
                m, seed = rest.split("_s"); k = 0
            n, m, k, seed = int(n), int(m), int(k), int(seed)
        except ValueError:
            continue
        rows = _records(f)
        meta_path = OUT_DIR / f"{rid}.meta.json"
        if meta_path.exists():
            model = json.loads(meta_path.read_text())["model"]
        med = [r for r in rows if r["kind"] == "medqa"]; mal = [r for r in rows if r["kind"] == "malicious"]
        off = [r for r in rows if r["kind"] == "nonmedical"]
        live = _live.get(rid)
        if (live and live.proc.poll() is None) or (live is None and _pid_alive(rid)):
            status = "running"
        elif live and live.stopped:
            status = "stopped"
        else:
            status = "finished" if len(rows) >= n + m + k else "stopped"
        summ_path = OUT_DIR / f"{rid}.summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else None
        out.append({
            "run_id": rid, "model": model, "n": n, "m": m, "k": k, "guard": guard,
            "seed": seed, "done": len(rows), "total": n + m + k, "status": status,
            "medqa_acc": (sum(r.get("correct", False) for r in med) / len(med)) if med else None,
            "false_refusal": (sum(r["refused"] for r in med) / len(med)) if med else None,
            "harmful_refused": (sum(r["refused"] for r in mal) / len(mal)) if mal else None,
            "scope_refused": (sum(r["refused"] for r in off) / len(off)) if off else None,
            "n_medqa_done": len(med), "n_mal_done": len(mal), "n_off_done": len(off), "summary": summary,
        })
    return out


def items(run_id: str, kind: str = "all", flt: str = "all") -> list[dict]:
    path = OUT_DIR / f"{run_id}.jsonl"
    if not path.exists():
        raise ValueError(f"no run {run_id}")
    rows = _records(path)
    if kind != "all":
        rows = [r for r in rows if r["kind"] == kind]
    if flt == "complied":
        rows = [r for r in rows if r["kind"] == "malicious" and not r["refused"]]
    elif flt == "answered":
        rows = [r for r in rows if r["kind"] == "nonmedical" and not r["refused"]]
    elif flt == "refused":
        rows = [r for r in rows if r["refused"]]
    elif flt == "wrong":
        rows = [r for r in rows if r["kind"] == "medqa" and not r.get("correct")]
    return rows


def log_tail(run_id: str, lines: int = 200) -> list[str]:
    p = OUT_DIR / f"{run_id}.log"
    if not p.exists():
        return []
    noise = ("resource_tracker", "warnings.warn(")     # interpreter shutdown chatter, not run output
    return [l for l in p.read_text(errors="replace").splitlines() if not any(n in l for n in noise)][-lines:]


def delete(run_id: str) -> dict:
    r = _live.get(run_id)
    if (r and r.proc.poll() is None) or _pid_alive(run_id):
        raise ValueError(f"{run_id} is still running")
    n = 0
    for suffix in (".jsonl", ".summary.json", ".meta.json", ".log", ".pid"):
        p = OUT_DIR / f"{run_id}{suffix}"
        if p.exists():
            p.unlink(); n += 1
    _live.pop(run_id, None)
    return {"run_id": run_id, "deleted": n}
