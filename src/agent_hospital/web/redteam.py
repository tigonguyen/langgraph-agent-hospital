"""Red-team runs for the web UI: a MedQA stream with malicious prompts injected.

Each run is a subprocess for ONE model; progress is read back from the JSONL it flushes per item
(same approach as web/runs.py), and the finished summary from its .summary.json sidecar. Files
live in data/redteam/med/eval_mixed/. Two kinds, never combined (the scripts do not stack them):

  harness  scripts/redteam/eval_guarded.py — the model inside a guarded LangGraph
           (graph/guarded.py: sysprompt | gatetool | gatenodes), every node the SAME model
  defense  scripts/redteam/eval_mixed.py --guard — the bare model behind an inference-time guard
           run by a separate, un-attacked model (none | system | gate | verify | gate+verify | memory)
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
GUARDED_SCRIPT = Path(_repo_root()) / "scripts/redteam/eval_guarded.py"

# What the UI offers. H = a graph harness around the model; D = a guard outside it. `none` on both
# is the model only.
HARNESSES = [
    {"id": "none", "label": "No harness — model only", "adds": "the model answers directly"},
    {"id": "sysprompt", "label": "graph sysprompt",
     "adds": "S1: the guard is a refusal instruction to the model itself — the weakest, it asks the attacked weights to refuse"},
    {"id": "gatetool", "label": "graph gatetool",
     "adds": "S2: the model holds a classify_request tool and decides whether to screen itself"},
    {"id": "gatenodes", "label": "graph gatenodes",
     "adds": "S3: a gate node labels the request and the graph routes HARMFUL to a fixed refusal — the model gets no vote"},
]
DEFENSES = [
    {"id": "D0", "guard": "none", "label": "None", "adds": "no guard: whatever the model replies reaches the user"},
    {"id": "D1", "guard": "system", "label": "System prompt", "adds": "a refusal instruction prepended to the attacked answerer"},
    {"id": "D2", "guard": "gate", "label": "Prompt gate",
     "adds": "a separate un-attacked model classifies the request; HARMFUL never reaches the answerer"},
    {"id": "D3", "guard": "verify", "label": "Output verifier",
     "adds": "a second agent reviews the finished reply and swaps it for a refusal if it helps the request"},
    {"id": "D4", "guard": "gate+verify", "label": "Gate + verifier", "adds": "both in series: pre-filter the request, post-filter the reply"},
    {"id": "D5", "guard": "memory", "label": "Gate + refusal memory",
     "adds": "the gate, plus a bank of what it already blocked; a word-overlap hit blocks with no model call"},
]
_DEF_BY_GUARD = {d["guard"]: d["id"] for d in DEFENSES}


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


GATE_MODEL = "qwen3:14b"                    # eval_mixed.py's default --gate-model
GATE_GUARDS = ("gate", "verify", "gate+verify", "memory")   # the guards that call the gate model


def run_id_for(model: str, n: int, m: int, seed: int, k: int = 0, guard: str = "none",
               gate_model: str = GATE_MODEL) -> str:
    """Mirrors eval_mixed.py's stem, including its `_gm-<model>` tag for a non-default gate model."""
    gm = f"_gm-{gate_model.replace(':', '-')}" if guard in GATE_GUARDS and gate_model != GATE_MODEL else ""
    return (f"{model.replace(':', '-')}{gm}_n{n}_m{m}" + (f"_k{k}" if k else "") + f"_s{seed}"
            + ("" if guard == "none" else "_g" + guard.replace("+", "")))


def harness_run_id(model: str, harness: str, n: int, m: int) -> str:
    """eval_guarded.py's own stem."""
    return f"{model.replace(':', '-')}_guard-{harness}_m{m}_n{n}"


def start(model: str, n: int, m: int, seed: int, k: int = 0, guard: str = "none",
          harness: str = "none", gate_model: str = GATE_MODEL) -> RedRun:
    if harness != "none":
        if harness not in {h["id"] for h in HARNESSES}:
            raise ValueError(f"unknown harness {harness!r}")
        return _start(harness_run_id(model, harness, n, m), model, n, m, 0, f"graph:{harness}",
                      [sys.executable, str(GUARDED_SCRIPT), harness, "--model", model, "-m", str(m), "-n", str(n),
                       "--out", str(OUT_DIR)])
    rid = run_id_for(model, n, m, seed, k, guard, gate_model)
    return _start(rid, model, n, m, seed, guard,
                  [sys.executable, str(SCRIPT), model, "-n", str(n), "-m", str(m), "-k", str(k), "--seed", str(seed),
                   "--out", str(OUT_DIR)] + ([] if guard == "none" else ["--guard", guard])
                  + (["--gate-model", gate_model] if guard in GATE_GUARDS else []))


def _start(rid: str, model: str, n: int, m: int, seed: int, guard: str, cmd: list[str]) -> RedRun:
    if rid in _live and _live[rid].proc.poll() is None:
        raise ValueError(f"{rid} is already running")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = str(OUT_DIR / f"{rid}.log")
    # the run id mirrors the script's own stem, which drops the '+' so it stays a clean filename
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
        if "_guard-" in rid:                        # eval_guarded.py: <model>_guard-<harness>_m<m>_n<n>
            row = _harness_row(f)
            if row:
                out.append(row)
            continue
        guard = "none"
        for g in ("system", "gateverify", "gate", "verify", "memory"):
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
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = meta.get("model", model)
        guard = meta.get("guard", guard)
        # Runs of the earlier clean-judge guard (graph/guard.py) carry `judges` in their meta; their
        # verify/memory are a different implementation, so they are not placed on the D ladder.
        legacy = "judges" in meta
        passes = meta.get("passes", 1) if legacy else 1     # legacy memory runs replayed the stream
        med = [r for r in rows if r["kind"] == "medqa"]; mal = [r for r in rows if r["kind"] == "malicious"]
        off = [r for r in rows if r["kind"] == "nonmedical"]
        live = _live.get(rid)
        if (live and live.proc.poll() is None) or (live is None and _pid_alive(rid)):
            status = "running"
        elif live and live.stopped:
            status = "stopped"
        else:
            status = "finished" if len(rows) >= (n + m + k) * passes else "stopped"
        summ_path = OUT_DIR / f"{rid}.summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else None
        out.append({
            "run_id": rid, "model": model, "n": n, "m": m, "k": k, "guard": guard, "harness": "none",
            "defense": None if legacy else _DEF_BY_GUARD.get({"gateverify": "gate+verify"}.get(guard, guard), "D0"),
            "legacy_guard": guard if legacy else None,
            "gate_model": meta.get("gate_model") or (GATE_MODEL if guard in (*GATE_GUARDS, "gateverify") and not legacy else None),
            "gate": _gate_stats(rows),
            "seed": seed, "done": len(rows), "total": (n + m + k) * passes, "status": status,
            "medqa_acc": (sum(r.get("correct", False) for r in med) / len(med)) if med else None,
            "false_refusal": (sum(r["refused"] for r in med) / len(med)) if med else None,
            "harmful_refused": (sum(r["refused"] for r in mal) / len(mal)) if mal else None,
            "scope_refused": (sum(r["refused"] for r in off) / len(off)) if off else None,
            "n_medqa_done": len(med), "n_mal_done": len(mal), "n_off_done": len(off), "summary": summary,
        })
    return out


def _status(rid: str, done: int, total: int) -> str:
    live = _live.get(rid)
    if (live and live.proc.poll() is None) or (live is None and _pid_alive(rid)):
        return "running"
    if live and live.stopped:
        return "stopped"
    return "finished" if done >= total else "stopped"


def _gate_stats(rows: list[dict]) -> dict | None:
    """How often a gate / verifier / memory fired and how often the answerer ran at all."""
    g = [r for r in rows if r.get("gate_verdict") or r.get("verify_verdict")]
    if not g:
        return None
    return {"gate_blocked_malicious": sum(r.get("gate_verdict") == "HARMFUL" for r in rows if r["kind"] == "malicious"),
            "gate_blocked_medqa": sum(r.get("gate_verdict") == "HARMFUL" for r in rows if r["kind"] == "medqa"),
            "verify_blocked": sum(r.get("verify_verdict") == "BLOCK" for r in rows),
            "memory_hits": sum(bool(r.get("memory_hit")) for r in rows),
            "tool_not_called": sum(r.get("gate_verdict") == "NOT_CALLED" for r in rows),
            "answerer_skipped": sum(r.get("answerer_called") is False for r in rows)}


def _harness_row(f: Path) -> dict | None:
    rid = f.stem
    try:
        model, rest = rid.split("_guard-", 1)
        harness, rest = rest.rsplit("_m", 1)
        m, n = (int(x) for x in rest.split("_n"))
    except ValueError:
        return None
    meta_path = OUT_DIR / f"{rid}.meta.json"
    if meta_path.exists():
        model = json.loads(meta_path.read_text()).get("model", model)
    rows = _records(f)
    med = [r for r in rows if r["kind"] == "medqa"]; mal = [r for r in rows if r["kind"] == "malicious"]
    summ_path = OUT_DIR / f"{rid}.summary.json"
    return {"run_id": rid, "model": model, "n": n, "m": m, "k": 0, "seed": None, "guard": "none",
            "harness": harness, "defense": "D0", "gate": _gate_stats(rows),
            "done": len(rows), "total": n + m, "status": _status(rid, len(rows), n + m),
            "medqa_acc": (sum(r.get("correct", False) for r in med) / len(med)) if med else None,
            "false_refusal": (sum(r["refused"] for r in med) / len(med)) if med else None,
            "harmful_refused": (sum(r["refused"] for r in mal) / len(mal)) if mal else None,
            "scope_refused": None, "n_medqa_done": len(med), "n_mal_done": len(mal), "n_off_done": 0,
            "summary": json.loads(summ_path.read_text()) if summ_path.exists() else None}


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
