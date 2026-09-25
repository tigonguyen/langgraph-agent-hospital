"""Revert everything the PoC created. Nothing here touches production data.

    PYTHONPATH=src .venv/bin/python scripts/redteam/inject/clean.py --all

Removes: the sandbox lesson bank, the sandbox Chroma collection, and (with --results)
the run outputs. The real data/longterm/lessons.sqlite and the real
knowledge_medmcqa_qwen3 collection are guarded -- this refuses to delete either.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/redteam/inject"))

OUT = ROOT / "data/redteam/inject"
REAL_DB = ROOT / "data/longterm/lessons.sqlite"
REAL_COLLECTION = "knowledge_medmcqa_qwen3"
SANDBOX_COLLECTION = "redteam_poisoned_medmcqa"


def drop_bank(db: Path) -> None:
    if db.resolve() == REAL_DB.resolve():
        print("REFUSING to delete the real lesson bank"); return
    if db.exists():
        db.unlink(); print(f"removed {db}")
    else:
        print(f"(no {db})")


def drop_collection() -> None:
    import chromadb
    c = chromadb.PersistentClient(path=str(ROOT / "data/chroma"))
    names = [x.name for x in c.list_collections()]
    if SANDBOX_COLLECTION not in names:
        print(f"(no {SANDBOX_COLLECTION})"); return
    assert SANDBOX_COLLECTION != REAL_COLLECTION
    c.delete_collection(SANDBOX_COLLECTION)
    print(f"dropped collection {SANDBOX_COLLECTION}")


def drop_results() -> None:
    n = 0
    for f in list(OUT.glob("*.jsonl")) + list(OUT.glob("*.json")):
        f.unlink(); n += 1
    print(f"removed {n} result files from {OUT}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="bank + collection (keeps results)")
    p.add_argument("--results", action="store_true", help="also delete run outputs")
    p.add_argument("--db", default=str(OUT / "lessons_poison.sqlite"))
    a = p.parse_args()
    if a.all or not a.results:
        drop_bank(Path(a.db))
        drop_collection()
    if a.results:
        drop_results()
    # Verify production is intact.
    print(f"\nproduction lesson bank {'PRESENT' if REAL_DB.exists() else 'MISSING'}: {REAL_DB}")
    try:
        import chromadb
        c = chromadb.PersistentClient(path=str(ROOT / "data/chroma"))
        names = [x.name for x in c.list_collections()]
        print(f"production collection {'PRESENT' if REAL_COLLECTION in names else 'MISSING'}: "
              f"{REAL_COLLECTION}  (collections: {names})")
    except Exception as e:
        print(f"(could not check chroma: {e})")


if __name__ == "__main__":
    main()
