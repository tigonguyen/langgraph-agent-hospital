"""Live Wikipedia search — the verifier's independent grounding source (V4).

Distinct from `knowledge/retriever.py`'s local Chroma corpus, which is itself built
from MedMCQA (the same benchmark family V0-V5 are scored on) — using it to "verify" an
answer isn't independent of the benchmark. Wikipedia is external, at the cost of a live
network call: this project is otherwise fully local/offline, so a Wikipedia outage (or
no network at all) must degrade gracefully to no grounding, never break an eval run.

One MediaWiki API round-trip does search + extract together (`generator=search`), rather
than a search call followed by N extract calls.
"""

from __future__ import annotations

from typing import Any

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia's API asks for a descriptive User-Agent identifying the client; a default
# python-requests one is sometimes rejected.
_HEADERS = {"User-Agent": "agent-hospital-research/1.0 (educational MedQA-USMLE project)"}
DEFAULT_K = 3
TIMEOUT_S = 5


def search_wikipedia(query: str, k: int = DEFAULT_K, max_chars: int = 500) -> str:
    """Top-k Wikipedia article intros for `query`, formatted like `format_evidence`.

    Returns "" on any failure (network, timeout, malformed response) — the caller treats
    that exactly like "nothing retrieved", same as the no-RAG fallback elsewhere in this
    project.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": k,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "format": "json",
        "formatversion": 2,
    }
    try:
        import requests

        resp = requests.get(API_URL, params=params, headers=_HEADERS, timeout=TIMEOUT_S)
        resp.raise_for_status()
        pages: list[dict[str, Any]] = resp.json().get("query", {}).get("pages", [])
    except Exception:
        return ""

    lines = []
    for page in pages:
        title = (page.get("title") or "").strip()
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        lines.append(f"- ({title}) {extract[:max_chars]}")
    if not lines:
        return ""
    return "Wikipedia evidence:\n" + "\n".join(lines)
