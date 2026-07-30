"""Check that .env is wired up and the configured model actually answers.

    PYTHONPATH=src .venv/bin/python scripts/check_env.py

Three stages, each reported separately so a failure localises itself:
  1. CONFIG   — what .env produced, and which provider the spec routes to
  2. RESOLVE  — the chat model object builds (no network yet)
  3. CALL     — one tiny live request; the only stage that needs the endpoint up

Exit 0 = the model answered. Exit 1 = something is misconfigured (the message says what).
Secrets are never printed — only a length and a short prefix.
"""

from __future__ import annotations

import os
import sys
import time

# Importing models triggers load_dotenv(), so .env is in os.environ after this line.
from agent_hospital.models import PROVIDERS, default_model, mode, resolve_model


def mask(value: str | None) -> str:
    if not value:
        return "(unset)"
    return f"{value[:7]}… ({len(value)} chars)"


def main() -> int:
    print("=" * 62)
    print("1. CONFIG")
    print("=" * 62)

    try:
        run_mode = mode()
    except ValueError as exc:
        print(f"  !! {exc}")
        return 1

    spec = default_model()
    prefix, _, rest = spec.partition(":")
    provider = PROVIDERS.get(prefix.lower()) if rest else None
    routed = provider or "ollama"

    embed_url = os.environ.get("AGENT_HOSPITAL_EMBED_BASE_URL")
    print(f"  AGENT_HOSPITAL_MODE  : {run_mode}")
    print(f"  embeddings (V1-V4)   : {embed_url or 'Ollama (local)'}")
    print(f"  AGENT_HOSPITAL_MODEL : {spec}")
    print(f"  ANTHROPIC_BASE_URL   : {os.environ.get('ANTHROPIC_BASE_URL') or '(unset -> public API)'}")
    print(f"  ANTHROPIC_API_KEY    : {mask(os.environ.get('ANTHROPIC_API_KEY'))}")
    print(f"  -> routes to provider: {routed}")

    # The single most common .env mistake: a cloud model name with no `provider:`
    # prefix silently falls through to Ollama, so the gateway is never contacted.
    if run_mode == "local" and routed == "ollama" and os.environ.get("ANTHROPIC_BASE_URL"):
        print()
        print("  !! ANTHROPIC_BASE_URL is set but mode is 'local', so the spec routes to")
        print(f"  !! Ollama and your gateway is ignored. Either set AGENT_HOSPITAL_MODE=api,")
        print(f"  !! or use an explicit spec: AGENT_HOSPITAL_MODEL=anthropic:{spec}")
        return 1

    if routed == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print()
        print("  !! Provider is anthropic but ANTHROPIC_API_KEY is unset.")
        return 1

    print()
    print("=" * 62)
    print("2. RESOLVE")
    print("=" * 62)
    try:
        model = resolve_model(spec)
    except Exception as exc:
        print(f"  FAILED to build the model object: {type(exc).__name__}: {exc}")
        return 1

    endpoint = getattr(model, "anthropic_api_url", None) or getattr(model, "base_url", None)
    print(f"  class    : {type(model).__name__}")
    print(f"  model    : {getattr(model, 'model', '?')}")
    print(f"  endpoint : {endpoint or '(provider default)'}")

    print()
    print("=" * 62)
    print("3. CALL  (one request: 'Reply with the single word OK.')")
    print("=" * 62)
    started = time.perf_counter()
    try:
        reply = model.invoke("Reply with the single word OK.")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}")
        print(f"  {exc}")
        print()
        print("  Common causes: endpoint unreachable (VPN?), bad key, or the endpoint")
        print("  does not serve this model name.")
        return 1

    elapsed = time.perf_counter() - started
    usage = getattr(reply, "usage_metadata", None) or {}
    preview = reply.content if isinstance(reply.content, str) else str(reply.content)
    print(f"  reply    : {preview.strip()[:80]!r}")
    print(f"  latency  : {elapsed:.2f}s")
    if usage:
        print(f"  tokens   : in={usage.get('input_tokens')} out={usage.get('output_tokens')}")

    print()
    print("=" * 62)
    print("4. RESPONSE SHAPE  (does the answer parser accept it?)")
    print("=" * 62)
    # Reaching the endpoint is not enough: the pipeline reads `.content` as a plain
    # string (Agent.say -> parse_choice). A model that returns content BLOCKS — e.g. an
    # extended-thinking model behind a proxy — connects fine and then raises TypeError
    # on the first scored item, so check it here rather than 1200 questions in.
    print(f"  .content type : {type(reply.content).__name__}")
    if isinstance(reply.content, str):
        print("  OK — plain string, parse_choice can read it.")
    else:
        kinds = sorted({b.get("type", "?") for b in reply.content if isinstance(b, dict)})
        print(f"  block types   : {kinds}")
        print()
        print("  !! This model returns content BLOCKS, not a string. parse_choice()")
        print("  !! raises TypeError on a list, so every variant will crash on item 1.")
        print("  !! Either pick a model that returns plain text, disable thinking on the")
        print("  !! endpoint, or flatten blocks to text in agents/base.py `Agent.say`.")
        return 1

    print()
    print(f"OK — {getattr(model, 'model', spec)} answered via {endpoint or 'provider default'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
