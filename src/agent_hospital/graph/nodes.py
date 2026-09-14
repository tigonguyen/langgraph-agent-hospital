"""Reusable graph nodes (state -> partial state), each built from a RunConfig.

Agents are lazy (`create_agent` builds on first call) and the Chroma store opens on
first retrieval, so compiling a graph touches no network / no vector store.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable

from agent_hospital import roles
from agent_hospital.agents.base import Agent
from agent_hospital.config import RunConfig
from agent_hospital.graph import longterm
from agent_hospital.knowledge import default_embeddings, format_evidence, open_store, retrieve
# Aliased: a local `@tool def search_wikipedia` below would shadow this import and recurse.
from agent_hospital.knowledge import search_wikipedia as _wikipedia_lookup
from agent_hospital.qa.mcq import (AGENTIC_ANSWER, AGENTIC_VERIFY, AGENTIC_VERIFY_WIKIPEDIA,
                                   ANALYSE_ONLY, DIGEST_EVIDENCE_ONLY, LESSON_SUFFIX, LETTER_ONLY,
                                   MISTAKE_LESSON_ONLY, REASON_THEN_ANSWER, UNDERSTAND_ONLY,
                                   format_mcq, parse_choice)
from agent_hospital.qa.reasoning import build_reasoning_agent, build_router

Node = Callable[[dict], dict]
_LETTERS = "ABCD"


def _prompt(state: dict, extra: str = "", closing: str = LETTER_ONLY) -> str:
    ev = state.get("evidence", "")
    head = f"{ev}\n\n" if ev else ""
    return f"{head}{format_mcq(state['item'], closing)}{extra}"


def make_reason_node(cfg: RunConfig) -> Node:
    reasoner = build_reasoning_agent(cfg.model_for("reasoner"))
    distill = bool(cfg.rag and cfg.rag.distill_query)

    def node(state: dict) -> dict:
        item = state["item"]
        return {"query": reasoner(item) if distill else item.question}

    return node


def make_retrieve_node(cfg: RunConfig) -> Node:
    """Invokes the search tool directly — no LLM call."""
    search, _reset = make_search_tool(cfg)

    def node(state: dict) -> dict:
        return {"evidence": search.invoke({"query": state["query"]})}

    return node


def make_evidence_digest_node(cfg: RunConfig) -> Node:
    """Turns raw retrieved passages into an interpreted summary, overwriting
    `state["evidence"]` so downstream nodes pick it up via `_prompt` unchanged."""
    agent = Agent("evidence-digest", roles.ROLE_PROMPTS["evidence-digest"],
                  model=cfg.model_for("evidence-digest"))

    def node(state: dict) -> dict:
        raw = state.get("evidence", "")
        if not raw:
            return {}
        text = f"{format_mcq(state['item'], DIGEST_EVIDENCE_ONLY)}\n\nRetrieved passages:\n{raw}"
        digest = agent.say(text).strip()
        return {"evidence": digest or raw}

    return node


def make_search_branch_node(cfg: RunConfig) -> Node:
    """Retrieve + digest bundled into ONE node so it shares a superstep with the reasoning
    node for real concurrency — a multi-node chain racing a single long call only has its
    first step actually overlap; the rest stalls until that call's superstep ends."""
    retrieve = make_retrieve_node(cfg)
    digest = make_evidence_digest_node(cfg)

    def node(state: dict) -> dict:
        s = {**state, **retrieve(state)}
        raw = s.get("evidence", "")     # retrieve's output, before digest overwrites it
        s = {**s, **digest(s)}
        # `retrieved` is surfaced for observability only; digest replaces `evidence` in
        # place, so without this the raw passages are unrecoverable downstream.
        return {"evidence": s.get("evidence", ""), "retrieved": raw}

    return node


def make_understand_node(cfg: RunConfig) -> Node:
    """Shared case understanding + search query, read by both the search and reasoning
    branches instead of each re-deriving its own."""
    agent = Agent("case-reasoner", roles.ROLE_PROMPTS["case-reasoner"], model=cfg.model_for("case-reasoner"))

    def node(state: dict) -> dict:
        reply = agent.say(format_mcq(state["item"], UNDERSTAND_ONLY)).strip()
        m = re.search(r"search query:\s*(.+)", reply, re.IGNORECASE | re.DOTALL)
        query = m.group(1).strip().splitlines()[0] if m else state["item"].question
        return {"case_understanding": reply, "query": query}

    return node


def make_adaptive_rag_node(cfg: RunConfig) -> Node:
    """Per-question router: skips retrieval (empty evidence, same as V0's prompt) when the
    case looks reasoning-heavy rather than fixing retrieval on/off for every item."""
    rag = cfg.rag
    router = build_router(cfg.model_for("reasoner"))
    reasoner = build_reasoning_agent(cfg.model_for("reasoner"))
    holder: dict[str, Any] = {}

    def node(state: dict) -> dict:
        item = state["item"]
        if not router(item):
            return {"evidence": "", "query": ""}
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        query = reasoner(item)
        hits = retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"])
        return {"evidence": format_evidence(hits), "query": query}

    return node


def _call_cap_guard(calls: dict, max_calls: int | None, tool_name: str) -> str | None:
    """Hard-stop for agentic tool use: a prompt-only "never call more than N times"
    instruction isn't reliable (measured: qwen2.5:7b ignored it, called a tool 4 times on
    one item). Redirecting via the TOOL'S OWN OUTPUT works better than a rule stated once
    at conversation start. Returns the redirect message if the cap is hit, else None."""
    if max_calls is None:
        return None
    if calls["n"] >= max_calls:
        return (f"You have already called {tool_name} the maximum of {max_calls} times. Do "
                "NOT call it again — answer the question now using your own reasoning and "
                "whatever evidence you already have.")
    calls["n"] += 1
    return None


def make_search_tool(cfg: RunConfig, max_calls: int | None = None):
    """Textbook-search tool, invoked directly by the retrieve node (no LLM call to choose
    it). Returns `(tool, reset)` — call `reset()` per item; the closure is reused across items."""
    from langchain_core.tools import tool

    rag = cfg.rag
    holder: dict[str, Any] = {}
    calls = {"n": 0}

    @tool
    def search_textbooks(query: str) -> str:
        """Search medical textbooks for passages relevant to a clinical query."""
        blocked = _call_cap_guard(calls, max_calls, "search_textbooks")
        if blocked:
            return blocked
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        hits = retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"])
        return format_evidence(hits)

    return search_textbooks, lambda: calls.__setitem__("n", 0)


def make_wikipedia_tool(cfg: RunConfig, max_calls: int | None = None):
    """Live Wikipedia search tool for the verifier — independent of this project's local
    corpus (itself built from MedMCQA, the benchmark being scored). Degrades to "" on any
    network failure, same as the local tool's no-RAG fallback."""
    from langchain_core.tools import tool

    calls = {"n": 0}

    @tool
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for an article relevant to a clinical query."""
        blocked = _call_cap_guard(calls, max_calls, "search_wikipedia")
        if blocked:
            return blocked
        return _wikipedia_lookup(query)

    return search_wikipedia, lambda: calls.__setitem__("n", 0)


def make_medmcqa_tool(cfg: RunConfig, max_calls: int | None = None):
    """`search_medmcqa` tool, bound to an agent so the model decides when to call it
    (agentic RAG) — only V1 holds it; V2-V5 retrieve non-agentically via `make_search_tool`."""
    from langchain_core.tools import tool

    rag = cfg.rag
    holder: dict[str, Any] = {}
    calls = {"n": 0}

    @tool
    def search_medmcqa(query: str) -> str:
        """Search a database of solved medical board questions (MedMCQA) for entries
        similar to the query. Returns the top matches with their correct answer and a
        brief explanation. Use a focused clinical query (the key findings and what is asked)."""
        blocked = _call_cap_guard(calls, max_calls, "search_medmcqa")
        if blocked:
            return blocked
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        hits = retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"])
        return format_evidence(hits) or "No similar questions found."

    return search_medmcqa, lambda: calls.__setitem__("n", 0)


_MAX_AGENTIC_SEARCHES = 2   # matches the "up to twice" confidence-gated retry the prompts describe

_AGENTIC_TOOL_BY_ROLE = {
    "rag-agent": (lambda cfg: make_medmcqa_tool(cfg, max_calls=_MAX_AGENTIC_SEARCHES), AGENTIC_ANSWER),
}


def make_agentic_rag_node(cfg: RunConfig) -> Node:
    """V1: a single tool-using agent decides whether/what to search (contrast V2-V5, where
    the retrieve node invokes the tool directly)."""
    make_tool, closing = _AGENTIC_TOOL_BY_ROLE[cfg.answer_role]
    search, reset_calls = make_tool(cfg)
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role],
                  model=cfg.model_for(cfg.answer_role), tools=[search])

    def node(state: dict) -> dict:
        reset_calls()
        reply = agent.say(format_mcq(state["item"], closing))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_reasoning_node(cfg: RunConfig) -> Node:
    """Reasons from the shared case understanding using only its own medical knowledge —
    never sees retrieved evidence, since it runs concurrently with the search node.
    Deliberately never names a final answer; the decider reads this report and commits."""
    agent = Agent("clinical-reasoner", roles.ROLE_PROMPTS["clinical-reasoner"],
                  model=cfg.model_for("clinical-reasoner"))

    def node(state: dict) -> dict:
        understanding = state.get("case_understanding", "")
        extra = f"\n\nCase summary:\n{understanding}" if understanding else ""
        analysis = agent.say(_prompt(state, extra=extra, closing=ANALYSE_ONLY)).strip()
        # clinical_report persists for the decider/verifier; rationale gets overwritten by
        # whichever of them runs next with its own short justification.
        return {"rationale": analysis, "clinical_report": analysis}

    return node


def make_answer_node(cfg: RunConfig) -> Node:
    """The decider. When `cfg.long_term` (V3-V5), also recalls/writes the general lesson
    bank (graph/longterm.py) — kept here rather than in the verifier so read+write happen
    in one place regardless of whether a verifier exists (V3 has none)."""
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    lt: dict[str, Any] = {}   # store opened once per graph, lazily, not per item

    def _store():
        if not cfg.long_term:
            return None
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced
        return lt["store"]

    def node(state: dict) -> dict:
        item = state["item"]
        prior = state.get("clinical_report", "")
        lessons = longterm.recall(_store(), cfg.long_term_split, item.question) if cfg.long_term else ""
        extra = ((f"\n\n{lessons}" if lessons else "")
                 + (f"\n\nColleague's clinical-reasoning report:\n{prior}" if prior else ""))
        closing = f"{REASON_THEN_ANSWER}\n\n{LESSON_SUFFIX}" if cfg.long_term else REASON_THEN_ANSWER
        reply = agent.say(_prompt(state, extra=extra, closing=closing))
        answer = parse_choice(reply)

        if cfg.long_term:
            longterm.remember(
                _store(), cfg.long_term_split, item.id, item.question,
                longterm.extract_lesson(reply),
                chosen=_LETTERS[answer] if answer is not None else "",
                read_only=cfg.long_term_read_only,
            )
        return {"answer": answer, "rationale": reply.strip()}

    return node


def make_report_verify_node(cfg: RunConfig) -> Node:
    """V4/V5's verifier: a cheap final check, not a second full derivation. Reads the
    clinical report, the shared case understanding (if `cfg.memory`), and the decider's
    chosen letter, straight from graph state — no separate memory agent needed.

    `cfg.verify_wikipedia` (V4) grounds against live Wikipedia instead of
    `cfg.verify_rag`'s local textbook tool, since the local corpus is itself built from
    MedMCQA — not an independent check.

    `cfg.long_term_mistakes` (V5) recalls the separate mistake bank; this node never reads
    or writes gold — only `make_mistake_distill_node`, after this one, does that.
    """
    if cfg.verify_wikipedia:
        search, reset_calls = make_wikipedia_tool(cfg, max_calls=_MAX_AGENTIC_SEARCHES)
        tools, role, closing = [search], "report-verifier-wikipedia", AGENTIC_VERIFY_WIKIPEDIA
    elif cfg.verify_rag:
        search, reset_calls = make_search_tool(replace(cfg, rag=cfg.verify_rag),
                                               max_calls=_MAX_AGENTIC_SEARCHES)
        tools, role, closing = [search], "report-verifier", AGENTIC_VERIFY
    else:
        tools, role, closing = (), "report-verifier", REASON_THEN_ANSWER
        reset_calls = lambda: None
    agent = Agent(role, roles.ROLE_PROMPTS[role], model=cfg.model_for("verifier"), tools=tools)

    lt: dict[str, Any] = {}

    def _store():
        if not cfg.long_term_mistakes:
            return None
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced
        return lt["store"]

    def node(state: dict) -> dict:
        reset_calls()
        cur = state.get("answer")
        cur_letter = _LETTERS[cur] if cur is not None else "unknown"
        report = state.get("clinical_report", "")
        understanding = state.get("case_understanding", "") if cfg.memory else ""
        item = state["item"]

        mistakes = (longterm.recall_mistakes(_store(), cfg.long_term_split, item.question)
                    if cfg.long_term_mistakes else "")

        extra = ((f"\n\n{mistakes}" if mistakes else "")
                 + (f"\n\nCase summary:\n{understanding}" if understanding else "")
                 + f"\n\nClinical-reasoning report (option-by-option verdicts):\n{report}"
                 + f"\n\nChosen answer: {cur_letter}"
                 + f"\n\nColleague's explanation: {state.get('rationale', '')}")
        reply = agent.say(_prompt(state, extra=extra, closing=closing))
        revised = parse_choice(reply)
        final = revised if revised is not None else cur
        return {"answer": final, "rationale": reply.strip()}

    return node


def make_mistake_distill_node(cfg: RunConfig) -> Node:
    """V5's node after `verify`: the ONLY node that reads gold (`item.answer_idx`). Skips
    entirely under `cfg.long_term_read_only`, since nothing would be persisted anyway.

    A correct final answer is a no-op. A wrong one gets one extra LLM call (`mistake-
    analyst`, shown the correct option) that distills a corrective lesson into the mistake
    bank. MUST NOT touch `state["answer"]`/`state["rationale"]` — this node only writes to
    the store for future episodes, never changes what gets scored for this one.
    """
    agent = Agent("mistake-analyst", roles.ROLE_PROMPTS["mistake-analyst"],
                  model=cfg.model_for("verifier"))

    lt: dict[str, Any] = {}

    def _store():
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced
        return lt["store"]

    def node(state: dict) -> dict:
        if cfg.long_term_read_only:
            return {}
        item = state["item"]
        chosen = state.get("answer")
        if chosen == item.answer_idx:
            return {}

        chosen_letter = _LETTERS[chosen] if chosen is not None else "unknown"
        correct_letter = _LETTERS[item.answer_idx]
        chosen_text = item.options[chosen] if chosen is not None else ""
        extra = (f"\n\nYour team chose: {chosen_letter}. {chosen_text}"
                 f"\n\nThe correct answer was: {correct_letter}. {item.options[item.answer_idx]}")
        reply = agent.say(_prompt(state, extra=extra, closing=MISTAKE_LESSON_ONLY))
        longterm.remember_mistake(
            _store(), cfg.long_term_split, item.id, item.question,
            longterm.extract_lesson(reply),
            wrong=chosen_letter, correct=correct_letter,
            read_only=cfg.long_term_read_only,
        )
        return {}

    return node
