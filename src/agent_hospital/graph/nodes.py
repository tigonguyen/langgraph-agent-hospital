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
# Aliased: a `@tool def search_wikipedia(...)` below would shadow this import in its own
# scope, turning `search_wikipedia(query)` inside the tool into infinite self-recursion.
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
    """Retrieval stage — invokes the `search_textbooks` tool directly (no LLM call)."""
    search, _reset = make_search_tool(cfg)

    def node(state: dict) -> dict:
        return {"evidence": search.invoke({"query": state["query"]})}

    return node


def make_evidence_digest_node(cfg: RunConfig) -> Node:
    """Layer 1, branch B's final step (V2-V4): turn the k raw retrieved passages into an
    interpreted summary rather than handing the decider/verifier raw snippets to re-read
    and re-weigh themselves. Overwrites `state["evidence"]` with the digest, so everything
    downstream picks it up via the existing `_prompt` auto-prepend with no further changes.
    """
    agent = Agent("evidence-digest", roles.ROLE_PROMPTS["evidence-digest"],
                  model=cfg.model_for("evidence-digest"))

    def node(state: dict) -> dict:
        raw = state.get("evidence", "")
        if not raw:
            return {}  # nothing retrieved (no-RAG fallback) -> nothing to digest
        text = f"{format_mcq(state['item'], DIGEST_EVIDENCE_ONLY)}\n\nRetrieved passages:\n{raw}"
        digest = agent.say(text).strip()
        return {"evidence": digest or raw}

    return node


def make_search_branch_node(cfg: RunConfig) -> Node:
    """Node 2 of the 4-node design (V2-V4): retrieve + digest, bundled into ONE graph node
    so it shares a superstep with Node 3 (reasoning) for real concurrency — LangGraph's
    synchronous execution advances in supersteps, and a multi-node chain racing a single
    long call only has its FIRST step actually overlap; the rest gets stranded until the
    long call's superstep ends. Uses the query `make_understand_node` already produced
    (no separate query-distillation call needed here).
    """
    retrieve = make_retrieve_node(cfg)
    digest = make_evidence_digest_node(cfg)

    def node(state: dict) -> dict:
        s = {**state, **retrieve(state)}
        s = {**s, **digest(s)}
        return {"evidence": s.get("evidence", "")}

    return node


def make_understand_node(cfg: RunConfig) -> Node:
    """Node 1 of the 4-node design (V2-V4): one shared case understanding (findings + what's
    asked) and search query, read by BOTH the search branch (Node 2) and the reasoning branch
    (Node 3) instead of each independently re-deriving its own — the shared first step of
    what V1's single agent does internally, split into its own node here.
    """
    agent = Agent("case-reasoner", roles.ROLE_PROMPTS["case-reasoner"], model=cfg.model_for("case-reasoner"))

    def node(state: dict) -> dict:
        reply = agent.say(format_mcq(state["item"], UNDERSTAND_ONLY)).strip()
        m = re.search(r"search query:\s*(.+)", reply, re.IGNORECASE | re.DOTALL)
        query = m.group(1).strip().splitlines()[0] if m else state["item"].question
        return {"case_understanding": reply, "query": query}

    return node


def make_adaptive_rag_node(cfg: RunConfig) -> Node:
    """V1b — adaptive RAG-tier routing (RAGCare-QA / Self-RAG style): a cheap router
    decides PER QUESTION whether a textbook lookup is likely to help (a fact-lookup
    question) versus do nothing or hurt (a reasoning-heavy question, where plain V0 wins
    per this project's own measurements). Skips retrieval entirely (empty evidence, same
    as V0's prompt) when the router says the case is reasoning-heavy rather than fixing
    retrieval on or off for every item.
    """
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
    """Shared hard-stop check for agentic tool use: a prompt-only "never call more than N
    times" instruction is not reliable — measured, qwen2.5:7b ignored it and called a tool
    4 times on one item. Returning a redirect message via the TOOL'S OWN OUTPUT (visible to
    the model at the exact moment it's deciding whether to search again) works far better
    than a rule stated once at the start of a long tool-calling conversation. Returns the
    redirect message if the cap is hit, else None (and increments the counter).
    """
    if max_calls is None:
        return None
    if calls["n"] >= max_calls:
        return (f"You have already called {tool_name} the maximum of {max_calls} times. Do "
                "NOT call it again — answer the question now using your own reasoning and "
                "whatever evidence you already have.")
    calls["n"] += 1
    return None


def make_search_tool(cfg: RunConfig, max_calls: int | None = None):
    """A `search_textbooks` tool over the knowledge store (gated retrieval).

    The retrieve node invokes it directly, so it costs no LLM call. Binding it to an
    agent instead would let the model choose when to search, at the cost of one extra
    model invocation per call (the model must be re-invoked on the tool result).

    `max_calls`, if set, hard-caps calls within one episode (see `_call_cap_guard`).
    Returns `(tool, reset)` — call `reset()` once per item before invoking the agent,
    since the tool closure is reused across items (uncapped callers can ignore `reset`).
    """
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
        return format_evidence(hits)          # "" = nothing cleared the gate (no-RAG fallback)

    return search_textbooks, lambda: calls.__setitem__("n", 0)


def make_wikipedia_tool(cfg: RunConfig, max_calls: int | None = None):
    """A `search_wikipedia` tool (V4's verifier) — live Wikipedia search, independent of
    this project's local corpus (itself built from MedMCQA, the benchmark being scored).

    Same shape as `make_search_tool`: a live network call instead of a Chroma lookup, but
    `knowledge.search_wikipedia` already degrades to "" on any failure, so a network
    outage behaves exactly like the local tool's own no-RAG fallback.
    """
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
    """Tool factory (NOT a graph node): returns a LangChain `@tool` (`search_medmcqa`)
    to BIND to an agent so the *model* decides when to call it (agentic RAG). Only V1's
    answer agent holds it — V2-V4 retrieve the same corpus non-agentically instead, via
    `make_retrieve_node`/`make_search_tool`. It retrieves the top-k similar solved board
    questions (MedMCQA) with answers/explanations — no gate (rag.threshold=0.0).

    `max_calls`/returns: see `make_search_tool`.
    """
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


# Hard cap for agentic search (V1 and the verifier's own search) — matches the "up to
# twice" confidence-gated retry the prompts describe; see _call_cap_guard for why a prompt-
# only cap isn't enough.
_MAX_AGENTIC_SEARCHES = 2

# role -> (tool factory, closing instruction) for the single-agent agentic-RAG node below.
_AGENTIC_TOOL_BY_ROLE = {
    "rag-agent": (lambda cfg: make_medmcqa_tool(cfg, max_calls=_MAX_AGENTIC_SEARCHES), AGENTIC_ANSWER),
}


def make_agentic_rag_node(cfg: RunConfig) -> Node:
    """V1 as a single tool-using agent (agentic RAG).

    One agent, one tool (which tool + corpus depends on `cfg.answer_role`, see
    `_AGENTIC_TOOL_BY_ROLE`): it decides whether/what to search, the `create_agent`
    loop runs the tool and feeds the results back, and the agent answers. The model —
    not the graph — drives retrieval (contrast V2-V4, where the retrieve node invokes
    the tool directly). The tool hard-caps itself at `_MAX_AGENTIC_SEARCHES` calls, reset
    per item below.
    """
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
    """Node 3 of the 4-node design (V2-V4): reasons from the shared case understanding
    (`make_understand_node`'s output) using only its own medical knowledge — it never sees
    retrieved evidence, since it runs concurrently with Node 2 (search) — and rates its own
    confidence, mirroring V1's step 4. Deliberately never names a final answer; the `decider`
    (make_answer_node) reads this report next and commits to a letter, so the reasoning stays
    inspectable on its own rather than an implicit side-effect of answering.
    """
    agent = Agent("clinical-reasoner", roles.ROLE_PROMPTS["clinical-reasoner"],
                  model=cfg.model_for("clinical-reasoner"))

    def node(state: dict) -> dict:
        understanding = state.get("case_understanding", "")
        extra = f"\n\nCase summary:\n{understanding}" if understanding else ""
        analysis = agent.say(_prompt(state, extra=extra, closing=ANALYSE_ONLY)).strip()
        # `clinical_report` persists untouched for the decider AND the verifier to read;
        # `rationale` is the current best user-facing explanation, which the decider (and
        # then the verifier, if present) will overwrite with its own short justification.
        return {"rationale": analysis, "clinical_report": analysis}

    return node


def make_answer_node(cfg: RunConfig) -> Node:
    """The decider (Node 4). When `cfg.long_term` (V3-V5), it also recalls lessons from
    similar EARLIER cases (the general bank, graph/longterm.py) and, after deciding,
    distills one back for future episodes — moved here from the verifier so it's the
    single place both "what usually works" is read AND written, independent of whether
    a verifier exists at all (V3 has none).
    """
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    # Long-term store: opened once per graph (not per item), lazily so building a graph
    # still touches no disk. See graph/longterm.try_open.
    lt: dict[str, Any] = {}

    def _store():
        if not cfg.long_term:
            return None
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced (longterm.try_open)
        return lt["store"]

    def node(state: dict) -> dict:
        item = state["item"]
        # A prior clinical-reasoning report (V2-V5) informs the choice; the short
        # justification written here replaces it as the user-facing explanation.
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
    """V4/V5's verifier: a cheap final check, not a second full derivation.

    Reads the clinical reasoner's report straight from `state["clinical_report"]` — plus,
    when `cfg.memory` is set, the shared case understanding too — and the decider's chosen
    letter and its own short explanation (`state["rationale"]`, not yet overwritten), and
    audits that choice against them. No separate memory agent: everything Nodes 1-3
    produced already lives in the shared graph state, so this just reads more of it directly
    (`state["evidence"]` is auto-prepended by `_prompt` like every other node).

    Independent grounding, if the report alone doesn't settle it: `cfg.verify_wikipedia`
    (V4) reaches for live Wikipedia search instead of `cfg.verify_rag`'s local textbook
    tool — the local corpus is itself built from MedMCQA (the same benchmark family being
    scored), so it isn't an independent check.

    `cfg.long_term_mistakes` (V5) recalls the SEPARATE mistake bank (graph/longterm.py) of
    cases the system got wrong before. This node NEVER reads or writes gold — it only
    recalls; `make_mistake_distill_node`, which runs after this one, is the only node that
    ever sees gold, and the only writer to that bank.

    Falls back to the prior answer on parse failure, like every other late-stage node.
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

    # Mistake-bank store: opened once per graph, lazily, same pattern as the decider's.
    lt: dict[str, Any] = {}

    def _store():
        if not cfg.long_term_mistakes:
            return None
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced (longterm.try_open)
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
    """V5's Node 6, after `verify`: the ONLY node in the graph that reads the gold answer
    (`item.answer_idx`) — the verifier itself never does (see `make_report_verify_node`).

    Skips entirely — no gold check, no LLM call, no store touch — when
    `cfg.long_term_read_only` (the default): nothing would be persisted anyway, so don't
    spend latency/tokens analyzing a case that won't be saved. That also means this node is
    a no-op during any scored eval run unless `--remember` was passed, exactly like the
    general bank's build/read-only split (graph/longterm.py module docstring).

    Otherwise: compares the FINAL answer to gold. A correct answer is a no-op (nothing to
    learn is written). A wrong answer gets one extra LLM call — a `mistake-analyst` agent
    reviews the miss with the correct option revealed (only here) and distills a `Lesson:`
    line, stored via `longterm.remember_mistake`.

    MUST NOT touch `state["answer"]`/`state["rationale"]` — this node's entire job is
    off to the side, writing to a store for FUTURE episodes; it must never change what
    gets scored for the CURRENT one.
    """
    agent = Agent("mistake-analyst", roles.ROLE_PROMPTS["mistake-analyst"],
                  model=cfg.model_for("verifier"))

    lt: dict[str, Any] = {}

    def _store():
        if "store" not in lt:
            lt["store"], lt["close"] = longterm.try_open(cfg.long_term_db)  # close unused but must stay referenced (longterm.try_open)
        return lt["store"]

    def node(state: dict) -> dict:
        if cfg.long_term_read_only:
            return {}
        item = state["item"]
        chosen = state.get("answer")
        if chosen == item.answer_idx:
            return {}                      # correct — nothing to learn

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
