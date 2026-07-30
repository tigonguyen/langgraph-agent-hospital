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
from agent_hospital.knowledge import default_embeddings, format_evidence, open_store, retrieve
from agent_hospital.qa.mcq import (AGENTIC_ANSWER, AGENTIC_ANSWER_TEXTBOOK, AGENTIC_VERIFY,
                                   ANALYSE_ONLY, DIGEST_EVIDENCE_ONLY, LETTER_ONLY,
                                   REASON_THEN_ANSWER, SCRIBE_NOTES, UNDERSTAND_ONLY, format_mcq,
                                   parse_choice)
from agent_hospital.qa.reasoning import build_followup_agent, build_reasoning_agent, build_router

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


def make_iterative_retrieve_node(cfg: RunConfig) -> Node:
    """i-MedRAG-style iterative retrieval (Xiong et al., 2024): distill an initial query,
    retrieve, then repeatedly ask for ONE follow-up query grounded in the evidence gathered
    so far, retrieving again, until the model signals it has enough or `rag.iterative_max`
    rounds are used. Evidence accumulates (deduped) across rounds — unlike `make_retrieve_node`,
    each follow-up round costs one LLM call (the price of chaining retrieval, not just gating it).
    """
    rag = cfg.rag
    reasoner = build_reasoning_agent(cfg.model_for("reasoner"))
    next_query = build_followup_agent(cfg.model_for("reasoner"))
    holder: dict[str, Any] = {}

    def node(state: dict) -> dict:
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        item = state["item"]

        query = reasoner(item)
        queries = [query]
        hits: list[tuple[Any, float]] = []
        seen: set[str] = set()

        for i in range(rag.iterative_max):
            for doc, score in retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"]):
                key = (doc.metadata.get("answer") or doc.page_content).strip().lower()
                if key not in seen:
                    seen.add(key)
                    hits.append((doc, score))
            if i == rag.iterative_max - 1:
                break
            query = next_query(item, format_evidence(hits))
            if not query:
                break
            queries.append(query)

        return {"evidence": format_evidence(hits), "query": " | ".join(queries)}

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


# Hard cap for agentic search (V1/V1a and the verifier's own search) — matches the "up to
# twice" confidence-gated retry the prompts describe; see _call_cap_guard for why a prompt-
# only cap isn't enough.
_MAX_AGENTIC_SEARCHES = 2

# role -> (tool factory, closing instruction) for the single-agent agentic-RAG node below.
# V1 searches MedMCQA (solved exam questions); V1a searches MedRAG Textbooks instead —
# same one-agent-does-everything architecture, different corpus.
_AGENTIC_TOOL_BY_ROLE = {
    "rag-agent": (lambda cfg: make_medmcqa_tool(cfg, max_calls=_MAX_AGENTIC_SEARCHES), AGENTIC_ANSWER),
    "rag-agent-textbook": (lambda cfg: make_search_tool(cfg, max_calls=_MAX_AGENTIC_SEARCHES),
                           AGENTIC_ANSWER_TEXTBOOK),
}


def make_agentic_rag_node(cfg: RunConfig) -> Node:
    """V1/V1a as a single tool-using agent (agentic RAG).

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
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    def node(state: dict) -> dict:
        # A prior clinical-reasoning report (V2-V4) informs the choice; the short
        # justification written here replaces it as the user-facing explanation. Reads
        # `clinical_report` directly rather than `_clinical_context` — the decider is
        # deliberately unaffected by the scribe's working memory (V3/V4), which exists
        # for the verifier only (see make_scribe_node).
        prior = state.get("clinical_report", "")
        extra = f"\n\nColleague's clinical-reasoning report:\n{prior}" if prior else ""
        reply = agent.say(_prompt(state, extra=extra, closing=REASON_THEN_ANSWER))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_scribe_node(cfg: RunConfig) -> Node:
    """Short-term memory (V3/V4): distil the case summary, retrieved evidence, and clinical
    reasoner's report — everything Nodes 1-3 produced — into shared working notes that ONLY
    the verifier reads (`_clinical_context`). Runs off the same join as `answer` (see
    build_graph), not nested inside the reasoning branch, so it can actually see `evidence`
    without adding decider latency; the decider itself reads `clinical_report` directly and
    is unaffected by this node. The notes live in `state["working_memory"]` for the rest of
    the episode (a per-question working memory, not cross-episode).
    """
    agent = Agent("scribe", roles.ROLE_PROMPTS["scribe"], model=cfg.model_for("scribe"))

    def node(state: dict) -> dict:
        understanding = state.get("case_understanding", "")
        report = state.get("clinical_report", "")
        extra = (f"\n\nCase summary:\n{understanding}" if understanding else "") + \
                f"\n\nClinical-reasoning report:\n{report}"
        notes = agent.say(_prompt(state, extra=extra, closing=SCRIBE_NOTES))
        return {"working_memory": notes.strip()}

    return node


def _clinical_context(state: dict) -> str:
    """Verifier input only: the scribe's condensed working memory if present (V3/V4), else
    the clinical reasoner's raw report (V2). The decider reads `clinical_report` directly
    instead (see make_answer_node) and never sees this."""
    return state.get("working_memory") or state.get("clinical_report", "")


def make_report_verify_node(cfg: RunConfig) -> Node:
    """V2/V3's verifier: a cheap final check, not a second full derivation.

    Reads `_clinical_context(state)` — the reasoner's report or the scribe's condensed
    notes, whichever is present — plus the decider's chosen letter and its own short
    explanation (`state["rationale"]`, not yet overwritten), and audits that choice
    against them. Only reaches for `cfg.verify_rag`'s independent `search_textbooks`
    tool, on a query targeted at the CHOSEN option specifically, if that context alone
    doesn't settle it. Falls back to the prior answer on parse failure, like every
    other late-stage node.
    """
    if cfg.verify_rag:
        search, reset_calls = make_search_tool(replace(cfg, rag=cfg.verify_rag),
                                               max_calls=_MAX_AGENTIC_SEARCHES)
        tools = [search]
    else:
        tools, reset_calls = (), lambda: None
    closing = AGENTIC_VERIFY if cfg.verify_rag else REASON_THEN_ANSWER
    agent = Agent("report-verifier", roles.ROLE_PROMPTS["report-verifier"],
                  model=cfg.model_for("verifier"), tools=tools)

    def node(state: dict) -> dict:
        reset_calls()
        cur = state.get("answer")
        cur_letter = _LETTERS[cur] if cur is not None else "unknown"
        context = _clinical_context(state)
        extra = (f"\n\nClinical-reasoning report (option-by-option verdicts):\n{context}"
                 f"\n\nChosen answer: {cur_letter}"
                 f"\n\nColleague's explanation: {state.get('rationale', '')}")
        reply = agent.say(_prompt(state, extra=extra, closing=closing))
        revised = parse_choice(reply)
        return {"answer": revised if revised is not None else cur, "rationale": reply.strip()}

    return node
