"""Reusable graph nodes (state -> partial state), each built from a RunConfig.

Agents are lazy (`create_agent` builds on first call) and the Chroma store opens on
first retrieval, so compiling a graph touches no network / no vector store.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from agent_hospital import roles
from agent_hospital.agents.base import Agent
from agent_hospital.config import RunConfig
from agent_hospital.knowledge import default_embeddings, format_evidence, open_store, retrieve
from agent_hospital.qa.mcq import (AGENTIC_ANALYSE, AGENTIC_ANSWER, AGENTIC_ANSWER_TEXTBOOK,
                                   AGENTIC_VERIFY, ANALYSE_ONLY, DIGEST_EVIDENCE_ONLY, LETTER_ONLY,
                                   REASON_THEN_ANSWER, SCRIBE_NOTES, format_mcq, parse_choice)
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
    search = make_search_tool(cfg)

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


def make_evidence_branch_node(cfg: RunConfig) -> Node:
    """Layer 1, branch B — reason -> retrieve -> digest, bundled into ONE graph node.

    Bundling matters for real concurrency with `clinical_reason` (branch A): LangGraph's
    synchronous execution advances in supersteps — all nodes in a superstep run
    concurrently, but the NEXT superstep only starts once every node in the CURRENT one
    has finished. Wired as three separate chained nodes, only the first step (~0.7s)
    actually overlaps with clinical_reason's much longer call — retrieve/digest are
    stranded in later supersteps, only starting once clinical_reason's superstep ends
    (measured: 12.8s total). Bundled into one node, the whole ~3.5s branch overlaps
    clinical_reason's slot entirely (measured: 10.9s total, same accuracy-relevant work).
    """
    reason = make_reason_node(cfg)
    retrieve = make_retrieve_node(cfg)
    digest = make_evidence_digest_node(cfg)

    def node(state: dict) -> dict:
        s = {**state, **reason(state)}
        s = {**s, **retrieve(s)}
        s = {**s, **digest(s)}
        return {"evidence": s.get("evidence", ""), "query": s.get("query", "")}

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


def make_search_tool(cfg: RunConfig):
    """A `search_textbooks` tool over the knowledge store (gated retrieval).

    The retrieve node invokes it directly, so it costs no LLM call. Binding it to an
    agent instead would let the model choose when to search, at the cost of one extra
    model invocation per call (the model must be re-invoked on the tool result).
    """
    from langchain_core.tools import tool

    rag = cfg.rag
    holder: dict[str, Any] = {}

    @tool
    def search_textbooks(query: str) -> str:
        """Search medical textbooks for passages relevant to a clinical query."""
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        hits = retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"])
        return format_evidence(hits)          # "" = nothing cleared the gate (no-RAG fallback)

    return search_textbooks


def make_medmcqa_tool(cfg: RunConfig):
    """Tool factory (NOT a graph node): returns a LangChain `@tool` (`search_medmcqa`)
    to BIND to an agent so the *model* decides when to call it (agentic RAG). V1's answer
    agent and V2-V4's clinical reasoner hold it. It retrieves the top-k similar solved
    board questions (MedMCQA) with answers/explanations — no gate (rag.threshold=0.0).
    """
    from langchain_core.tools import tool

    rag = cfg.rag
    holder: dict[str, Any] = {}

    @tool
    def search_medmcqa(query: str) -> str:
        """Search a database of solved medical board questions (MedMCQA) for entries
        similar to the query. Returns the top matches with their correct answer and a
        brief explanation. Use a focused clinical query (the key findings and what is asked)."""
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        hits = retrieve(query, k=rag.k, threshold=rag.threshold, store=holder["store"])
        return format_evidence(hits) or "No similar questions found."

    return search_medmcqa


# role -> (tool factory, closing instruction) for the single-agent agentic-RAG node below.
# V1 searches MedMCQA (solved exam questions); V1a searches MedRAG Textbooks instead —
# same one-agent-does-everything architecture, different corpus.
_AGENTIC_TOOL_BY_ROLE = {
    "rag-agent": (lambda cfg: make_medmcqa_tool(cfg), AGENTIC_ANSWER),
    "rag-agent-textbook": (lambda cfg: make_search_tool(cfg), AGENTIC_ANSWER_TEXTBOOK),
}


def make_agentic_rag_node(cfg: RunConfig) -> Node:
    """V1/V1a as a single tool-using agent (agentic RAG).

    One agent, one tool (which tool + corpus depends on `cfg.answer_role`, see
    `_AGENTIC_TOOL_BY_ROLE`): it decides whether/what to search, the `create_agent`
    loop runs the tool and feeds the results back, and the agent answers. The model —
    not the graph — drives retrieval (contrast V2-V4, where the retrieve node invokes
    the tool directly).
    """
    make_tool, closing = _AGENTIC_TOOL_BY_ROLE[cfg.answer_role]
    search = make_tool(cfg)
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role],
                  model=cfg.model_for(cfg.answer_role), tools=[search])

    def node(state: dict) -> dict:
        reply = agent.say(format_mcq(state["item"], closing))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


# V2-V4's clinical reasoner may hold the same search_medmcqa tool as V1's agent, so its
# system prompt gets this appended only when a tool is actually bound (see make_clinical_reason_node).
_CLINICAL_REASONER_TOOL_HINT = (
    "\n\nYou have a tool, search_medmcqa, that retrieves similar solved board questions with "
    "their correct answer and explanation — call it in step 3 if it would sharpen your reasoning. "
    "Treat any hit as an analogy to weigh, not a guaranteed match."
)


def make_clinical_reason_node(cfg: RunConfig) -> Node:
    """Dedicated clinical-reasoning stage (spec §4.2, V2-V4): works the case up like a physician —
    findings, what's asked, reasoning, option-by-option, summary — and deliberately never names
    a final answer. The `decider` (make_answer_node) reads this report next and commits to a
    letter, so the reasoning stays inspectable on its own rather than an implicit side-effect of
    answering.
    """
    use_tool = bool(cfg.rag and cfg.rag.tool)
    tools = [make_medmcqa_tool(cfg)] if use_tool else ()
    prompt = roles.ROLE_PROMPTS["clinical-reasoner"] + (_CLINICAL_REASONER_TOOL_HINT if use_tool else "")
    agent = Agent("clinical-reasoner", prompt, model=cfg.model_for("clinical-reasoner"), tools=tools)
    closing = AGENTIC_ANALYSE if use_tool else ANALYSE_ONLY
    # Fallback agent for the empty-completion retry below: same role, no tool bound. Built
    # eagerly (not only on failure) so its `create_agent` graph is compiled up front like
    # every other agent, keeping "compiling a graph touches no network" true for this node too.
    fallback = Agent("clinical-reasoner", roles.ROLE_PROMPTS["clinical-reasoner"],
                     model=cfg.model_for("clinical-reasoner")) if use_tool else None

    def node(state: dict) -> dict:
        analysis = agent.say(_prompt(state, closing=closing)).strip()
        if not analysis and fallback is not None:
            # Rare, item-specific failure observed with qwen2.5:7b: a tool-bound agent
            # occasionally returns an empty completion (0 chars, no tool call) for a
            # particular case, reproducible at temperature=0. Retrying the same case
            # without the tool binding reliably produces real analysis.
            analysis = fallback.say(_prompt(state, closing=ANALYSE_ONLY)).strip()
        # `clinical_report` persists untouched for the decider AND the verifier to read;
        # `rationale` is the current best user-facing explanation, which the decider (and
        # then the verifier, if present) will overwrite with its own short justification.
        return {"rationale": analysis, "clinical_report": analysis}

    return node


def make_answer_node(cfg: RunConfig) -> Node:
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    def node(state: dict) -> dict:
        # A prior clinical-reasoning report (V2-V4) informs the choice; the short
        # justification written here replaces it as the user-facing explanation.
        prior = _clinical_context(state)
        extra = f"\n\nColleague's clinical-reasoning report:\n{prior}" if prior else ""
        reply = agent.say(_prompt(state, extra=extra, closing=REASON_THEN_ANSWER))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_scribe_node(cfg: RunConfig) -> Node:
    """Short-term memory (V3/V4): distil the clinical reasoner's report into shared working
    notes that the decider and verifier then read instead of the raw report (`_clinical_context`).
    The notes live in `state["working_memory"]` for the rest of the episode (a per-question
    working memory, not cross-episode).
    """
    agent = Agent("scribe", roles.ROLE_PROMPTS["scribe"], model=cfg.model_for("scribe"))

    def node(state: dict) -> dict:
        report = state.get("clinical_report", "")
        notes = agent.say(_prompt(state, extra=f"\n\nClinical-reasoning report:\n{report}",
                                  closing=SCRIBE_NOTES))
        return {"working_memory": notes.strip()}

    return node


def _clinical_context(state: dict) -> str:
    """Decider/verifier input: the scribe's working memory if present (V3/V4), else the
    clinical reasoner's raw report (V2)."""
    return state.get("working_memory") or state.get("clinical_report", "")


def make_report_verify_node(cfg: RunConfig) -> Node:
    """V2/V3's verifier: a cheap final check, not a second full derivation.

    Reads `_clinical_context(state)` — the reasoner's report or the scribe's condensed
    notes, whichever is present, never overwritten by the decider — and audits the
    decider's chosen letter against it. Only reaches for `cfg.verify_rag`'s independent
    `search_textbooks` tool, on a query targeted at the CHOSEN option specifically, if
    that context alone doesn't settle it. Falls back to the prior answer on parse
    failure, like every other late-stage node.
    """
    tools = [make_search_tool(replace(cfg, rag=cfg.verify_rag))] if cfg.verify_rag else ()
    closing = AGENTIC_VERIFY if cfg.verify_rag else REASON_THEN_ANSWER
    agent = Agent("report-verifier", roles.ROLE_PROMPTS["report-verifier"],
                  model=cfg.model_for("verifier"), tools=tools)

    def node(state: dict) -> dict:
        cur = state.get("answer")
        cur_letter = _LETTERS[cur] if cur is not None else "unknown"
        context = _clinical_context(state)
        extra = (f"\n\nClinical-reasoning report (option-by-option verdicts):\n{context}"
                 f"\n\nChosen answer: {cur_letter}")
        reply = agent.say(_prompt(state, extra=extra, closing=closing))
        revised = parse_choice(reply)
        return {"answer": revised if revised is not None else cur, "rationale": reply.strip()}

    return node
