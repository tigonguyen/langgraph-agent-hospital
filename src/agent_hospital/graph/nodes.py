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
from agent_hospital.qa.mcq import (AGENTIC_ANSWER, AGENTIC_VERIFY, DELIBERATE, LETTER_ONLY,
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
    agent and V2/V3's panel hold it. It retrieves the top-k similar solved
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


def make_agentic_rag_node(cfg: RunConfig) -> Node:
    """V1 as a single tool-using agent (agentic RAG).

    One agent, one tool: it decides whether/what to search in the MedMCQA database,
    the `create_agent` loop runs the tool and feeds the results back, and the agent
    answers. The model — not the graph — drives retrieval (contrast V2-V4, where the
    retrieve node invokes the tool directly).
    """
    search = make_medmcqa_tool(cfg)
    agent = Agent("rag-agent", roles.ROLE_PROMPTS[cfg.answer_role],
                  model=cfg.model_for(cfg.answer_role), tools=[search])

    def node(state: dict) -> dict:
        reply = agent.say(format_mcq(state["item"], AGENTIC_ANSWER))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_answer_node(cfg: RunConfig) -> Node:
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    def node(state: dict) -> dict:
        reply = agent.say(_prompt(state, closing=REASON_THEN_ANSWER))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_panel_node(cfg: RunConfig) -> Node:
    # Agentic RAG (rag.tool): each specialist holds search_medmcqa and searches on its own
    # while forming an opinion (MedAgents-style experts). Otherwise the panel reads the
    # graph-retrieved evidence prepended by _prompt (V3/V4).
    tools = [make_medmcqa_tool(cfg)] if (cfg.rag and cfg.rag.tool) else ()
    agents = []
    for i in range(cfg.panel_size):
        persona = roles.PERSPECTIVES[i % len(roles.PERSPECTIVES)]
        prompt = f"{roles.ROLE_PROMPTS['specialist']}\nPerspective: {persona}"
        agents.append(Agent(f"specialist-{i}", prompt, model=cfg.model_for("specialist"), tools=tools))

    def node(state: dict) -> dict:
        base = _prompt(state, closing=DELIBERATE)
        return {"opinions": [a.say(base) for a in agents]}

    return node


def make_scribe_node(cfg: RunConfig) -> Node:
    """Short-term memory (V3): distil the panel discussion into shared working notes that
    the attending and verifier then reason over. The notes live in `state["working_memory"]`
    for the rest of the episode (a per-question working memory, not cross-episode).
    """
    agent = Agent("scribe", roles.ROLE_PROMPTS["scribe"], model=cfg.model_for("scribe"))

    def node(state: dict) -> dict:
        ops = "\n\n".join(f"Panelist {i + 1}:\n{o}" for i, o in enumerate(state.get("opinions", [])))
        notes = agent.say(_prompt(state, extra=f"\n\nPanel opinions:\n{ops}", closing=SCRIBE_NOTES))
        return {"working_memory": notes.strip()}

    return node


def _panel_context(state: dict) -> str:
    """Attending/verifier input: the scribe's working memory if present (V3), else the
    raw panel opinions (V2/V4)."""
    wm = state.get("working_memory", "")
    if wm:
        return f"\n\nWorking notes (team memory):\n{wm}"
    ops = "\n\n".join(f"Panelist {i + 1}:\n{o}" for i, o in enumerate(state.get("opinions", [])))
    return f"\n\nPanel opinions:\n{ops}"


def make_aggregate_node(cfg: RunConfig) -> Node:
    agent = Agent("attending", roles.ROLE_PROMPTS["attending"], model=cfg.model_for("attending"))

    def node(state: dict) -> dict:
        reply = agent.say(_prompt(state, extra=_panel_context(state), closing=REASON_THEN_ANSWER))
        return {"answer": parse_choice(reply), "rationale": reply.strip()}

    return node


def make_verify_node(cfg: RunConfig) -> Node:
    """Verifier stage. If `cfg.verify_rag` is set, the verifier holds its OWN
    `search_textbooks` tool (independent of the panel's RAG) to double-check the
    proposed answer against reference textbook passages — separate corpus from the
    panel's MedMCQA exemplars: specialists get analogous solved cases, the verifier
    checks against expository textbook fact.
    """
    tools = [make_search_tool(replace(cfg, rag=cfg.verify_rag))] if cfg.verify_rag else ()
    closing = AGENTIC_VERIFY if cfg.verify_rag else REASON_THEN_ANSWER
    agent = Agent("verifier", roles.ROLE_PROMPTS["verifier"], model=cfg.model_for("verifier"), tools=tools)

    def node(state: dict) -> dict:
        cur = state.get("answer")
        cur_letter = _LETTERS[cur] if cur is not None else "unknown"
        mem = f"\n\nWorking notes (team memory):\n{state['working_memory']}" if state.get("working_memory") else ""
        reply = agent.say(_prompt(state, extra=f"{mem}\n\nProposed answer: {cur_letter}",
                                  closing=closing))
        revised = parse_choice(reply)
        return {"answer": revised if revised is not None else cur, "rationale": reply.strip()}

    return node
