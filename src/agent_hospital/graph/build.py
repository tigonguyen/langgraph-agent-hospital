"""Assemble a compiled StateGraph from a RunConfig.

V2-V5 (rag.tool=False): understand -> (search || reasoning) -> answer -> [verify] ->
[distill_mistake]. search and reasoning are wired straight off `understand` so LangGraph
runs them in the same superstep — bundling retrieve+digest into one `search` node matters
here, since a multi-step chain racing a single long call only overlaps on its first step.

A join with >1 predecessor MUST use `add_edge([a, b], c)` (list form) so `c` runs ONCE
after both complete — separate `add_edge(a, c)`/`add_edge(b, c)` calls each trigger `c`
independently, causing two concurrent writes to the same state key.

`rag.tool=True` (V1) instead binds retrieval as a tool on a single agent (agentic RAG,
sequential — the model must be re-invoked to consume its own tool call).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent_hospital.config import RunConfig
from agent_hospital.graph import nodes
from agent_hospital.graph.state import QAState


def build_graph(cfg: RunConfig):
    # In-process only, keyed by thread_id (build_variant uses the item id) — this is
    # per-episode checkpointing, not cross-episode memory (see graph/longterm.py for that).
    # MCQItem must be allowlisted or every checkpoint read logs a deprecation warning.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("agent_hospital.diseases.medqa_usmle", "MCQItem")])
    # Opt-in: one compiled graph serves every item, so an always-on saver accumulates a
    # thread per question that nothing reads back (~100KB/item, ~128MB over 1273 items).
    checkpointer = InMemorySaver(serde=serde) if cfg.checkpoint else None

    g = StateGraph(QAState)

    if cfg.rag and cfg.rag.tool and not cfg.clinical_reason:
        g.add_node("agent", nodes.make_agentic_rag_node(cfg))
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g.compile(checkpointer=checkpointer)

    rag_start = START
    if cfg.clinical_reason:
        g.add_node("understand", nodes.make_understand_node(cfg))
        g.add_edge(START, "understand")
        rag_start = "understand"

    # Branches wired straight off rag_start run concurrently, not in series.
    predecessors: list[str] = []

    if cfg.rag and not cfg.rag.tool:
        if cfg.rag.adaptive:
            g.add_node("retrieve", nodes.make_adaptive_rag_node(cfg))
            g.add_edge(rag_start, "retrieve")
            predecessors.append("retrieve")
        elif cfg.clinical_reason:
            g.add_node("search", nodes.make_search_branch_node(cfg))
            g.add_edge("understand", "search")
            predecessors.append("search")
        else:
            g.add_node("reason", nodes.make_reason_node(cfg))
            g.add_node("retrieve", nodes.make_retrieve_node(cfg))
            g.add_edge(START, "reason")
            g.add_edge("reason", "retrieve")
            predecessors.append("retrieve")

    if cfg.clinical_reason:
        g.add_node("reasoning", nodes.make_reasoning_node(cfg))
        g.add_edge("understand", "reasoning")
        predecessors.append("reasoning")

    g.add_node("answer", nodes.make_answer_node(cfg))
    if len(predecessors) > 1:
        g.add_edge(predecessors, "answer")     # list form = wait for ALL, run once
    else:
        g.add_edge(predecessors[0] if predecessors else START, "answer")
    prev = "answer"

    if cfg.verify:
        g.add_node("verify", nodes.make_report_verify_node(cfg))
        g.add_edge(prev, "verify")
        prev = "verify"

    if cfg.long_term_mistakes:
        g.add_node("distill_mistake", nodes.make_mistake_distill_node(cfg))
        g.add_edge(prev, "distill_mistake")
        prev = "distill_mistake"

    g.add_edge(prev, END)
    return g.compile(checkpointer=checkpointer)
