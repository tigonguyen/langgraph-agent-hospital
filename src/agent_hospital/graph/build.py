"""Assemble a compiled StateGraph from a RunConfig.

Linear pipeline, nodes present per config:
  [reason → retrieve] → (answer | panel → aggregate) → [verify]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_hospital.config import RunConfig
from agent_hospital.graph import nodes
from agent_hospital.graph.state import QAState


def build_graph(cfg: RunConfig):
    g = StateGraph(QAState)

    # V1: a single tool-using agent that drives its own retrieval (agentic RAG).
    if cfg.rag and cfg.rag.tool and cfg.panel_size == 1 and not cfg.aggregate:
        g.add_node("agent", nodes.make_agentic_rag_node(cfg))
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g.compile()

    seq: list[str] = []

    # Graph-invoked RAG (V4) uses explicit reason+retrieve nodes. Agentic RAG
    # (rag.tool) folds retrieval into the answering/panel agents as a tool instead
    # (V1-V3), so no separate reason/retrieve nodes are wired.
    if cfg.rag and not cfg.rag.tool:
        if cfg.rag.iterative_max > 0:
            g.add_node("retrieve", nodes.make_iterative_retrieve_node(cfg))
            seq.append("retrieve")
        elif cfg.rag.adaptive:
            g.add_node("retrieve", nodes.make_adaptive_rag_node(cfg))
            seq.append("retrieve")
        else:
            g.add_node("reason", nodes.make_reason_node(cfg))
            g.add_node("retrieve", nodes.make_retrieve_node(cfg))
            seq += ["reason", "retrieve"]

    if cfg.panel_size > 1 or cfg.aggregate:
        g.add_node("panel", nodes.make_panel_node(cfg))
        seq.append("panel")
        if cfg.memory:  # short-term memory: scribe distils the panel into working notes (V3)
            g.add_node("scribe", nodes.make_scribe_node(cfg))
            seq.append("scribe")
        g.add_node("aggregate", nodes.make_aggregate_node(cfg))
        seq.append("aggregate")
    else:
        g.add_node("answer", nodes.make_answer_node(cfg))
        seq.append("answer")

    if cfg.verify:
        g.add_node("verify", nodes.make_verify_node(cfg))
        seq.append("verify")

    prev = START
    for name in seq:
        g.add_edge(prev, name)
        prev = name
    g.add_edge(prev, END)
    return g.compile()
