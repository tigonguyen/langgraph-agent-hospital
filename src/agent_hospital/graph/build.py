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
    seq: list[str] = []

    if cfg.rag:
        g.add_node("reason", nodes.make_reason_node(cfg))
        g.add_node("retrieve", nodes.make_retrieve_node(cfg))
        seq += ["reason", "retrieve"]

    if cfg.panel_size > 1 or cfg.aggregate:
        g.add_node("panel", nodes.make_panel_node(cfg))
        g.add_node("aggregate", nodes.make_aggregate_node(cfg))
        seq += ["panel", "aggregate"]
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
