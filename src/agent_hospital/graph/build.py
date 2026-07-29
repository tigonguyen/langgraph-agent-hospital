"""Assemble a compiled StateGraph from a RunConfig.

Layered pipeline (V2-V4, rag.tool=False):
              START
             /      \\
  clinical_reason  evidence_branch   <- Layer 1: two CONCURRENT branches, each ONE graph
        |              |                node. Neither depends on the other's output, and
      [scribe]          |               LangGraph runs same-superstep nodes concurrently —
         \\            /                but ONLY nodes in the SAME superstep, so branch B's
          answer (decider)   <- Layer 2  three internal steps (reason/retrieve/digest) are
              |                          bundled into one node rather than three chained
          [verify]           <- Layer 3  ones (see `make_evidence_branch_node` for why: a
                                          multi-step chain racing a single long call has its
                                          LATER steps stranded in later supersteps, only
                                          starting once the long call's superstep ends).

A join with >1 predecessor MUST be wired as `add_edge([a, b], c)` (list form) so `c` runs
ONCE after both complete — separate `add_edge(a, c)` / `add_edge(b, c)` calls each trigger
`c` independently, causing two concurrent writes to the same state key.

`rag.tool=True` (V1) instead binds retrieval AS a tool directly to the answering/reasoning
agent (agentic RAG, sequential by nature — the model must be re-invoked to consume its own
tool call) rather than running it as a separate concurrent branch.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_hospital.config import RunConfig
from agent_hospital.graph import nodes
from agent_hospital.graph.state import QAState


def build_graph(cfg: RunConfig):
    g = StateGraph(QAState)

    # V1: a single tool-using agent that both retrieves AND answers (agentic RAG).
    # Skipped when a dedicated clinical-reasoning stage is wired (V2-V4) — there, retrieval
    # runs as its own concurrent branch instead (see module docstring).
    if cfg.rag and cfg.rag.tool and not cfg.clinical_reason:
        g.add_node("agent", nodes.make_agentic_rag_node(cfg))
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g.compile()

    # Nodes that must complete before `answer` runs. Each branch below is wired straight
    # off START, so when more than one is present they execute concurrently, not in series.
    predecessors: list[str] = []

    # Layer 1, branch B: fetch + digest RAG evidence, its own branch from START. `digest`
    # overwrites state["evidence"] with an interpreted summary of the k retrieved docs (not
    # raw snippets), so `answer`/`verify` pick it up via the existing `_prompt` auto-prepend
    # with no changes needed there.
    if cfg.rag and not cfg.rag.tool:
        if cfg.rag.iterative_max > 0:
            g.add_node("retrieve", nodes.make_iterative_retrieve_node(cfg))
            g.add_edge(START, "retrieve")
            predecessors.append("retrieve")
        elif cfg.rag.adaptive:
            g.add_node("retrieve", nodes.make_adaptive_rag_node(cfg))
            g.add_edge(START, "retrieve")
            predecessors.append("retrieve")
        elif cfg.clinical_reason:
            # Bundled into ONE node so it shares clinical_reason's superstep — see
            # make_evidence_branch_node for why three separate chained nodes would NOT
            # give real concurrency here.
            g.add_node("evidence_branch", nodes.make_evidence_branch_node(cfg))
            g.add_edge(START, "evidence_branch")
            predecessors.append("evidence_branch")
        else:
            g.add_node("reason", nodes.make_reason_node(cfg))
            g.add_node("retrieve", nodes.make_retrieve_node(cfg))
            g.add_edge(START, "reason")
            g.add_edge("reason", "retrieve")
            predecessors.append("retrieve")

    # Layer 1, branch A: clinical reasoning, its own branch from START — runs alongside
    # branch B rather than waiting on it. The reasoner never sees `state["evidence"]` (not
    # written yet at this point), so it digests the case purely from its own knowledge.
    if cfg.clinical_reason:
        g.add_node("clinical_reason", nodes.make_clinical_reason_node(cfg))
        g.add_edge(START, "clinical_reason")
        last = "clinical_reason"
        if cfg.memory:  # short-term memory: scribe condenses the report into working notes (V3/V4)
            g.add_node("scribe", nodes.make_scribe_node(cfg))
            g.add_edge(last, "scribe")
            last = "scribe"
        predecessors.append(last)

    # Layer 2: the decider, the join point for Layer 1's branch(es).
    g.add_node("answer", nodes.make_answer_node(cfg))
    if predecessors:
        g.add_edge(predecessors, "answer")     # list form = wait for ALL, run once
    else:
        g.add_edge(START, "answer")
    prev = "answer"

    # Layer 3: verifier.
    if cfg.verify:
        g.add_node("verify", nodes.make_report_verify_node(cfg))
        g.add_edge(prev, "verify")
        prev = "verify"

    g.add_edge(prev, END)
    return g.compile()
