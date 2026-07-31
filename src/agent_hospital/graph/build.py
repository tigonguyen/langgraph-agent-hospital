"""Assemble a compiled StateGraph from a RunConfig.

The 4-node design (V2-V4, rag.tool=False) — the same 4 jobs V1's single agent does
internally (understand, search, reason, decide), split into 4 nodes:

                    understand          <- Node 1: shared case summary + search query
                   /            \\
              search          reasoning  <- Nodes 2 & 3: CONCURRENT, each ONE graph node,
                |                 |          neither reads the other's output. Node 2 bundles
                 \\                /          retrieve+digest into one node (a multi-step chain
                  answer (decider)  <- Node 4  racing a single long call has its LATER steps
                       |                       stranded in later supersteps otherwise — see
                    [verify]                   make_search_branch_node for the measured why).

Short-term memory (`cfg.memory`, V4/V5) is NOT a separate agent: `case_understanding`,
`evidence`, and `clinical_report` already sit in the shared graph state that every node
reads, so `verify` (the only consumer) reads them straight from `state` — no LLM call
needed just to pass information one node already produced to another.

A join with >1 predecessor MUST be wired as `add_edge([a, b], c)` (list form) so `c` runs
ONCE after both complete — separate `add_edge(a, c)` / `add_edge(b, c)` calls each trigger
`c` independently, causing two concurrent writes to the same state key.

`rag.tool=True` (V1) instead binds retrieval AS a tool directly to a single answering
agent (agentic RAG, sequential by nature — the model must be re-invoked to consume its own
tool call) rather than running it as a separate node/branch at all.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent_hospital.config import RunConfig
from agent_hospital.graph import nodes
from agent_hospital.graph.state import QAState


def build_graph(cfg: RunConfig):
    # Short-term (per-episode) checkpointing: snapshots QAState after every node, keyed by
    # the thread_id the caller passes at invoke time (build_variant uses the item id, one
    # thread per question — never shared across items, so this adds no cross-episode memory).
    # In-process only (not written to disk), so it doesn't survive past this Python process.
    # MCQItem (state["item"]) is a plain dataclass, not a registered msgpack type, so it must
    # be explicitly allowlisted or every checkpoint read logs a deprecation warning.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("agent_hospital.diseases.medqa_usmle", "MCQItem")])
    # Opt-in (cfg.checkpoint): one compiled graph serves every item, so an always-on saver
    # accumulates a thread per question that nothing ever reads back — measured ~100 KB per
    # item, ~128 MB over the 1273-item test split. Long-term memory does NOT come from here
    # (a checkpointer is keyed by thread_id = one question); see graph/longterm.py.
    checkpointer = InMemorySaver(serde=serde) if cfg.checkpoint else None

    g = StateGraph(QAState)

    # V1: a single tool-using agent that both retrieves AND answers (agentic RAG).
    # Skipped when the 4-node design is wired (V2-V4) — there, retrieval runs as its own
    # concurrent node instead (see module docstring).
    if cfg.rag and cfg.rag.tool and not cfg.clinical_reason:
        g.add_node("agent", nodes.make_agentic_rag_node(cfg))
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g.compile(checkpointer=checkpointer)

    # Node 1 (V2-V4 only): shared case understanding + search query, read by Nodes 2 and 3
    # below instead of each independently re-deriving its own.
    rag_start = START
    if cfg.clinical_reason:
        g.add_node("understand", nodes.make_understand_node(cfg))
        g.add_edge(START, "understand")
        rag_start = "understand"

    # Nodes that must complete before `answer` (Node 4) runs. Each branch is wired straight
    # off `rag_start`, so when more than one is present they execute concurrently, not in series.
    predecessors: list[str] = []

    # Node 2 (or the plain graph-invoked RAG path when there's no clinical_reason stage at
    # all): fetch + digest RAG evidence. `digest` overwrites state["evidence"] with an
    # interpreted summary of the k retrieved docs (not raw snippets), so `answer`/`verify`
    # pick it up via the existing `_prompt` auto-prepend with no changes needed there.
    if cfg.rag and not cfg.rag.tool:
        if cfg.rag.adaptive:
            g.add_node("retrieve", nodes.make_adaptive_rag_node(cfg))
            g.add_edge(rag_start, "retrieve")
            predecessors.append("retrieve")
        elif cfg.clinical_reason:
            # Node 2: uses the query `understand` already produced (no separate
            # query-distillation call here) — bundled into ONE node so it shares
            # `understand`'s superstep with Node 3 for real concurrency.
            g.add_node("search", nodes.make_search_branch_node(cfg))
            g.add_edge("understand", "search")
            predecessors.append("search")
        else:
            g.add_node("reason", nodes.make_reason_node(cfg))
            g.add_node("retrieve", nodes.make_retrieve_node(cfg))
            g.add_edge(START, "reason")
            g.add_edge("reason", "retrieve")
            predecessors.append("retrieve")

    # Node 3: clinical reasoning from the shared understanding, blind to evidence (runs
    # concurrently with Node 2 — neither depends on the other's output).
    if cfg.clinical_reason:
        g.add_node("reasoning", nodes.make_reasoning_node(cfg))
        g.add_edge("understand", "reasoning")
        predecessors.append("reasoning")

    # Node 4: the decider, the join point for Nodes 2 and 3.
    g.add_node("answer", nodes.make_answer_node(cfg))
    if len(predecessors) > 1:
        g.add_edge(predecessors, "answer")     # list form = wait for ALL, run once
    else:
        g.add_edge(predecessors[0] if predecessors else START, "answer")
    prev = "answer"

    # Verifier (V4/V5). `cfg.memory` just tells it to also read case_understanding from
    # state (see make_report_verify_node) — no extra node or LLM call, the state is
    # already there.
    if cfg.verify:
        g.add_node("verify", nodes.make_report_verify_node(cfg))
        g.add_edge(prev, "verify")
        prev = "verify"

    # Node 6 (V5 only): the only node that ever reads gold. Runs after verify, writes a
    # corrective lesson to the SEPARATE mistake bank only when the final answer was wrong
    # (see make_mistake_distill_node) — never touches state["answer"]/["rationale"].
    if cfg.long_term_mistakes:
        g.add_node("distill_mistake", nodes.make_mistake_distill_node(cfg))
        g.add_edge(prev, "distill_mistake")
        prev = "distill_mistake"

    g.add_edge(prev, END)
    return g.compile(checkpointer=checkpointer)
