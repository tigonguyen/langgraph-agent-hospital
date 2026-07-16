"""Reusable graph nodes (state -> partial state), each built from a RunConfig.

Agents are lazy (`create_agent` builds on first call) and the Chroma store opens on
first retrieval, so compiling a graph touches no network / no vector store.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_hospital import roles
from agent_hospital.agents.base import Agent
from agent_hospital.config import RunConfig
from agent_hospital.knowledge import default_embeddings, format_evidence, open_store, retrieve
from agent_hospital.qa.mcq import format_mcq, parse_choice
from agent_hospital.qa.reasoning import build_reasoning_agent

Node = Callable[[dict], dict]
_LETTERS = "ABCD"


def _prompt(state: dict, extra: str = "") -> str:
    ev = state.get("evidence", "")
    head = f"{ev}\n\n" if ev else ""
    return f"{head}{format_mcq(state['item'])}{extra}"


def make_reason_node(cfg: RunConfig) -> Node:
    reasoner = build_reasoning_agent(cfg.model_for("reasoner"))
    distill = bool(cfg.rag and cfg.rag.distill_query)

    def node(state: dict) -> dict:
        item = state["item"]
        return {"query": reasoner(item) if distill else item.question}

    return node


def make_retrieve_node(cfg: RunConfig) -> Node:
    rag = cfg.rag
    holder: dict[str, Any] = {}

    def node(state: dict) -> dict:
        if "store" not in holder:
            holder["store"] = open_store(rag.collection, embeddings=default_embeddings(rag.embedder))
        hits = retrieve(state["query"], k=rag.k, threshold=rag.threshold, store=holder["store"])
        return {"evidence": format_evidence(hits)}

    return node


def make_answer_node(cfg: RunConfig) -> Node:
    agent = Agent(cfg.answer_role, roles.ROLE_PROMPTS[cfg.answer_role], model=cfg.model_for(cfg.answer_role))

    def node(state: dict) -> dict:
        return {"answer": parse_choice(agent.say(_prompt(state)))}

    return node


def make_panel_node(cfg: RunConfig) -> Node:
    agents = []
    for i in range(cfg.panel_size):
        persona = roles.PERSPECTIVES[i % len(roles.PERSPECTIVES)]
        prompt = f"{roles.ROLE_PROMPTS['specialist']}\nPerspective: {persona}"
        agents.append(Agent(f"specialist-{i}", prompt, model=cfg.model_for("specialist")))

    def node(state: dict) -> dict:
        base = _prompt(state)
        return {"opinions": [a.say(base) for a in agents]}

    return node


def make_aggregate_node(cfg: RunConfig) -> Node:
    agent = Agent("attending", roles.ROLE_PROMPTS["attending"], model=cfg.model_for("attending"))

    def node(state: dict) -> dict:
        ops = "\n\n".join(f"Panelist {i + 1}:\n{o}" for i, o in enumerate(state.get("opinions", [])))
        return {"answer": parse_choice(agent.say(_prompt(state, extra=f"\n\nPanel opinions:\n{ops}")))}

    return node


def make_verify_node(cfg: RunConfig) -> Node:
    agent = Agent("verifier", roles.ROLE_PROMPTS["verifier"], model=cfg.model_for("verifier"))

    def node(state: dict) -> dict:
        cur = state.get("answer")
        cur_letter = _LETTERS[cur] if cur is not None else "unknown"
        revised = parse_choice(agent.say(_prompt(state, extra=f"\n\nProposed answer: {cur_letter}")))
        return {"answer": revised if revised is not None else cur}

    return node
