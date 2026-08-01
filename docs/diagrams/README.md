# Diagrams

Standalone SVGs for the midterm report and slides. Self-contained — no external CSS,
fonts or scripts — so they can be dropped straight into Word, LaTeX, Google Slides or a
web page, and scale without pixelation.

**Stale — depict a pre-rebuild design.** `v1_flow.svg` and `v2_flow.svg` were hand-drawn against the
MedAgents-style agentic-panel design (`clinical_reason → decider` V2, per-specialist tool calls). The
graph was since rebuilt again into the current 4-node design (`understand → (search ‖ reasoning) →
decider [→ verify]`, an explicit asymmetric-trust rule instead of a numeric gate), so these two no
longer match the current graph topology. They have not been redrawn as part of this pass — regenerate
them (or draw new ones) from the current `build_graph(cfg).get_graph().draw_mermaid()` output before
using them in the report or slides. `v0_flow.svg` is unaffected (V0 hasn't changed).

`ladder_overview.svg` and `system_stack.svg` are current — redrawn/added against the present 4-node
design and reflect the unified MedMCQA corpus across V1–V4.

| file | shows | report section |
|---|---|---|
| [system_stack.svg](system_stack.svg) | the full technology stack — data, embedding, ChromaDB, LangGraph/LangChain, LLM inference, evaluation | §2 System Architecture |
| [ladder_overview.svg](ladder_overview.svg) | V0–V4 side by side, what each rung adds | §2 System Architecture |
| [v0_flow.svg](v0_flow.svg) | V0 — single `answer` node, and what happens inside it | §3 System Design |
| [v1_flow.svg](v1_flow.svg) | V1 — retrieval, the gate, and the two prompt paths (stale, see above) | §3 RAG design |
| [v2_flow.svg](v2_flow.svg) | V2 — the three agents and where each LLM call falls (stale, see above) | §3 Agent architecture |
| [memory_design.svg](memory_design.svg) | short-term memory — the shared `QAState` table: who writes each field, who reads it | §3 Memory design |
| [longterm_memory.svg](longterm_memory.svg) | long-term memory — the two namespaced banks (`medqa-lessons` read/written by the decider; `medqa-mistakes` read by the verifier, written only by `distill_mistake`) and the contamination guards | §3 Memory design |

**Colour convention** (consistent across the flow diagrams; `system_stack.svg` uses its own legend for
platform layers — data / embedding / orchestration / LLM call / evaluation):

| colour | meaning |
|---|---|
| purple | an LLM call |
| teal | retrieval — no LLM call |
| pink | the stage that is new in this variant |
| green / amber | the two prompt paths after the gate (evidence vs no-RAG fallback) |
| grey | START / END / control flow |

**To convert:** most tools take SVG directly. If a format needs raster,
`rsvg-convert -w 2000 v1_flow.svg -o v1_flow.png` or open in a browser and export.

**Keeping them accurate:** these are hand-drawn, not generated, so they do not update when
the code changes. The mermaid sources in [ARCHITECTURE.md](../ARCHITECTURE.md) and
[REPORT_NOTES.md](../REPORT_NOTES.md) are the versioned text equivalents; the live topology
can always be recovered with:

```python
build_graph(cfg).get_graph().draw_mermaid()
```
