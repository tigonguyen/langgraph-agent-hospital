# Diagrams

Standalone SVGs for the midterm report and slides. Self-contained — no external CSS,
fonts or scripts — so they can be dropped straight into Word, LaTeX, Google Slides or a
web page, and scale without pixelation.

| file | shows | report section |
|---|---|---|
| [ladder_overview.svg](ladder_overview.svg) | V0–V4 side by side, what each rung adds | §2 System Architecture |
| [v0_flow.svg](v0_flow.svg) | V0 — single `answer` node, and what happens inside it | §3 System Design |
| [v1_flow.svg](v1_flow.svg) | V1 — retrieval, the gate, and the two prompt paths | §3 RAG design |
| [v2_flow.svg](v2_flow.svg) | V2 — the three agents and where each LLM call falls | §3 Agent architecture |

**Colour convention** (consistent across all four):

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
