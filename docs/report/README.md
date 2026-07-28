# Midterm report (LaTeX)

Source for the midterm report, structured against the framework in the assignment PDF (§12).

## Files
- `report.tex` — the report (Introduction → Conclusion + reproducibility appendix).
- `references.bib` — citations (MedQA, MedMCQA, MedCPT, MedRAG, Medprompt, Qwen2.5, LangGraph).
- `figures/*.pdf` — diagrams, converted from `docs/diagrams/*.svg` via `rsvg-convert`.
- `report-overleaf.zip` — everything above, packaged for Overleaf upload.

`[TODO: …]` markers (red in the PDF) flag every place waiting on a measurement that is not yet
collected (full test-set run, V3/V4 at scale, prediction files, token cost).

## Compile locally
```bash
tectonic report.tex          # self-contained, no TeX install needed
# or, with a full TeX distribution:
latexmk -pdf report.tex
```

## Deploy to Overleaf
Overleaf has **no public API to push into someone's account** — programmatic sync needs the paid
Git/GitHub integration on the account. So the practical paths are:

1. **Upload the zip (recommended).** Overleaf → *New Project* → *Upload Project* → pick
   `report-overleaf.zip`. It compiles as-is (standard packages, PDF figures — no `svg`/shell-escape).
2. **Git integration (paid).** On an Overleaf Premium project, use its Git or GitHub bridge and push
   this `docs/report/` directory.

Regenerate the zip after edits:
```bash
cd docs/report && zip -r report-overleaf.zip report.tex references.bib figures/
```

Regenerate figures if the SVGs change:
```bash
for f in ladder_overview v0_flow v1_flow v2_flow; do
  rsvg-convert -f pdf -o docs/report/figures/$f.pdf docs/diagrams/$f.svg
done
```
