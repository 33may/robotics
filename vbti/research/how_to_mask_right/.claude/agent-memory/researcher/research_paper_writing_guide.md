---
name: research_paper_writing_guide
description: Complete style and structure guide for writing ML research papers — section-by-section guidance, word counts, common mistakes, abstract rules
type: reference
---

# How to Write ML Research Papers

## General Principles

- Successful communication is central to the scientific process
- Prefer active voice and first-person over passive
- Each paragraph discusses one distinct idea and flows naturally
- Keep sentences digestible; paragraphs 3-6 sentences
- Use specific section headings, not generic ones
- Name and explain things in relation to existing concepts
- Minimize abbreviations; avoid them in titles and headings
- Every claim needs evidence — experimental results, theoretical proof, or citation

## Paper Structure (IMRAD + extensions)

### Abstract (150-250 words, write LAST)
- One paragraph, max 6-7 sentences
- Structure: Background (1-2 sentences) → Gap/Purpose (1 sentence) → Method (1-2 sentences) → Key Results (1-2 sentences) → Implication (1 sentence)
- Results section should be the longest part of the abstract
- Use past tense, active voice where possible, short concise sentences
- Do NOT include: citations, abbreviations, figures, extensive background, jargon
- Do NOT repeat abstract content in introduction
- Do NOT claim more than data demonstrates
- Specific findings with data, not vague statements ("49% vs 30%, P<0.01" not "differed significantly")

### Introduction (broad → specific → your contribution)
- Start with research area and importance (broad context)
- Narrow to specific problem or gap in knowledge
- State what is already known
- State what is NOT known (the gap)
- State your research purpose — usually to fill the gap
- End with clear statement of study aims
- For ML: "The purpose of this study is to use <data> for <problem> to <output>"

### Methods / Experimental Setup
- Detailed enough for replication
- Use subheadings: Data, Model Architecture, Evaluation Metrics, etc.
- Past tense, can use passive voice here
- No results or justifications (those go in Discussion)
- For ML: describe datasets, model configs, hardware, hyperparameters, evaluation protocol

### Results (what you found — no interpretation)
- Present in order of importance, not chronological
- State findings plainly, with numbers
- Use figures and tables for complex data
- All figures: labeled axes, titles, legends, detailed captions that stand alone
- Do NOT interpret — just report

### Discussion (what the findings mean)
- Start with brief summary of important findings
- Compare with related work
- Explain WHY results are what they are
- Connect findings across experiments
- State limitations honestly
- Suggest future work based on limitations

### Conclusion
- Primary take-home message
- 1-2 additional important findings
- Practical implications
- Brief, precise — no new information

## Seven Essential Content Areas (ICML guide)

1. Research goals + evaluation criteria
2. Task definition
3. Knowledge representation / data description
4. System details (enough to reimplement)
5. Evaluation evidence (never unsupported claims)
6. Related work (similarities, differences, advances)
7. Limitations + proposed future solutions

## Figures & Tables

- Label ALL components: axes, legends, units
- Captions should be self-contained — reader understands without reading text
- Always reference and discuss figures in the text
- Tables for summarized text/numbers; figures for graphical material
- Never say "see figure above/below" — use figure numbers

## Common Mistakes to Avoid

- Making unsupported claims about superiority
- Vague results ("improved significantly") without numbers
- Confusing correlation with causation
- Omitting limitations
- Generic section headings ("Results" with no specificity)
- Single-paragraph subsections
- Overusing system names as sentence subjects
- Inadequate figure captions
- Abstract that's just a compressed version of every section
- Discussion that just restates results without interpretation

## Style

- Active voice: "We find that..." not "It was found that..."
- First person is fine: "We conducted..." "Our results show..."
- Break long adjective chains
- No contractions in technical writing
- Transition sentences at section beginnings
- Consistent section lengths (don't have one 3-page section and one paragraph)

## Sources
- [ICML: Crafting Papers on ML](https://icml.cc/Conferences/2002/craft.html)
- [PMC: How to write a good abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC3136027/)
- [USC: The Abstract](https://libguides.usc.edu/writingguide/abstract)
- [UMN: Structure of a Research Paper](https://libguides.umn.edu/StructureResearchPaper)
- [UBC: IMRAD Format](https://scwrl.ubc.ca/stem-writing-resources/features-of-academic-stem-research-writing/imrad/)
