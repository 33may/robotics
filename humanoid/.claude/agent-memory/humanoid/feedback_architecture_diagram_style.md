---
name: feedback-architecture-diagram-style
description: Architecture diagrams must match the arm-project C4/C2 style (hand-composed HTML+SVG, code-grounded) — NOT warm deck cards, NOT graphviz auto-layout
metadata:
  type: feedback
---

For software/architecture diagrams, Anton wants the **arm-project C4 style**: reference `vbti/oficial/docs/diagrams/c2_containers.png` — cool light canvas, dashed SYSTEM boundary, labeled process sub-boundaries, orange cards = running/validated, dashed cards = prepared/gated, gray cards = external actors, thin gray labeled arrows, legend top-right, `C4 · LEVEL N` kicker, code-grounded footnote.

**Why:** two attempts were rejected on 2026-07-06 — (1) the warm theme.css "railed card" version read as marketing ("complete bullshit... more scientific, UML maybe, more software looking"), (2) a graphviz-dot auto-layout version was rejected too ("structure it better... closer to the containers idea"). What worked: hand-composed HTML + absolutely-positioned cards + hand-routed SVG orthogonal connectors, replicating c2_containers.png exactly.

**How to apply:** before drawing any architecture figure, READ the actual code first (components/transports/contracts must be real — cite frame sizes, ports, rates), then compose in the C2 language. Working example: `vbti/oficial/docs/diagrams/oli_c2_runtime.html`, rendered via `cd visuals && node render.mjs ../diagrams/oli_c2_runtime.html --selector ".canvas" --pad 0`. Methodology spec: `docs/diagrams/c3_uml_agent_prompt.md` (code-grounded, close-the-visual-loop). See [[feedback-show-dont-tell]].
