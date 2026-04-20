---
name: Don't read rendered gallery images unless explicitly asked
description: When rendering galleries of many diagnostic images, the coordinator should not open/Read each PNG — delegate review to the render agent via findings.md
type: feedback
---

When spawning an agent to render a batch of diagnostic images (galleries for filter validation, sanity checks, sample renders, etc.), the coordinator must NOT use the Read tool to open each PNG afterwards. Only read images if the user explicitly asks "show me X" or references a specific file.

**Why:** Reading dozens of images eats a lot of time and context for little incremental value — the agent already reviewed them. The user wants the coordinator to stay focused on decisions, not image-by-image inspection.

**How to apply:**
- Have the render agent take review responsibility: instruct it to Read every image it rendered and produce a thorough findings.md with per-image labels, flagged filenames, and summary stats
- In the coordinator's response after the agent returns, relay the agent's summary numbers, not personal impressions
- Spot-check only when the user explicitly asks (e.g., "show me the 3 borderline ones")
- If the user says "look at N images" they mean "read exactly N", not "N or more"
