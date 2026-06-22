---
name: humanoid-oli-docs-before-sim
description: "Oli workflow sequence: build AI-ready docs, answer SDK/control questions, then adapt simulation."
metadata: 
  node_type: memory
  type: project
  originSessionId: 5d82aaba-6c05-4ee2-a722-f7c3e26b188d
---

For Oli onboarding, build the AI-native knowledge base from official LimX/Oli documentation before trying to fully answer the SDK/control-interface questions or adapt simulation.

**Why:** The MAY-137 SDK note should be answered from real documentation rather than assumptions, and simulation work needs clear control/state interface targets.

**How to apply:** When Oli work resumes, prioritize MAY-139 documentation extraction/source mapping first, use it to fill MAY-137, then move to Isaac/simulation adaptation once the interface map is grounded.