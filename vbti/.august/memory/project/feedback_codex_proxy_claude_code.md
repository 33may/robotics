---
name: codex-proxy-claude-code-preference
description: User prefers Claude Code's TUI but routes it to OpenAI Codex through a proxy.
metadata:
  type: feedback
  originSessionId: 4cde1ebf-036c-48f8-9ddf-a6ebb1ddcf25
---

"I was using the claude code but recenstluy I dont like the speead and quality of acvyual opus model. So i strated using the openai codex model, however I am giant fan of the claude code application TUI, so we created the codex proxy so now inside the claude code I have the openai model."

**Why:** The model choice changed because the user preferred Codex speed/quality, but the Claude Code terminal experience stayed the same.
**How to apply:** When editing or debugging this setup, treat the harness/UI and the backend model separately and verify the proxy still makes the stack feel native.