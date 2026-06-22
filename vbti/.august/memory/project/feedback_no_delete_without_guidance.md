---
name: Never delete files without explicit user guidance
description: Cleanup/disk operations — Claude surveys and reports only; user decides every deletion
type: feedback
originSessionId: fb1c4af0-5d7a-4759-9511-3fa563600fdc
---
Never run `rm`, `trash-empty`, `conda env remove`, `pip cache purge`, dataset deletions, or any destructive cleanup command without the user explicitly approving that specific action. Even "obvious" wins like trash or pip cache require a green light.

**Why:** The user is the one responsible for the state of their PC. Cleanup decisions depend on context Claude doesn't have (what's mid-experiment, what's about to be re-used, what they care about keeping). A wrong delete is unrecoverable.

**How to apply:** During PC cleanup or any disk-management session — survey, present findings with sizes and what each item is, recommend if asked, but stop there. Wait for the user to say "delete X" before any destructive action. The user runs the deletes themselves or tells Claude exactly what to remove.
