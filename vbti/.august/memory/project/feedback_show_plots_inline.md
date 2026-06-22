---
name: feedback_show_plots_inline
description: After generating any plot/image, always show it inline (Read tool) AND cite the file path so user can re-open
type: feedback
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
After generating any plot, image, or figure, **always show it inline by reading the file with the Read tool, AND state the absolute file path** in the message text.

**Why:** user reviews session transcripts asynchronously and wants to see what was generated without re-running anything. Citing only the path means they have to context-switch to a viewer; showing only the inline image means they can't re-open or share the file. Both are needed.

**How to apply:**
- After every `plt.savefig(...)`, `cv2.imwrite(...)`, or any image-producing tool, follow up with a `Read` of that file in the same response, then mention the absolute path in prose.
- For multiple plots in one response: show at least one inline (the most informative), and cite paths for the rest.
- Even when an experiment subagent reports "saved plot X", relay both the path AND the inline view in the next message.
- Same applies to other artifacts (rendered preview frames, side-by-side grids, etc.).

**EVEN IF YOU SHOWED THE SAME FILE EARLIER IN THE CONVERSATION** — when you regenerate / re-render / re-edit it, show it again inline. The user reviews each response in isolation; "I already showed this 3 messages ago" doesn't count. User reinforced this verbatim 2026-05-04 with "ALWAYS INCLUDE THE PLOTS IN RESPONSE" after I once again only cited the path on a re-render.

**Recovery clue:** if you find yourself thinking "I already showed this above", that's the failure mode — Read it again anyway. The cost of a duplicate image inline is zero; the cost of missing one is the user repeating themselves.
