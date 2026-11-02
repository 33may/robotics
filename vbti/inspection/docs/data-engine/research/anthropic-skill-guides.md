# Anthropic Skill-Authoring Guidance — Research Notes

Compiled 2026-09-03. Sources fetched live (not from training memory) to inform the `data-engineer` skill build.

---

## 1. Official Claude Code docs — `code.claude.com/docs/en/skills`

Source: https://code.claude.com/docs/en/skills

### What a skill is, and when to make one
> "Create a skill when you keep pasting the same instructions, checklist, or multi-step procedure into chat, or when a section of CLAUDE.md has grown into a procedure rather than a fact. Unlike CLAUDE.md content, a skill's body loads only when it's used, so long reference material costs almost nothing until you need it."

Skills follow the [Agent Skills open standard](https://agentskills.io) (portable across tools). Claude Code adds extensions on top: invocation control, subagent execution (`context: fork`), dynamic context injection (`` !`cmd` ``).

### Frontmatter fields (Claude Code superset)
Only `description` is *recommended*; every field is technically optional.

| Field | Purpose |
|---|---|
| `name` | Display name in skill listings; directory name still decides the invoked command for personal/project skills |
| `description` | What the skill does + when to use it. Claude uses this for triggering. If omitted, Claude Code falls back to the first markdown paragraph |
| `when_to_use` | Extra trigger phrases/example requests, appended to `description` in the listing |
| `argument-hint` | Autocomplete hint, e.g. `[issue-number]` |
| `arguments` | Named positional args for `$name` substitution |
| `disable-model-invocation` | `true` = only the human can invoke it (`/name`); Claude never auto-triggers it. Use for side-effecting actions (`/deploy`, `/commit`) |
| `user-invocable` | `false` = only Claude can invoke it; hidden from `/` menu. Use for background knowledge, not an actionable command |
| `allowed-tools` | Tools Claude may use without an approval prompt **for the turn that invokes the skill only** (clears on next message) |
| `disallowed-tools` | Tools removed from the pool while the skill is active |
| `model` / `effort` | Override model/effort for the skill's turn |
| `context: fork` | Runs the skill's body as the prompt for a forked **subagent** rather than inline in the main conversation |
| `agent` | Which subagent type executes a forked skill (`Explore`, `Plan`, `general-purpose`, or a custom `.claude/agents/*` type). Defaults to `general-purpose` |
| `background` | With `context: fork`, whether the fork runs in the background (default `true`) or blocks the turn |
| `paths` | Glob patterns — skill auto-activates only when Claude is working with matching files |
| `metadata`, `license`, `compatibility` | Agent-Skills-spec fields for portability outside Claude Code; Claude Code stores but doesn't act on them |

Important constraint for **portability outside Claude Code** (claude.ai uploads, Skills API, `package_skill.py`): only 6 fields are legal — `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`. Anything else (e.g. `argument-hint`) causes a hard packaging error.

### Description length / truncation mechanics
- The combined `description` + `when_to_use` text is **truncated at 1,536 characters** in the skill listing.
- The whole skill listing (all skills' name+description) shares a token budget = **1% of the model's context window** (`skillListingBudgetFraction`). When it overflows, Claude Code drops descriptions starting from your *least-invoked* skills first — so a skill you rarely use can silently lose its description text and stop triggering.
- Practical implication: **put the key use case first** in the description, since truncation and overflow both cut from the end.

### Progressive disclosure / content lifecycle
- At startup only `name`+`description` are preloaded (level 1).
- Full `SKILL.md` body loads into context **as a single message** only when invoked (level 2); referenced files load only when Claude actually reads them (level 3+).
- Once loaded, a skill's content **stays in context every turn afterward** — Claude Code does *not* re-read the file each turn. So: "write guidance that should apply throughout a task as standing instructions rather than one-time steps."
- Re-invoking a skill with identical rendered content just adds a short "already loaded" note (dedup), not a full duplicate.
- After auto-compaction, only the **most recent invocation of each skill** is re-attached, capped at 5,000 tokens each, sharing a **25,000-token combined budget** across all skills — oldest-invoked skills can be dropped entirely.

### Directory / size conventions
```
my-skill/
├── SKILL.md        (required — overview and navigation)
├── reference.md     (loaded when needed)
├── examples.md       (loaded when needed)
└── scripts/
    └── helper.py     (executed, not loaded into context)
```
> **"Keep SKILL.md under 500 lines. Move detailed reference material to separate files."**

### Subagent execution (`context: fork`)
- Use when the skill is an explicit, self-contained task (not "here are some conventions"). Warning in the docs: *"context: fork only makes sense for skills with explicit instructions. If your skill contains guidelines like 'use these API conventions' without a task, the subagent receives the guidelines but no actionable prompt, and returns without meaningful output."*
- Two distinct integration directions:
  - **Skill → subagent** (`context: fork` + `agent:`): SKILL.md content becomes the subagent's task; system prompt comes from the agent type; CLAUDE.md still loads (except for `Explore`/`Plan` agents, which skip it to stay lean).
  - **Subagent → skill** (a custom subagent's frontmatter lists `skills:`): the skill is *preloaded* as reference material into the subagent's own context at startup (different from the lazy-load a normal session does).
- Forked-skill edits bypass session checkpoints (`/rewind` won't undo them — only git will).

### Evaluation loop (official pattern)
The `skill-creator` plugin (`anthropics/claude-plugins-official`) automates: writing `evals/evals.json` test cases → isolated per-test subagent runs → grading → **with-skill vs without-skill benchmark** (pass-rate improvement vs token/time overhead) → blind A/B between skill versions → auto-generated "should-trigger" / "should-not-trigger" prompts to tune the description's hit rate.

### Troubleshooting triggering
- **Not triggering**: description likely missing the keywords users actually say; verify with "What skills are available?"; malformed YAML frontmatter silently loads the skill with *empty* metadata (so `/name` still works but auto-trigger can't).
- **Triggers too often**: description too broad — narrow it, or force manual-only with `disable-model-invocation: true`.

---

## 2. Anthropic engineering blog — "Equipping agents for the real world with Agent Skills"

Source: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills (authors: Barry Zhang, Keith Lazuka, Mahesh Murag)

- Core framing: a skill is like **"putting together an onboarding guide for a new hire"** — organized folders of instructions/scripts/resources an agent discovers and loads dynamically, turning a general-purpose agent into a specialized one without hand-building a bespoke agent per use case.
- Progressive disclosure explicitly named as the mechanism that makes **"the amount of context that can be bundled into a skill effectively unbounded"** — cost is paid only for what's actually read.
- Reference files vs. scripts distinction: reference files are for Claude to *read and understand*; scripts can be either *executed tools* or *read as documentation* — the skill author must make which one is intended unambiguous.
- Best-practice loop given: **evaluate first** (find gaps by running representative tasks without the skill) → **structure for scale** (split when unwieldy) → **think from Claude's perspective**, watch real usage → **iterate with Claude itself** (ask Claude to capture a successful ad hoc approach into a reusable skill).
- Security anti-patterns flagged explicitly: install skills **only from trusted sources**; never write instructions that make Claude "connect to potentially untrusted external network sources"; watch for "unexpected trajectories or overreliance on certain contexts" as a signal something's wrong.

---

## 3. `anthropics/skills` GitHub repo + official best-practices doc

Sources: https://github.com/anthropics/skills, https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices

This platform-docs best-practices page is the canonical, most detailed source — linked directly from the Claude Code skills page as "writing guidance that applies across Claude products." Distilled:

### Frontmatter validation rules (hard limits)
- `name`: max **64 chars**, lowercase letters/numbers/hyphens only, no XML tags, cannot contain the reserved words `"anthropic"` or `"claude"`.
- `description`: max **1,024 chars**, non-empty, no XML tags.

### Naming convention
Recommended: **gerund form** (verb + `-ing`) — `processing-pdfs`, `analyzing-spreadsheets`, `managing-databases`. Acceptable alternatives: noun phrases (`pdf-processing`) or imperative (`process-pdfs`). Avoid vague (`helper`, `utils`), overly generic (`data`, `files`), or reserved-word names.

### Description-writing rules (the load-bearing ones)
- **Always third person.** "Processes Excel files and generates reports" — not "I can help you..." The description is injected into the system prompt; inconsistent point-of-view "can cause discovery problems."
- **State what it does AND when to use it**, with concrete trigger terms/contexts, not just a category name.
- Good: `"Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction."`
- Bad: `"Helps with documents"`, `"Processes data"`, `"Does stuff with files"`.
- Claude picks among "potentially 100+ available Skills" using only this field — it is explicitly called "critical for skill selection."

### Progressive disclosure — concrete patterns
1. **High-level guide with references** — SKILL.md has quick-start inline, deep content in `FORMS.md` / `REFERENCE.md` / `EXAMPLES.md`, linked directly.
2. **Domain-specific organization** — split reference material by domain (`reference/finance.md`, `reference/sales.md`, ...) so a query about one domain never loads the others' context.
3. **Conditional details** — basic path inline, edge cases link out ("For tracked changes: see REDLINING.md").

Rules that go with this:
- **One level deep only.** All reference files must link directly from SKILL.md; a reference file must not link to another reference file — Claude may only partially read (`head -100`) a nested file, silently truncating what it sees.
- **Table of contents required for any reference file over 100 lines**, so a partial read still reveals the full scope.
- **SKILL.md body: under 500 lines** for optimal performance — same number cited in the Claude Code docs.

### "Concise is key" — the framing device
> "The context window is a public good." Default assumption: Claude is already very smart. Only add context Claude doesn't already have. For every paragraph, ask: "Does Claude really need this explanation? Can I assume Claude knows this? Does this paragraph justify its token cost?"
- ~50-token example beats a ~150-token example that over-explains what a PDF or a pip install is.

### Degrees of freedom (novel, load-bearing concept)
Match instruction specificity to task fragility:
- **High freedom** (prose/heuristics) — when multiple approaches are valid, decisions are context-dependent.
- **Medium freedom** (pseudocode/parameterized scripts) — a preferred pattern exists but some variation is fine.
- **Low freedom** (exact script, no params, "do not modify the command") — operations are fragile/error-prone and consistency is critical (e.g. DB migrations).
Analogy given: narrow bridge with cliffs (low freedom, exact guardrails) vs. open field (high freedom, general direction).

### Workflows / checklists inside a skill
For complex multi-step tasks, give Claude a **literal markdown checklist to copy into its response and check off**, e.g.:
```
Task Progress:
- [ ] Step 1: Analyze the form (run analyze_form.py)
- [ ] Step 2: Create field mapping (edit fields.json)
...
```
Paired with a **feedback-loop pattern**: "run validator → fix errors → repeat," with an explicit "only proceed when validation passes" gate. This is called out as significantly improving output quality, both for code-backed skills (run a `validate.py`) and non-code skills (check against a style-guide checklist).

### Anti-patterns explicitly named
- **Windows-style paths** (`scripts\helper.py`) — always forward slashes, cross-platform.
- **Offering too many options** ("you can use pypdf, or pdfplumber, or PyMuPDF, or...") — give one recommended default with a narrow escape hatch for the genuine edge case (e.g. OCR fallback), not an open menu.
- **Time-sensitive instructions** ("before August 2025 use X, after use Y") — will silently go stale; put deprecated methods in a collapsed "Old patterns" `<details>` section instead.
- **Inconsistent terminology** — pick one term ("API endpoint", not a mix of "endpoint/URL/route/path") and use it everywhere; inconsistency measurably hurts Claude's parsing.
- **Deeply nested references** (see above).
- For scripts specifically: **"solve, don't defer"** — handle `FileNotFoundError`/`PermissionError` explicitly in the script rather than letting it crash for Claude to improvise around; and no **"voodoo constants"** — every timeout/retry-count value must be commented with *why* that number, not just what it is.
- **Assuming a package is installed** — always state the exact `pip install X` before showing usage.
- **MCP tool references must be fully qualified** as `ServerName:tool_name`, or Claude may fail to locate the tool when multiple MCP servers are present.

### Evaluation-driven development (the mandated order of operations)
> "Create evaluations BEFORE writing extensive documentation." The five-step loop: (1) run Claude on representative tasks *without* the skill, document actual failures; (2) build ≥3 scenarios that test those specific gaps; (3) establish a no-skill baseline; (4) write the *minimal* instructions that address exactly those gaps; (5) iterate by re-running evals against the baseline. This exists specifically to stop authors from "documenting imagined problems."

Also: **test across model tiers you'll actually use** — Haiku (does it have enough guidance?), Sonnet (clear/efficient?), Opus (does it over-explain, wasting Opus's own reasoning budget?).

### Skill-authoring checklist (from the doc, abridged)
Core quality: description is specific + states what/when; body <500 lines; extra detail is in separate files; no time-sensitive content; consistent terminology; concrete (not abstract) examples; references one level deep; workflows have clear steps.
Code/scripts: explicit error handling; no voodoo constants; required packages listed; no Windows paths; validation steps for critical operations.
Testing: ≥3 evaluations; tested across model tiers; tested on real usage; team feedback incorporated.

### `anthropics/skills` repo conventions
- Two-field-only frontmatter is the *baseline template* (`name`, `description`) — see `template/SKILL.md`, which is a deliberately bare skeleton (just the two frontmatter fields + an "Insert instructions below" heading) meant to be copied and filled in.
- Repo layout: `skills/` (worked examples by category: Creative & Design, Development & Technical, Enterprise & Communication, Document Skills), `spec/` (the Agent Skills open standard itself), `template/` (starting skeleton).
- Philosophy stated in the README: skills teach **specific, repeatable tasks**; keep each skill self-contained and focused; include concrete examples/guidelines rather than abstract descriptions.

---

## 4. `obra/superpowers` — `writing-skills` meta-skill

Sources: https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md, https://github.com/obra/superpowers/blob/main/skills/writing-skills/anthropic-best-practices.md

This is a third-party (not Anthropic-authored) plugin, but it explicitly summarizes and builds on the same official best-practices doc fetched in §3 above (its `anthropic-best-practices.md` is essentially a compressed mirror of that page, confirming the two sources agree). Its own contribution is a stricter, more opinionated **process** for producing a skill, framed as TDD applied to documentation:

### The Iron Law
> "NO SKILL WITHOUT A FAILING TEST FIRST" — applies to new skills and edits alike.

RED (write a pressure-scenario subagent test, watch the *undocumented* agent violate the rule, capture the failure verbatim) → GREEN (write the *minimal* skill content that fixes exactly that failure, re-test until it passes) → REFACTOR (probe for new rationalizations/loopholes under pressure, close them, re-test until "bulletproof").

### Description rule that's stricter than Anthropic's own doc
> **"Use when..."** format focused purely on triggering conditions. **"NEVER summarize the skill's process or workflow in the description"** — because agents may act on the description alone and skip reading the body. Cited failure case: a description that said "dispatches subagent per task with code review between tasks" caused the agent to perform a single review, ignoring the skill's actual flowchart mandating two reviews.

### Token budgets (more concrete than Anthropic's ">500 lines" cutoff)
- Getting-started / always-loaded skills: **under 150 words**.
- Frequently-loaded skills: **under 200 words**.
- Everything else: **under 500 words** (a tighter number than the docs' 500-*line* ceiling).
- Split into a separate reference file once the reference content alone would exceed **~100 lines**.

### Content anti-patterns beyond Anthropic's list
- **Narrative examples** — "here's how I solved it once for project X" is explicitly disallowed; a skill must be a reusable technique/pattern, not a war story.
- **Multi-language dilution** — one excellent example beats five mediocre ones across five languages.
- **Code inside flowcharts** — unreadable/uncopyable; keep flowcharts for decision points only, put code in normal fenced blocks.
- **Generic step labels** ("helper1", "step3") — labels must be semantic.

### Skill Discovery Optimization (SDO)
Describes the actual retrieval path a future Claude instance follows: hits a problem → searches skill descriptions → scans the SKILL.md overview → reads pattern/reference tables → only loads examples when actually implementing. Keyword strategy: include literal error messages, symptom words (flaky, hanging, zombie), tool names, and synonym variants (timeout/hang/freeze; cleanup/teardown/afterEach) so the description matches however a user or agent phrases the problem.

---

## Cross-source agreement (high confidence — appears identically in ≥2 independent sources)

- `description` field is the single most load-bearing artifact: third person, states what+when, keyword-rich, first sentence carries the trigger.
- Progressive disclosure: SKILL.md under ~500 lines; split to reference files; reference files one level deep from SKILL.md; ToC required past 100 lines.
- Scripts should be either clearly "run this" or clearly "read this as reference" — never ambiguous.
- Time-sensitive/date-conditional instructions and Windows-style paths are explicit anti-patterns.
- Evaluation-before-documentation (or test-before-skill, in superpowers' stricter TDD framing) is the recommended authoring order in every source that discusses process.

## Where the sources diverge

- Anthropic's official docs give a **line-count** ceiling (500 lines) for SKILL.md; `obra/superpowers` gives a **word-count** ceiling (150–500 words depending on load frequency) and additionally caps individual reference files at ~100 lines before they must exist as separate files. superpowers' numbers are the stricter of the two — treat them as the tighter target, Anthropic's as the hard outer bound.
- Anthropic's official guidance never says "don't summarize the workflow in the description"; superpowers adds this as a hard rule based on an observed failure mode (agents acting on the description alone). Worth adopting even though it's not in the primary-source docs.
- `context: fork` (running a skill as an isolated subagent) is a **Claude-Code-specific extension**, absent from the portable Agent Skills spec and not discussed by either the engineering blog or the best-practices doc, which are written for the model-agnostic skill format (claude.ai / API). It matters for a Claude Code project like ours but wouldn't survive an export to the API/claude.ai skill format.

---

## Implications for our `data-engineer` skill

1. **Name**: existing project convention (confirmed by reading `~/.claude/skills/dataset-copilot/SKILL.md`, `eval-copilot/SKILL.md`, `hardware-copilot/SKILL.md`) is noun-phrase + `-copilot`, not Anthropic's recommended gerund form. For consistency with the sibling skills already governing this same robotics stack, prefer `data-engineer` or `data-engineer-copilot` over a gerund like `engineering-data` — internal consistency with neighboring skills outweighs Anthropic's general naming guidance here, but the frontmatter char/charset limits (≤64 chars, lowercase+hyphens, no "claude"/"anthropic") still apply.

2. **Description**: model it on the existing `dataset-copilot` pattern (`"X copilot for Y. Use when [...]. Triggers on words like [...]."`), which already satisfies third-person + what-it-does + trigger-list. Front-load the single most common trigger phrase first — both the 1,536-char truncation and the skill-listing budget cut from the end, and least-invoked skills lose their description text first under budget pressure. Do not summarize the internal workflow/checklist in the description (superpowers' rule) — only the trigger conditions belong there.

3. **Structure**: sibling skills keep supporting files flat in the skill directory (`cli_reference.md`, `conversion_knowledge.md` next to `SKILL.md`, not nested under `references/`) — match that, and keep SKILL.md itself as thin session-start + pointers, consistent with `dataset-copilot`'s ~84-line SKILL.md.

4. **Workflow**: if `data-engineer` needs a multi-step pipeline (ingest → validate → convert → verify), use the literal copyable checklist pattern with an explicit "only proceed when validation passes" gate — this is the one workflow pattern independently endorsed by every source.

5. **Subagent use**: only reach for `context: fork` if the skill has one explicit, self-contained task to hand off (e.g., a long-running dataset conversion or a bulk validation pass) — not for skills whose job is to hold reference conventions/knowledge in the main conversation. Given this skill will likely mix "hold context about our dataset conventions" (inline) and "run a conversion/validation job" (forkable), consider splitting those into two skills or gating the fork behind a specific sub-workflow rather than forking the whole skill.

6. **Before writing prose**: build 3+ before/after evaluation scenarios first (representative data-engineering asks run with the skill absent, capture actual failure) — per both Anthropic's and superpowers' explicit ordering, skill content should be written to close observed gaps, not to pre-document imagined ones.

Source list:
- https://code.claude.com/docs/en/skills
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://github.com/anthropics/skills
- https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md
- https://github.com/obra/superpowers/blob/main/skills/writing-skills/anthropic-best-practices.md
