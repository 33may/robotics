/**
 * The agentic loop, as a readable stream of what actually happened.
 *
 * APP-OWNED PANEL. "plan / inspect / move / answer" and the fact/plan/finding
 * trust tiers are this project's vocabulary, not a toolkit concept.
 *
 * Design rules, each one taken from how shipping agent UIs actually work
 * (surveyed 2026-08-25 across Claude Code, ChatGPT/ChatKit, Gemini + Google
 * ADK, and the trace viewers: Langfuse, Phoenix, Weave, Braintrust):
 *
 * 1. **The model's prose is the baseline voice — unlabelled.** Only machinery
 *    gets marked. A row that says "TEXT" beside a sentence is noise.
 * 2. **A tool call and its result are ONE block.** Rendering them as two rows
 *    is the duplication LangSmith shipped a dedup fix for, and "shown as a
 *    widget *and* a raw blob" is a named anti-pattern. One card, one concept.
 * 3. **Tools are indented one level (24px) and carry their own name.** Indent
 *    is the median across six trace viewers (20-24px); the tool's identity
 *    belongs inside the block, not in a gutter column beside it.
 * 4. **Thinking is not rendered at all.** `display: "omitted"` is the DEFAULT
 *    on Opus 5 — the content never arrives, only a token estimate. A beat with
 *    nothing behind it is worse than silence.
 * 5. **Images cap at 200px with click-to-expand.** Phoenix, Helicone and Weave
 *    converged on this independently.
 * 6. **Mono for payloads, sans for prose; two weights; 11-14px.** Density is
 *    the whole point of a trace.
 * 7. **Sub-agents are cards you open, not subtrees you indent into.** Ten
 *    sub-agents make ten cards; nesting runs out of horizontal room.
 *
 * The payload is the WHOLE trace, retained (`ui/publisher.py:publish_trace`),
 * so a panel opened halfway through a run — or three days later — shows
 * everything. Images are URLs over the `/captures` mount, never bus payloads.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import { useBus, useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';

interface SubTurn {
  readonly tool?: string;
  readonly args?: Readonly<Record<string, unknown>>;
  readonly result?: string;
  readonly image?: string | null;
  readonly evidence?: readonly unknown[];
  readonly reasoning?: string;
  /** NOT always a string. When the orchestrator supplies an `answer_schema`
   *  the vision model answers in that shape, so this can be an object —
   *  `{"fully_in_frame": "yes", "text_verbatim": "VBTI"}` is a real one that
   *  killed the app when rendered as a React child. Render via <Value>. */
  readonly answer?: unknown;
  /** What this viewpoint offered — `{saw, recommendation}`, camera-centric.
   *  Always an object, so always render via <Value>. */
  readonly view?: unknown;
  /** Pre-2026-08-26 runs carried `{target, facing, better}` under this key. */
  readonly framing?: unknown;
  readonly raw?: string;
}

interface SubTranscript {
  readonly cell?: readonly number[] | null;
  readonly task?: string;
  readonly answer_schema?: unknown;
  readonly image?: string | null;
  readonly view_text?: string;
  readonly prompt?: string;
  readonly turns?: readonly SubTurn[];
  readonly answer?: string;
  /** The final turn's `view`, copied to the transcript top by the subagent
   *  loop ("disk and Finding always agree", eyes/agents/vlm_agent.py). The
   *  render falls back to this when the reply turn's own key is missing. */
  readonly view?: unknown;
}

/** One resolved evidence hunt (brain/loop.py:_collect_evidence). Image paths
 *  arrive as /captures URLs — publisher.publish_trace maps them — and either
 *  image can be null. found:false is a real record, the "we looked here"
 *  receipt, not an error. Old runs carry none of this and must still render. */
interface EvidenceRec {
  readonly cell?: readonly number[] | string;
  readonly find?: string;
  readonly found?: boolean;
  readonly report?: string;
  readonly crop?: string | null;
  readonly frame?: string | null;
  readonly transcript?: string;
}

export interface TraceEvent {
  readonly seq: number;
  readonly t: number;
  readonly kind: string;
  readonly text?: string;
  readonly tool?: string;
  readonly args?: Readonly<Record<string, unknown>>;
  readonly images?: readonly string[];
  readonly sub?: SubTranscript;
  readonly tier?: string;
  readonly what?: string;
  readonly question?: string;
  readonly model?: string;
  readonly captured?: number;
  readonly tokens?: number;
  readonly verdict?: string;
  readonly reasoning?: string;
  readonly evidence?: string;
  /** A cell pair, or the string "survey" — the survey pose is off-grid. */
  readonly cell?: readonly number[] | string;
  readonly failed?: boolean;
  /** requested | awaiting_approval | approved | redirected | cancelled | captured */
  readonly state?: string;
  readonly to?: readonly number[] | string;
  /** sub_step: which turn of the subagent's own loop this is. */
  readonly n?: number;
  readonly of?: number;
  readonly event?: string;
  readonly task?: string;
  readonly image?: string;
  readonly result?: string;
  readonly answer?: string;
  /** sub_step inside answer(): which citation is being hunted right now. */
  readonly hunt?: string;
  /** `evidence` events carry one EvidenceRec each, flattened. */
  readonly find?: string;
  readonly found?: boolean;
  readonly report?: string;
  readonly crop?: string | null;
  readonly frame?: string | null;
  /** The final `answer` event: one record per cited hunt, misses included. */
  readonly evidence_images?: readonly EvidenceRec[];
  /** The final `answer` event: the card's structured readout — codes,
   *  serials, model numbers — as the model established them from pixels. */
  readonly fields?: Readonly<Record<string, unknown>>;
}

interface TracePayload {
  readonly events?: readonly TraceEvent[];
}

export interface TracePanelConfig extends Record<string, unknown> {
  topic: string;
  paneWidth: string;
  /** Pre-filled so iterating costs one keypress. In production there is no
   *  question box at all — the question comes from the inspection job. */
  defaultQuestion: string;
  /** State rows are context the harness injected, not anything the model
   *  said. Useful when debugging the harness, noise when reading behaviour. */
  showState: boolean;
}

export const tracePanelDefaultConfig: TracePanelConfig = {
  topic: 'trace/state',
  paneWidth: '460px',
  showState: true,
  defaultQuestion: 'is there a logo on the cup?',
};

const str = (v: unknown): string =>
  typeof v === 'string' ? v : v === undefined || v === null ? '' : JSON.stringify(v);

/** A target is a cell pair OR the string "survey" — the survey pose is not on
 *  the viewsphere, so anything that renders a target must accept both. */
const cellLabel = (c: readonly number[] | string | undefined): string =>
  c === undefined ? '' : typeof c === 'string' ? c : `[${c.join(', ')}]`;

/** The whole run starts here: type a question, press ask. Nothing in the
 *  loop happens before this, which is why an empty trace shows the composer
 *  instead of an empty-state message. */
function Ask({ initial }: { initial: string }) {
  const bus = useBus();
  const [q, setQ] = useState(initial);
  const [sent, setSent] = useState(false);
  const ask = () => {
    const question = q.trim();
    if (!question || sent) return;
    bus.send({ cmd: 'brain/ask', question });
    setSent(true);
  };
  return (
    <div className="trace-ask">
      <div className="trace-ask-label">what should the robot find out?</div>
      <div className="trace-ask-row">
        <input
          className="trace-ask-input"
          data-testid="ask-input"
          value={q}
          autoFocus
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') ask();
          }}
        />
        <button
          className="trace-ask-go"
          data-testid="ask-send"
          type="button"
          onClick={ask}
          disabled={sent}
        >
          {sent ? 'asked' : 'ask'}
        </button>
      </div>
      <div className="trace-ask-hint">
        the arm moves only when you approve each viewpoint
      </div>
    </div>
  );
}

/* ── icons ────────────────────────────────────────────────────────────────
   Inline SVG rather than a dependency: four glyphs do not justify a package,
   and `currentColor` lets one CSS rule tint icon and label together. */

const ICON: Record<string, ReactElement> = {
  plan: (
    <svg viewBox="0 0 16 16" aria-hidden>
      <path d="M3 4h10M3 8h10M3 12h6" />
    </svg>
  ),
  inspect: (
    <svg viewBox="0 0 16 16" aria-hidden>
      <circle cx="7" cy="7" r="4.2" />
      <path d="M10.2 10.2 14 14" />
    </svg>
  ),
  move: (
    <svg viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="8" r="5.2" />
      <path d="M8 1.2v2.2M8 12.6v2.2M1.2 8h2.2M12.6 8h2.2" />
    </svg>
  ),
  answer: (
    <svg viewBox="0 0 16 16" aria-hidden>
      <path d="M3 8.4 6.4 12 13 4.6" />
    </svg>
  ),
};

/* ── block model ──────────────────────────────────────────────────────────
   The trace is a flat event log; the UI is not. A tool call, the ledger write
   it caused and the result it returned are one thing that happened, so they
   are folded into one block before anything renders. */

interface Block {
  key: number;
  kind: 'run' | 'prose' | 'tool' | 'state' | 'answer' | 'approval' | 'survey';
  ev?: TraceEvent;
  /** The `tool_start` stream event: the model has BEGUN writing this call —
   *  its name is known, its args are still generating. The shell this
   *  renders is replaced by the real call block when `tool_call` lands. */
  start?: TraceEvent;
  call?: TraceEvent;
  result?: TraceEvent;
  write?: TraceEvent;
  /** Live progress from the vision subagent, while it is still running. */
  steps?: TraceEvent[];
  /** Hunts resolved inside an answer() call, in the order they streamed. */
  evidence?: TraceEvent[];
  /** The `answer` event, absorbed into its call: the only carrier of the
   *  full evidence_images record (cached hunts emit no fresh event). */
  answerEv?: TraceEvent;
}

function toBlocks(events: readonly TraceEvent[]): Block[] {
  const out: Block[] = [];
  const used = new Set<number>();

  events.forEach((ev, i) => {
    if (used.has(ev.seq)) return;

    switch (ev.kind) {
      case 'thinking':
        return; // rule 4: nothing behind it, so nothing to draw
      case 'run':
        out.push({ key: ev.seq, kind: 'run', ev });
        return;
      case 'text':
        out.push({ key: ev.seq, kind: 'prose', ev });
        return;
      case 'state':
        out.push({ key: ev.seq, kind: 'state', ev });
        return;
      case 'approval':
        out.push({ key: ev.seq, kind: 'approval', ev });
        return;
      case 'sub_step': {
        // Inside a tool call these are absorbed by the scan below. The
        // SURVEY declaration agent streams its steps with cell:null BEFORE
        // any tool_call exists, so with no owner they were dropped here and
        // the survey run was invisible — the one sub-run with no verb gets
        // a block of its own instead.
        const prev = out[out.length - 1];
        if (prev?.kind === 'survey') prev.steps?.push(ev);
        else out.push({ key: ev.seq, kind: 'survey', steps: [ev] });
        return;
      }
      case 'answer':
        // Owned by the answer TOOL CALL — absorbed below, evidence_images
        // and all. Reached only with no call to own it; rendering it here
        // too is the duplication LangSmith had to ship a dedup fix for.
        return;
      case 'evidence':
        // Emitted only inside an answer() call and absorbed there; an
        // orphan's record rides the `answer` event too, so nothing is lost.
        return;
      case 'tool_start':
        // The model began writing a tool call: show its shell NOW. The args
        // (an answer's whole verdict and fields) stream for seconds after,
        // and until this event existed the answer block waited for them.
        out.push({ key: ev.seq, kind: 'tool', start: ev });
        return;
      case 'tool_call': {
        // The call this start announced has arrived — the shell served its
        // purpose and must not linger as a duplicate row.
        for (let k = out.length - 1; k >= 0; k -= 1) {
          const b = out[k];
          if (b?.start && !b.call && b.start.tool === ev.tool) {
            out.splice(k, 1);
            break;
          }
        }
        const block: Block = { key: ev.seq, kind: 'tool', call: ev };
        // Scan forward for this call's result, absorbing the ledger write it
        // triggered. Stop at the next call — a missing result (crash, budget
        // exhaustion) must not steal the following call's.
        for (let j = i + 1; j < events.length; j += 1) {
          const next = events[j];
          if (!next || next.kind === 'tool_call') break;
          if (next.kind === 'write') {
            block.write = next;
            used.add(next.seq);
          } else if (next.kind === 'sub_step') {
            (block.steps ??= []).push(next);
            used.add(next.seq);
          } else if (next.kind === 'evidence') {
            (block.evidence ??= []).push(next);
            used.add(next.seq);
          } else if (next.kind === 'answer') {
            block.answerEv = next;
            used.add(next.seq);
          } else if (next.kind === 'tool_result') {
            block.result = next;
            used.add(next.seq);
            break;
          }
        }
        out.push(block);
        return;
      }
      case 'write':
        // Only reached when a write had no owning call; keep it rather than
        // silently dropping a ledger entry.
        out.push({ key: ev.seq, kind: 'state', ev });
        return;
      default:
        out.push({ key: ev.seq, kind: 'state', ev });
    }
  });
  return out;
}

/** The subagent's final, structured — never re-parsed out of its prose. */
function finalTurn(result?: TraceEvent): SubTurn | undefined {
  return result?.sub?.turns?.find((t) => t.evidence || t.reasoning || t.answer);
}

/** The primary argument, the one worth showing on the header line. */
function leadArg(call: TraceEvent): string {
  const a = call.args ?? {};
  if (call.tool === 'move') return `cell ${str(a.cell)}`;
  if (call.tool === 'inspect') return str(a.question);
  if (call.tool === 'plan') return '';
  if (call.tool === 'answer') return str(a.verdict);
  return '';
}

/** Arguments as a human reads them.
 *
 *  `JSON.stringify` turned `plan`'s criteria into one line of escaped `\n`
 *  — measured in the pane, not assumed. A long string is prose and gets a
 *  section; a short one is a value and gets a key/value row. JSON is the
 *  fallback for genuinely structured data, never the default. */
function Args({ args }: { args: Readonly<Record<string, unknown>> }) {
  const entries = Object.entries(args).filter(([, v]) => v !== undefined && v !== null && v !== '');
  if (!entries.length) return null;
  return (
    <>
      {entries.map(([k, v]) => {
        const isProse = typeof v === 'string' && (v.length > 64 || v.includes('\n'));
        if (isProse) {
          return (
            <div key={k}>
              <div className="trace-section">{k}</div>
              {k === 'criteria' ? (
                <Criteria text={v as string} />
              ) : (
                <div className="trace-body-text">{v as string}</div>
              )}
            </div>
          );
        }
        return (
          <div className="trace-kv" key={k}>
            <span className="trace-kv-key">{k}</span>
            <code className="trace-kv-val">{str(v)}</code>
          </div>
        );
      })}
    </>
  );
}

/** Renders anything the models can put in a field.
 *
 *  A structured answer is not a failure to be stringified — it is the schema
 *  working. Objects render as key/value rows, which reads better than the
 *  string ever did; everything else renders as text. What must never happen
 *  is handing React a raw object, which is what took the app down mid-run. */
function Value({ v }: { v: unknown }) {
  if (v === null || v === undefined || v === '') return null;
  if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
    return <span>{String(v)}</span>;
  }
  if (Array.isArray(v)) {
    return (
      <ul className="trace-evidence">
        {v.map((item, i) => (
          <li key={i}>
            <Value v={item} />
          </li>
        ))}
      </ul>
    );
  }
  return (
    <>
      {Object.entries(v as Record<string, unknown>).map(([k, val]) => (
        <div className="trace-kv" key={k}>
          <span className="trace-kv-key">{k}</span>
          <span className="trace-kv-val trace-kv-plain">
            <Value v={val} />
          </span>
        </div>
      ))}
    </>
  );
}

function Media({
  images,
  onOpen,
  size = 'strip',
}: {
  images: readonly string[];
  onOpen: (src: string) => void;
  size?: 'strip' | 'full';
}) {
  if (!images.length) return null;
  return (
    <div className={`trace-media trace-media-${size}`}>
      {images.map((src) => (
        <button
          key={src}
          type="button"
          className="trace-media-item"
          onClick={(e) => {
            e.stopPropagation();
            onOpen(src);
          }}
        >
          <img src={src} alt="" loading="lazy" />
        </button>
      ))}
    </div>
  );
}

/** Long payloads collapse to a fixed height with an explicit affordance —
 *  never a silent clip, which reads as "that was all of it". */
function Clamp({ text, lines = 6 }: { text: string; lines?: number }) {
  const [open, setOpen] = useState(false);
  const long = text.split('\n').length > lines || text.length > 420;
  if (!long) return <div className="trace-body-text">{text}</div>;
  return (
    <>
      <div className={open ? 'trace-body-text' : 'trace-body-text trace-clamped'}>{text}</div>
      <button
        type="button"
        className="trace-more"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
      >
        {open ? 'show less' : 'show more'}
      </button>
    </>
  );
}

/** The plan is a heading plus numbered criteria — render that shape.
 *
 *  It arrived as one grey slab of pre-wrapped text, which is exactly as hard
 *  to scan as the JSON it replaced. Numbered lines become a list, `Label:`
 *  lines become headings, and everything else stays prose. */
function Criteria({ text }: { text: string }) {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  return (
    <div className="trace-criteria">
      {lines.map((line, i) => {
        const num = /^(\d+)[.)]\s*(.*)$/.exec(line);
        if (num) {
          return (
            <div className="trace-criterion" key={i}>
              <span className="trace-criterion-n">{num[1]}</span>
              <span>{num[2]}</span>
            </div>
          );
        }
        const head = /^([A-Za-z][\w ]{0,24}):\s*(.*)$/.exec(line);
        if (head) {
          return (
            <div className="trace-criteria-head" key={i}>
              <b>{head[1]}</b>
              {head[2] ? <span> {head[2]}</span> : null}
            </div>
          );
        }
        return (
          <div className="trace-criteria-line" key={i}>
            {line}
          </div>
        );
      })}
    </div>
  );
}

function ToolBlock({
  block,
  selected,
  onSelect,
  onOpenImage,
}: {
  block: Block;
  selected: boolean;
  onSelect: () => void;
  onOpenImage: (src: string) => void;
}) {
  const call = block.call;
  if (!call) return null;
  const tool = call.tool ?? 'tool';
  const schema = str((call.args ?? {}).answer_schema);
  const lead = leadArg(call);
  const result = block.result;
  // A call with no result yet IS the in-flight state — no extra event needed.
  const steps = block.steps ?? [];
  const running = !result;
  const last = steps[steps.length - 1];
  // A subagent that ran out of turns returns `answer: unknown` — a real
  // failure the orchestrator swallowed, so it must not read as a normal
  // result. That silent case is the whole reason this panel exists.
  const failed =
    result?.failed ||
    /^(cannot|no capture|call plan)/i.test(result?.text ?? '') ||
    /answer: unknown/.test(result?.text ?? '');
  const turns = result?.sub?.turns?.length ?? 0;
  const final = finalTurn(result);
  // A finding is fields, not paragraphs. The stream shows the verdict and its
  // first piece of evidence; the rest is one click away. Dumping the whole
  // evidence/reasoning/answer blob here is what made this a wall of text.
  const body = final ? '' : (result?.text ?? '').replace(/^view \[[^\]]*\] —\n?/, '');

  return (
    <div
      className="trace-tool"
      data-tool={tool}
      data-selected={selected ? 'yes' : undefined}
      data-failed={failed ? 'yes' : undefined}
      data-running={running ? 'yes' : undefined}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onSelect();
      }}
    >
      <div className="trace-tool-head">
        <span className="trace-tool-icon">{ICON[tool] ?? ICON.answer}</span>
        <span className="trace-tool-name">{tool}</span>
        {lead ? <span className="trace-tool-lead">{lead}</span> : null}
        {block.write ? <span className="trace-tier">{block.write.tier}</span> : null}
      </div>

      {/* The schema is what makes `inspect` a measurement rather than a chat
          turn, so it gets its own labelled row instead of hiding in JSON. */}
      {schema ? (
        <div className="trace-kv">
          <span className="trace-kv-key">schema</span>
          <code className="trace-kv-val">{schema}</code>
        </div>
      ) : null}

      {call.tool === 'plan' ? <Criteria text={str((call.args ?? {}).criteria)} /> : null}

      {result ? (
        <div className="trace-tool-result">
          {final ? (
            <>
              <div className="trace-finding">
                <span className="trace-finding-answer"><Value v={final.answer} /></span>
                {final.evidence?.[0] ? (
                  <span className="trace-finding-ev"><Value v={final.evidence[0]} /></span>
                ) : null}
              </div>
            </>
          ) : null}
          {body ? <Clamp text={body} /> : null}
          <Media images={result.images ?? []} onOpen={onOpenImage} />
          {turns ? <div className="trace-tool-foot">vision subagent · {turns} steps</div> : null}
        </div>
      ) : (
        <div className="trace-running-wrap">
        {steps.filter((st) => st.image).slice(-3).length ? (
          <Media
            images={steps.filter((st) => st.image).slice(-3).map((st) => st.image as string)}
            onOpen={onOpenImage}
          />
        ) : null}
        <div className="trace-running">
          <span className="trace-pulse" />
          <span>
            {last?.event === 'tool'
              ? `looking · step ${last.n}/${last.of} · ${last.tool}`
              : last
                ? `looking · step ${last.n}/${last.of}`
                : 'running'}
          </span>
        </div>
        </div>
      )}
    </div>
  );
}

/** One evidence hunt, as a receipt: the picture, where it was taken, and the
 *  hunter's verbatim report. The crop is what the hunter chose to look at; a
 *  miss (found:false) or a record with no crop shows the full frame instead —
 *  proof of where we looked, which is the point of a NO. */
function EvidenceRow({
  rec,
  onOpenImage,
}: {
  rec: EvidenceRec;
  onOpenImage: (src: string) => void;
}) {
  const img = rec.found !== false && rec.crop ? rec.crop : rec.frame ?? rec.crop ?? null;
  return (
    <div className="trace-evrec" data-found={rec.found === false ? 'no' : 'yes'}>
      {img ? <Media images={[img]} onOpen={onOpenImage} /> : null}
      <div className="trace-evrec-meta">
        <span className="trace-evrec-cell">
          {cellLabel(rec.cell)}
          {rec.found === false ? ' · not found' : ''}
        </span>
        {rec.report ? <span className="trace-evrec-report">{str(rec.report)}</span> : null}
      </div>
    </div>
  );
}

/** The survey declaration agent, rendered like the inspect sub-runs even
 *  though no verb owns it: same card, same in-flight strip, and the `done`
 *  step's answer IS the declaration the whole plan gets built from. */
function SurveyBlock({
  block,
  selected,
  onSelect,
  onOpenImage,
}: {
  block: Block;
  selected: boolean;
  onSelect: () => void;
  onOpenImage: (src: string) => void;
}) {
  const steps = block.steps ?? [];
  const done = steps.find((st) => st.event === 'done');
  const last = steps[steps.length - 1];
  const images = steps.filter((st) => st.image).map((st) => st.image as string);
  return (
    <div
      className="trace-tool"
      data-tool="survey"
      data-selected={selected ? 'yes' : undefined}
      data-running={done ? undefined : 'yes'}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onSelect();
      }}
    >
      <div className="trace-tool-head">
        <span className="trace-tool-icon">{ICON.inspect}</span>
        <span className="trace-tool-name">survey</span>
      </div>
      {done ? (
        <div className="trace-tool-result">
          <Clamp text={str(done.answer)} />
          <Media images={images} onOpen={onOpenImage} />
          <div className="trace-tool-foot">vision subagent · {done.n ?? steps.length} steps</div>
        </div>
      ) : (
        <div className="trace-running-wrap">
          {images.length ? <Media images={images.slice(-3)} onOpen={onOpenImage} /> : null}
          <div className="trace-running">
            <span className="trace-pulse" />
            <span>{last ? `looking · step ${last.n}/${last.of}` : 'running'}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/** A tool call the model has STARTED writing (tool_start) whose args are
 *  still streaming. The answer's shell is the card being born — the label
 *  and a pulse the instant the tool is invoked (Anton 2026-09-21); other
 *  tools get the same immediate acknowledgement in their own dress. */
function StartBlock({ ev }: { ev: TraceEvent }) {
  const tool = ev.tool ?? 'tool';
  if (tool === 'answer') {
    return (
      <div className="trace-answer" data-running="yes">
        <div className="trace-answer-label">answer</div>
        <div className="trace-running">
          <span className="trace-pulse" />
          <span>writing the answer…</span>
        </div>
      </div>
    );
  }
  return (
    <div className="trace-tool" data-tool={tool} data-running="yes">
      <div className="trace-tool-head">
        <span className="trace-tool-icon">{ICON[tool] ?? ICON.answer}</span>
        <span className="trace-tool-name">{tool}</span>
      </div>
      <div className="trace-running">
        <span className="trace-pulse" />
        <span>preparing…</span>
      </div>
    </div>
  );
}

/** One cited hunt as the answer tool call named it. */
interface CitedHunt {
  readonly cell?: readonly number[] | string;
  readonly find?: string;
}

const huntKey = (cell: unknown, find: unknown): string =>
  `${cellLabel(cell as readonly number[] | string | undefined)}|${str(find).toLowerCase()}`;

/** The answer IN FLIGHT, as stages (Anton 2026-09-21): one row per cited
 *  hunt, checked off as its evidence event lands, the active one carrying
 *  the hunter's live step counter, then a closing "compose card" stage.
 *  Deliberately NO verdict text and NO images here — the main stream shows
 *  only that the answer is happening; the details are one click away in the
 *  right pane, and the card renders once, complete, when the call ends. */
function AnswerStages({ block }: { block: Block }) {
  const cited: readonly CitedHunt[] =
    (block.call?.args?.evidence_cells as readonly CitedHunt[] | undefined) ?? [];
  const landed = block.evidence ?? [];
  // A hunt cached from a bounced attempt emits no fresh `evidence` event —
  // its row simply stays pending until the result swaps this for the card.
  const done = new Set(landed.map((e) => huntKey(e.cell, e.find)));
  const steps = block.steps ?? [];
  const last = steps[steps.length - 1];
  const active = last ? huntKey(last.cell, last.hunt) : null;
  const rows = cited.length ? cited : landed.map((e) => ({ cell: e.cell, find: e.find }));
  const allDone = rows.length > 0 && rows.every((h) => done.has(huntKey(h.cell, h.find)));
  return (
    <div className="trace-stages">
      {rows.map((h, i) => {
        const key = huntKey(h.cell, h.find);
        const state = done.has(key) ? 'done' : key === active ? 'active' : 'pending';
        return (
          <div className="trace-stage" data-state={state} key={i}>
            <span className="trace-stage-mark">
              {state === 'done' ? '✓' : state === 'active' ? <span className="trace-pulse" /> : '·'}
            </span>
            <span className="trace-stage-cell">{cellLabel(h.cell)}</span>
            <span className="trace-stage-find">{str(h.find)}</span>
            {state === 'active' && last?.of ? (
              <span className="trace-stage-n">{last.n}/{last.of}</span>
            ) : null}
          </div>
        );
      })}
      <div className="trace-stage" data-state={allDone ? 'active' : 'pending'}>
        <span className="trace-stage-mark">
          {allDone ? <span className="trace-pulse" /> : '·'}
        </span>
        <span className="trace-stage-find">compose card</span>
      </div>
    </div>
  );
}

/** The finished answer, rendered ONCE and whole: the verdict, the structured
 *  fields (codes, serials, counts — the card's reason to exist), and the
 *  evidence pictures. This is the terminal artifact of a run, so it renders
 *  from the authoritative `answer` event, falling back to the call args and
 *  the result's image strip for runs recorded before fields existed. */
function AnswerCard({
  block,
  onOpenImage,
}: {
  block: Block;
  onOpenImage: (src: string) => void;
}) {
  const a = block.call?.args ?? {};
  const ansEv = block.answerEv;
  const fields =
    ansEv?.fields ?? (a.fields as Readonly<Record<string, unknown>> | undefined) ?? {};
  const recs: readonly EvidenceRec[] = ansEv?.evidence_images ?? block.evidence ?? [];
  const images = recs
    .map((r) => (r.found !== false && r.crop ? r.crop : r.frame ?? r.crop))
    .filter((s): s is string => Boolean(s));
  const fallback = !images.length ? block.result?.images ?? [] : [];
  return (
    <div className="trace-card">
      <div className="trace-answer-label">answer</div>
      <div className="trace-answer-verdict">{str(ansEv?.verdict ?? a.verdict)}</div>
      {Object.keys(fields).length ? (
        <div className="trace-card-fields">
          {Object.entries(fields).map(([k, v]) => (
            <div className="trace-card-field" key={k}>
              <span className="trace-card-field-k">{k}</span>
              <span className="trace-card-field-v">
                <Value v={v} />
              </span>
            </div>
          ))}
        </div>
      ) : null}
      <Media images={images.length ? images : fallback} onOpen={onOpenImage} />
    </div>
  );
}

/** answer() is not instant: the cited evidence hunts run INSIDE the call and
 *  take tens of seconds each. Until the tool_result lands this block reads
 *  as stages in progress — no verdict text, no images (Anton 2026-09-21:
 *  the text appearing before the evidence made the answer look done while
 *  it was still being verified). The finished card renders only when the
 *  call completes; a bounced answer (failed=true) is an attempt on record,
 *  never the verdict. */
function AnswerBlock({
  block,
  selected,
  onSelect,
  onOpenImage,
}: {
  block: Block;
  selected: boolean;
  onSelect: () => void;
  onOpenImage: (src: string) => void;
}) {
  const result = block.result;
  const failed = result?.failed === true;
  return (
    <div
      className="trace-answer"
      data-failed={failed ? 'yes' : undefined}
      data-running={result ? undefined : 'yes'}
      data-selected={selected ? 'yes' : undefined}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onSelect();
      }}
    >
      {!result ? (
        <>
          <div className="trace-answer-label">answer</div>
          <AnswerStages block={block} />
        </>
      ) : failed ? (
        <>
          <div className="trace-answer-label">answer · bounced</div>
          {result.text ? <Clamp text={result.text} /> : null}
        </>
      ) : (
        <AnswerCard block={block} onOpenImage={onOpenImage} />
      )}
    </div>
  );
}

function SubAgent({
  sub,
  onOpenImage,
}: {
  sub: SubTranscript;
  onOpenImage: (src: string) => void;
}) {
  const turns = sub.turns ?? [];
  // The model's replies and their results alternate in the transcript. Pair
  // them so one step is one row rather than two, which is what turned this
  // pane into a wall of text.
  const steps: { call?: SubTurn; out?: SubTurn }[] = [];
  turns.forEach((turn) => {
    if (turn.tool) steps.push({ call: turn });
    else if (steps.length && !steps[steps.length - 1]?.out) {
      const last = steps[steps.length - 1];
      if (last) last.out = turn;
    } else steps.push({ out: turn });
  });
  const final = turns.find((t) => t.evidence || t.reasoning || t.answer);
  // Three places the viewpoint appendix can live, oldest run last: the final
  // reply turn's `view`; that turn's `framing` (the pre-2026-08-26 key); the
  // transcript top-level copy the loop writes, which survives even when the
  // reply turn's shape drifted (or, exhausted runs, when no final turn
  // exists at all). Reading only one of these is how it went unrendered.
  const view = final?.view ?? final?.framing ?? sub.view;

  return (
    <>
      <div className="trace-sub-question">{sub.task}</div>
      {sub.answer_schema ? (
        <div className="trace-kv">
          <span className="trace-kv-key">schema</span>
          <code className="trace-kv-val"><Value v={sub.answer_schema} /></code>
        </div>
      ) : null}

      {sub.image ? (
        <Media images={[sub.image]} onOpen={onOpenImage} size="full" />
      ) : null}
      {sub.view_text ? <div className="trace-caption">{sub.view_text}</div> : null}

      <div className="trace-section">steps</div>
      <ol className="trace-steps">
        {steps.map((step, i) => (
          <li className="trace-step" key={i}>
            <span className="trace-step-n">{i + 1}</span>
            <div className="trace-step-body">
              {step.call?.tool ? (
                <div className="trace-step-head">
                  <b>{step.call.tool}</b>
                  {step.call.args ? (
                    <code className="trace-step-args">{str(step.call.args)}</code>
                  ) : null}
                </div>
              ) : null}
              {step.out?.result ? (
                <div className="trace-step-out">{step.out.result}</div>
              ) : null}
              {step.out?.image ? (
                <Media images={[step.out.image]} onOpen={onOpenImage} />
              ) : null}
              {step.out?.raw ? <div className="trace-step-out">{step.out.raw}</div> : null}
            </div>
          </li>
        ))}
      </ol>

      {final ? (
        <>
          {final.evidence?.length ? (
            <>
              <div className="trace-section">evidence</div>
              <Value v={final.evidence} />
            </>
          ) : null}
          {final.reasoning ? (
            <>
              <div className="trace-section">reasoning</div>
              <div className="trace-body-text">{final.reasoning}</div>
            </>
          ) : null}
          {final.answer ? (
            <>
              <div className="trace-section">answer</div>
              <div className="trace-sub-answer">
                <Value v={final.answer} />
              </div>
            </>
          ) : null}
        </>
      ) : null}
      {view ? (
        <>
          {/* The only part of a finding that is about the VIEW rather than
              the object — and the part the orchestrator steers on, so it is
              worth being able to check what was actually said. Outside the
              `final` guard: the top-level copy renders even when no reply
              turn qualified as a final. */}
          <div className="trace-section">view</div>
          <Value v={view} />
        </>
      ) : null}
    </>
  );
}

function Detail({
  block,
  onOpenImage,
}: {
  block: Block | null;
  onOpenImage: (src: string) => void;
}) {
  if (!block) return <div className="trace-detail-empty">select a step</div>;
  const { call, result, ev } = block;

  return (
    <div className="trace-detail">
      <div className="trace-detail-head">
        {call ? (
          <>
            <span className="trace-tool-icon">{ICON[call.tool ?? ''] ?? ICON.answer}</span>
            <span className="trace-tool-name">{call.tool}</span>
          </>
        ) : (
          <span className="trace-tool-name">{block.kind}</span>
        )}
        {result?.cell ? (
          <span className="trace-tool-lead">cell {cellLabel(result.cell)}</span>
        ) : null}
      </div>

      {call?.args ? <Args args={call.args} /> : null}

      {ev?.text && !call ? (
        <>
          <div className="trace-section">{ev.kind}</div>
          <div className="trace-body-text">{ev.text}</div>
        </>
      ) : null}

      {ev?.kind === 'answer' ? (
        <>
          <div className="trace-section">reasoning</div>
          <div className="trace-body-text">{ev.reasoning}</div>
          <div className="trace-section">evidence</div>
          <div className="trace-body-text">{ev.evidence}</div>
        </>
      ) : null}

      {!result && block.steps?.length ? (
        <>
          {/* A survey block never gets a result; its steps ARE the record,
              not an in-flight glimpse, so the label must not say otherwise. */}
          <div className="trace-section">
            {block.kind === 'survey' ? 'steps' : 'looking now'}
          </div>
          <ol className="trace-steps">
            {block.steps.map((st, i) => (
              <li className="trace-step" key={i}>
                <span className="trace-step-n">{st.n}</span>
                <div className="trace-step-body">
                  <div className="trace-step-head">
                    <b>{st.event === 'tool' ? st.tool : st.event}</b>
                    {st.args ? <code className="trace-step-args">{str(st.args)}</code> : null}
                    {st.hunt ? <code className="trace-step-args">{st.hunt}</code> : null}
                  </div>
                  {st.result ? <div className="trace-step-out">{st.result}</div> : null}
                  {st.answer ? <div className="trace-step-out">{str(st.answer)}</div> : null}
                  {st.image ? <Media images={[st.image]} onOpen={onOpenImage} /> : null}
                </div>
              </li>
            ))}
          </ol>
        </>
      ) : null}

      {block.answerEv?.evidence_images?.length || block.evidence?.length ? (
        <>
          <div className="trace-section">evidence</div>
          {(block.answerEv?.evidence_images ?? block.evidence ?? []).map((rec, i) => (
            <EvidenceRow key={i} rec={rec} onOpenImage={onOpenImage} />
          ))}
        </>
      ) : null}

      {result?.sub ? (
        <SubAgent sub={result.sub} onOpenImage={onOpenImage} />
      ) : result?.text ? (
        <>
          <div className="trace-section">result</div>
          <div className="trace-body-text">{result.text}</div>
          <Media images={result.images ?? []} onOpen={onOpenImage} size="full" />
        </>
      ) : null}
    </div>
  );
}

export function TracePanel({ config }: PanelProps<TracePanelConfig>) {
  const payload = useTopicPayload<TracePayload>(config.topic);
  const [selected, setSelected] = useState<number | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);

  const blocks = useMemo(() => {
    const all = toBlocks(payload?.events ?? []);
    return config.showState ? all : all.filter((b) => b.kind !== 'state');
  }, [payload, config.showState]);

  // Follow the tail only while the reader is already at the bottom: scrolling
  // back to study a step must not be undone by the next event landing.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [blocks]);

  if (!blocks.length) {
    return <Ask initial={config.defaultQuestion} />;
  }

  const head = blocks.find((b) => b.kind === 'run')?.ev;
  const current = blocks.find((b) => b.key === selected) ?? null;

  return (
    <div className="trace-panel" data-testid="trace-panel">
      <div className="trace-main">
        {head ? (
          <div className="trace-head">
            <div className="trace-head-q">{head.question}</div>
            <div className="trace-head-meta">
              <span>{head.model}</span>
              <span>{head.captured} cells captured</span>
            </div>
          </div>
        ) : null}

        <div
          className="trace-scroll"
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
        >
          {blocks.map((block) => {
            if (block.kind === 'run') return null;
            if (block.kind === 'prose') {
              return (
                <p className="trace-prose" key={block.key}>
                  {block.ev?.text}
                </p>
              );
            }
            if (block.kind === 'approval') {
              const st = block.ev?.state ?? '';
              const label: Record<string, string> = {
                requested: 'requested',
                awaiting_approval: 'waiting for your approval',
                approved: 'you approved',
                redirected: 'you chose a different cell',
                cancelled: 'not approved',
                captured: 'captured',
              };
              return (
                <div className="trace-approval" data-state={st} key={block.key}>
                  <span>{label[st] ?? st}</span>
                  {block.ev?.cell ? <span>{cellLabel(block.ev.cell)}</span> : null}
                  {block.ev?.to ? <span>→ {cellLabel(block.ev.to)}</span> : null}
                </div>
              );
            }
            if (block.kind === 'state') {
              // The stapled block is a delta plus an action menu. The delta is
              // what changed and belongs in the stream; the menu is a list of
              // options that is long, repeats every move, and rots instantly.
              const raw = (block.ev?.text ?? '').replace(/^STATE · /, '');
              const [delta = '', menu = ''] = raw.split(/\n?MOVES ·\s*/);
              const offered = menu ? menu.split(/\s(?=\[|move\()/).length : 0;
              return (
                <div className="trace-state" key={block.key} title={raw}>
                  {delta.trim()}
                  {offered ? <span className="trace-state-menu"> · {offered} moves offered</span> : null}
                </div>
              );
            }
            if (block.kind === 'answer') {
              return (
                <div
                  className="trace-answer"
                  key={block.key}
                  onClick={() => setSelected(block.key)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') setSelected(block.key);
                  }}
                >
                  <div className="trace-answer-label">answer</div>
                  <div className="trace-answer-verdict">{block.ev?.verdict}</div>
                </div>
              );
            }
            if (block.kind === 'survey') {
              return (
                <SurveyBlock
                  key={block.key}
                  block={block}
                  selected={block.key === selected}
                  onSelect={() => setSelected(block.key)}
                  onOpenImage={setLightbox}
                />
              );
            }
            if (block.start && !block.call) {
              return <StartBlock key={block.key} ev={block.start} />;
            }
            if (block.call?.tool === 'answer') {
              return (
                <AnswerBlock
                  key={block.key}
                  block={block}
                  selected={block.key === selected}
                  onSelect={() => setSelected(block.key)}
                  onOpenImage={setLightbox}
                />
              );
            }
            return (
              <ToolBlock
                key={block.key}
                block={block}
                selected={block.key === selected}
                onSelect={() => setSelected(block.key)}
                onOpenImage={setLightbox}
              />
            );
          })}
        </div>
      </div>

      <aside className="trace-pane" style={{ width: config.paneWidth }}>
        <Detail block={current} onOpenImage={setLightbox} />
      </aside>

      {lightbox ? (
        <div className="trace-lightbox" onClick={() => setLightbox(null)} role="presentation">
          {/* The larger side of the bitmap gets pinned to 60% of its screen
              axis (CSS keys on data-orient); the other side follows the
              image's own ratio. */}
          <img
            src={lightbox}
            alt=""
            onLoad={(e) => {
              const im = e.currentTarget;
              im.dataset.orient =
                im.naturalWidth >= im.naturalHeight ? 'landscape' : 'portrait';
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
