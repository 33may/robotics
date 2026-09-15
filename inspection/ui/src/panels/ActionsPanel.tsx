/**
 * Operator actions: survey + view grid + Stop + Exit. Pure IO — every button
 * renders from `views/state` + `run/status`; a click sends a command and the
 * truth comes back on the bus. See ../AGENTS.md and the v2 design doc.
 *
 * Two modes. While a brain run is driving (a `run` event on the trace with no
 * `answer` yet), the grid disappears: the AI names the target, so the only
 * decision left is yes/no on ITS request — the same shape as the survey
 * approval, one previewing button. A 36-cell grid during that wait is 35 ways
 * to send a redirect by accident. The toggle brings the grid back (a redirect
 * IS sometimes the right correction — live.py treats it as one); with no
 * brain run the grid is the default.
 *
 * There is no deny command in the machine, on purpose (brain/live.py: "brain
 * may press view/request. Only a human may press view/confirm."). Not
 * approving is the deny — the mover blocks un-timed — and STOP/exit stay
 * below, never disabled.
 */
import { useState } from 'react';
import { useBus, useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';
import type { ViewCellState } from './CloudInspectPanel';

interface ViewCell {
  h: number;
  v: number;
  state: ViewCellState;
  /** Capture URL, present iff the cell was visited — the preview shown while
   *  approving a re-visit or a replay-backed request. */
  image?: string;
}
/** The gate vocabulary (machine.py `_faces_display`), NOT the cell one:
 *  `settling` = boxguard not stable yet, `ready` = requestable. The in-flight
 *  face arrives as pending/previewing, phase-derived exactly like cells. */
type FaceStatus =
  | 'settling' | 'ready' | 'pending' | 'previewing' | 'visited' | 'unreachable';
interface FaceState {
  status: FaceStatus;
  /** Achieved off-angle from the resolution ladder, once resolved. */
  off_angle_deg?: number | null;
  /** Camera standoff point, base frame — absent while settling. */
  pos?: number[];
}
interface ViewsState {
  survey?: { state: ViewCellState };
  cells: ViewCell[];
  faces?: Record<string, FaceState>;
}
interface RunStatus { phase?: string; target?: unknown }
/** Only the kinds matter here; the trace panel owns the full shape. */
interface TraceState { events?: { kind: string }[] }
/** Only `source` matters here; InspectionApp owns the full shape. */
interface RunMeta { source?: string }

const ACTIONABLE: ViewCellState[] = ['available', 'blocked', 'pending', 'previewing'];
/** `pending` stays clickable like cells: a re-request of the in-flight face
 *  is deduped server-side. */
const FACE_ACTIONABLE: FaceStatus[] = ['ready', 'pending', 'previewing'];
/** Render order — machine.py faceview.FACES; payload keys are unordered. */
const FACE_ORDER = ['x+', 'x-', 'y+', 'y-', 'top'];

function label(state: ViewCellState | FaceStatus): string {
  if (state === 'pending') return '…';
  if (state === 'previewing') return 'preview';
  return '';
}

function faceLabel(name: string, face: FaceState): string {
  const theta = face.off_angle_deg;
  const deg = theta === null || theta === undefined ? '' : ` ${Math.round(theta)}°`;
  return `${name}${deg} ${label(face.status)}`.trimEnd();
}

export function ActionsPanel(_props: PanelProps) {
  const bus = useBus();
  const views = useTopicPayload<ViewsState>('views/state');
  const status = useTopicPayload<RunStatus>('run/status') ?? {};
  const trace = useTopicPayload<TraceState>('trace/state');
  // null = follow the brain. Per-mount on purpose: the next run lands back
  // on the safe default instead of whatever the last operator forced.
  const [forced, setForced] = useState<boolean | null>(null);
  const send = (cmd: string, target?: unknown) =>
    bus.send(target === undefined ? { cmd } : { cmd, target });
  const press = (state: ViewCellState, target: 'survey' | [number, number]) =>
    send(state === 'previewing' ? 'view/confirm' : 'view/request', target);

  const events = trace?.events ?? [];
  const driving =
    events.some((e) => e.kind === 'run') && !events.some((e) => e.kind === 'answer');
  // A collect sweep is "driving" the whole run (Anton 2026-09-08): the
  // SweepDriver names every target, so the decision left is the same yes/no
  // as a brain request and the grid is 35 accidental redirects. It has no
  // trace to infer that from, hence the explicit `run/meta` gate; the
  // toggle still brings the grid back for a deliberate redirect.
  const collect = useTopicPayload<RunMeta>('run/meta')?.source === 'data-engine';
  const approveOnly = forced ?? (driving || collect);

  const cells = views?.cells ?? [];
  const rows = [...new Set(cells.map((c) => c.v))].sort((a, b) => b - a);
  const survey = views?.survey?.state;
  const faces = views?.faces ?? {};
  const faceEntries = Object.entries(faces).sort(
    ([a], [b]) => FACE_ORDER.indexOf(a) - FACE_ORDER.indexOf(b));
  const pressFace = (name: string, status: FaceStatus) =>
    send(status === 'previewing' ? 'view/confirm' : 'view/request', ['face', name]);
  // At most one cell is pending/previewing (machine.py: one target), so the
  // first hit IS the AI's request. Survey, cells and faces are mutually
  // exclusive targets for the same reason.
  const requested = cells.find((c) => c.state === 'pending' || c.state === 'previewing');
  const requestedFace = faceEntries.find(
    ([, s]) => s.status === 'pending' || s.status === 'previewing');
  const surveyActive = survey === 'pending' || survey === 'previewing';

  return (
    <div className="actions-panel">
      <div className="actions-head">
        <div className="actions-status">{String(status.phase ?? '—')}</div>
        <button className="act act-mode" onClick={() => setForced(!approveOnly)}>
          {approveOnly ? 'grid' : 'approve'}
        </button>
      </div>

      {approveOnly ? (
        // Inline flex, not a css edit: the container must grow like the
        // grid-mode survey button does (`.act { flex: 1 }` on a panel-level
        // child), so its one button is the same full-height target.
        <div className="actions-approve" style={{ flex: 1, minHeight: 0 }}>
          {surveyActive && survey ? (
            <button
              className={`act act-${survey}`}
              data-testid={survey === 'previewing' ? 'confirm-button' : undefined}
              disabled={survey !== 'previewing'}
              onClick={() => send('view/confirm', 'survey')}
            >
              survey {label(survey)}
            </button>
          ) : requested ? (
            <>
              {requested.image ? (
                <img className="actions-preview" src={requested.image} alt="" />
              ) : null}
              <button
                className={`act act-${requested.state}`}
                data-testid={requested.state === 'previewing' ? 'confirm-button' : undefined}
                disabled={requested.state !== 'previewing'}
                onClick={() => send('view/confirm', [requested.h, requested.v])}
              >
                {requested.h},{requested.v} {label(requested.state)}
              </button>
            </>
          ) : requestedFace ? (
            <button
              className={`act act-${requestedFace[1].status}`}
              data-testid={requestedFace[1].status === 'previewing'
                ? 'confirm-button' : undefined}
              disabled={requestedFace[1].status !== 'previewing'}
              onClick={() => send('view/confirm', ['face', requestedFace[0]])}
            >
              face {faceLabel(requestedFace[0], requestedFace[1])}
            </button>
          ) : (
            <div className="actions-status">—</div>
          )}
        </div>
      ) : (
        <>
          {survey && (
            <button
              className={`act act-${survey}`}
              data-testid={survey === 'previewing' ? 'confirm-button' : undefined}
              disabled={!ACTIONABLE.includes(survey)}
              onClick={() => press(survey, 'survey')}
            >
              survey {label(survey)}
            </button>
          )}
          {faceEntries.length > 0 && (
            // The five orthogonal face targets, above the shell grid: a
            // different address space, so a different row — never mixed
            // into the h,v cells. Settling faces render disabled on
            // purpose (the operator sees the gate hasn't opened yet).
            <div className="actions-row actions-faces">
              {faceEntries.map(([f, face]) => (
                <button
                  key={f}
                  className={`act act-face act-${face.status}`}
                  data-testid={face.status === 'previewing' ? 'confirm-button' : undefined}
                  disabled={!FACE_ACTIONABLE.includes(face.status)}
                  onClick={() => pressFace(f, face.status)}
                  title={`face ${f}: ${face.status}`}
                >
                  {faceLabel(f, face)}
                </button>
              ))}
            </div>
          )}
          <div className="actions-grid">
            {rows.map((v) => (
              <div className="actions-row" key={v}>
                {cells.filter((c) => c.v === v).sort((a, b) => a.h - b.h).map((c) => (
                  <button
                    key={`${c.h}-${c.v}`}
                    className={`act act-${c.state}`}
                    data-testid={c.state === 'previewing' ? 'confirm-button' : undefined}
                    disabled={!ACTIONABLE.includes(c.state)}
                    onClick={() => press(c.state, [c.h, c.v])}
                    title={`h${c.h} v${c.v}: ${c.state}`}
                  >
                    {c.h},{c.v} {label(c.state)}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="actions-footer">
        {/* Never disabled. A stop button that has to be re-enabled before it
            works is a stop button you cannot hit in a hurry, and the phase it
            would key off is a retained topic that can lag the arm. The
            dispatcher validates it: outside `executing` it is a logged
            no-op. */}
        <button className="act act-stop"
                onClick={() => send('run/stop')}>STOP</button>
        {collect ? (
          // `run/finish` = exit's safe shutdown, closed as "completed":
          // enough views is a decision, not an abort. Collect-only — a live
          // run's completion is its answer, not a button. `act-visited`
          // borrows the green "done" colour without a css edit.
          <button
            className="act act-visited"
            data-testid="finish-button"
            onClick={() => send('run/finish')}
          >
            finish
          </button>
        ) : null}
        <button className="act act-exit" onClick={() => send('run/exit')}>exit</button>
      </div>
    </div>
  );
}
