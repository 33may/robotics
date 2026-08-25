/**
 * The inspection dashboard.
 *
 * Five panels: cell, camera, cloud, log, actions. `actions` is a real command
 * path — clicks send `view/request` / `view/confirm` / `run/stop` / `run/exit`
 * onto the bus — but authority stays entirely backend-side: `run/machine.py`'s
 * `Supervisor` validates every command before anything moves, so a stale
 * button here can never move the arm. Stop hierarchy, strongest first: pendant
 * e-stop (hardware) > Ctrl-C (process) > this panel's Stop button (software,
 * `executing` only).
 */

import { BusProvider, PortholeDashboard, useBusStatus, useTopicPayload } from '@porthole/framework';
import type { LayoutSpec } from '@porthole/framework';

import { inspectionPanelDefinitions } from './panelDefinitions.js';

/**
 * Two columns: the robot on the left, the agent's thinking on the right.
 *
 * ORDER IS THE LAYOUT. Dockview splits whatever group the reference panel is
 * in, so these placements are not a list of positions — they are a sequence of
 * cuts, and rearranging the lines rearranges the app:
 *
 *   1  cloud                    the root group
 *   2  trace  right of cloud    cuts the root into two COLUMNS. Doing this
 *                               before anything else is what makes `trace`
 *                               full height; add it later and it only ever
 *                               gets the height of whatever it split.
 *   3  cell   within cloud      tabs into the same group. Added LAST, so it
 *                               is the ACTIVE tab — dockview activates
 *                               whatever arrived most recently, and there is
 *                               no `active` flag on a placement. That is the
 *                               only reason `cloud` creates the group and
 *                               `cell` follows it rather than the reverse.
 *   4  log    below cloud       cuts the LEFT column only, never the right.
 *   5  actions within log       added LAST so ACTIONS is the active tab.
 *                               Deliberate, and it overrides the sketch: STOP
 *                               must never be hidden behind a tab (the same
 *                               reason it is never greyed out — Anton
 *                               2026-08-20). `log` is `keepMounted` so it
 *                               still collects rows while it sits behind.
 *   6  chain  right of log      the bottom strip becomes three groups...
 *   7  camera right of chain    ...left to right.
 *
 * `trace` gets a full-height column because it is the reading surface: a
 * thought stream plus a detail pane needs vertical room far more than the
 * glanceable panels do.
 *
 * NOTE: this is only the DEFAULT. `PortholeDashboard` restores a saved layout
 * from `localStorage['porthole.layout.inspection']` and skips this entirely
 * when one exists, so editing here changes nothing until that key is cleared.
 */
const inspectionLayout: LayoutSpec = [
  { id: 'cloud', type: 'cloud' },
  { id: 'trace', type: 'trace', position: { referencePanel: 'cloud', direction: 'right' } },
  { id: 'cell', type: 'cell', position: { referencePanel: 'cloud', direction: 'within' } },
  { id: 'log', type: 'log', position: { referencePanel: 'cloud', direction: 'below' } },
  { id: 'actions', type: 'actions', position: { referencePanel: 'log', direction: 'within' } },
  { id: 'chain', type: 'chain', position: { referencePanel: 'log', direction: 'right' } },
  { id: 'camera', type: 'camera', position: { referencePanel: 'chain', direction: 'right' } },
];

/**
 * Which bus to dial, from `?bus=` on the URL.
 *
 * Defaults to `ws://<page host>:8765`, which is right when the loop runs on
 * this machine. Override it to run two backends at once (a mock beside a real
 * run, which is how the checks avoid fighting over the default port), or to
 * open this window on a laptop and point it at the robot's machine:
 *
 *     ?bus=8781                  same host, other port
 *     ?bus=ws://192.168.2.10:8765
 */
function busUrlFromQuery(): string | undefined {
  const value = new URLSearchParams(window.location.search).get('bus');
  if (!value) return undefined;
  if (/^wss?:\/\//.test(value)) return value;
  return `ws://${window.location.hostname || '127.0.0.1'}:${value}`;
}

interface RunStatus {
  readonly phase?: string;
  readonly step?: number;
  readonly max_turns?: number;
  readonly question?: string;
  readonly visited?: number;
  readonly reachable?: number;
  readonly total?: number;
  readonly detail?: string;
}

function InspectionStatusBar() {
  const connection = useBusStatus();
  const status = useTopicPayload<RunStatus>('run/status');

  return (
    <div className="inspection-statusbar">
      <span className="porthole-status-dot" data-status={connection} />
      {status?.phase ? (
        <span className="inspection-phase" data-phase={status.phase}>
          {status.phase}
        </span>
      ) : null}
      {status?.step !== undefined ? (
        <span>
          step {status.step}
          {status.max_turns !== undefined ? `/${status.max_turns}` : ''}
        </span>
      ) : null}
      {status?.visited !== undefined ? (
        <span>
          {status.visited}/{status.reachable ?? '?'} visited
          {status.total !== undefined ? ` of ${status.total}` : ''}
        </span>
      ) : null}
      {status?.detail ? <span>{status.detail}</span> : null}
      {status?.question ? <span className="inspection-question">{status.question}</span> : null}
    </div>
  );
}

export function InspectionApp() {
  const url = busUrlFromQuery();
  return (
    // Conditional spread: `exactOptionalPropertyTypes` rejects `url={undefined}`.
    <BusProvider {...(url !== undefined ? { url } : {})}>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <PortholeDashboard
            panels={inspectionPanelDefinitions}
            layout={inspectionLayout}
            storageKey="inspection"
          />
        </div>
        <InspectionStatusBar />
      </div>
    </BusProvider>
  );
}
