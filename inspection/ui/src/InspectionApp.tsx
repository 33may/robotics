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

import { useMemo } from 'react';
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

/**
 * `run/meta`, RETAINED (`ui/publisher.py:publish_run_meta`) — published once at
 * boot, before the UI is even served, so by the time this connects the message
 * is already sitting on the bus. `source` is what gates the collect-mode UI:
 * a `"data-engine"` run has no brain in the loop, so the trace panel and its
 * Ask composer have nothing to show and are dropped from the layout entirely
 * (not just hidden — see `collectModePanels` below).
 */
interface RunMeta {
  readonly source?: 'live' | 'data-engine';
  readonly name?: string;
  readonly object?: string | null;
  readonly question?: string | null;
}

/**
 * `trace` out, everything else identical. Filtering the panel list AND the
 * layout (rather than rendering `TracePanel` empty) is what makes it not
 * appear as a tab at all — an empty tab would still say "this run has an
 * agent" to anyone glancing at the dockview strip.
 */
function collectModePanels(
  panels: typeof inspectionPanelDefinitions,
): typeof inspectionPanelDefinitions {
  return panels.filter((p) => p.type !== 'trace');
}

function collectModeLayout(layout: LayoutSpec): LayoutSpec {
  return layout.filter((p) => p.id !== 'trace');
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

/**
 * The dashboard proper, inside `BusProvider` so it can read `run/meta`.
 *
 * Defaults to the LIVE panel set whenever `source` is not (yet, or ever)
 * `"data-engine"` — undefined included. `run/meta` is retained and published
 * before the UI is served, so in practice it is already cached by the time
 * this ever renders; defaulting to live rather than blocking on it is what
 * keeps every OTHER boot path (mock, trace-viewer) working unchanged if one
 * is ever booted without publishing `run/meta` at all.
 *
 * `key` forces a full remount — and a fresh `applyDefaultLayout` — the
 * instant `source` resolves to `data-engine`: `PortholeDashboard` only
 * applies its layout prop once, in `onReady`, so a bare prop swap after the
 * dashboard has already mounted would leave a stale `trace` tab sitting
 * there. A distinct `storageKey` per mode keeps the two saved layouts from
 * fighting over the same `localStorage` entry.
 */
function InspectionDashboard() {
  const meta = useTopicPayload<RunMeta>('run/meta');
  const collectMode = meta?.source === 'data-engine';

  const panels = useMemo(
    () => (collectMode ? collectModePanels(inspectionPanelDefinitions) : inspectionPanelDefinitions),
    [collectMode],
  );
  const layout = useMemo(
    () => (collectMode ? collectModeLayout(inspectionLayout) : inspectionLayout),
    [collectMode],
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, minHeight: 0 }}>
        <PortholeDashboard
          key={collectMode ? 'collect' : 'live'}
          panels={panels}
          layout={layout}
          storageKey={collectMode ? 'inspection-collect' : 'inspection'}
        />
      </div>
      <InspectionStatusBar />
    </div>
  );
}

export function InspectionApp() {
  const url = busUrlFromQuery();
  return (
    // Conditional spread: `exactOptionalPropertyTypes` rejects `url={undefined}`.
    <BusProvider {...(url !== undefined ? { url } : {})}>
      <InspectionDashboard />
    </BusProvider>
  );
}
