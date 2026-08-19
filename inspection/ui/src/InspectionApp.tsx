/**
 * The inspection dashboard.
 *
 * Four panels, no command path. v1 keeps the approval gate and the look/answer
 * decision on the terminal (`run/decider.py`), so this window is a monitor —
 * every question about authority is answered in `run/`, where it already was.
 */

import { BusProvider, PortholeDashboard, useBusStatus, useTopicPayload } from '@porthole/framework';
import type { LayoutSpec } from '@porthole/framework';

import { inspectionPanelDefinitions } from './panelDefinitions.js';

/**
 * `cloud` is added second within the same group as `cell`, which makes it the
 * active tab. Both are kept mounted, so switching tabs costs nothing.
 */
const inspectionLayout: LayoutSpec = [
  { id: 'cell', type: 'cell' },
  { id: 'camera', type: 'camera', position: { referencePanel: 'cell', direction: 'right' } },
  { id: 'cloud', type: 'cloud', position: { referencePanel: 'camera', direction: 'below' } },
  { id: 'log', type: 'log', position: { referencePanel: 'cell', direction: 'below' } },
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
