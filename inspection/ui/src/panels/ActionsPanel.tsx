/**
 * Operator actions: survey + view grid + Stop + Exit. Pure IO — every button
 * renders from `views/state` + `run/status`; a click sends a command and the
 * truth comes back on the bus. See ../AGENTS.md and the v2 design doc.
 */
import { useBus, useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';
import type { ViewCellState } from './CloudInspectPanel';

interface ViewsState {
  survey?: { state: ViewCellState };
  cells: { h: number; v: number; state: ViewCellState }[];
}
interface RunStatus { phase?: string; target?: unknown }

const ACTIONABLE: ViewCellState[] = ['available', 'blocked', 'pending', 'previewing'];

function label(state: ViewCellState): string {
  if (state === 'pending') return '…';
  if (state === 'previewing') return 'preview';
  return '';
}

export function ActionsPanel(_props: PanelProps) {
  const bus = useBus();
  const views = useTopicPayload<ViewsState>('views/state');
  const status = useTopicPayload<RunStatus>('run/status') ?? {};
  const send = (cmd: string, target?: unknown) =>
    bus.send(target === undefined ? { cmd } : { cmd, target });
  const press = (state: ViewCellState, target: 'survey' | [number, number]) =>
    send(state === 'previewing' ? 'view/confirm' : 'view/request', target);

  const cells = views?.cells ?? [];
  const rows = [...new Set(cells.map((c) => c.v))].sort((a, b) => b - a);
  const survey = views?.survey?.state;
  return (
    <div className="actions-panel">
      <div className="actions-status">{String(status.phase ?? '—')}</div>
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
      <div className="actions-footer">
        {/* Never disabled. A stop button that has to be re-enabled before it
            works is a stop button you cannot hit in a hurry, and the phase it
            would key off is a retained topic that can lag the arm. The
            dispatcher validates it: outside `executing` it is a logged
            no-op. */}
        <button className="act act-stop"
                onClick={() => send('run/stop')}>STOP</button>
        <button className="act act-exit" onClick={() => send('run/exit')}>exit</button>
      </div>
    </div>
  );
}
