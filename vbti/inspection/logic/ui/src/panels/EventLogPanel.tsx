/**
 * Streaming event log.
 *
 * REGISTRY COMPONENT — copy and edit freely; see docs/component-model.md.
 *
 * Subscribes reactively (`useTopic`) rather than imperatively, because this
 * panel's output *is* DOM: every message genuinely changes what is on screen,
 * so there is nothing for React to waste. That is the opposite call from
 * CameraPanel, and the reason is the rule in the framework's hooks module.
 */

import { useEffect, useRef, useState } from 'react';
import { useBus } from '@porthole/framework';
import type { BusEnvelope, PanelProps } from '@porthole/framework';

export interface EventLogPanelConfig extends Record<string, unknown> {
  /** Topic, or a `prefix/*` pattern, or `*` for every topic. */
  topic: string;
  /** Rows kept in the DOM. Older rows are dropped. */
  limit: number;
  /** Stick to the newest row unless the human has scrolled up. */
  follow: boolean;
}

export const eventLogPanelDefaultConfig: EventLogPanelConfig = {
  topic: 'log/events',
  limit: 500,
  follow: true,
};

interface LogRow {
  key: string;
  ts: number;
  topic: string;
  level: string;
  text: string;
}

function toRow(envelope: BusEnvelope): LogRow {
  const payload = envelope.payload as Record<string, unknown> | null;
  const level = typeof payload?.['level'] === 'string' ? (payload['level'] as string) : 'info';
  const message = payload?.['msg'] ?? payload?.['message'];
  return {
    key: `${envelope.topic}#${envelope.seq}`,
    ts: envelope.ts,
    topic: envelope.topic,
    level,
    text: typeof message === 'string' ? message : JSON.stringify(envelope.payload),
  };
}

function formatClock(ts: number): string {
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString('en-GB', { hour12: false }) +
    '.' + String(date.getMilliseconds()).padStart(3, '0');
}

export function EventLogPanel({ config }: PanelProps<EventLogPanelConfig>) {
  const bus = useBus();
  const [rows, setRows] = useState<LogRow[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    setRows([]);
    return bus.subscribe(config.topic, (envelope) => {
      setRows((previous) => {
        const next = [...previous, toRow(envelope)];
        return next.length > config.limit ? next.slice(-config.limit) : next;
      });
    });
  }, [bus, config.topic, config.limit]);

  // Only auto-scroll while the human is already at the bottom. Yanking the
  // view down while someone is reading history is the classic log-panel sin.
  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !config.follow || !pinnedRef.current) return;
    element.scrollTop = element.scrollHeight;
  }, [rows, config.follow]);

  function handleScroll() {
    const element = scrollRef.current;
    if (!element) return;
    const distanceFromBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight;
    pinnedRef.current = distanceFromBottom < 24;
  }

  return (
    <div className="porthole-event-log" ref={scrollRef} onScroll={handleScroll}>
      <table>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} data-level={row.level}>
              <td className="porthole-event-log-time">{formatClock(row.ts)}</td>
              <td className="porthole-event-log-topic">{row.topic}</td>
              <td className="porthole-event-log-text">{row.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
