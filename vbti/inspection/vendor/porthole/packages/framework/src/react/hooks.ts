/**
 * Subscription hooks. There are two, and choosing the wrong one is the single
 * most common way to make a Porthole dashboard slow.
 *
 *   useTopic()       reactive   — re-renders the component on every message
 *   useTopicEffect() imperative — never re-renders; hands you the message
 *
 * **Rule: if the panel paints to a canvas, WebGL surface, or any imperative
 * drawing API, use `useTopicEffect`.** A 30 Hz point cloud driven through
 * `useTopic` asks React to reconcile a tree 30 times a second to produce zero
 * DOM changes, because the pixels are drawn by three.js, not by React. The
 * reconciliation is pure waste and it is what makes streaming dashboards
 * stutter.
 *
 * Use `useTopic` for DOM that genuinely changes: log lists, readouts, menus,
 * status chips. At event rates it is the clearer code and costs nothing.
 */

import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';

import type { BusCommand, ConnectionStatus, BusEnvelope, BusHello } from '../bus/types.js';
import { useBus } from './context.js';

/**
 * Latest envelope on a topic, re-rendering when it changes.
 * Returns `undefined` until the first message arrives.
 */
export function useTopic<P = unknown>(topic: string): BusEnvelope<P> | undefined {
  const bus = useBus();

  const subscribe = useCallback(
    (onChange: () => void) => bus.subscribe(topic, onChange),
    [bus, topic],
  );
  const snapshot = useCallback(() => bus.latest(topic), [bus, topic]);

  return useSyncExternalStore(subscribe, snapshot, snapshot) as BusEnvelope<P> | undefined;
}

/** `useTopic` without the envelope wrapper, when you only want the data. */
export function useTopicPayload<P = unknown>(topic: string): P | undefined {
  return useTopic<P>(topic)?.payload;
}

/**
 * Run a handler on every message without re-rendering.
 *
 * The handler is held in a ref, so it may close over fresh props without
 * resubscribing — only a change of `topic` does that.
 */
export function useTopicEffect<P = unknown>(
  topic: string,
  handler: (envelope: BusEnvelope<P>) => void,
): void {
  const bus = useBus();
  const latest = useRef(handler);
  latest.current = handler;

  useEffect(
    () => bus.subscribe(topic, (envelope) => latest.current(envelope as BusEnvelope<P>)),
    [bus, topic],
  );
}

/** Connection status, for a status chip or a "backend gone" overlay. */
export function useBusStatus(): ConnectionStatus {
  const bus = useBus();
  const subscribe = useCallback((onChange: () => void) => bus.onStatus(onChange), [bus]);
  const snapshot = useCallback(() => bus.status, [bus]);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

/** The backend's `$/hello`: app name, server version, declared topics. */
export function useBusHello(): BusHello | null {
  const bus = useBus();
  const subscribe = useCallback((onChange: () => void) => bus.onHello(onChange), [bus]);
  const snapshot = useCallback(() => bus.hello, [bus]);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

/**
 * A stable `send` function for commands.
 *
 * Returns false when the socket is down. Show that to the human — a button that
 * silently does nothing while the backend is restarting is a bug report.
 */
export function useSendCommand(): (command: BusCommand) => boolean {
  const bus = useBus();
  return useCallback((command: BusCommand) => bus.send(command), [bus]);
}
