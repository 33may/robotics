/**
 * The WebSocket client. Framework-agnostic on purpose — React bindings live in
 * `../react`, and nothing here imports React.
 *
 * Two behaviours are worth knowing about before writing a panel:
 *
 * 1. **Client-side retention.** The client caches the last envelope per topic.
 *    The server already replays retained state on connect, but a docking UI
 *    unmounts hidden panels, so a panel remounting on tab switch would
 *    otherwise stare at a blank canvas until the next message. `latest()` makes
 *    mount-time rendering synchronous.
 * 2. **Reconnection is expected, not exceptional.** A backend restarts many
 *    times an hour during development. The client reconnects with backoff and
 *    keeps its cache, so the UI shows the last known state greyed out rather
 *    than collapsing.
 */

import { decodeEnvelope } from './decode.js';
import type { BusCommand, ConnectionStatus, BusEnvelope, BusHello } from './types.js';

export interface BusClientOptions {
  /** Defaults to `ws://<current host>:8765`, so a browser on another machine works. */
  url?: string;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
}

type EnvelopeHandler = (envelope: BusEnvelope) => void;
type StatusHandler = (status: ConnectionStatus) => void;

const DEFAULT_PORT = 8765;

function defaultUrl(): string {
  if (typeof window === 'undefined') return `ws://127.0.0.1:${DEFAULT_PORT}`;
  return `ws://${window.location.hostname || '127.0.0.1'}:${DEFAULT_PORT}`;
}

/** `camera/wrist` matches `camera/wrist`, `camera/*` and `*`. */
function topicMatches(pattern: string, topic: string): boolean {
  if (pattern === topic || pattern === '*') return true;
  if (pattern.endsWith('/*')) return topic.startsWith(pattern.slice(0, -1));
  return false;
}

export class BusClient {
  readonly url: string;

  private socket: WebSocket | null = null;
  private readonly handlers = new Map<string, Set<EnvelopeHandler>>();
  private readonly statusHandlers = new Set<StatusHandler>();
  private readonly helloHandlers = new Set<() => void>();
  private readonly cache = new Map<string, BusEnvelope>();

  private _status: ConnectionStatus = 'closed';
  private _hello: BusHello | null = null;
  private attempt = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private closedByUser = false;

  private readonly baseMs: number;
  private readonly maxMs: number;

  constructor(options: BusClientOptions = {}) {
    this.url = options.url ?? defaultUrl();
    this.baseMs = options.reconnectBaseMs ?? 250;
    this.maxMs = options.reconnectMaxMs ?? 5000;
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  /** The server's `$/hello`, or null before the first successful connection. */
  get hello(): BusHello | null {
    return this._hello;
  }

  connect(): this {
    this.closedByUser = false;
    this.open();
    return this;
  }

  close(): void {
    this.closedByUser = true;
    if (this.retryTimer !== null) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.socket?.close();
    this.socket = null;
    this.setStatus('closed');
  }

  /**
   * Subscribe to a topic, a `prefix/*` pattern, or `*` for everything.
   * Returns an unsubscribe function. If the topic already has a cached value,
   * the handler is invoked immediately with it.
   */
  subscribe(pattern: string, handler: EnvelopeHandler): () => void {
    let set = this.handlers.get(pattern);
    if (!set) {
      set = new Set();
      this.handlers.set(pattern, set);
    }
    set.add(handler);

    for (const envelope of this.cache.values()) {
      if (topicMatches(pattern, envelope.topic)) {
        safeCall(handler, envelope);
      }
    }

    return () => {
      const current = this.handlers.get(pattern);
      if (!current) return;
      current.delete(handler);
      if (current.size === 0) this.handlers.delete(pattern);
    };
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this._status);
    return () => this.statusHandlers.delete(handler);
  }

  /**
   * Notified when `$/hello` arrives.
   *
   * Separate from `onStatus` because the two are not simultaneous: the socket
   * reports `open` before the server's first frame is decoded, so anything
   * watching status alone samples `hello` while it is still null and is never
   * told otherwise.
   */
  onHello(handler: () => void): () => void {
    this.helloHandlers.add(handler);
    return () => this.helloHandlers.delete(handler);
  }

  /** Last envelope seen on a topic, if any. */
  latest(topic: string): BusEnvelope | undefined {
    return this.cache.get(topic);
  }

  /** Topics seen so far, whether or not they were declared in `$/hello`. */
  knownTopics(): string[] {
    return [...this.cache.keys()].sort();
  }

  /**
   * Send a command. Returns false if the socket is not open — commands are
   * at-most-once by design (protocol §7) and are never queued, because a
   * button press replayed on reconnect is a lie about when the human clicked.
   */
  send(command: BusCommand): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ ts: Date.now() / 1000, ...command }));
    return true;
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private open(): void {
    this.setStatus('connecting');

    const socket = new WebSocket(this.url);
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.onopen = () => {
      this.attempt = 0;
      this.setStatus('open');
    };

    socket.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        console.warn('[porthole] ignoring text frame from server', event.data.slice(0, 120));
        return;
      }
      let envelope: BusEnvelope;
      try {
        envelope = decodeEnvelope(event.data as ArrayBuffer);
      } catch (error) {
        console.error('[porthole] undecodable frame', error);
        return;
      }
      this.dispatch(envelope);
    };

    socket.onclose = () => {
      this.socket = null;
      this.setStatus('closed');
      if (!this.closedByUser) this.scheduleRetry();
    };

    socket.onerror = () => {
      // `onclose` always follows; retry logic lives there so it runs once.
    };
  }

  private dispatch(envelope: BusEnvelope): void {
    if (envelope.topic === '$/hello') {
      this._hello = envelope.payload as BusHello;
      for (const handler of this.helloHandlers) {
        try {
          handler();
        } catch (error) {
          console.error('[porthole] hello handler threw', error);
        }
      }
      return;
    }
    this.cache.set(envelope.topic, envelope);

    for (const [pattern, set] of this.handlers) {
      if (!topicMatches(pattern, envelope.topic)) continue;
      for (const handler of set) safeCall(handler, envelope);
    }
  }

  private scheduleRetry(): void {
    const delay = Math.min(this.baseMs * 2 ** this.attempt, this.maxMs);
    this.attempt += 1;
    this.retryTimer = setTimeout(() => this.open(), delay);
  }

  private setStatus(status: ConnectionStatus): void {
    if (this._status === status) return;
    this._status = status;
    for (const handler of this.statusHandlers) {
      try {
        handler(status);
      } catch (error) {
        console.error('[porthole] status handler threw', error);
      }
    }
  }
}

/** One panel's broken handler must not stop every other panel from updating. */
function safeCall(handler: EnvelopeHandler, envelope: BusEnvelope): void {
  try {
    handler(envelope);
  } catch (error) {
    console.error(`[porthole] handler for "${envelope.topic}" threw`, error);
  }
}
