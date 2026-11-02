/**
 * React context holding the one BusClient an app owns.
 *
 * One client per app, not per panel. Panels multiplex over it — a docking UI
 * can mount a dozen panels, and opening a dozen sockets would multiply the
 * server's fan-out cost for no benefit.
 */

import { createContext, useContext, useEffect, useMemo } from 'react';
import type { ReactNode } from 'react';

import { BusClient } from '../bus/client.js';
import type { BusClientOptions } from '../bus/client.js';

const BusContext = createContext<BusClient | null>(null);

export interface BusProviderProps extends BusClientOptions {
  children: ReactNode;
  /** Inject a client instead of creating one — used by tests and replay tools. */
  client?: BusClient;
}

export function BusProvider({ children, client, ...options }: BusProviderProps) {
  const bus = useMemo(
    () => client ?? new BusClient(options),
    // A changed URL means a different backend, so a new client is correct.
    [client, options.url],
  );

  useEffect(() => {
    if (client) return; // caller owns the injected client's lifecycle
    bus.connect();
    return () => bus.close();
  }, [bus, client]);

  return <BusContext.Provider value={bus}>{children}</BusContext.Provider>;
}

export function useBus(): BusClient {
  const bus = useContext(BusContext);
  if (!bus) {
    throw new Error('useBus() called outside <BusProvider>. Wrap your app in one.');
  }
  return bus;
}
