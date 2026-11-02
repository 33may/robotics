/**
 * The docking shell: turns a list of panel definitions plus a layout into a
 * running dashboard.
 *
 * Layout persistence is deliberately client-side (localStorage). It works
 * identically in the native window and in a browser on another machine, needs
 * no backend round-trip, and keeps the Python side free of UI state — which is
 * the whole point of a thin backend.
 */

import { useCallback, useMemo, useRef } from 'react';
import { DockviewReact } from 'dockview-react';
import type { DockviewApi, DockviewReadyEvent, IDockviewPanelProps } from 'dockview-react';
import type { ComponentType, FunctionComponent } from 'react';

import { buildPanelRegistry } from '../panel/definePanel.js';
import type { AnyPanelDefinition, LayoutSpec, PanelProps } from '../panel/types.js';
import { clearDashboardLayout, loadDashboardLayout, saveDashboardLayout } from './layout.js';

export interface PortholeDashboardProps {
  /** Every panel kind this dashboard can show. */
  panels: readonly AnyPanelDefinition[];
  /** Layout used on first run, and whenever a saved layout cannot be restored. */
  layout: LayoutSpec;
  /**
   * localStorage key for the saved layout. Omit to disable persistence —
   * useful for tests and screenshots, which must start from a known state.
   */
  storageKey?: string;
  /** A dockview theme object, e.g. `themeAbyss`. */
  theme?: object;
  className?: string;
}

interface HostParams {
  config?: Record<string, unknown>;
}

/**
 * Bridge one PanelDefinition into the component shape dockview expects.
 *
 * Config lives in dockview's panel `params`, so it is serialized into the saved
 * layout for free — a panel's per-instance settings survive a reload without
 * any separate storage.
 */
function makeHost(definition: AnyPanelDefinition): FunctionComponent<IDockviewPanelProps> {
  const Panel = definition.component as ComponentType<PanelProps<Record<string, unknown>>>;

  return function PanelHost(rawProps: IDockviewPanelProps) {
    const props = rawProps as IDockviewPanelProps<HostParams>;
    const config = useMemo(
      () => ({ ...definition.defaultConfig, ...props.params?.config }),
      [props.params?.config],
    );

    const setConfig = useCallback(
      (patch: Record<string, unknown>) => {
        props.api.updateParameters({ config: { ...config, ...patch } });
      },
      [props.api, config],
    );

    const setTitle = useCallback((title: string) => props.api.setTitle(title), [props.api]);

    return (
      <Panel
        instanceId={props.api.id}
        config={config}
        setConfig={setConfig}
        setTitle={setTitle}
      />
    );
  };
}

export function PortholeDashboard({
  panels,
  layout,
  storageKey,
  theme,
  className,
}: PortholeDashboardProps) {
  const registry = useMemo(() => buildPanelRegistry(panels), [panels]);
  const components = useMemo(() => {
    const map: Record<string, FunctionComponent<IDockviewPanelProps>> = {};
    for (const [type, definition] of registry) map[type] = makeHost(definition);
    return map;
  }, [registry]);

  const apiRef = useRef<DockviewApi | null>(null);

  const applyDefaultLayout = useCallback(
    (api: DockviewApi) => {
      api.clear();
      for (const placement of layout) {
        const definition = registry.get(placement.type);
        if (!definition) {
          console.error(`[porthole] layout references unknown panel type "${placement.type}"`);
          continue;
        }
        api.addPanel({
          id: placement.id,
          component: placement.type,
          title: placement.title ?? definition.title,
          params: { config: placement.config ?? {} },
          renderer: definition.keepMounted ? 'always' : 'onlyWhenVisible',
          ...(placement.position
            ? { position: placement.position as never, floating: false as const }
            : {}),
        });
      }
    },
    [layout, registry],
  );

  const onReady = useCallback(
    (event: DockviewReadyEvent) => {
      apiRef.current = event.api;

      const saved = storageKey ? loadDashboardLayout(storageKey) : null;
      if (saved) {
        try {
          event.api.fromJSON(saved as unknown as Parameters<DockviewApi['fromJSON']>[0]);
        } catch (error) {
          // A saved layout can reference a panel type that has since been
          // renamed or removed. Falling back beats a blank window, and the
          // stale layout is cleared so the failure does not repeat.
          console.warn('[porthole] discarding unrestorable saved layout', error);
          if (storageKey) clearDashboardLayout(storageKey);
          applyDefaultLayout(event.api);
        }
      } else {
        applyDefaultLayout(event.api);
      }

      if (storageKey) {
        event.api.onDidLayoutChange(() => saveDashboardLayout(storageKey, event.api.toJSON()));
      }
    },
    [applyDefaultLayout, storageKey],
  );

  return (
    <DockviewReact
      components={components}
      onReady={onReady}
      {...(className !== undefined ? { className } : {})}
      {...(theme !== undefined ? { theme: theme as never } : {})}
    />
  );
}

/** Wipe the saved layout and reload. Wire this to a "reset layout" action. */
export function resetDashboardLayout(storageKey: string): void {
  clearDashboardLayout(storageKey);
  if (typeof window !== 'undefined') window.location.reload();
}
