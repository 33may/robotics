/**
 * The panel contract.
 *
 * A panel is a React component plus a declared config shape. The config is the
 * mechanism that makes panels *reusable rather than copied*: a camera panel
 * whose topic is config can be instantiated twice, on two topics, in one
 * dashboard. A camera panel with `"camera/wrist"` hardcoded can only be
 * duplicated by duplicating its source.
 *
 * The rule that follows: **anything that would otherwise make you copy a panel
 * belongs in its config.** Topic names, units, colour scales, axis limits,
 * frame ids. Anything that changes how the panel *works* rather than what it
 * points at belongs in code, and that is when forking is the honest answer.
 */

import type { ComponentType } from 'react';

/** Props every panel component receives. */
export interface PanelProps<C = Record<string, unknown>> {
  /** Unique per panel instance — use it to key per-instance resources. */
  readonly instanceId: string;
  /** This instance's config, merged over the definition's defaults. */
  readonly config: C;
  /** Merge a patch into this instance's config. Persisted with the layout. */
  readonly setConfig: (patch: Partial<C>) => void;
  /** Override the tab title for this instance. */
  readonly setTitle: (title: string) => void;
}

/** A panel kind, registered once and instantiable many times. */
export interface PanelDefinition<C = Record<string, unknown>> {
  /** Stable identifier used in layouts. Renaming it invalidates saved layouts. */
  readonly type: string;
  /** Default tab title. */
  readonly title: string;
  readonly component: ComponentType<PanelProps<C>>;
  /** Every config key a panel reads must have a default here. */
  readonly defaultConfig: C;
  /** One line, shown in an "add panel" menu. */
  readonly description?: string;
  /**
   * Keep the panel mounted while its tab is hidden.
   *
   * Off by default: hidden panels unmount, which is right for DOM panels and
   * frees their subscriptions. Turn it on for panels that own expensive
   * resources a remount would rebuild — a WebGL context, a decoded mesh, a
   * plot's accumulated history. The cost is that the panel keeps consuming
   * messages while invisible.
   */
  readonly keepMounted?: boolean;
}

/** One panel instance in a layout. */
export interface PanelPlacement {
  /** Instance id, unique within the layout. */
  readonly id: string;
  /** Which `PanelDefinition.type` to instantiate. */
  readonly type: string;
  readonly title?: string;
  readonly config?: Record<string, unknown>;
  /** Where to put it, relative to a previously placed instance. */
  readonly position?: {
    readonly referencePanel?: string;
    readonly direction?: 'left' | 'right' | 'above' | 'below' | 'within';
  };
}

/** A whole default layout: instances placed in order. */
export type LayoutSpec = readonly PanelPlacement[];

/**
 * A panel definition with its config type erased.
 *
 * A registry holds panels with different config shapes, and `PanelDefinition<C>`
 * is invariant in `C` because `C` appears in the component's props. There is no
 * sound supertype, so the erasure is explicit and confined to this alias rather
 * than sprinkled as casts at every call site. Panel authors never write it —
 * `definePanel` keeps their config fully typed.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyPanelDefinition = PanelDefinition<any>;
