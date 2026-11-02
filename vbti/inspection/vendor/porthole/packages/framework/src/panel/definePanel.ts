/**
 * Panel registration.
 *
 * `definePanel` exists mainly so the generic parameter is inferred from
 * `defaultConfig`, which is what makes `config` strongly typed inside a panel
 * without anyone writing a type argument by hand.
 */

import type { AnyPanelDefinition, PanelDefinition } from './types.js';

export function definePanel<C extends Record<string, unknown>>(
  definition: PanelDefinition<C>,
): PanelDefinition<C> {
  if (!/^[a-z][a-z0-9-]*$/.test(definition.type)) {
    throw new Error(
      `panel type "${definition.type}" must be kebab-case — it appears in saved layouts`,
    );
  }
  return definition;
}

/** Index definitions by type, rejecting duplicates loudly. */
export function buildPanelRegistry(
  definitions: readonly AnyPanelDefinition[],
): Map<string, AnyPanelDefinition> {
  const registry = new Map<string, AnyPanelDefinition>();
  for (const definition of definitions) {
    if (registry.has(definition.type)) {
      throw new Error(`two panels registered as "${definition.type}"`);
    }
    registry.set(definition.type, definition);
  }
  return registry;
}
