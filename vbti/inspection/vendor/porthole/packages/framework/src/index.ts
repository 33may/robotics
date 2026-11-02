/**
 * @porthole/framework — the tier you depend on and never edit.
 *
 * Everything exported here is stable API. If you find yourself wanting to
 * change something in this package to build a panel, that is a signal the
 * panel needs an escape hatch, not that the framework needs a special case —
 * open the discussion rather than forking this package.
 *
 * Components and panels are the other tier: they arrive as source you own.
 */

export { BusClient } from './bus/client.js';
export type { BusClientOptions } from './bus/client.js';
export { decodeEnvelope, revivePayload } from './bus/decode.js';
export {
  PROTOCOL_VERSION,
  isEncodedImage,
  isNdArray,
} from './bus/types.js';
export type {
  AnyTypedArray,
  BusCommand,
  ConnectionStatus,
  EncodedImage,
  BusEnvelope,
  BusHello,
  NdArray,
  NdDtype,
  TopicQoS,
  TopicDeclaration,
} from './bus/types.js';

export { buildPanelRegistry, definePanel } from './panel/definePanel.js';
export type {
  LayoutSpec,
  PanelDefinition,
  PanelPlacement,
  PanelProps,
} from './panel/types.js';

export { PortholeDashboard, resetDashboardLayout } from './shell/PortholeDashboard.js';
export type { PortholeDashboardProps } from './shell/PortholeDashboard.js';

export { BusProvider, useBus } from './react/context.js';
export type { BusProviderProps } from './react/context.js';
export {
  useBusStatus,
  useSendCommand,
  useBusHello,
  useTopicPayload,
  useTopic,
  useTopicEffect,
} from './react/hooks.js';
export { useImageTopicCanvas } from './react/useImageTopicCanvas.js';
export type { ImageTopicCanvasOptions } from './react/useImageTopicCanvas.js';
