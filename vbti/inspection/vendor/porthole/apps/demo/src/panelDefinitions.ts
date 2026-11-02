/**
 * Panel definitions for the demo app.
 *
 * Note what this file demonstrates: the app registers panels, the registry only
 * supplies components. That separation is why `camera` can be instantiated
 * twice on two different topics without duplicating a line of panel source.
 */

import { definePanel } from '@porthole/framework';
import {
  CameraPanel,
  cameraPanelDefaultConfig,
} from '@porthole/registry/panels/CameraPanel.js';
import {
  CommandPanel,
  commandPanelDefaultConfig,
} from '@porthole/registry/panels/CommandPanel.js';
import {
  EventLogPanel,
  eventLogPanelDefaultConfig,
} from '@porthole/registry/panels/EventLogPanel.js';
import {
  MediaGridPanel,
  mediaGridPanelDefaultConfig,
} from '@porthole/registry/panels/MediaGridPanel.js';
import type { MediaGridPanelConfig } from '@porthole/registry/panels/MediaGridPanel.js';
import {
  PointCloudPanel,
  pointCloudPanelDefaultConfig,
} from '@porthole/registry/panels/PointCloudPanel.js';
import {
  SceneViewPanel,
  sceneViewPanelDefaultConfig,
} from '@porthole/registry/panels/SceneViewPanel.js';

// Annotated rather than inferred: without the annotation the object literal
// widens to its own shape, and `definePanel` then infers a config type that is
// structurally incompatible with the panel's own (readonly arrays, in
// particular). Naming the type is what keeps `config` correct inside the panel.
const demoMediaGridConfig: MediaGridPanelConfig = {
  ...mediaGridPanelDefaultConfig,
  sources: [
    { topic: 'camera/wrist', label: 'wrist' },
    { topic: 'camera/scene', label: 'scene' },
    { topic: 'camera/table', label: 'table' },
    { topic: 'camera/shelf', label: 'shelf' },
  ],
};

export const demoPanelDefinitions = [
  definePanel({
    type: 'scene-view',
    title: 'cell',
    component: SceneViewPanel,
    defaultConfig: sceneViewPanelDefaultConfig,
    description: 'Backend-driven 3D scene with animated poses',
    // Meshes are parsed once and uploaded to the GPU; a remount on every tab
    // switch would re-clone and re-upload them.
    keepMounted: true,
  }),
  definePanel({
    type: 'media-grid',
    title: 'views',
    component: MediaGridPanel,
    defaultConfig: demoMediaGridConfig,
    description: 'Grid of live image/video cells',
  }),
  definePanel({
    type: 'point-cloud',
    title: 'scene',
    component: PointCloudPanel,
    defaultConfig: pointCloudPanelDefaultConfig,
    description: 'Orbitable 3D point cloud',
    // A WebGL context and its buffers are expensive to rebuild on every tab
    // switch, so this panel stays mounted while hidden.
    keepMounted: true,
  }),
  definePanel({
    type: 'camera',
    title: 'camera',
    component: CameraPanel,
    defaultConfig: cameraPanelDefaultConfig,
    description: 'Live JPEG stream',
  }),
  definePanel({
    type: 'command',
    title: 'decide',
    component: CommandPanel,
    defaultConfig: commandPanelDefaultConfig,
    description: 'Pick an option, add a note, send a command',
  }),
  definePanel({
    type: 'event-log',
    title: 'events',
    component: EventLogPanel,
    defaultConfig: eventLogPanelDefaultConfig,
    description: 'Streaming event log',
  }),
];
