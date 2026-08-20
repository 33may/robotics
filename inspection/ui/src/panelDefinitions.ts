/**
 * Panel registration for the inspection UI.
 *
 * Three of these come from porthole's registry and were copied in by
 * `porthole add` — they are ordinary source files in this repo now, editable,
 * and `porthole update` three-way-merges upstream fixes into them. The other
 * two, `cloud` and `actions`, are app-owned because they encode viewsphere
 * vocabulary (view/request, view/confirm, the cell state machine).
 */

import { definePanel } from '@porthole/framework';

import { ActionsPanel } from './panels/ActionsPanel.js';
import { CameraPanel, cameraPanelDefaultConfig } from './panels/CameraPanel.js';
import {
  CloudInspectPanel,
  cloudInspectPanelDefaultConfig,
} from './panels/CloudInspectPanel.js';
import { EventLogPanel, eventLogPanelDefaultConfig } from './panels/EventLogPanel.js';
import { SceneViewPanel, sceneViewPanelDefaultConfig } from './panels/SceneViewPanel.js';
import type { SceneViewPanelConfig } from './panels/SceneViewPanel.js';

/**
 * Annotated, not inferred.
 *
 * `definePanel` infers its config type from `defaultConfig`, so an inline
 * object literal widens to *its own* shape — `[1.4, -1.4, 1.1]` becomes the
 * tuple type `[1.4, -1.4, 1.1]`, which is not assignable to the panel's
 * `readonly [number, number, number]`. Naming the type is what keeps `config`
 * correctly typed inside the panel. Do this for every panel you re-configure.
 */
const cellPanelConfig: SceneViewPanelConfig = {
  ...sceneViewPanelDefaultConfig,
  // The workcell, not a tabletop: back off far enough to see the table, the
  // posts and the shelf envelope at once.
  cameraPosition: [1.4, -1.4, 1.1],
  cameraTarget: [0.0, -0.35, 0.25],
  originAxes: 0.15,
};

export const inspectionPanelDefinitions = [
  definePanel({
    type: 'cell',
    title: 'cell',
    component: SceneViewPanel,
    defaultConfig: cellPanelConfig,
    description: 'Robot, workcell, object and viewpoint markers',
    // A WebGL context plus seven parsed meshes is expensive to rebuild; keep it
    // alive while the tab is hidden.
    keepMounted: true,
  }),
  definePanel({
    type: 'camera',
    title: 'camera',
    component: CameraPanel,
    defaultConfig: { ...cameraPanelDefaultConfig, topic: 'camera/wrist' },
    description: 'Live wrist camera',
  }),
  definePanel({
    type: 'cloud',
    title: 'cloud',
    component: CloudInspectPanel,
    defaultConfig: cloudInspectPanelDefaultConfig,
    description: 'Fused cloud, viewpoints, and the shot from the picked one',
    keepMounted: true,
  }),
  definePanel({
    type: 'log',
    title: 'log',
    component: EventLogPanel,
    defaultConfig: eventLogPanelDefaultConfig,
    description: 'Run event log',
  }),
  definePanel({
    type: 'actions',
    title: 'actions',
    component: ActionsPanel,
    defaultConfig: {},
    description: 'Survey + view grid + Stop + Exit',
  }),
];
