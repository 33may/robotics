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
import {
  ImageChainPanel,
  imageChainPanelDefaultConfig,
} from './panels/ImageChainPanel.js';
import { SceneViewPanel, sceneViewPanelDefaultConfig } from './panels/SceneViewPanel.js';
import { TracePanel, tracePanelDefaultConfig } from './panels/TracePanel.js';
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
  // Clicking a viewsphere marker starts the PREVIEW for that cell — the same
  // `view/request` the actions panel sends, so there is one command path with
  // two buttons. `view/confirm` is deliberately NOT here: the press that
  // actually moves the arm stays in one place (Anton 2026-08-24).
  pickCommand: { prefix: 'views/', cmd: 'view/request' },
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
    type: 'chain',
    title: 'chain',
    component: ImageChainPanel,
    defaultConfig: imageChainPanelDefaultConfig,
    description: 'Capture → prompt → mask → accepted points, for the last view',
  }),
  definePanel({
    type: 'log',
    title: 'log',
    component: EventLogPanel,
    defaultConfig: eventLogPanelDefaultConfig,
    description: 'Run event log',
    // Kept mounted so it accumulates events while it sits behind `actions`.
    // A log that only starts recording when you look at it is not a log —
    // and it is the tab you switch to precisely when something went wrong.
    keepMounted: true,
  }),
  definePanel({
    type: 'actions',
    title: 'actions',
    component: ActionsPanel,
    defaultConfig: {},
    description: 'Survey + view grid + Stop + Exit',
  }),
  definePanel({
    type: 'trace',
    title: 'trace',
    component: TracePanel,
    defaultConfig: tracePanelDefaultConfig,
    description: 'Every stage of the agentic loop: thoughts, tool calls, images, findings',
  }),
];
