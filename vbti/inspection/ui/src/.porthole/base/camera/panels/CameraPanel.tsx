/**
 * A single live image panel.
 *
 * REGISTRY COMPONENT. This file is meant to be copied into your app and edited.
 * See docs/component-model.md before you change it: small tweaks belong in
 * `config`, real behaviour changes are a fork, and `porthole update` can still
 * merge upstream fixes into a fork as long as you keep the recorded base.
 *
 * The decode path lives in the framework (`useImageTopicCanvas`) rather than
 * here, because three separate non-obvious behaviours — drop stale frames, copy
 * bytes out of the receive buffer, use createImageBitmap over object URLs —
 * are needed identically by every panel that shows imagery, and each one fails
 * as a slow leak rather than a visible bug. Panels own layout; the framework
 * owns the wire.
 *
 * For more than one view at a time, use MediaGridPanel; it is the same decode
 * path with a layout solver on top.
 */

import { useRef } from 'react';
import { useImageTopicCanvas } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';

export interface CameraPanelConfig extends Record<string, unknown> {
  /** Topic publishing `$img` payloads. */
  topic: string;
  /** Fit the frame inside the panel, or fill and crop. */
  fit: 'contain' | 'cover';
}

export const cameraPanelDefaultConfig: CameraPanelConfig = {
  topic: 'camera/wrist',
  fit: 'contain',
};

export function CameraPanel({ config }: PanelProps<CameraPanelConfig>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useImageTopicCanvas(config.topic, canvasRef);

  return (
    <div className="porthole-camera-panel">
      <canvas ref={canvasRef} data-fit={config.fit} />
    </div>
  );
}
