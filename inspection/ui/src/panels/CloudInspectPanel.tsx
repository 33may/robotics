/**
 * The fused object cloud, the viewpoints around it, and the picture taken from
 * whichever viewpoint you click.
 *
 * APP-OWNED PANEL. This one is not a porthole registry component and should not
 * become one: `visited / current / available / blocked / unreachable` is
 * inspection vocabulary, and the toolkit deliberately knows nothing about
 * robots. The generic half it stands on — a point cloud fed by a topic — is in
 * the toolkit already.
 *
 * Clicking sends nothing. The captured images are served over HTTP by the same
 * process that serves this bundle, so picking a viewpoint is pure frontend
 * state and works even while the robot is mid-move. See ../AGENTS.md §5, §7.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useTopicEffect, useTopicPayload } from '@porthole/framework';
import type { NdArray, PanelProps } from '@porthole/framework';
import * as THREE from 'three';

export type ViewCellState =
  | 'visited'
  | 'current'
  | 'available'
  | 'blocked'
  | 'unreachable';

export interface ViewCell {
  readonly h: number;
  readonly v: number;
  readonly state: ViewCellState;
  readonly pos: readonly [number, number, number];
  readonly gloss?: string;
  readonly step?: number;
  /** URL on this server, present iff the cell was visited. */
  readonly image?: string;
  readonly comment?: string;
}

export interface ViewsState {
  readonly center?: readonly [number, number, number];
  readonly radius?: number;
  readonly current?: readonly [number, number] | null;
  readonly cells?: readonly ViewCell[];
}

/**
 * Marker colours.
 *
 * `blocked` and `unreachable` are different on purpose. Blocked means planning
 * refused it from where the arm is standing and it may work after the next
 * move; unreachable means no IK branch exists at any roll for this object
 * position. One is transient, one is not, and painting them the same colour
 * hides that `plan_failed` is cleared after every real move.
 */
const STATE_COLOR: Record<ViewCellState, string> = {
  visited: '#7ee2a8',
  current: '#8ab4f8',
  available: '#6d737d',
  blocked: '#f2c76b',
  unreachable: '#5a3a3a',
};

export interface CloudInspectPanelConfig extends Record<string, unknown> {
  cloudTopic: string;
  cloudField: string;
  viewsTopic: string;
  pointSize: number;
  maxPoints: number;
  markerRadius: number;
  /** Width of the image pane, as a CSS length. */
  paneWidth: string;
}

export const cloudInspectPanelDefaultConfig: CloudInspectPanelConfig = {
  cloudTopic: 'cloud/fused',
  cloudField: 'points',
  viewsTopic: 'views/state',
  pointSize: 0.0025,
  maxPoints: 400_000,
  markerRadius: 0.018,
  paneWidth: '300px',
};

function cellKey(cell: { h: number; v: number }): string {
  return `h${String(cell.h).padStart(2, '0')}v${cell.v}`;
}

// ── 3D ──────────────────────────────────────────────────────────────────────

function FusedCloud({ config }: { config: CloudInspectPanelConfig }) {
  const geometry = useMemo(() => {
    const buffer = new THREE.BufferGeometry();
    buffer.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(config.maxPoints * 3), 3),
    );
    buffer.setDrawRange(0, 0);
    return buffer;
  }, [config.maxPoints]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  // Mutated in place. Allocating a BufferAttribute per cloud update produces
  // megabytes of garbage and the GC pause lands while someone is orbiting.
  useTopicEffect<Record<string, unknown>>(config.cloudTopic, (envelope) => {
    const nd = envelope.payload?.[config.cloudField] as NdArray | undefined;
    if (!nd || !(nd.data instanceof Float32Array)) return;
    const count = Math.min(nd.shape[0] ?? 0, config.maxPoints);
    const attribute = geometry.getAttribute('position') as THREE.BufferAttribute;
    (attribute.array as Float32Array).set(nd.data.subarray(0, count * 3));
    attribute.needsUpdate = true;
    geometry.setDrawRange(0, count);
    geometry.computeBoundingSphere();
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial size={config.pointSize} sizeAttenuation color="#c8d4e4" />
    </points>
  );
}

interface ViewMarkersProps {
  readonly cells: readonly ViewCell[];
  readonly radius: number;
  readonly selected: string | null;
  readonly onPick: (cell: ViewCell) => void;
}

function ViewMarkers({ cells, radius, selected, onPick }: ViewMarkersProps) {
  // 36 spheres is well under the count where instancing pays for itself, and
  // individual meshes keep per-marker raycasting free.
  return (
    <group>
      {cells.map((cell) => {
        const key = cellKey(cell);
        const picked = key === selected;
        return (
          <mesh
            key={key}
            position={[cell.pos[0], cell.pos[1], cell.pos[2]]}
            onClick={(event) => {
              event.stopPropagation();
              onPick(cell);
            }}
          >
            <sphereGeometry args={[picked ? radius * 1.6 : radius, 16, 12]} />
            <meshBasicMaterial
              color={STATE_COLOR[cell.state]}
              transparent
              opacity={cell.state === 'unreachable' ? 0.45 : 0.95}
              wireframe={picked}
            />
          </mesh>
        );
      })}
    </group>
  );
}

/** Frame the object once, when its centre first arrives. */
function CloudCameraRig({
  center,
  radius,
}: {
  center: readonly [number, number, number] | undefined;
  radius: number;
}) {
  const camera = useThree((state) => state.camera);
  const framed = useRef(false);

  useEffect(() => {
    if (!center || framed.current) return;
    framed.current = true;
    const distance = Math.max(radius, 0.2) * 2.4;
    camera.position.set(center[0] + distance, center[1] - distance, center[2] + distance * 0.6);
    camera.lookAt(center[0], center[1], center[2]);
  }, [camera, center, radius]);

  return null;
}

// ── panel ───────────────────────────────────────────────────────────────────

export function CloudInspectPanel({ config }: PanelProps<CloudInspectPanelConfig>) {
  const views = useTopicPayload<ViewsState>(config.viewsTopic);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const cells = views?.cells ?? [];
  const center = views?.center;
  const radius = views?.radius ?? 0.35;
  const selected = cells.find((cell) => cellKey(cell) === selectedKey) ?? null;

  return (
    <div className="inspection-cloud-panel">
      <div className="inspection-cloud-view">
        <Canvas
          // Z-up: OrbitControls reads object.up in its constructor, so it has
          // to be set at camera creation, not in an effect.
          camera={{ position: [0.8, -0.8, 0.6], up: [0, 0, 1], fov: 45, near: 0.005, far: 50 }}
        >
          <ambientLight intensity={1.2} />
          <CloudCameraRig center={center} radius={radius} />
          <FusedCloud config={config} />
          <ViewMarkers
            cells={cells}
            radius={config.markerRadius}
            selected={selectedKey}
            onPick={(cell) => setSelectedKey(cellKey(cell))}
          />
          {center ? (
            <mesh position={[center[0], center[1], center[2]]}>
              <sphereGeometry args={[0.006, 10, 8]} />
              <meshBasicMaterial color="#f2777a" />
            </mesh>
          ) : null}
          <axesHelper args={[0.1]} />
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.15}
            target={center ? [center[0], center[1], center[2]] : [0, 0, 0]}
          />
        </Canvas>
      </div>

      <aside className="inspection-cloud-pane" style={{ width: config.paneWidth }}>
        {selected ? (
          <>
            <div className="inspection-cloud-pane-head">
              <span className="inspection-cell-id">h{selected.h} v{selected.v}</span>
              <span className="inspection-cell-state" data-state={selected.state}>
                {selected.state}
              </span>
            </div>
            {selected.image ? (
              <img className="inspection-cloud-shot" src={selected.image} alt="" />
            ) : (
              <div className="inspection-cloud-empty">not visited</div>
            )}
            {selected.gloss ? <p className="inspection-cell-gloss">{selected.gloss}</p> : null}
            {selected.comment ? (
              <p className="inspection-cell-comment">{selected.comment}</p>
            ) : null}
            {selected.step !== undefined ? (
              <p className="inspection-cell-step">step {selected.step}</p>
            ) : null}
          </>
        ) : null}
      </aside>
    </div>
  );
}
