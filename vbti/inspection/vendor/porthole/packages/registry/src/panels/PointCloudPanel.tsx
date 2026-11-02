/**
 * Orbitable 3D point cloud.
 *
 * REGISTRY COMPONENT — copy and edit freely; see docs/component-model.md.
 * This is the panel most likely to be forked, because "what else is in the
 * scene" is inherently app-specific. Fork it, add your meshes, and keep the
 * recorded base so `porthole update` can still merge upstream fixes.
 *
 * The geometry is mutated in place and never re-created: allocating a
 * BufferAttribute per frame at 30 Hz produces megabytes a second of garbage,
 * and the GC pause lands exactly when the human is dragging to orbit.
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { Grid, OrbitControls } from '@react-three/drei';
import { useTopicEffect } from '@porthole/framework';
import type { NdArray, PanelProps } from '@porthole/framework';
import * as THREE from 'three';

export interface PointCloudPanelConfig extends Record<string, unknown> {
  /** Topic carrying `{ points: $nd }` with shape [N, 3]. */
  topic: string;
  /** Field within the payload holding the array. */
  field: string;
  pointSize: number;
  /** Upper bound on points held on the GPU. Extra points are ignored. */
  maxPoints: number;
  showGrid: boolean;
}

export const pointCloudPanelDefaultConfig: PointCloudPanelConfig = {
  topic: 'scene/cloud',
  field: 'points',
  pointSize: 0.004,
  maxPoints: 200_000,
  showGrid: true,
};

interface CloudPayload {
  [key: string]: unknown;
}

function PointCloud({ config }: { config: PointCloudPanelConfig }) {
  const pointsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const buffer = new THREE.BufferGeometry();
    const positions = new Float32Array(config.maxPoints * 3);
    buffer.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    buffer.setDrawRange(0, 0);
    return buffer;
  }, [config.maxPoints]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useTopicEffect<CloudPayload>(config.topic, (envelope) => {
    const nd = envelope.payload?.[config.field] as NdArray | undefined;
    if (!nd || !(nd.data instanceof Float32Array)) return;

    const count = Math.min(nd.shape[0] ?? 0, config.maxPoints);
    const attribute = geometry.getAttribute('position') as THREE.BufferAttribute;
    (attribute.array as Float32Array).set(nd.data.subarray(0, count * 3));
    attribute.needsUpdate = true;
    geometry.setDrawRange(0, count);
    geometry.computeBoundingSphere();
  });

  return (
    <points ref={pointsRef} geometry={geometry} frustumCulled={false}>
      <pointsMaterial size={config.pointSize} sizeAttenuation color="#8ab4f8" />
    </points>
  );
}

export function PointCloudPanel({ config }: PanelProps<PointCloudPanelConfig>) {
  return (
    <div className="porthole-pointcloud-panel">
      <Canvas camera={{ position: [0.4, 0.3, 0.4], fov: 50, near: 0.01, far: 50 }}>
        <ambientLight intensity={0.8} />
        <PointCloud config={config} />
        {config.showGrid ? (
          <Grid
            args={[2, 2]}
            cellSize={0.05}
            sectionSize={0.25}
            cellColor="#2a2f3a"
            sectionColor="#39404f"
            fadeDistance={4}
            infiniteGrid
          />
        ) : null}
        <axesHelper args={[0.1]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.15} />
      </Canvas>
    </div>
  );
}
