/**
 * A 3D scene driven entirely by the backend.
 *
 * REGISTRY COMPONENT. Copy it into your app and edit it; see
 * docs/component-model.md. The contract it implements is
 * docs/scene-protocol.md — read that before changing anything here, because
 * the Python side is written against it.
 *
 * ## What this panel does and does not do
 *
 * It renders transforms. It does not compute them. No URDF parsing, no forward
 * kinematics, no joint interpolation: the backend publishes where every node
 * is, at whatever rate it likes, and this draws it. A second kinematic
 * implementation in the browser is a second thing that can disagree with the
 * planner about where the robot is standing.
 *
 * ## Two coordinate traps, both load-bearing
 *
 * 1. **The world is Z-up.** three.js is Y-up. Rather than rotating the scene —
 *    which would make browser coordinates differ from the backend's — the
 *    camera's up vector is set to (0, 0, 1). `OrbitControls` reads `object.up`
 *    *in its constructor* and never re-reads it, so the up vector must be set
 *    when the camera is created, which is why it is a `<Canvas camera>` prop
 *    and not an effect.
 * 2. **ColladaLoader rotates Z-up assets.** A `.dae` declaring
 *    `<up_axis>Z_UP</up_axis>` — which every robot mesh does — comes back with
 *    `scene.rotation.x = -π/2` applied, because the loader converts to its own
 *    Y-up world. In a Z-up world that must be undone, or every mesh lies on its
 *    side while every primitive is correct.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Canvas } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { Grid, OrbitControls } from '@react-three/drei';
import { useBus, useTopicEffect, useTopicPayload } from '@porthole/framework';
import type { NdArray, PanelProps } from '@porthole/framework';
import * as THREE from 'three';

// ── protocol types (docs/scene-protocol.md) ─────────────────────────────────

export type SceneGeometry =
  | { readonly kind: 'box'; readonly size: readonly [number, number, number] }
  | { readonly kind: 'sphere'; readonly radius: number }
  | { readonly kind: 'cylinder'; readonly radius: number; readonly length: number }
  | { readonly kind: 'axes'; readonly size: number }
  | {
      readonly kind: 'mesh';
      readonly url: string;
      readonly scale?: readonly [number, number, number];
    }
  | {
      readonly kind: 'points';
      /** Topic streaming `{ [field]: $nd float32 [N,3] }`. */
      readonly topic: string;
      readonly field?: string;
      readonly size?: number;
      readonly maxPoints?: number;
    };

export interface SceneMaterial {
  readonly color?: string;
  readonly opacity?: number;
  readonly wireframe?: boolean;
  readonly side?: 'front' | 'back' | 'double';
}

export interface SceneNode {
  readonly path: string;
  readonly geometry: SceneGeometry;
  readonly material?: SceneMaterial;
  readonly transform?: readonly number[];
  readonly visible?: boolean;
}

export interface SceneDescription {
  readonly frame?: string;
  readonly nodes: readonly SceneNode[];
}

export interface ScenePoseFrame {
  readonly names?: readonly string[];
  readonly transforms?: NdArray;
  readonly overrides?: Readonly<Record<string, SceneMaterial & { visible?: boolean }>>;
  readonly background?: readonly [string, string];
}

// ── config ──────────────────────────────────────────────────────────────────

export interface SceneViewPanelConfig extends Record<string, unknown> {
  descriptionTopic: string;
  posesTopic: string;
  /** Camera start position in scene coordinates (metres, Z-up). */
  cameraPosition: readonly [number, number, number];
  /** Point the camera orbits around. */
  cameraTarget: readonly [number, number, number];
  showGrid: boolean;
  /** Length of the world-origin triad, metres. 0 hides it. */
  originAxes: number;
  /** Path prefixes to hide, e.g. `["shelf"]`. Toggled by the overlay. */
  hiddenGroups: readonly string[];
  /** Show the group visibility overlay. */
  showGroupToggles: boolean;
  /**
   * Make described nodes clickable. A click on a node whose path starts with
   * `prefix` sends `{cmd, target}` on the bus, where target is the path with
   * the prefix removed. Null (the default) leaves the scene read-only.
   */
  pickCommand: { readonly prefix: string; readonly cmd: string } | null;
}

export const sceneViewPanelDefaultConfig: SceneViewPanelConfig = {
  descriptionTopic: 'scene/description',
  posesTopic: 'scene/poses',
  cameraPosition: [1.6, -1.6, 1.2],
  cameraTarget: [0, 0, 0.3],
  showGrid: true,
  originAxes: 0.2,
  hiddenGroups: [],
  showGroupToggles: true,
  pickCommand: null,
};

// ── mesh loading ────────────────────────────────────────────────────────────

/**
 * Parsed meshes, keyed by URL and shared across every node and every rebuild.
 *
 * Parsing the UR5e's seven Collada files takes about a second and produces the
 * same result every time. A scene description arrives whenever the world is
 * re-modelled, so without this cache every edit to a cell file would re-parse
 * 9 MB. Clones share geometry, so the cache also keeps one copy on the GPU.
 */
const sceneMeshCache = new Map<string, Promise<THREE.Object3D>>();

async function loadSceneMesh(url: string): Promise<THREE.Object3D> {
  const cached = sceneMeshCache.get(url);
  if (cached) return cached;

  const pending = (async () => {
    const extension = url.split('?')[0]?.split('.').pop()?.toLowerCase() ?? '';
    switch (extension) {
      case 'dae': {
        const { ColladaLoader } = await import('three/examples/jsm/loaders/ColladaLoader.js');
        const collada = await new ColladaLoader().loadAsync(url);
        if (!collada?.scene) throw new Error(`no scene in ${url}`);
        // Undo the loader's Z_UP → Y_UP correction; see the header note.
        collada.scene.rotation.set(0, 0, 0);
        return collada.scene as unknown as THREE.Object3D;
      }
      case 'glb':
      case 'gltf': {
        const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
        const gltf = await new GLTFLoader().loadAsync(url);
        return gltf.scene;
      }
      case 'stl': {
        const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js');
        const geometry = await new STLLoader().loadAsync(url);
        return new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: 0xb0b6c0 }));
      }
      case 'obj': {
        const { OBJLoader } = await import('three/examples/jsm/loaders/OBJLoader.js');
        return (await new OBJLoader().loadAsync(url)) as unknown as THREE.Object3D;
      }
      default:
        throw new Error(`unsupported mesh extension "${extension}" for ${url}`);
    }
  })();

  sceneMeshCache.set(url, pending);
  return pending;
}

// ── building ────────────────────────────────────────────────────────────────

function buildMaterial(spec: SceneMaterial | undefined): THREE.MeshLambertMaterial {
  const opacity = spec?.opacity ?? 1;
  return new THREE.MeshLambertMaterial({
    color: new THREE.Color(spec?.color ?? '#8a92a0'),
    opacity,
    transparent: opacity < 1,
    wireframe: spec?.wireframe ?? false,
    side:
      spec?.side === 'double'
        ? THREE.DoubleSide
        : spec?.side === 'back'
          ? THREE.BackSide
          : THREE.FrontSide,
    depthWrite: opacity >= 1,
  });
}

function buildPrimitive(geometry: SceneGeometry): THREE.BufferGeometry | null {
  switch (geometry.kind) {
    case 'box':
      // Full extents, not half-extents: that is what a tape measure and a URDF
      // record. pinocchio's Box.halfSide is the odd one out — double it there.
      return new THREE.BoxGeometry(...geometry.size);
    case 'sphere':
      return new THREE.SphereGeometry(geometry.radius, 24, 16);
    case 'cylinder': {
      const cylinder = new THREE.CylinderGeometry(
        geometry.radius,
        geometry.radius,
        geometry.length,
        24,
      );
      // three builds cylinders along +Y; every robotics convention uses +Z.
      cylinder.rotateX(Math.PI / 2);
      return cylinder;
    }
    default:
      return null;
  }
}

/**
 * Give a node its own materials and remember their described values.
 *
 * Per-frame overrides mutate materials in place, so nodes must not share them —
 * a cloned mesh shares its source's materials by default, and tinting one link
 * red would tint every link built from the same file. Stashing the base values
 * is what makes an override revertible without rebuilding the node.
 */
function isolateMaterials(node: THREE.Object3D, override?: SceneMaterial): void {
  node.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    const material = override
      ? buildMaterial(override)
      : ((source as THREE.Material).clone() as THREE.MeshLambertMaterial);
    mesh.material = material;
    mesh.userData.baseColor = (material as THREE.MeshLambertMaterial).color?.clone();
    mesh.userData.baseOpacity = material.opacity;
    mesh.userData.baseTransparent = material.transparent;
  });
}

function applyMatrix(object: THREE.Object3D, rowMajor: ArrayLike<number>, offset = 0): void {
  // Matrix4.set takes row-major arguments (fromArray is the column-major one),
  // which is exactly the layout a C-contiguous numpy [4,4] arrives in.
  object.matrix.set(
    rowMajor[offset + 0]!, rowMajor[offset + 1]!, rowMajor[offset + 2]!, rowMajor[offset + 3]!,
    rowMajor[offset + 4]!, rowMajor[offset + 5]!, rowMajor[offset + 6]!, rowMajor[offset + 7]!,
    rowMajor[offset + 8]!, rowMajor[offset + 9]!, rowMajor[offset + 10]!, rowMajor[offset + 11]!,
    rowMajor[offset + 12]!, rowMajor[offset + 13]!, rowMajor[offset + 14]!, rowMajor[offset + 15]!,
  );
  object.matrixWorldNeedsUpdate = true;
}

function disposeSceneNode(node: THREE.Object3D): void {
  node.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    // Geometry is shared with the mesh cache for loaded files; only primitives
    // own theirs, and those are marked at build time.
    if (mesh.userData.ownsGeometry) mesh.geometry.dispose();
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) material.dispose();
  });
}

// ── the scene graph ─────────────────────────────────────────────────────────

interface SceneGraphProps {
  readonly config: SceneViewPanelConfig;
  readonly onGroups: (groups: readonly string[]) => void;
  /**
   * The panel container, tinted directly from the pose stream.
   *
   * The canvas is transparent and the verdict colour lives in CSS, so a
   * backend flipping the background 30 times a second costs one style write —
   * no React render, no three.js material churn.
   */
  readonly backgroundRef: RefObject<HTMLDivElement | null>;
}

function SceneGraph({ config, onGroups, backgroundRef }: SceneGraphProps) {
  const bus = useBus();
  const rootRef = useRef<THREE.Group>(null);
  const nodesRef = useRef(new Map<string, THREE.Object3D>());
  const overriddenRef = useRef(new Set<string>());
  /**
   * The newest pose frame, kept so a node built AFTER it arrived can still be
   * placed.
   *
   * Mesh nodes are `await`ed — a `.dae` takes hundreds of milliseconds — so a
   * pose frame published while they load reaches the handler before those
   * nodes exist. Without this the frame is lost for them: a backend that
   * publishes poses continuously self-heals on the next frame, but one that
   * publishes a single frame per state change (a robot parked at idle, say)
   * leaves every mesh stranded at its described placement, typically stacked
   * on the origin, until something moves. Found on a real robot: the tool
   * primitives were placed correctly while all seven arm meshes sat in a pile.
   */
  const lastPoseRef = useRef<{
    names: readonly string[];
    data: Float32Array;
    overrides: NonNullable<ScenePoseFrame['overrides']>;
  } | null>(null);
  // Points nodes stream their data from another topic; their subscriptions are
  // keyed by node path so a rebuild that drops a node also drops its listener.
  const cloudSubsRef = useRef(new Map<string, () => void>());

  function dropCloudSubscription(path: string): void {
    cloudSubsRef.current.get(path)?.();
    cloudSubsRef.current.delete(path);
  }

  /** Place a freshly built node using the newest pose frame, if it named it. */
  function applyLastPose(object: THREE.Object3D, path: string): void {
    const last = lastPoseRef.current;
    if (!last) return;
    const index = last.names.indexOf(path);
    if (index >= 0) applyMatrix(object, last.data, index * 16);

    // Per-frame material overrides are lost the same way a transform is, and
    // a missed one is a red "this link is colliding" tint that never appears.
    const override = last.overrides[path];
    if (!override) return;
    overriddenRef.current.add(path);
    if (override.visible !== undefined) object.visible = override.visible;
    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const material = mesh.material as THREE.MeshLambertMaterial;
      if (override.color !== undefined) material.color.set(override.color);
      if (override.opacity !== undefined) {
        material.opacity = override.opacity;
        material.transparent = override.opacity < 1;
      }
    });
  }

  // The description is structural and arrives rarely, so a reactive
  // subscription is right here — it genuinely changes the scene graph, and the
  // build is async. Poses are the hot path and use useTopicEffect below; that
  // is the split docs/architecture.md is describing.
  const description = useTopicPayload<SceneDescription>(config.descriptionTopic);

  useEffect(() => {
    const root = rootRef.current;
    if (!root || !description?.nodes) return;

    let cancelled = false;
    const nodes = nodesRef.current;
    const wanted = new Set(description.nodes.map((node) => node.path));

    for (const [path, object] of nodes) {
      if (wanted.has(path)) continue;
      root.remove(object);
      disposeSceneNode(object);
      dropCloudSubscription(path);
      nodes.delete(path);
    }

    async function build(spec: SceneNode): Promise<void> {
      const existing = nodes.get(spec.path);
      if (existing) {
        root!.remove(existing);
        disposeSceneNode(existing);
        dropCloudSubscription(spec.path);
        nodes.delete(spec.path);
      }

      let object: THREE.Object3D;
      if (spec.geometry.kind === 'points') {
        const { topic, field = 'points', size = 0.003, maxPoints = 200_000 } = spec.geometry;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          'position',
          new THREE.BufferAttribute(new Float32Array(maxPoints * 3), 3),
        );
        geometry.setDrawRange(0, 0);

        const points = new THREE.Points(
          geometry,
          new THREE.PointsMaterial({
            size,
            sizeAttenuation: true,
            color: new THREE.Color(spec.material?.color ?? '#9ecbff'),
          }),
        );
        points.frustumCulled = false;
        points.userData.ownsGeometry = true;

        // Subscribed imperatively rather than through a hook: the node set is
        // data, so the number of clouds is not known at render time and cannot
        // drive a fixed number of hook calls.
        cloudSubsRef.current.set(
          spec.path,
          bus.subscribe(topic, (envelope) => {
            const payload = envelope.payload as Record<string, unknown> | undefined;
            const nd = payload?.[field] as NdArray | undefined;
            if (!nd || !(nd.data instanceof Float32Array)) return;
            const count = Math.min(nd.shape[0] ?? 0, maxPoints);
            const attribute = geometry.getAttribute('position') as THREE.BufferAttribute;
            (attribute.array as Float32Array).set(nd.data.subarray(0, count * 3));
            attribute.needsUpdate = true;
            geometry.setDrawRange(0, count);
            geometry.computeBoundingSphere();
          }),
        );
        object = points;
      } else if (spec.geometry.kind === 'mesh') {
        const source = await loadSceneMesh(spec.geometry.url);
        if (cancelled) return;
        object = source.clone(true);
        if (spec.geometry.scale) object.scale.set(...spec.geometry.scale);
        isolateMaterials(object, spec.material);
      } else if (spec.geometry.kind === 'axes') {
        object = new THREE.AxesHelper(spec.geometry.size);
      } else {
        const geometry = buildPrimitive(spec.geometry);
        if (!geometry) return;
        const mesh = new THREE.Mesh(geometry, buildMaterial(spec.material));
        mesh.userData.ownsGeometry = true;
        isolateMaterials(mesh, spec.material);
        object = mesh;
      }

      object.name = spec.path;
      object.matrixAutoUpdate = false;
      if (spec.transform?.length === 16) applyMatrix(object, spec.transform);
      else object.matrix.identity();
      object.visible = spec.visible ?? true;
      object.userData.describedVisible = object.visible;

      // A pose frame that arrived while this node was loading was skipped —
      // catch up to it now, so an async mesh ends up where a synchronous
      // primitive already is. The described transform above is the fallback;
      // the pose stream wins, exactly as it does for a node built in time.
      applyLastPose(object, spec.path);

      // The scene path travels with the object so a pointer hit can name
      // what it hit: a raycast lands on a leaf mesh, several levels below the
      // node, and walking up to a tagged ancestor is what turns that into an
      // identity the backend published and can parse back.
      object.userData.portholePath = spec.path;
      nodes.set(spec.path, object);
      root!.add(object);
    }

    void Promise.all(description.nodes.map((spec) => build(spec).catch(reportBuildFailure)));
    onGroups([...new Set(description.nodes.map((node) => node.path.split('/')[0]!))]);

    return () => {
      cancelled = true;
    };
  }, [description, onGroups, bus]);

  // Dispose everything on unmount, not on every description.
  useEffect(() => {
    const nodes = nodesRef.current;
    const subscriptions = cloudSubsRef.current;
    return () => {
      for (const object of nodes.values()) disposeSceneNode(object);
      nodes.clear();
      for (const unsubscribe of subscriptions.values()) unsubscribe();
      subscriptions.clear();
    };
  }, []);

  useTopicEffect<ScenePoseFrame>(config.posesTopic, (envelope) => {
    const nodes = nodesRef.current;
    const { names, transforms, overrides, background } = envelope.payload ?? {};

    const container = backgroundRef.current;
    if (container) {
      container.style.background = background
        ? `linear-gradient(180deg, ${background[0]} 0%, ${background[1]} 100%)`
        : '';
    }

    if (names && transforms?.data instanceof Float32Array) {
      const data = transforms.data;
      for (let index = 0; index < names.length; index += 1) {
        const object = nodes.get(names[index]!);
        // Not necessarily a backend bug: a mesh node still loading is not in
        // the map yet. `applyLastPose` places it once its build finishes.
        if (!object) continue;
        applyMatrix(object, data, index * 16);
      }
      // Copied, not referenced: the decoder hands out views into the message
      // buffer, and keeping one alive would pin that whole message. A frame is
      // 16 floats per node — copying it is cheaper than the retention.
      lastPoseRef.current = {
        names,
        data: data.slice(),
        overrides: overrides ?? {},
      };
    }

    // Overrides are not sticky: anything overridden last frame and not this
    // frame reverts, so a backend cannot forget to clear a red tint.
    const stillOverridden = new Set<string>();
    for (const [path, spec] of Object.entries(overrides ?? {})) {
      const object = nodes.get(path);
      if (!object) continue;
      stillOverridden.add(path);
      if (spec.visible !== undefined) object.visible = spec.visible;
      object.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        const material = mesh.material as THREE.MeshLambertMaterial;
        if (spec.color !== undefined) material.color.set(spec.color);
        if (spec.opacity !== undefined) {
          material.opacity = spec.opacity;
          material.transparent = spec.opacity < 1;
        }
      });
    }
    for (const path of overriddenRef.current) {
      if (stillOverridden.has(path)) continue;
      const object = nodes.get(path);
      if (!object) continue;
      object.visible = object.userData.describedVisible ?? true;
      object.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        const material = mesh.material as THREE.MeshLambertMaterial;
        if (mesh.userData.baseColor) material.color.copy(mesh.userData.baseColor);
        material.opacity = mesh.userData.baseOpacity ?? 1;
        material.transparent = mesh.userData.baseTransparent ?? false;
      });
    }
    overriddenRef.current = stillOverridden;
  });

  // Group visibility is config, applied after every render so it survives a
  // rebuild without the build path having to know about it.
  useEffect(() => {
    const hidden = new Set(config.hiddenGroups);
    for (const [path, object] of nodesRef.current) {
      const group = path.split('/')[0]!;
      object.visible = !hidden.has(group) && (object.userData.describedVisible ?? true);
    }
  });

  /**
   * Forward a click on a described node as a command.
   *
   * Deliberately dumb: this panel does not know what a viewsphere cell is. It
   * strips the configured prefix and sends the REST of the path — the very id
   * the backend minted when it published the marker — so the vocabulary stays
   * on the backend side of the bus and `_normalize_target` parses its own
   * naming. `pickCommand` defaults to null, so a scene view is read-only
   * unless an app opts it in.
   */
  function handlePick(event: ThreeEvent<MouseEvent>): void {
    const pick = config.pickCommand;
    if (!pick) return;
    for (let obj: THREE.Object3D | null = event.object; obj; obj = obj.parent) {
      const path = obj.userData?.portholePath as string | undefined;
      if (path === undefined) continue;
      if (!path.startsWith(pick.prefix)) return;   // a described node, not a target
      event.stopPropagation();                     // don't also orbit the camera
      bus.send({ cmd: pick.cmd, target: path.slice(pick.prefix.length) });
      return;
    }
  }

  return <group ref={rootRef} onClick={handlePick} />;
}

function reportBuildFailure(error: unknown): void {
  // A scene that is missing one mesh is far more useful than a blank canvas,
  // so a failed node is reported and skipped rather than rejecting the batch.
  console.error('[porthole] scene node failed to build:', error);
}

// ── panel ───────────────────────────────────────────────────────────────────

export function SceneViewPanel({ config, setConfig }: PanelProps<SceneViewPanelConfig>) {
  const [groups, setGroups] = useState<readonly string[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const hidden = useMemo(() => new Set(config.hiddenGroups), [config.hiddenGroups]);

  function toggleGroup(group: string): void {
    const next = new Set(hidden);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    setConfig({ hiddenGroups: [...next] });
  }

  return (
    <div className="porthole-scene-panel" ref={containerRef}>
      <Canvas
        // `up` must be set at camera creation: OrbitControls snapshots
        // object.up in its constructor and never re-reads it.
        camera={{
          position: [...config.cameraPosition],
          up: [0, 0, 1],
          fov: 45,
          near: 0.01,
          far: 100,
        }}
        // Transparent so the verdict background is CSS on the container, which
        // costs one style write per frame instead of a scene mutation.
        gl={{ alpha: true, antialias: true }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={1.1} />
        <directionalLight position={[2, -2, 4]} intensity={1.4} />
        <directionalLight position={[-3, 3, 1]} intensity={0.5} />

        <SceneGraph config={config} onGroups={setGroups} backgroundRef={containerRef} />

        {config.showGrid ? (
          <Grid
            // drei's Grid lies in XZ; a quarter turn about X puts it in the
            // XY ground plane of a Z-up world.
            rotation={[Math.PI / 2, 0, 0]}
            cellSize={0.1}
            sectionSize={0.5}
            cellColor="#242a34"
            sectionColor="#39414f"
            fadeDistance={12}
            infiniteGrid
          />
        ) : null}
        {config.originAxes > 0 ? <axesHelper args={[config.originAxes]} /> : null}

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.15}
          target={[...config.cameraTarget]}
          maxDistance={30}
          minDistance={0.05}
        />
      </Canvas>

      {config.showGroupToggles && groups.length > 1 ? (
        <div className="porthole-scene-groups">
          {groups.map((group) => (
            <button
              key={group}
              type="button"
              data-hidden={hidden.has(group)}
              onClick={() => toggleGroup(group)}
            >
              {group}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
