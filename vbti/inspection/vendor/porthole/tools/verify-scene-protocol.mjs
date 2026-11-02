/**
 * Scene protocol conformance check (docs/scene-protocol.md §9).
 *
 * Starts a real Python backend publishing a real scene, decodes it with the
 * real TypeScript decoder, and asserts the things a renderer depends on and
 * cannot recover from:
 *
 *   - transforms arrive as float32 [N, 4, 4] with one row per name
 *   - transforms are row-major, so element [0][3] is the X translation
 *   - no pose frame names a node absent from the description
 *   - overrides are per-frame, not sticky
 *   - mesh URLs are server paths the UI can actually fetch
 *
 * Row-majorness is the one worth stating twice: a transposed 4x4 still renders,
 * still animates, and puts everything in the wrong place with a plausible-
 * looking rotation. Nothing downstream can detect it.
 *
 *   node tools/verify-scene-protocol.mjs
 */

import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

import { decodeEnvelope } from '../packages/framework/dist/bus/decode.js';

const PORT = 8791;
const PYTHON = process.env.PORTHOLE_PYTHON ?? `${process.env.HOME}/miniconda3/envs/robo/bin/python`;

const BACKEND = `
import time
import numpy as np
from porthole import PortholeBus
from porthole.scene import box_node, mesh_node, pose_frame, scene_description

bus = PortholeBus(app="scene-check", port=${PORT}).start()
bus.declare("scene/description", qos="stream", kind="scene")
bus.declare("scene/poses", qos="stream", kind="scene")

bus.publish("scene/description", scene_description([
    box_node("cell/slab", size=(1.0, 0.5, 0.03), color="#888888"),
    mesh_node("robot/wrist", url="/meshes/wrist3.dae"),
], frame="base"))

# A transform with a distinguishable translation and a distinguishable
# rotation: if either axis convention flips, both change.
T = np.eye(4)
T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T[:3, 3] = [0.11, 0.22, 0.33]

# Alternate tinted and untinted frames for the whole run. Publishing them once
# at startup would prove nothing: scene/poses is a stream topic, so only the
# newest is retained, and a client connecting later sees exactly one frame.
for i in range(200):
    tint = {"robot/wrist": {"color": "#ff0000"}} if i % 2 == 0 else None
    bus.publish("scene/poses", pose_frame({"robot/wrist": T}, overrides=tint))
    time.sleep(0.1)
`;

const checks = [];
function check(name, condition, detail = '') {
  checks.push({ name, ok: Boolean(condition), detail });
}

const python = spawn(PYTHON, ['-c', BACKEND], { stdio: ['ignore', 'inherit', 'inherit'] });
process.on('exit', () => python.kill());

await sleep(1800);

const socket = new WebSocket(`ws://127.0.0.1:${PORT}`);
socket.binaryType = 'arraybuffer';

let description;
const poseFrames = [];

socket.onmessage = (event) => {
  const envelope = decodeEnvelope(event.data);
  if (envelope.topic === 'scene/description') description = envelope.payload;
  if (envelope.topic === 'scene/poses') poseFrames.push(envelope.payload);
};

await new Promise((resolve, reject) => {
  socket.onopen = resolve;
  socket.onerror = () => reject(new Error(`cannot reach ws://127.0.0.1:${PORT}`));
});

await sleep(2000);

// ── assertions ──────────────────────────────────────────────────────────────

check('description arrives', description !== undefined);
check('description declares its frame', description?.frame === 'base', description?.frame);
check('description carries every node', description?.nodes?.length === 2,
  `got ${description?.nodes?.length}`);

const box = description?.nodes?.find((node) => node.path === 'cell/slab');
check('box uses full extents', JSON.stringify(box?.geometry?.size) === '[1,0.5,0.03]',
  JSON.stringify(box?.geometry?.size));
check('every node has a 16-value transform',
  description?.nodes?.every((node) => node.transform?.length === 16));

const mesh = description?.nodes?.find((node) => node.path === 'robot/wrist');
check('mesh url is a server path', mesh?.geometry?.url?.startsWith('/'), mesh?.geometry?.url);

const first = poseFrames[0];
check('pose frame arrives', first !== undefined, `got ${poseFrames.length} frames`);
check('transforms are float32', first?.transforms?.data instanceof Float32Array,
  first?.transforms?.data?.constructor?.name);
check('transforms are [N,4,4]', JSON.stringify(first?.transforms?.shape) === '[1,4,4]',
  JSON.stringify(first?.transforms?.shape));
check('one transform per name',
  first?.names?.length === (first?.transforms?.shape?.[0] ?? -1),
  `${first?.names?.length} names vs ${first?.transforms?.shape?.[0]} rows`);

// Row-major: translation occupies indices 3, 7, 11 of the flattened 4x4.
const m = first?.transforms?.data;
check('transform is row-major (translation at 3,7,11)',
  Math.abs((m?.[3] ?? 0) - 0.11) < 1e-6 &&
  Math.abs((m?.[7] ?? 0) - 0.22) < 1e-6 &&
  Math.abs((m?.[11] ?? 0) - 0.33) < 1e-6,
  m ? `got ${m[3]}, ${m[7]}, ${m[11]} — column-major would give ${m[12]}, ${m[13]}, ${m[14]}` : '');
check('rotation survives untransposed', Math.abs((m?.[1] ?? 0) + 1) < 1e-6,
  `m[0][1]=${m?.[1]} (should be -1; +1 means transposed)`);

const describedPaths = new Set((description?.nodes ?? []).map((node) => node.path));
check('no pose names an undescribed node',
  poseFrames.every((frame) => (frame.names ?? []).every((name) => describedPaths.has(name))));

const tinted = poseFrames.filter((frame) => frame.overrides?.['robot/wrist']?.color === '#ff0000');
const untinted = poseFrames.filter((frame) => frame.overrides === undefined);
check('overrides ride in the pose frame', tinted.length > 0,
  `${tinted.length} of ${poseFrames.length} frames tinted`);
check('overrides are not sticky', untinted.length > 0,
  `${untinted.length} of ${poseFrames.length} frames untinted`);

socket.close();
python.kill();

// ── report ──────────────────────────────────────────────────────────────────

let failed = 0;
for (const { name, ok, detail } of checks) {
  if (!ok) failed += 1;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${detail && !ok ? `  (${detail})` : ''}`);
}
console.log(`\n${checks.length - failed}/${checks.length} scene-protocol checks passed`);
process.exit(failed === 0 ? 0 : 1);
