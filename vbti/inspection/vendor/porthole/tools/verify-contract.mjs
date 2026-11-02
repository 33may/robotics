/**
 * Cross-language contract check.
 *
 * Starts a real Python bus, connects with the real TypeScript decoder, and
 * asserts the values that arrive are the values that were published. Unit
 * tests on either side can both pass while the two disagree about the wire —
 * this is the test that catches that.
 *
 *   node tools/verify-contract.mjs
 *
 * Requires the framework to be built (`npm run build -w @porthole/framework`)
 * and the `porthole` Python package importable.
 */

import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

import { decodeEnvelope } from '../packages/framework/dist/bus/decode.js';

const PORT = 8788;
const PYTHON = process.env.PORTHOLE_PYTHON ?? `${process.env.HOME}/miniconda3/envs/robo/bin/python`;

const PUBLISHER = `
import time, numpy as np
from porthole import PortholeBus
from porthole.mock import gradient_frame

bus = PortholeBus(app="contract-check", port=${PORT}).start()
bus.declare("scene/cloud", qos="stream", kind="ndarray")
bus.declare("camera/wrist", qos="stream", kind="image/jpeg")
bus.declare("log/events", qos="event")

points = np.arange(30, dtype=np.float32).reshape(10, 3)
bus.publish("scene/cloud", {"points": bus.array_payload(points), "n": 10})
bus.publish("camera/wrist", bus.jpeg_payload(gradient_frame(0.0, 64, 48)))
for i in range(3):
    bus.publish("log/events", {"i": i, "msg": "hello"})
time.sleep(8)
`;

const checks = [];
function check(name, condition, detail = '') {
  checks.push({ name, ok: Boolean(condition), detail });
}

const python = spawn(PYTHON, ['-c', PUBLISHER], { stdio: ['ignore', 'inherit', 'inherit'] });
process.on('exit', () => python.kill());

await sleep(1500);

const socket = new WebSocket(`ws://127.0.0.1:${PORT}`);
socket.binaryType = 'arraybuffer';

const received = new Map();
const events = [];

socket.onmessage = (event) => {
  const envelope = decodeEnvelope(event.data);
  if (envelope.topic === 'log/events') events.push(envelope.payload.i);
  received.set(envelope.topic, envelope);
};

await new Promise((resolve, reject) => {
  socket.onopen = resolve;
  socket.onerror = () => reject(new Error(`cannot reach ws://127.0.0.1:${PORT}`));
});

await sleep(1200);

// ── assertions ──────────────────────────────────────────────────────────────

const hello = received.get('$/hello');
check('$/hello arrives', hello !== undefined);
check('protocol version is 1', hello?.payload.protocol === 1);
check('declares 3 topics', hello?.payload.topics.length === 3,
  `got ${hello?.payload.topics.length}`);

const cloud = received.get('scene/cloud');
check('cloud arrives', cloud !== undefined);
const nd = cloud?.payload.points;
check('array is Float32Array', nd?.data instanceof Float32Array,
  `got ${nd?.data?.constructor?.name}`);
check('shape survives', JSON.stringify(nd?.shape) === '[10,3]', JSON.stringify(nd?.shape));
check('values survive', nd?.data?.[0] === 0 && nd?.data?.[29] === 29,
  `first=${nd?.data?.[0]} last=${nd?.data?.[29]}`);
check('sibling scalars survive', cloud?.payload.n === 10);

const frame = received.get('camera/wrist');
check('image arrives', frame !== undefined);
check('image is jpeg', frame?.payload.mime === 'image/jpeg');
check('image dims survive', frame?.payload.w === 64 && frame?.payload.h === 48,
  `${frame?.payload.w}x${frame?.payload.h}`);
check('jpeg magic bytes', frame?.payload.bytes?.[0] === 0xff && frame?.payload.bytes?.[1] === 0xd8);

check('all 3 events delivered in order', JSON.stringify(events) === '[0,1,2]',
  JSON.stringify(events));

socket.close();
python.kill();

// ── report ──────────────────────────────────────────────────────────────────

let failed = 0;
for (const { name, ok, detail } of checks) {
  if (!ok) failed += 1;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${detail && !ok ? `  (${detail})` : ''}`);
}
console.log(`\n${checks.length - failed}/${checks.length} contract checks passed`);
process.exit(failed === 0 ? 0 : 1);
