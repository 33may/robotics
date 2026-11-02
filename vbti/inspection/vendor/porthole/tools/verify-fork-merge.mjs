/**
 * Prove the fork model works.
 *
 * The scenario that decides whether owning source is viable:
 *
 *   1. install a component
 *   2. edit it locally (a fork)
 *   3. upstream independently fixes something else in the same file
 *   4. update
 *
 * A tool that keeps no base must either clobber your edit or refuse. Keeping
 * the base makes this a three-way merge, so both changes survive and only
 * genuinely overlapping edits conflict. This test asserts exactly that, plus
 * the conflict case.
 *
 *   node tools/verify-fork-merge.mjs
 */

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));
const CLI = join(REPO_ROOT, 'tools/porthole.mjs');
const UPSTREAM_FILE = join(REPO_ROOT, 'packages/registry/src/panels/CameraPanel.tsx');

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: Boolean(ok), detail });

const porthole = (args, cwd) =>
  execFileSync('node', [CLI, ...args], { cwd, encoding: 'utf8' });

const pristineUpstream = readFileSync(UPSTREAM_FILE, 'utf8');
const workdir = mkdtempSync(join(tmpdir(), 'porthole-fork-'));

try {
  // 1. install
  porthole(['add', 'camera', '--into', workdir]);
  const installed = join(workdir, 'panels/CameraPanel.tsx');
  check('component installed', readFileSync(installed, 'utf8').includes('CameraPanel'));
  check(
    'base recorded',
    readFileSync(join(workdir, '.porthole/base/camera/panels/CameraPanel.tsx'), 'utf8').length > 0,
  );

  // 2. fork it — change the default topic, near the top of the file
  const forked = readFileSync(installed, 'utf8').replace(
    "topic: 'camera/wrist',",
    "topic: 'my_robot/front_cam',",
  );
  writeFileSync(installed, forked);
  check('local fork applied', forked.includes('my_robot/front_cam'));

  const statusAfterFork = porthole(['status', '--into', workdir]);
  check('status reports the fork', statusAfterFork.includes('forked'), statusAfterFork.trim());

  // 3. upstream changes something else, near the bottom of the same file
  writeFileSync(
    UPSTREAM_FILE,
    pristineUpstream.replace(
      '  return (\n    <div className="porthole-camera-panel">',
      '  // upstream: added a frame counter hook here\n' +
        '  return (\n    <div className="porthole-camera-panel">',
    ),
  );

  const statusAfterUpstream = porthole(['status', '--into', workdir]);
  check(
    'status sees upstream moved',
    statusAfterUpstream.includes('upstream-changed'),
    statusAfterUpstream.trim(),
  );

  // 4. update — the moment of truth
  const updateOutput = porthole(['update', 'camera', '--into', workdir]);
  const mergedText = readFileSync(installed, 'utf8');

  check('update reported a clean merge', updateOutput.includes('1 conflict(s)') === false, updateOutput.trim());
  check('local fork survived the update', mergedText.includes('my_robot/front_cam'));
  check('upstream change was applied', mergedText.includes('upstream: added a frame counter hook'));
  check('no conflict markers', !mergedText.includes('<<<<<<<'));

  // 5. the conflict case: both sides edit the same line
  writeFileSync(
    UPSTREAM_FILE,
    readFileSync(UPSTREAM_FILE, 'utf8').replace(
      "fit: 'contain',",
      "fit: 'cover', // upstream changed the default",
    ),
  );
  const conflicting = readFileSync(installed, 'utf8').replace(
    "fit: 'contain',",
    "fit: 'contain', // mine",
  );
  writeFileSync(installed, conflicting);

  let conflictOutput = '';
  try {
    conflictOutput = porthole(['update', 'camera', '--into', workdir]);
  } catch (error) {
    conflictOutput = String(error.stdout ?? error.message);
  }
  const conflictedText = readFileSync(installed, 'utf8');
  check('overlapping edits conflict', conflictedText.includes('<<<<<<< yours'), conflictOutput.trim());
  check('conflict names both sides', conflictedText.includes('>>>>>>> upstream'));
} finally {
  writeFileSync(UPSTREAM_FILE, pristineUpstream);
  rmSync(workdir, { recursive: true, force: true });
}

let failed = 0;
for (const { name, ok, detail } of results) {
  if (!ok) failed += 1;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${detail && !ok ? `\n        ${detail}` : ''}`);
}
console.log(`\n${results.length - failed}/${results.length} fork-model checks passed`);
process.exit(failed === 0 ? 0 : 1);
