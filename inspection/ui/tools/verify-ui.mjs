/**
 * Render the inspection UI against the mock run and save a screenshot.
 *
 * This is the feedback loop for frontend work here: run it, Read the PNG, see
 * whether what you built actually draws. The assertions exist so the common
 * failures — a payload shape the panel does not expect, a topic nobody
 * publishes, WebGL refusing to start — fail loudly instead of producing a
 * plausible-looking empty rectangle.
 *
 *   npm run check            # or: node tools/verify-ui.mjs [out.png]
 *
 * Uses the system Chrome, so there is no browser download.
 *
 * v2: the backend is a command-driven Supervisor that IDLEs until a command
 * arrives — nothing auto-plays. So after boot this script drives the mock
 * itself, the same way an operator would: click the survey button (send
 * `view/request`), wait for `previewing`, click it again (send
 * `view/confirm`), wait for the boot capture to settle. Only once that has
 * happened does the world have a sphere, an object, cloud points and log
 * rows for the later checks to find.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';

const uiDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = fileURLToPath(new URL('../../../', import.meta.url));
const outPath = process.argv[2] ?? `${uiDir}tools/ui-screenshot.png`;
const PORT = 8772;
const BUS_PORT = 8781;   // never the default: a real run may own 8765
const PYTHON = process.env.INSPECTION_PYTHON ?? `${process.env.HOME}/miniconda3/envs/robo/bin/python`;
const CHROME = ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']
  .find(existsSync);

if (!CHROME) throw new Error('no system Chrome found');
if (!existsSync(`${uiDir}dist/index.html`)) throw new Error('not built — npm run build');

const children = [];
process.on('exit', () => children.forEach((child) => child.kill()));

// The mock owns the bus and drives every topic. Same process the operator runs.
const backend = spawn(
  PYTHON,
  ['-u', `${uiDir}app.py`, 'mock', '--no_window', `--port=${PORT}`, `--bus_port=${BUS_PORT}`],
  { cwd: repoRoot, stdio: ['ignore', 'inherit', 'inherit'], env: { ...process.env, BROWSER: 'true' } },
);
children.push(backend);

// RobotCell()/UR5eIK() construction and the frontend's own mesh fetch both
// take real seconds. The Supervisor itself idles the instant it starts — it
// is the meshes and the workcell, not any auto-playing turn, that this waits
// out.
await sleep(12000);

const browser = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1700, height: 1000 } });

const consoleErrors = [];
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});
page.on('pageerror', (error) => consoleErrors.push(String(error)));

const meshResponses = [];
page.on('response', (response) => {
  if (/\.dae$/i.test(new URL(response.url()).pathname)) {
    meshResponses.push({ url: response.url(), status: response.status() });
  }
});

await page.goto(`http://127.0.0.1:${PORT}/?bus=${BUS_PORT}`, { waitUntil: 'networkidle' });
await sleep(6000);

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: Boolean(ok), detail });

/**
 * Poll `fn` until it returns truthy or `timeoutMs` elapses.
 *
 * The v2 backend is a real dispatcher thread behind a real OMPL planner and a
 * slow-but-real FakeRig (speed=0.6) — every phase transition after a click
 * takes a real, variable number of milliseconds, never zero and never a fixed
 * constant. A single fixed sleep either wastes time on the common case or
 * flakes on the slow one; this waits only as long as actually needed, up to a
 * generous ceiling.
 */
async function waitFor(fn, timeoutMs, intervalMs = 150) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await fn()) return true;
    await sleep(intervalMs);
  }
  return false;
}

const actionsStatus = page.locator('.actions-status');
const statusText = async () => (await actionsStatus.innerText().catch(() => '')).trim();
// Direct child combinator: excludes the grid's per-cell buttons and the
// footer's stop/exit buttons, all of which also carry the `.act` class.
const surveyBtn = page.locator('.actions-panel > button.act');
const surveyClass = async () => (await surveyBtn.getAttribute('class').catch(() => '')) ?? '';

check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

const statusbarText = await page.locator('.inspection-statusbar').innerText().catch(() => '');
check('bus connected', await page.locator('.porthole-status-dot[data-status="open"]').count() > 0,
  statusbarText.replace(/\n/g, ' '));

// v2's `run/status` is `{phase, target, visited, total}` — there is no
// `step` field, so the old `/step \d+/` assertion is stale by construction.
// Restore the same proof strength as the v1 check (numeric fields actually
// flowing through the topic, not just a truthy string): InspectionApp's
// statusbar renders `visited`/`total` as `${visited}/${reachable ?? '?'}
// visited of ${total}` — v2 never sets `reachable`, so that slot is always
// the literal "?", but `visited` and `total` are real numbers from the
// dispatcher (0/0 at boot). Confirmed against this harness's own baseline
// capture: "idle 0/? visited of 0". Keep the phase-chip assertion too.
const phaseText = await page.locator('.inspection-phase').innerText().catch(() => '');
const visitedTotalMatch = /\d+\/\S+ visited of \d+/.test(statusbarText);
check('run status published', phaseText.trim().length > 0 && visitedTotalMatch,
  `phase="${phaseText}" statusbar="${statusbarText.replace(/\n/g, ' ')}"`);

// ── cell panel: the 3D workcell ──────────────────────────────────────────────
check('cell panel present', await page.locator('.porthole-scene-panel canvas').count() > 0);

check('robot meshes fetched', meshResponses.length >= 7, `${meshResponses.length} .dae requests`);
check('robot meshes served 200',
  meshResponses.length > 0 && meshResponses.every((r) => r.status === 200),
  JSON.stringify(meshResponses.slice(0, 3)));

// ── camera panel ─────────────────────────────────────────────────────────────
const cameraSize = await page
  .locator('.porthole-camera-panel canvas')
  .evaluate((el) => `${el.width}x${el.height}`)
  .catch(() => 'none');
// The mock renders 640x360 but `publish_frame` stride-samples the LIVE stream
// by `live_stride` (2 by default — full-size frames froze the 3D panel, see
// publisher.py). So 320x180 is the correct expectation; asserting the
// pre-stride size made this check permanently red and therefore ignored.
check('camera painted a real frame', cameraSize === '320x180', cameraSize);

// ── actions panel: renders from the first views/state, no drive needed ──────
check('actions panel renders (survey button present)',
  await surveyBtn.count() === 1 && /\bact-(available|visited)\b/.test(await surveyClass()),
  `class="${await surveyClass()}"`);

// STOP is never disabled — not even here, at idle, where it has nothing to
// stop. A stop the UI has to re-enable before it works is not a stop; the
// dispatcher is what validates it (logged no-op outside `executing`).
const stopBtn = page.locator('.act-stop');
check('STOP is always enabled', await stopBtn.count() === 1 && await stopBtn.isEnabled(),
  `phase="${await statusText()}"`);

// ── drive the two-press survey flow, exactly like an operator would ─────────
// Press 1: idle -(view/request survey)-> planning -> previewing.
const idleStatus = await statusText();
await surveyBtn.click();

// This is the check that proves the whole path: DOM click -> bus.send ->
// websocket -> pump -> dispatcher. `_start_planning` sets the phase
// synchronously before the planner thread is even spawned, so this never
// waits on OMPL — only on one command round-trip over the bus.
const leftIdle = await waitFor(async () => {
  const t = await statusText();
  return t === 'planning' || t === 'previewing';
}, 5000);
check('click sends a command (status leaves idle within 5s)', leftIdle,
  `status was "${idleStatus}", now "${await statusText()}"`);

// Scaffolding for the rest of the drive, not a product assertion — but
// reported as a check (not thrown) so a stall here still leaves every other
// check's real pass/fail visible instead of aborting the run.
const reachedPreviewing = await waitFor(async () => (await statusText()) === 'previewing', 30000);
check('survey plan reaches previewing', reachedPreviewing, `status="${await statusText()}"`);

// Press 2: previewing -(view/confirm)-> executing -> capturing -> fusing -> idle.
if (reachedPreviewing) await surveyBtn.click();

const reachedExecuting = await waitFor(async () => (await statusText()) === 'executing', 10000);
check('confirm reaches executing', reachedExecuting, `status="${await statusText()}"`);

// `_on_view_confirm` never re-publishes `views/state`, so the survey button
// stays labelled "previewing" (and clickable — `previewing` is ACTIONABLE)
// all the way through executing/capturing/fusing, until settle. Clicking it
// again here sends a now-stale `view/confirm`, which the dispatcher refuses
// (phase is no longer `previewing`) — FR11, "a stale button can never move
// the arm". Real operator mistake, not a script fixture, and it is the one
// bus.send in this whole script that produces a `log/events` row on the
// happy path (every other `pub.log` call is an error/refusal path we are
// not otherwise triggering). Fired the instant `executing` is observed —
// the move is short (a small wrist offset), so this loses the race if it
// waits for anything else first. Gated on `reachedExecuting`, mirroring the
// `if (reachedPreviewing)` confirm click above: under contention, if the
// previous wait timed out just as the backend actually reached `previewing`,
// an unconditional click here would land as the FIRST legitimate confirm
// (moving the arm) instead of a guaranteed-stale one.
if (reachedExecuting) await surveyBtn.click({ timeout: 2000 }).catch(() => {});

// Two shots apart, taken while the rig is actually moving: proves geometry
// was drawn AND that poses are streaming. readPixels cannot be used —
// preserveDrawingBuffer is false, so it returns zeros exactly when the panel
// is visible. This is anchored to `executing` on purpose: it is the one
// phase PoseStreamer is active (`sup.pose_active`), so it is the only phase
// in which two frames 600ms apart are guaranteed to differ.
const cellPanel = page.locator('.porthole-scene-panel');
const cellA = await cellPanel.screenshot();
await sleep(600);
const cellB = await cellPanel.screenshot();
check('cell panel drew geometry', cellA.length > 8000, `${cellA.length} byte png`);
check('cell panel is animating', !cellA.equals(cellB),
  'two frames 600ms apart during executing are identical');

// Boot settles: survey visited, sphere + object created, cloud/fused and
// views/state (with cells) published for the first time.
const settled = await waitFor(async () => /\bact-visited\b/.test(await surveyClass()), 60000);
check('boot settles: survey visited', settled, `class="${await surveyClass()}"`);

// ── scene groups: only populated once the sphere/object exist (post-boot) ───
const groups = await page.locator('.porthole-scene-groups button').allInnerTexts();
check('scene has every node group',
  ['world', 'cell', 'robot', 'tool', 'object', 'views'].every((g) => groups.includes(g)),
  groups.join(','));

// ── cloud panel: markers, picking, and the image pane ────────────────────────
await page.locator('[data-testid="cloud-tab"], .dv-tab:has-text("cloud")').first().click()
  .catch(() => {});
await sleep(1500);

check('cloud panel present', await page.locator('.inspection-cloud-panel canvas').count() > 0);

const cloudPanel = page.locator('.inspection-cloud-view');
const cloudShot = await cloudPanel.screenshot();
check('cloud panel drew geometry', cloudShot.length > 8000, `${cloudShot.length} byte png`);

/**
 * Markers are WebGL: there is no DOM node to target, and blind grid-clicking
 * does not work — 36 discs of ~23 px sit on a ring ~470 px across, so a grid
 * coarse enough to be fast misses all of them and a grid fine enough to hit
 * one takes thousands of clicks.
 *
 * So aim at pixels that are actually marker-coloured. Screenshot the canvas,
 * hand the PNG back into the page, decode it on a 2D canvas (the browser
 * already has a decoder; Node does not), and find a pixel matching one of the
 * state colours. `canvas.toDataURL` is not an option — preserveDrawingBuffer is
 * false, so it comes back blank.
 */
const MARKER_RGB = [
  [126, 226, 168], // visited
  [138, 180, 248], // current
  [104, 110, 120], // available, as composited at 0.95 opacity over the panel bg
];

async function findMarkerPixel(panel = cloudPanel) {
  const shot = await panel.screenshot();
  return page.evaluate(
    async ([dataUrl, targets]) => {
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = dataUrl;
      });
      const surface = document.createElement('canvas');
      surface.width = image.width;
      surface.height = image.height;
      const context = surface.getContext('2d');
      context.drawImage(image, 0, 0);
      const { data } = context.getImageData(0, 0, surface.width, surface.height);

      const matches = (x, y) => {
        if (x < 0 || y < 0 || x >= surface.width || y >= surface.height) return false;
        const i = (y * surface.width + x) * 4;
        return targets.some(
          ([r, g, b]) =>
            Math.abs(data[i] - r) < 22 &&
            Math.abs(data[i + 1] - g) < 22 &&
            Math.abs(data[i + 2] - b) < 22,
        );
      };

      // Require the neighbourhood to match too. Antialiasing blends a marker's
      // silhouette with the background, so edge pixels are marker-coloured while
      // the ray through them misses the sphere — which looks exactly like a
      // broken raycaster and is not one.
      for (let y = 0; y < surface.height; y += 2) {
        for (let x = 0; x < surface.width; x += 2) {
          if (
            matches(x, y) &&
            matches(x - 3, y) && matches(x + 3, y) &&
            matches(x, y - 3) && matches(x, y + 3)
          ) {
            return { x, y };
          }
        }
      }
      return null;
    },
    [`data:image/png;base64,${shot.toString('base64')}`, MARKER_RGB],
  );
}

const box = await cloudPanel.boundingBox();
const target = box ? await findMarkerPixel() : null;
check('a marker is visible on screen', target !== null, 'no marker-coloured pixel found');

let picked = false;
if (box && target) {
  await page.mouse.click(box.x + target.x, box.y + target.y);
  await sleep(400);
  picked = (await page.locator('.inspection-cloud-pane-head').count()) > 0;
}
check('clicking a marker opens the pane', picked,
  target ? `clicked ${target.x},${target.y} within the canvas` : 'no target');

if (picked) {
  const state = await page.locator('.inspection-cell-state').innerText().catch(() => '');
  const hasImage = (await page.locator('.inspection-cloud-shot').count()) > 0;
  const hasEmpty = (await page.locator('.inspection-cloud-empty').count()) > 0;
  check('pane shows an image or says not visited', hasImage || hasEmpty, `state=${state}`);
}

// ── 3D click: a marker in the CELL panel starts the preview ──────────────────
// The same `view/request` the actions panel sends, from the scene view. Any
// backend REACTION proves the wiring: a legal cell moves the phase off idle,
// an already-visited one answers with a "dropped" log line. Asserting on
// either keeps the check from depending on which of 36 markers the pixel scan
// happened to land on.
await page.locator('[data-testid="cell-tab"], .dv-tab:has-text("cell")').first().click()
  .catch(() => {});
await sleep(400);
const scenePanel = page.locator('.porthole-scene-panel');
const sceneBox = await scenePanel.boundingBox();
const sceneTarget = sceneBox ? await findMarkerPixel(scenePanel) : null;
check('a viewsphere marker is visible in the cell panel', sceneTarget !== null);

let reacted = false;
if (sceneBox && sceneTarget) {
  const before = await page.locator('.porthole-event-log tbody tr').count();
  const phaseBefore = await page.locator('.inspection-phase').innerText().catch(() => '');
  await page.mouse.click(sceneBox.x + sceneTarget.x, sceneBox.y + sceneTarget.y);
  await sleep(900);
  const after = await page.locator('.porthole-event-log tbody tr').count();
  const phaseAfter = await page.locator('.inspection-phase').innerText().catch(() => '');
  reacted = after > before || phaseAfter !== phaseBefore;
}
check('clicking a 3D marker reaches the backend', reacted,
  'no phase change and no new log row after the click');

// ── chain panel: the identity chain for the last capture ─────────────────────
// The mock runs the real segmentation path over a stub backend and writes real
// capture files, so by now the survey has produced a full chain on disk.
await page.locator('[data-testid="chain-tab"], .dv-tab:has-text("chain")').first().click()
  .catch(() => {});
await sleep(1200);

const chainImgs = page.locator('.inspection-chain-img');
const nChain = await chainImgs.count();
check('chain panel shows the 2x2 grid', nChain === 4,
  `${nChain} images (missing stages render as .inspection-chain-missing)`);

// Counting <img> tags is not enough — a broken URL still renders one. Ask the
// browser whether the bytes actually decoded.
let decoded = 0;
for (let i = 0; i < nChain; i += 1) {
  if (await chainImgs.nth(i).evaluate((el) => el.complete && el.naturalWidth > 0)) {
    decoded += 1;
  }
}
check('chain images actually loaded', decoded === 4, `${decoded}/${nChain} decoded`);

const chainSource = await page.locator('.inspection-chain-source').first().textContent()
  .catch(() => '');
check('chain reports the identity source', /mask|depth/.test(chainSource ?? ''),
  `header="${chainSource}"`);

// ── log panel ────────────────────────────────────────────────────────────────
const logRows = await page.locator('.porthole-event-log tbody tr').count();
check('log has rows', logRows > 0, `${logRows} rows`);

await page.screenshot({ path: outPath });
await browser.close();
children.forEach((child) => child.kill());

let failed = 0;
for (const { name, ok, detail } of results) {
  if (!ok) failed += 1;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${detail && !ok ? `  (${detail})` : ''}`);
}
console.log(`\n${results.length - failed}/${results.length} checks passed`);
console.log(`screenshot: ${outPath}`);
process.exit(failed === 0 ? 0 : 1);
