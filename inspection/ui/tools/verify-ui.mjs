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
  ['-u', `${uiDir}app.py`, 'mock', '--no_window', `--port=${PORT}`, `--bus_port=${BUS_PORT}`, '--turns=6'],
  { cwd: repoRoot, stdio: ['ignore', 'inherit', 'inherit'], env: { ...process.env, BROWSER: 'true' } },
);
children.push(backend);

// The workcell, IK and the first few planned moves have to happen before the
// panels have anything to show.
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

check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

const status = await page.locator('.inspection-statusbar').innerText().catch(() => '');
check('bus connected', await page.locator('.porthole-status-dot[data-status="open"]').count() > 0,
  status.replace(/\n/g, ' '));
check('run status published', /step \d+/.test(status), status.replace(/\n/g, ' '));

// ── cell panel: the 3D workcell ──────────────────────────────────────────────
check('cell panel present', await page.locator('.porthole-scene-panel canvas').count() > 0);

const groups = await page.locator('.porthole-scene-groups button').allInnerTexts();
check('scene has every node group',
  ['world', 'cell', 'robot', 'tool', 'object', 'views'].every((g) => groups.includes(g)),
  groups.join(','));

check('robot meshes fetched', meshResponses.length >= 7, `${meshResponses.length} .dae requests`);
check('robot meshes served 200',
  meshResponses.length > 0 && meshResponses.every((r) => r.status === 200),
  JSON.stringify(meshResponses.slice(0, 3)));

// Two shots apart: proves geometry was drawn AND that poses are streaming.
// readPixels cannot be used — preserveDrawingBuffer is false, so it returns
// zeros exactly when the panel is visible.
const cellPanel = page.locator('.porthole-scene-panel');
const cellA = await cellPanel.screenshot();
await sleep(600);
const cellB = await cellPanel.screenshot();
check('cell panel drew geometry', cellA.length > 8000, `${cellA.length} byte png`);
check('cell panel is animating', !cellA.equals(cellB), 'two frames 600ms apart are identical');

// ── camera panel ─────────────────────────────────────────────────────────────
const cameraSize = await page
  .locator('.porthole-camera-panel canvas')
  .evaluate((el) => `${el.width}x${el.height}`)
  .catch(() => 'none');
check('camera painted a real frame', cameraSize === '640x360', cameraSize);

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

async function findMarkerPixel() {
  const shot = await cloudPanel.screenshot();
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
