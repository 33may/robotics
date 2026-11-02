/**
 * Render the demo dashboard headlessly and save a screenshot.
 *
 * This is the verification loop the whole toolkit is built around: an agent
 * that writes a panel can run this, Read the PNG, and see whether the thing it
 * built actually draws. Assertions on console errors mean the common failures
 * (undefined payload, bad hook, WebGL refusing to start) fail loudly instead of
 * producing a plausible-looking blank rectangle.
 *
 *   node tools/screenshot-demo.mjs [out.png]
 *
 * Uses the system Chrome, so there is no browser download.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const outPath = process.argv[2] ?? `${repoRoot}tools/demo-screenshot.png`;
const PYTHON = process.env.PORTHOLE_PYTHON ?? `${process.env.HOME}/miniconda3/envs/robo/bin/python`;
const CHROME = ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']
  .find(existsSync);

if (!CHROME) throw new Error('no system Chrome found');
if (!existsSync(`${repoRoot}apps/demo/dist/index.html`)) {
  throw new Error('demo not built — run: npm run build -w @porthole/demo');
}

const children = [];
function cleanup() {
  for (const child of children) child.kill();
}
process.on('exit', cleanup);

// 1. the backend — the very same process the native app runs, minus the window.
//    Running a different backend here is how a check drifts into testing
//    something nobody ships: the mesh mount lives in the app, so a separate
//    publisher would silently never exercise it.
const backend = spawn(
  PYTHON,
  ['-u', `${repoRoot}python/examples/demo_app.py`, '--serve-only', '--port', '8767'],
  { stdio: ['ignore', 'inherit', 'inherit'] },
);
children.push(backend);

await sleep(2500);

// 3. the browser
const browser = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });

const consoleErrors = [];
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});
page.on('pageerror', (error) => consoleErrors.push(String(error)));

// Mesh geometry is fetched over HTTP, not carried on the bus, so the only place
// a broken asset mount shows up is here.
const meshResponses = [];
page.on('response', (response) => {
  if (/\.(dae|glb|gltf|stl|obj)$/i.test(new URL(response.url()).pathname)) {
    meshResponses.push({ url: response.url(), status: response.status() });
  }
});

await page.goto('http://127.0.0.1:8767', { waitUntil: 'networkidle' });
// Long enough for the socket to open, retained state to replay, and several
// live frames to paint.
await sleep(4000);

// ── assertions ──────────────────────────────────────────────────────────────

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: Boolean(ok), detail });

check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

const statusText = await page.locator('.porthole-statusbar').innerText().catch(() => '');
check('bus reports open', statusText.includes('open'), statusText.replace(/\n/g, ' '));
check('hello advertised topics', /\d+ topics/.test(statusText), statusText.replace(/\n/g, ' '));

check('point cloud canvas present', await page.locator('.porthole-pointcloud-panel canvas').count() > 0);
check('scene view canvas present', await page.locator('.porthole-scene-panel canvas').count() > 0);

// A 3D panel exposes nothing to the DOM, and reading the WebGL buffer back does
// not work: with preserveDrawingBuffer false — the default, and the right
// default — the buffer's contents are undefined once the frame is composited,
// so readPixels returns zeros whenever the panel is actually visible.
//
// Screenshot the element instead. Two properties are worth asserting and both
// come free: a PNG of a flat colour compresses to almost nothing, so size shows
// that something was drawn; and two shots taken 400 ms apart must differ,
// because the backend is streaming poses at 30 Hz. Identical shots mean a
// static scene, which is the failure that matters — geometry with no animation
// looks perfectly fine in a single screenshot.
const scenePanel = page.locator('.porthole-scene-panel');
const sceneShotA = await scenePanel.screenshot();
await sleep(400);
const sceneShotB = await scenePanel.screenshot();

check('scene view drew geometry', sceneShotA.length > 6000, `${sceneShotA.length} byte png`);
check('scene view is animating', !sceneShotA.equals(sceneShotB),
  'two frames 400ms apart are byte-identical');

const gridCells = await page.locator('.porthole-media-cell').count();
check('media grid rendered every cell', gridCells === 4, `${gridCells} cells`);

const gridSizes = await page
  .locator('.porthole-media-cell canvas')
  .evaluateAll((els) => els.map((el) => `${el.width}x${el.height}`));
check('grid cells painted real frames',
  gridSizes.filter((size) => size !== '0x0').length === 4, gridSizes.join(' '));
check('grid cells keep their own resolution', gridSizes.includes('480x480'), gridSizes.join(' '));

const gridColumns = await page
  .locator('.porthole-media-grid')
  .evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(' ').length)
  .catch(() => 0);
check('auto layout chose a real column count', gridColumns >= 1 && gridColumns <= 4,
  `${gridColumns} columns`);

check('mesh fetched over http', meshResponses.length > 0,
  'no mesh request seen — asset mount missing?');
check('mesh served 200',
  meshResponses.length > 0 && meshResponses.every((response) => response.status === 200),
  JSON.stringify(meshResponses));

const logRows = await page.locator('.porthole-event-log tbody tr').count();
check('event log has rows', logRows > 0, `${logRows} rows`);

const optionCount = await page.locator('.porthole-command-options button').count();
check('decision options rendered', optionCount === 4, `${optionCount} options`);

// The command path, end to end: click an option, send, watch Python answer on
// a topic that only it can publish.
await page.locator('.porthole-command-options button').first().click();
await page.locator('.porthole-command-submit').click();
await sleep(1200);
const logText = await page.locator('.porthole-event-log').innerText();
check('command reached Python and came back', logText.includes('operator chose h3v1'));

await page.screenshot({ path: outPath });
await browser.close();
cleanup();

let failed = 0;
for (const { name, ok, detail } of results) {
  if (!ok) failed += 1;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${name}${detail && !ok ? `  (${detail})` : ''}`);
}
console.log(`\n${results.length - failed}/${results.length} checks passed`);
console.log(`screenshot: ${outPath}`);
process.exit(failed === 0 ? 0 : 1);
