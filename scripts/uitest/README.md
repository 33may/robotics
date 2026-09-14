# uitest — a UI debug loop for GUI apps, driven by real input events

A small, dependency-light toolkit for end-to-end testing a GUI application by
**actually clicking it** and **verifying with pixels**. Built generically; the
first client is the **Rerun 0.37 web viewer**, which is the hard case: a WASM
app that renders into a single `<canvas>` with no DOM to query.

```
launch  ──▶  observe  ──▶  act  ──▶  observe  ──▶  verify
(start)      (screenshot)  (click/    (screenshot)  (imgdiff, pass/fail)
                            drag/key)
```

Everything derived — screenshots, crops, diffs, logs, PIDs — goes to
`$UITEST_DIR` (default `/tmp/rrd_ui_test`). Nothing is ever written into the
repo or into any dataset directory.

---

## Quick start

```bash
PY=/home/anton/miniconda3/envs/robo/bin/python
cd scripts/uitest

$PY make_toy_rrd.py                       # -> /tmp/rrd/_toy.rrd
$PY launch.py /tmp/rrd/_toy.rrd           # -> JSON with url, pids, ws_url
$PY demo_rerun.py                         # -> 9/9 checks passed
$PY cleanup.py                            # kill everything it started
```

The loop by hand:

```bash
$PY observe.py 01_before --park 100,700
$PY act.py drag 203 900 903 900           # scrub the timeline to step 30
$PY observe.py 02_after --park 100,700
$PY imgdiff.py /tmp/rrd_ui_test/01_before.png /tmp/rrd_ui_test/02_after.png \
      --region 1135,55,465,690 --min-frac 0.10
echo $?                                   # 0 = the UI changed as required
```

---

## The scripts

| script | role | notes |
|---|---|---|
| `make_toy_rrd.py` | build test material | growing point cloud + per-step image + 2 entities, fixed blueprint |
| `launch.py` | **launch** | starts Rerun web server + isolated Chrome, 3-stage readiness wait |
| `observe.py` | **observe** | screenshot to `$UITEST_DIR/<name>.png`, optional `--park`, `--region` |
| `act.py` | **act** | `click / dblclick / move / drag / wheel / key / text / reload` at pixel coords |
| `imgdiff.py` | **verify** | standalone CLI, `--region`, `--min-frac`, `--max-frac`, `--heatmap`; exit 0/1 |
| `crop.py` | read closely | cut a region, `--scale` (NEAREST), `--grid` with absolute coord labels |
| `cleanup.py` | teardown | PID-file based, `--purge` to also delete artifacts |
| `demo_rerun.py` | worked example / regression | the 9 assertions that prove the loop works |
| `cdp.py` | the engine | minimal CDP client; import this to script your own flows |
| `test_review_workspace.py` | the real suite | T1–T6 against `inspection/investigation/rr/` output; 21 checks |
| `native_hypr.py` | **observe, natively** | float+pin a real window on Hyprland and `grim` it. Verification only — no input injection |

```bash
$PY launch.py /tmp/rrd/2408-cup3.rrd --width 1600 --height 1000
$PY test_review_workspace.py            # -> 21/21 checks passed
$PY cleanup.py
```

---

## Route decision: web viewer + CDP (chosen) vs native window (rejected)

### Chosen: Rerun **web** viewer + Chrome DevTools **Protocol**

```bash
rerun /tmp/rrd/_toy.rrd --serve-web --web-viewer-port 9099 --port 9877 --hide-welcome-screen
# -> http://127.0.0.1:9099/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9877%2Fproxy
```

Verified working. `curl` on the index returns 16 KB of HTML and
`/re_viewer_bg.wasm` is a 40 MB 200. Renders correctly in headless Chrome.

**Use `--serve-web`, not `--web-viewer`.** `--web-viewer` implies `--serve-web`
but *also launches the user's default desktop browser* onto the viewer URL,
hijacking their real browsing session. We hit this. `launch.py` additionally
sets `BROWSER=/bin/true` as a second line of defence.

### Rejected: native window on Hyprland + `grim` + input injection

Concrete failure evidence, all measured on this machine:

1. **No input injection exists.** `wtype` and `ydotool` are apt-installable but
   neither solves it:
   - `wtype` implements `virtual-keyboard-v1` only — **keyboard, no pointer**.
     It cannot click or drag at all, which kills the timeline-drag requirement.
   - `ydotool` needs `/dev/uinput`, which is `crw------- root root`. The user is
     in the `input` group but that is irrelevant at mode 0600. It would need a
     udev rule plus a root `ydotoold` daemon — invasive, and still gives you
     only *global* cursor warping with no window-relative coordinates.
2. **`xdotool` is useless here.** `hyprctl clients` reports the native viewer as
   `class=rerun xwayland=false` — it is a native Wayland client, invisible to X11
   tooling.
3. **`hyprctl dispatch` cannot click.** It dispatches keybind dispatchers; there
   is no pointer-button dispatcher.
4. **The window size is not yours to choose.** `rerun --window-size 900x600`
   produced a `1366x1068` window — Hyprland tiled it and ignored the request.
   No stable pixel coordinates.
5. **`grim` output does not match the geometry you ask for.** Capturing
   `-g "6,38 866x1068"` produced a **1082×1335** PNG: `hyprctl` reports *logical*
   coordinates while `grim` writes *physical* pixels, and this display runs at
   fractional scale 1.25. Every coordinate would need a manual, display-specific
   correction factor.

The CDP route dissolves 4 and 5 outright (`Emulation.setDeviceMetricsOverride`
pins both size and scale), and 1–3 do not apply.

### Why raw CDP and not the `chrome-devtools` MCP tools

The MCP tools are excellent for *observing* (`take_screenshot`,
`evaluate_script`, `list_console_messages`) and were used throughout this
investigation. They cannot *drive* this app:

- **`click` takes `uid` only** — an element id from the a11y snapshot. `drag`
  takes `from_uid`/`to_uid`. A WASM canvas app has exactly one element and no
  descendants, so there is nothing to address. There is no x/y parameter.
- **`resize_page` failed**: `Protocol error (Browser.setContentsSize): Restore
  window to normal state before setting content size` — the attached browser
  window was maximised, and there is no MCP tool to unmaximise it. Fixed pixel
  geometry, the whole basis of the strategy, was unobtainable.
- **Scripts cannot call MCP tools.** The deliverables are CLIs; they need a
  library.

`Input.dispatchMouseEvent` is also strictly better than the documented fallback
of dispatching synthetic `PointerEvent`s from `evaluate_script`: CDP input is
**trusted** (`isTrusted: true`) and goes through the browser's real hit-testing,
so it behaves identically to a human's mouse. We never needed the synthetic
route, so the toolkit does not carry it.

---

## Coordinates — the strategy that proved stable

**Pin the device scale factor to 1 and never think about DPR again.**

`launch.py` calls:

```python
Emulation.setDeviceMetricsOverride(width=1600, height=1000, deviceScaleFactor=1)
```

After that, **one number means one thing everywhere**:

```
screenshot pixel == CSS pixel == the x/y you pass to act.py
```

Without this the display's `devicePixelRatio = 1.25` silently scales
everything: the raw canvas backing store here is `1708x1153` for a `1366x922`
CSS viewport, and a screenshot taken through the MCP comes back at 1708x1153
while clicks are in CSS pixels. Every coordinate would be wrong by 25%.

The window size does **not** need to match — the emulation override is
independent of the real window, which is why headless works and why a tiling WM
cannot interfere.

### Finding targets on a canvas

There is no DOM, so targets are found by *looking*:

```bash
$PY crop.py shot.png --region 0,190,200,180 --scale 3 --grid 20 --out grid.png
```

`--grid` labels the lines with **absolute original-image coordinates**, so you
read the target's position straight off the enlarged crop. This is how every
coordinate in `demo_rerun.py` was derived.

For a linear widget, calibrate instead of guessing: click at several x, read
the resulting value, fit a line. The Rerun timeline came out as

```
x(step) = 203.0 + 23.35 * step        # at 1600x1000
```

confirmed to the exact step (clicking `x_for_step(30)` reads `#30`).

---

## Traps hit, in the order they bit

### 1. The viewer auto-plays — this is the big one

By default the Rerun viewer **plays the recording on load**. Every screenshot
catches a different `step`, so every assertion is flaky and every measurement is
nonsense. It cost the most time here: a calibration sweep returned
`x=250→#14, 450→#26, 650→#34, 850→#40, 1050→#52, 1250→#60` — irregular,
non-monotonic, and off the end of a 60-step recording. All of it was playback
drift during the settle delay, not a coordinate problem.

Two fixes, use both:

- **Bake it into the blueprint** (best, deterministic by construction):
  ```python
  rrb.TimePanel(state="expanded", timeline="step", play_state="paused")
  ```
  The recording then opens paused at `#0`.
- **Click pause** at `(17, 765)` for recordings whose blueprint you don't own.

With playback paused, a drag from `x_for_step(0)` to `x_for_step(30)` lands on
`#30` exactly, every time.

### 2. Park the cursor before every screenshot

Rerun pops a data-inspector tooltip whenever the cursor rests over a view, and
highlights panel splitters and headers. An un-parked cursor silently corrupts
diffs — a "toggle is reversible" check failed at `changed_frac 0.082` purely
because the mouse was parked at `(800, 400)` over the 3D view and a tooltip was
showing.

Park somewhere inert and **measure that it is inert**. Measured at 1600x1000:

| park spot | change vs untouched frame |
|---|---|
| **(100, 700)** — empty blueprint panel | **0.00000** |
| (100, 180) — panel header | 0.00012 |
| (1590, 995) — near scrollbar | 0.00014 |
| (5, 500) — panel splitter | 0.00121 |

`observe.py --park 100,700`, or `shot()` in `demo_rerun.py`, does this for you.

### 3. Dragging the timeline *ruler* makes a loop selection, not a scrub

Rerun's time panel has two horizontal bands and they behave differently:

- **y ≈ 792** (the ruler strip with `#0 #10 …` labels) — dragging here creates a
  **loop selection** (`Start #19 / Stop #58`), turns the loop toggle blue, and
  the selection then blocks subsequent clicks in that band.
- **y ≈ 900** (the streams area below) — dragging here **scrubs** the playhead
  cleanly, no side effects.

Use `TIME_ROW_Y = 900`.

### 4. Small hit targets silently do nothing

The `>` expand arrow in the blueprint tree at `(19, 344)` has a ~10 px hit rect;
clicking it produced `changed_frac 0.003` — indistinguishable from noise, and it
never expanded. Prefer fat targets (a row label at `(105, 296)` gives a solid
`0.14`). If a click seems to do nothing, suspect the hit rect before the input
layer.

### 5. Hover state is what makes some controls exist at all

The per-entity visibility eye at `(179, 296)` is **only rendered while the row
is hovered**. `CDP.click()` therefore sends `mouseMoved` → sleep → `mousePressed`
→ `mouseReleased`. A bare press at the coordinate hits nothing. egui also
resolves *which widget* from the previous pointer position, so the leading move
is required in general, not just for reveal-on-hover controls.

### 6. Drags need intermediate moves and are rate-limited by the app

egui starts a drag only after it sees motion while the button is held, so a
press→release pair is a click, never a drag. `CDP.drag()` interpolates 24 steps
by default with 12 ms between them. Under the software rasterizer the app
processes roughly one pointer position per rendered frame; if you see a drag
undershoot its target, raise `--steps` and check that playback is paused (see
trap 1 — undershoot is far more often playback drift than dropped events).

### 7. `pkill -f` / `pgrep -f` kills the caller

`pkill -f "rerun /tmp/rrd/_toy.rrd"` matched the shell that was *running that
very command* and killed the session — twice, exit code 144. The pattern also
matches an agent's wrapper process. `cleanup.py` is deliberately PID-file based
(`$UITEST_DIR/pids`, `pid<TAB>label` per line). Never pattern-match on process
names here.

### 8. `--web-viewer` opens the user's real browser

See the route section. Use `--serve-web`.

### 9. The canvas reports `300x150` right after navigation

That is the HTML default; `eframe` resizes it a moment later. Reading canvas
dimensions during startup gives a meaningless answer. `launch.py` re-queries
after the settle stage.

### 10. Readiness needs three stages

Polling the HTTP port only proves the *server* is up. The 40 MB WASM still has
to download, boot, connect to the gRPC proxy, receive the recording, and paint.
`launch.py` waits for: port open → `<canvas>` present → two screenshots 0.6 s
apart differing by `< 0.002`. Stopping early yields screenshots of a blank
viewer.

### 11. The `rerun` CLI is a Python wrapper — kill the process *group*

`/home/anton/miniconda3/envs/robo/bin/rerun` is a Python shim that spawns the
real binary (`rerun_sdk/rerun_cli/rerun`) as a child. `subprocess.Popen` hands
you the shim's PID; a plain `kill` on it leaves the actual server running and
holding the port. `launch.py` uses `start_new_session=True` and `cleanup.py`
uses `os.killpg`, so both die together.

Related: if you share `$UITEST_DIR` or `scripts/uitest/` with another agent or
session, `cleanup.py` will only ever touch PIDs *it* recorded — verified in
practice, it correctly left another session's `rerun` processes alone. This is
another reason not to fall back on pattern-matching.

### 12. Chrome's `websockets` frame limit truncates screenshots

CDP returns screenshots as base64 in a single WS message, several MB at
1600x1000. The `websockets` default `max_size` is 1 MiB and the connection dies
without a useful error. `CDP` passes `max_size=None`.

### 13. Panel layout is in the blueprint; TREE EXPANSION is not

The saved blueprint pins the panels, so region constants are safe. It does
**not** pin how far the blueprint panel's entity tree is unfolded, and that is
not stable between viewer sessions: the same `.rrd` opened expanded once and
collapsed the next time. With `world` collapsed, the row at the y where `cam`
used to be is `chain`, so "hide the camera frusta" quietly hid the chain panel
— and *that changed 0.38 of the 3D region*, so it looked like a pass.

Never trust a remembered row y. `ensure_tree_expanded()` clicks each expander
and keeps the result only if the tree got **longer**, then `find_rows()`
measures where the rows actually landed.

### 14. Row detection is an off-by-one waiting to happen

`find_rows()` first took "the longest run of ink bands spaced 24 px apart".
The panel's own **"Blueprint" header sits exactly one pitch above the first
row** and joins the run, shifting every name down one: `prim`→`overlay`,
`cam`→`world`. The symptom was a `0.99` changed_frac and a blank white view —
which is what an off-by-one looks like when your only feedback is pixels.

Fix: anchor the run on a click target you have already *proven* lands where you
think (here the expander that successfully opened the tree), and walk from
there. Row *pitch* is reliable; row *origin* is not — it rides on the height of
the Sources panel above, which grows with the number of loaded recordings.

### 15. Area metrics lie in a layered 3D scene

Three separate attempts to assert "the camera preview dome is gone" all failed,
in different directions, because the dome **occludes** brighter things:

| metric | before → after hiding | expected |
|---|---|---|
| warm (R≫B) pixels | 3403 → **4807** ↑ | it was hiding the red axis gizmos |
| non-background pixels | 151k → **235k** ↑ | it was hiding the floor grid |
| eroded mid-tone area | 56.9k → 40.8k (−28%) | the table patch is also a filled surface |

What works is scoring a **colour nothing else in the scene uses**: the cyan
frustum colour goes `21774 → 0`, and the yellow AABB goes `755 → 8538` as it
comes out from under the dome. Assert on identity, not on area. If a metric
needs its threshold chosen after seeing the answer, it is not evidence — say so
and pick a different metric.

### 16. Native capture on Hyprland: scale, tearing, and re-tiling

`native_hypr.py` exists because the *deliverable* is the native app even though
the *harness* is the browser. Three things had to be solved:

- **Logical vs physical pixels.** `hyprctl clients` reports `at`/`size` in
  logical points (a 3440x1440 monitor at `scale 1.25` is a 2752x1152 space);
  `grim` writes the physical framebuffer. Cropping one with the other's numbers
  gives a plausible-looking picture that is off by 25% — the left panel was
  sliced in half. Multiply by `hyprctl monitors -j | .[].scale`.
- **`grim -g` returns torn frames** right after a resize: two thirds of a
  1400x900 capture was the previous frame. Grab the **whole output** and crop in
  PIL, and throw the first grab away.
- **Hyprland silently re-tiles.** A window pinned to 2200x1300 came back as
  1366x1068 the moment another client opened. `place()` asserts the geometry in
  a retry loop and is called **again** right before the capture.

Also: `native_hypr.py` floats the window because fixed geometry demands it, but
that is a harness behaviour. The daily-use path (`rr/view.py show`) launches via
`hyprctl dispatch exec "[tile] …"` so the viewer arrives **tiled** and needs no
`MainMod+O`.

### 17. Never name a scratch script after a stdlib module

`/tmp/bisect.py` shadowed the stdlib `bisect`, which `random` imports, which
`rerun` imports — producing a circular-import error inside the *SDK* that looks
like a broken install. Ten wasted minutes. Same hazard: `code.py`, `types.py`,
`select.py`.

---

## Determinism achieved

Worth stating, because it makes strict thresholds usable:

- **`act.py reload` returns the app to a pixel-identical baseline**
  (`changed_frac == 0.0`), including undoing entity-visibility toggles. Use it
  between test cases instead of restarting the viewer.
- **Toggling an entity off and back on restores the exact original pixels**
  (`changed_frac == 0.0`).
- Headless Chrome falls back to **SwiftShader** (the viewer shows a "Software
  rasterizer" badge). Slower, but it removes GPU-driver nondeterminism, which is
  a good trade for a test rig. `--headful` uses the real GPU if you need it.
- `demo_rerun.py` passes 9/9 on repeated runs.

## Threshold guidance

`changed_frac` is the fraction of pixels in the region whose max-channel
difference exceeds `--tol` (default 12/255). Observed magnitudes:

| change | region | `changed_frac` |
|---|---|---|
| identical frames | any | `0.000` |
| `#0` → `#30` text | tight 60x20 box on the digits | `0.070` |
| `#0` → `#30` text | loose 205x22 field | `0.019` |
| 2D image swap | the 2D view | `0.507` |
| point cloud grows | the 3D view | `0.089` |
| entity hidden | the 3D view | `0.115` |
| tree row selected | the tree panel | `0.141` |

**Tighten the region around small text.** The same change reads `0.019` in a
loose box and `0.070` in a tight one — the loose box failed a `min-frac 0.02`
assertion that the tight box passes comfortably.

Reasonable defaults: `--min-frac 0.02` for "something changed",
`--max-frac 0.002` for "this region must not change".

Pick control regions carefully: **hiding an entity re-frames the 3D view**
(Rerun auto-fits the camera to visible content), so an "unrelated part of the 3D
view" assertion does not hold. The 2D view is a valid control for 3D changes and
vice versa.

---

## Reusing this for a different app

The only Rerun-specific pieces are `launch.start_rerun()`, `make_toy_rrd.py`
and the coordinates in `demo_rerun.py`. For any other web/WASM/Electron UI:

```bash
$PY launch.py --url http://localhost:3000    # skip rerun entirely
```

then `observe.py` / `act.py` / `imgdiff.py` / `crop.py` work unchanged. The
readiness heuristic (`wait_ready`) is app-agnostic — it waits for a canvas and
for the frame to stop changing; drop the canvas requirement for a DOM app.

For a **native** app the loop shape survives but the transports change: replace
`observe` with a compositor screenshot tool and `act` with a real input
injector. On Wayland that means solving the `/dev/uinput` permission problem
first — see the rejected-route evidence above.

## Environment

Verified on: Ubuntu 24.04, Hyprland 0.56.2 (Wayland), Chrome from
`/opt/google/chrome`, `rerun-sdk == 0.37.0` and the `rerun` CLI from
`/home/anton/miniconda3/envs/robo/bin/`. Python deps: `websockets`, `numpy`,
`Pillow` — all already present in the `robo` env. No `pip install` required.
