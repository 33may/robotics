# D405 frame-timeout investigation — USB path, kernel logs, bandwidth

Host `vbti-MS-7E66`, Ubuntu 24.04, kernel `6.17.0-1018-realtime` (PREEMPT_RT).
Investigated 2026-08-24. **Read-only inspection — the camera was never opened, no
streaming program was run, the robot was not touched.**

Symptom under investigation: `pipe.wait_for_frames(timeout_ms=2000)` raised
`RuntimeError: Frame didn't arrive within 2000` three times during an ~18 min run,
each recovering on the very next 10 Hz retry. Two clustered near startup, one later.

---

## 0. Headline

**The USB path is not the problem, and I can show that with a counter rather than an
argument.** `uvcvideo`'s own per-stream statistics for the exact run in question show
**16583 / 16582 / 16572 frames delivered with `errors: 0`, `empty: 0`, `invalid: 0`**
on all three UVC streams. 16583 frames ÷ 15 fps = **1105.5 s**, against a run that
`run.json` timestamps at 1100.7 s — a 0.4 % match. The camera delivered a
mathematically complete, error-free 15.00 fps stream for the entire run.

So the three timeouts happened **above the kernel** — inside librealsense or in how the
application drives it. Source research (§5b) supplies the mechanism: librealsense's V4L2
backend silently drops frames in *userspace* — `verify_vd_md_sync()` discards a frame
whenever the video and metadata buffer sequence numbers disagree, logging only at
`LOG_WARNING` and never touching `dmesg` or the kernel counters. Intel documents exactly
this as a **known issue on kernel 5+** ("consecutive frame drops … when streaming
Depth+IR+RGB … within 1-3 minutes … affected by CPU and resource utilization"), and their
recommended mitigation is the RSUSB backend — which the installed wheel is **not**.

The three secondary findings (autosuspend enabled, the three-hub chain, kernel 6.17
outside the supported matrix) are real and worth addressing, but only the last of them is
implicated by the evidence.

**The single most valuable next step is not a hardware change — it is turning on
librealsense debug logging for one run** (action B1). Every mechanism named above logs;
none of it is currently captured, because the app logs to stdout only.

---

## 1. USB topology, power and speed per tier

Device path: `/sys/bus/usb/devices/4-2.4.4.4`, on xHCI controller `0000:10:00.3`
(PCI `00:08.1`, an AMD chipset controller). Buses 3 (480M) and 4 (5000M) are the
USB2/USB3 halves of that same controller.

| Tier | sysfs | Device | ID | Speed | `bMaxPower` | `bmAttributes` | Power class |
|---|---|---|---|---|---|---|---|
| root | `usb4` | xHCI root hub (`0000:10:00.3`) | `1d6b:0003` | 10000 | — | — | host |
| hub 1 | `4-2` | VIA Labs VL813 "USB3.0 Hub" `bcdDevice 90.11` | `2109:0813` | 5000 | 0 mA | `0xe0` | **self-powered**, remote-wakeup |
| hub 2 | `4-2.4` | VIA Labs VL813 "USB3.0 Hub" `bcdDevice 90.11` | `2109:0813` | 5000 | 0 mA | `0xe0` | **self-powered**, remote-wakeup |
| hub 3 | `4-2.4.4` | VIA Labs VL813 "USB3.0 Hub" `bcdDevice 90.15` | `2109:0813` | 5000 | 0 mA | `0xe0` | **self-powered**, remote-wakeup |
| device | `4-2.4.4.4` | Intel RealSense D405, `bcdUSB 3.20` | `8086:0b5b` | **5000** | **720 mA** | `0x80` | bus-powered, no remote-wakeup |

Decoding: `bmAttributes 0xe0` = bit7 reserved-set + bit6 **Self Powered** + bit5 Remote
Wakeup. `bMaxPower 0 mA` on all three hubs is the correct declaration for a
self-powered hub (it draws nothing from upstream). The D405's `0x80` = bus-powered,
which is expected.

Negotiated link speed for the camera: **`speed = 5000`, `rx_lanes = 1`, `tx_lanes = 1`**
— i.e. it trained at full USB 3.2 Gen1 x1 (SuperSpeed 5 Gbps). It did **not** fall back
to 480M. This is the single most important power/cabling sanity check and it passes.

Hub descriptors (all three identical):

```
wHubCharacteristic 0x0009
  Per-port power switching
  Per-port overcurrent protection
bPwrOn2PwrGood      100 * 2 milli seconds
bHubContrCurrent      0 milli Ampere
 Port 4: 0000.0263 5Gbps power suspend enable connect
Self Powered
```

Per-port over-current protection is present, and the camera's port reports:

```
/sys/bus/usb/devices/4-2.4.4/4-2.4.4:1.0/4-2.4.4-port4/over_current_count = 0
```

**Is 720 mA safe through this chain? On the evidence available, yes.**
- *Verified:* all three hubs declare themselves self-powered with per-port power
  switching; a compliant self-powered USB3 hub must offer 900 mA per downstream port,
  which covers 720 mA with margin.
- *Verified:* `over_current_count = 0` — the hub's own over-current detector has never
  tripped on that port.
- *Verified:* the link trained and stayed at 5000 Mbps. Marginal power on a D4xx
  classically shows up as a fallback to 480M or repeated re-enumeration; neither happened.
- **Inferred / NOT verified:** I read *descriptors*, not a multimeter. Cheap hubs
  routinely declare `Self Powered` while running bus-powered with no barrel jack
  connected. I cannot tell from software whether a DC adapter is physically plugged
  into each of the three hubs. **Anton should physically confirm this.** If any hub in
  the chain is actually bus-powered, the 720 mA budget is genuinely at risk and all of
  the above descriptor evidence is worthless.

Parallel USB2 chain: `3-2 → 3-2.4 → 3-2.4.4`, the same three physical VL813s
(`2109:2813`, the USB2 half), **with no devices attached to any of them**. Bus 1/2
(keyboard, mouse, Bluetooth, MSI Mystic Light) is on a *different* xHCI controller
(`0000:0e:00.0`), so it shares no bandwidth or interrupt path with the camera.

**Nothing else is on the camera's host controller at all.** Bus 4 contains exactly the
three hubs and the D405; bus 3 contains three empty hubs. `/proc/interrupts` confirms
IRQ 56 (`0000:10:00.3`) is the camera's only traffic source.

---

## 2. Kernel-level USB errors — the logs are clean

Checked: `sudo dmesg -T`, `sudo journalctl -k -b 0`, `-b -1` (Aug 21), `-b -2` (Aug 20),
and `journalctl -k -b 0 -p warning`. Grepped for `reset`, `-EPROTO`, `-ENOENT`, `babble`,
`device not accepting address`, `over-current`, `disconnect`, `link state`, `Cannot enable`,
`halted`, `Stall`, `LPM`.

### Statement: there are **no** USB errors during or around any run.

Boot 0 is `Mon 2026-08-24 10:14:36 CEST`. The complete set of USB lines involving the
camera or its hubs after enumeration is:

```
[Mon Aug 24 10:14:27 2026] usb 4-2.4:   reset SuperSpeed USB device number 3 using xhci_hcd
[Mon Aug 24 10:14:28 2026] usb 4-2.4.4: reset SuperSpeed USB device number 4 using xhci_hcd
[Mon Aug 24 10:26:01 2026] usb 4-2.4.4.4: new SuperSpeed USB device number 5 using xhci_hcd
[Mon Aug 24 10:26:01 2026] usb 4-2.4.4.4: New USB device found, idVendor=8086, idProduct=0b5b, bcdDevice=50.e0
[Mon Aug 24 10:26:01 2026] usb 4-2.4.4.4: SerialNumber: 125423070926
[Mon Aug 24 10:26:01 2026] uvcvideo 4-2.4.4.4:1.1: Unknown video format 00000050-0000-0010-8000-00aa00389b71
[Mon Aug 24 10:26:01 2026] usb 4-2.4.4.4: Found UVC 1.50 device Intel(R) RealSense(TM) Depth Camera 405  (8086:0b5b)
[Mon Aug 24 10:26:01 2026] usbcore: registered new interface driver uvcvideo
[Mon Aug 24 10:26:01 2026] usb 4-2.4.4.4: UVC non compliance: permanently disabling control 981ae2 (Region of Interest Auto Ctrls), due to error -5
```

**That is the entire list.** Nothing at all between 10:26:01 and the end of the boot —
and the run in question was 12:11–12:29. No resets, no `-EPROTO`, no babble, no
disconnects, no link-state changes, no `uvcvideo` frame warnings.

Interpretation of the three non-clean-looking lines, none of which is a fault:
- The two `reset SuperSpeed` lines at 10:14:27/28 are **hub enumeration resets**, 12
  minutes *before* the camera was even plugged in (10:26:01). Normal VL813 behaviour
  when the kernel reads the hub descriptor.
- `Unknown video format 00000050-...` — the in-tree `uvcvideo` does not recognise one of
  the D405's proprietary formats on interface 1. Cosmetic; librealsense handles that
  format itself.
- `UVC non compliance: permanently disabling control 981ae2 (Region of Interest Auto
  Ctrls), due to error -5` — the D405 firmware rejects a UVC control probe the kernel
  makes at bind time. This is a well-known, benign D4xx/uvcvideo interaction that
  happens once at plug-in, not during streaming.

Earlier boots: Aug 21 (`boot -1`) and Aug 20 (`boot -2`) show **the same clean pattern** —
enumeration plus the same single `981ae2` line, nothing else.

One further positive: `usb usb4: We don't know the algorithms for LPM for this host,
disabling LPM.` — **USB3 link power management (U1/U2) is globally disabled on the
camera's host controller.** LPM is a classic source of SuperSpeed stalls; it is off
here, so it can be struck off the list entirely.

### Kernel-level frame statistics (the decisive evidence)

`uvcvideo` keeps per-stream counters in debugfs, reset at each stream start and
retained after the stream stops. These are from **the run in question**:

```
## /sys/kernel/debug/usb/uvcvideo/4-5-1/stats     (interface 1 — depth, Z16)
frames:  16583   packets: 16584   empty: 0   errors: 0   invalid: 0
pts: 0 early, 16583 initial, 16582 ok
scr: 16583 count ok, 16583 diff ok

## /sys/kernel/debug/usb/uvcvideo/4-5-2/stats     (interface 2 — IR pair, Y8I)
frames:  16582   packets: 16583   empty: 0   errors: 0   invalid: 0
pts: 0 early, 16582 initial, 16581 ok
scr: 16582 count ok, 16582 diff ok

## /sys/kernel/debug/usb/uvcvideo/4-5-3/stats     (interface 3 — colour, YUY2)
frames:  16572   packets: 16573   empty: 0   errors: 0   invalid: 0
pts: 0 early, 16572 initial, 16571 ok
scr: 16572 count ok, 16572 diff ok
```

Reading these:
- `errors: 0`, `invalid: 0`, `empty: 0` — **not one corrupt, truncated or empty UVC
  payload in ~50,000 frames across three streams.** Signal-integrity problems, power
  brownouts and flaky hubs all manifest in these counters. They are all zero.
- `scr: N count ok, N diff ok` — every single frame's source-clock delta was consistent.
  A device-side stall would perturb this.
- The three streams stayed within 11 frames of each other over 18 minutes.
- **Timing check:** 16583 frames ÷ 15 fps = 1105.5 s. `run.json`'s final step is
  `t = 1100.716 s`, and the run's capture directories run 12:11:35 → 12:29:21. The
  stream ran ~4.5 s longer than the run itself, which is exactly the `open_camera()`
  warm-up plus teardown. **The camera produced a complete 15.00 fps stream with no
  multi-second gaps.**

Corroboration that these counters really are the run in question (they reset at each
stream start and nothing has streamed since):

```
/sys/bus/usb/devices/4-2.4.4.4/power/runtime_active_time = 1327497 ms = 22.12 min
  seeded run streaming (16583 / 15 fps)                  = 1105.5 s = 18.43 min
  remainder for the 2408-geomtest run + probing          =  222.0 s =  3.70 min
/sys/bus/usb/devices/4-2.4.4.4/power/runtime_status      = suspended
lsof /dev/video*                                         = (nothing)
```

The 18.43 min of streaming plus the ~1.5 min `2408-geomtest` run at 11:50–11:51 and
plug-in probing account for the device's entire 22.12 min of lifetime activity.

*Caveat, stated honestly:* these counters tick when `uvcvideo` completes a frame
buffer, which is independent of whether userspace dequeued it. So they prove the
**device and the USB link** were healthy; they do **not** prove librealsense's
userspace thread kept up. That asymmetry is precisely what points the finger upward.

---

## 3. USB autosuspend — enabled on the camera *and* all three hubs

This is a real misconfiguration. It is not supported as the cause by the logs, but it
should be fixed anyway.

```
4-2         ctrl=auto  delay=0      status=suspended   USB3.0 Hub
4-2.4       ctrl=auto  delay=0      status=suspended   USB3.0 Hub
4-2.4.4     ctrl=auto  delay=0      status=suspended   USB3.0 Hub
4-2.4.4.4   ctrl=auto  delay=2000   status=suspended   Intel(R) RealSense(TM) Depth Camera 405
```

- `/sys/module/usbcore/parameters/autosuspend = 2` — the Ubuntu default, which is where
  the camera's 2000 ms delay comes from.
- The camera is **runtime-suspended right now**, and so is the entire hub chain. The
  port state confirms it: `Port 4: 0000.0263 5Gbps power suspend enable connect`.
- `power/runtime_suspended_time = 6866236 ms` vs `runtime_active_time = 1320885 ms` —
  the device spends most of its life suspended.
- For contrast, the keyboard and mouse show `ctrl=on` (usbhid pins them); **nothing
  pins the camera.** No udev rule sets `power/control=on` for `8086:0b5b`.
- I checked `/etc/udev/rules.d/99-realsense-cams.rules`: it only creates `cam_*`
  symlinks by V4L2 serial and **does not touch power management at all**. Note also
  that this rule file lists four *other* cameras (`125423070759`, `130523070141`,
  `125423070468`, `125423070032`) and does not cover the attached one.
- `/lib/udev/rules.d/60-autosuspend.rules` and the ChromiumOS hwdb do **not** match
  `8086:0b5b`, so the `auto` setting is simply the kernel default, not an explicit
  policy decision.

**Why it is only rank 4 rather than rank 1:** `uvcvideo` takes a PM reference while a
video node is open, so autosuspend cannot fire mid-stream. It can only bite in the
window between librealsense probing the device and starting the stream — which would
fit "clusters near startup". But a resume across three tiers is ~100 ms, not 2000 ms,
and a *failed* resume would have logged. **Fix it regardless: it is a one-line,
zero-risk change that removes a whole class of first-frame stalls.**

---

## 4. Bandwidth analysis

### What the app actually streams — verified

`/home/anton/projects/robotics/inspection/perception/camera.py`:

```python
WIDTH, HEIGHT, FPS = 848, 480, 15          # <- 15 fps, not 30
cfg.enable_stream(rs.stream.color,      width, height, rs.format.rgb8, fps)
cfg.enable_stream(rs.stream.depth,      width, height, rs.format.z16,  fps)
cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8,   fps)
cfg.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8,   fps)
```

Confirmed: 848x480, **15 fps** (the brief said "verify" — it is 15, not 30), four
logical streams: depth + IR left + IR right + colour.

### What that becomes on the wire — verified from UVC descriptors

Four *logical* streams do **not** mean four USB streams. Parsing the device's
VideoStreaming descriptors (`lsusb -v -d 8086:0b5b`) gives three streaming interfaces:

| Interface | Formats offered | Used for | bpp on the wire |
|---|---|---|---|
| IF 1 | `Z16` (16 bpp), `0x50` (16 bpp) | depth | 16 |
| IF 2 | `Y8` (8), **`Y8I` (16)**, `Y12I` (24), `UYVY` (16), `0x32` (8) | IR left + right | 16 |
| IF 3 | `YUY2` (16 bpp) | colour | 16 |

The IR pair travels as a single **`Y8I` interleaved 16-bpp stream** which librealsense
splits into two Y8 frames in software. Colour is `YUY2` on the wire, converted to
`rgb8` in software. So: **three UVC streams, all 848x480 @ 16 bpp @ 15 fps.**

That is corroborated by the debugfs layout — exactly three stream directories
(`4-5-1`, `4-5-2`, `4-5-3`) — and by the six V4L2 nodes, which are three capture nodes
plus three metadata nodes:

```
video0 ID_V4L_CAPABILITIES=:capture:      video1 ID_V4L_CAPABILITIES=:        (metadata)
video2 ID_V4L_CAPABILITIES=:capture:      video3 ID_V4L_CAPABILITIES=:        (metadata)
video4 ID_V4L_CAPABILITIES=:capture:      video5 ID_V4L_CAPABILITIES=:        (metadata)
```

### The arithmetic

```
per frame per stream : 848 x 480 x 2 B      =    814,080 B
per second per stream: 814,080 x 15         = 12,211,200 B/s = 97.7 Mbit/s
three streams        : 12,211,200 x 3       = 36,633,600 B/s = 293.1 Mbit/s  (34.9 MiB/s)
```

Against the link:

| Reference | Capacity | Utilisation |
|---|---|---|
| SuperSpeed raw line rate | 5000 Mbit/s | 5.9 % |
| After 8b/10b encoding | 4000 Mbit/s | **7.3 %** |
| Practical bulk throughput (~400 MB/s) | 3200 Mbit/s | **9.2 %** |

**~11x headroom. Bandwidth is categorically not the problem.**

### And there is no bandwidth *reservation* to fail

A finding I think is the most important structural fact about this device:

```
bInterfaceNumber 1/2/3:  bmAttributes 2  ->  Transfer Type   Bulk
                         wMaxPacketSize 0x0400 (1024 bytes), bMaxBurst 15
```

**The D405 streams over BULK endpoints, not isochronous.** That matters because the
classic USB-camera failure mode — the xHCI scheduler refusing an isochronous bandwidth
reservation, producing `-ENOSPC` / "not enough bandwidth" on a hub chain — **cannot
occur here at all.** Bulk takes whatever the bus has spare and is link-level retried
on error, which is also why `errors: 0` in §2 is such a strong result.

The trade-off is that bulk has no *latency* guarantee: it is best-effort, so a
congested bus can delay it. But with 9 % utilisation and literally no other device on
the controller, there is nothing to congest it.

### Competing load

- **On the same bus/hub chain: nothing.** Bus 4 = 3 hubs + the camera. Bus 3 (the USB2
  half of the same hubs) = 3 hubs, zero devices.
- **On the same host controller: nothing** beyond the above.
- Keyboard/mouse/Bluetooth are on `0000:0e:00.0`, a separate controller.
- The Chromium/QtWebEngine UI on software Vulkan (`lvp_icd.json`, lavapipe, is
  installed) consumes **CPU, not USB bandwidth**. It cannot contend for the bus. It
  can, however, contend for CPU — see cause #2 below.

---

## 5. Software stack and configuration

| Item | Value | Note |
|---|---|---|
| Kernel | `6.17.0-1018-realtime` | PREEMPT_RT, Ubuntu 24.04 |
| `uvcvideo` | in-tree, version `1.1.1` | **not** the Intel-patched module |
| pyrealsense2 | `2.58.3.10794` (pip wheel) | `~/miniconda3/envs/robo` |
| librealsense backend | **V4L2** | verified: the `.so` contains `src/linux/backend-v4l2.cpp` and `src/linux/backend-hid.cpp`; there are **no** RSUSB/libuvc backend source markers. libusb is linked but only for the firmware-update path. So `uvcvideo` is genuinely in the data path and the debugfs counters in §2 are authoritative. |
| Kernel cmdline | `usbcore.usbfs_memory_mb=1000` | the classic librealsense fix is **already applied** |
| `uvcvideo` params | `nodrop=1`, `timeout=5000`, `quirks=4294967295` | `quirks=0xFFFFFFFF` is the "unset" sentinel (default), not an override. `timeout=5000` is the *control* timeout, not a frame timeout. |
| USB metadata | **available** | three metadata V4L2 nodes exist, so librealsense can read hardware frame counters/timestamps |
| CPU | Ryzen 9 9900X, 24 threads | load avg 4.11 — not saturated |
| xHCI IRQ | `irq/56-xhci_hcd`, SCHED_FIFO prio 50, pinned CPU20 | `irqbalance` inactive |
| App RT priority | **none** — no `sched_setscheduler`/`chrt` anywhere in the codebase | so no app thread can preempt the FIFO-50 IRQ threads |

### Two false alarms I want to clear explicitly

**1. The serial numbers are not mismatched.** `camera.py` has
`WRIST_SERIAL = "123622270954"`, while `lsusb`/sysfs report `iSerial 125423070926`.
These are *different namespaces*: the USB descriptor carries the V4L2/module serial and
librealsense reports the ASIC serial. The existing udev rules document exactly this
pairing convention ("top (RS 123622270073 / V4L2 125423070759)"), and
`data/runs/2408-seeded/session.json` — written by the successful run — records
`"serial": "123622270954"`. The code is correct.

**2. The missing pose `003` is not a camera failure.** `2408-seeded` contains
`000,001,002,004,005,...` with no `003`. `run.json` step 4 reads
`"result": "software stop — arm halted mid-move", "stopped": true`, and step 5 retries
the same target `[10,0]` successfully. That gap is the arm, not the camera.

---

## 5b. What librealsense actually does above the kernel (from source research)

This section is from a source/issue-tracker review of librealsense 2.58, not from this
host. It matters because §2 proved the failure is above `uvcvideo`, and these are the
only three stages between `uvcvideo` and `wait_for_frames`.

**Correction to a common assumption:** the D405 has **no separate RGB imager**. Depth,
IR1, IR2 *and* colour all come from the **Stereo Module (MI 0)**, wrapped in
`platform::multi_pins_uvc_device` (`src/ds/d400/d400-device.cpp`: *"used when color
stream comes from depth sensor (as in D405)"*). So this is four logical streams from
**one** sensor, not two — which makes cross-sensor drift a *non*-issue and rules out one
of my earlier hypotheses. Intel's constraint that all D405 streams share resolution and
FPS is satisfied by the app.

**Gate 1 — the pipeline aggregator (`src/pipeline/aggregator.cpp`):**

```cpp
// in case not all required streams were aggregated don't publish the frame set
for (int s : _streams_to_aggregate_ids)
    if (!_last_set[s]) return;
```

`_last_set` is **never cleared**, so this gate blocks only until every enabled stream has
produced its *first* frame, then is permanently satisfied. With four streams, one late
first frame stalls `wait_for_frames` completely. **This bites inside `open_camera`'s
warm-up loop, not in the steady-state `CameraWorker`** — by the time `start_camera()`
runs, the gate is already satisfied for the life of the pipeline.

**Gate 2 — `timestamp_composite_matcher::skip_missing_stream` (`src/sync.cpp`):**

```cpp
auto gap = 1000. / fps;
auto threshold = 7 * gap;          // ~467 ms at 15 fps
if (now - next_expected.value < threshold) return false;
LOG(... "exceeded cutout of {NE+7*gap} ... deactivating matcher!");
```

**This is the most useful number in the whole investigation.** At 848x480@15 the syncer
will hold a frameset for at most **~467 ms** waiting for a late stream, then give up and
release without it. **A single dropped frame therefore cannot produce a 2000 ms
timeout.** To hit 2000 ms you need roughly **30 consecutive misses on one stream**, or a
genuine multi-second stall of that stream. That is a hard lower bound on the severity of
whatever is happening.

**Gate 3 — librealsense's own silent drop path, `verify_vd_md_sync()`:** the V4L2 backend
compares the video buffer's `sequence` against the *metadata* buffer's `sequence`; on
mismatch it emits only `LOG_WARNING("Video frame dropped, video and metadata buffers
inconsistency")` and **drops the frame without invoking the callback**. The kernel
completed both buffers perfectly — so **this drop is invisible in `uvcvideo`'s debugfs
counters and invisible in `dmesg`.** Separately, the backend detects bad frames by
comparing `buf.bytesused` to the expected payload and logs `Incomplete frame received`,
also a silent userspace drop. (librealsense never checks `V4L2_BUF_FLAG_ERROR` at all.)

**Intel's own documented known issue** (librealsense release notes, v2.40–2.45 era) is
the closest published match to this symptom: on kernel 5+ versus kernel 4 there is *"a
higher frame drop rate intensified by consecutive frame drops, mostly between 2-4 frames
in a row, and reaching 7 frames in certain cases"*; *"when streaming Depth+IR+RGB+IMU,
the frame drops appear within 1-3 minutes"*; and *"the recurrence rate is affected by CPU
and resource utilization"*. **Intel's recommended mitigation is to build with
`-DFORCE_RSUSB_BACKEND`** — which the installed wheel is *not* (§5).

**Version support:** the v2.58.3 release notes list *"Kernel versions: 6.[2, 5, 8, 11,
14], 5.[0, 3, 4, 8, 13, 15, 19]"*. **Kernel 6.17 is not on that list.** Metadata itself
is fine unpatched (`V4L2_META_FMT_D4XX` has been mainline since ~4.20, and this host has
the metadata nodes), so this is a validation gap rather than a known breakage.

**`nodrop`:** the kernel default flipped to `1` in Dec 2024 ("media: uvcvideo: Invert
default value for nodrop module param"), so this host's `nodrop=1` is the 6.17 default,
not a local override. Setting it back to `0` would **not** help — librealsense drops the
bad frame either way; `nodrop=0` just moves the drop into the kernel.

**PREEMPT_RT:** no librealsense issue links PREEMPT_RT to frame timeouts (the RT-adjacent
issues are all DKMS build failures). But the mechanism is real and Intel half-documents
it via "affected by CPU and resource utilization". On this host the risk is low: the app
sets no RT priorities at all, so nothing can preempt the FIFO-50 `irq/56-xhci_hcd` thread.

---

## 6. Ranked probable causes

### #1 — A silent librealsense-userspace frame drop, sustained for ~30 frames

Most likely `verify_vd_md_sync()` (video/metadata buffer sequence inconsistency) or the
`bytesused` incomplete-frame path, i.e. **Intel's documented kernel-5+ consecutive-drop
known issue** manifesting on an unsupported kernel 6.17.

**Evidence for — this is the only hypothesis consistent with every single measurement:**
- It drops frames **entirely in userspace, after the kernel has completed both the video
  and the metadata buffer**. That is precisely why `uvcvideo`'s counters read
  `errors: 0, empty: 0, invalid: 0` (§2) and `dmesg` is clean (§2) while
  `wait_for_frames` still starves.
- Intel's own note says the drops are *consecutive* and cluster when streaming
  **Depth+IR+RGB** — exactly this app's configuration — and that the rate is *"affected
  by CPU and resource utilization"*, which supplies the "clusters near startup" pattern
  when the software-Vulkan UI launches.
- The ~467 ms syncer bound (§5b) says a 2000 ms timeout **requires** ~30 consecutive
  misses on one stream. "Consecutive drops" is the defining feature of Intel's known
  issue.
- Kernel 6.17 is outside librealsense 2.58.3's validated matrix (§5b).
- Three events in 18 min with instant recovery is the right order of magnitude for a KPI
  that Intel measures in fractions of a percent.

**Evidence against / not yet proven:** every one of these drop paths logs at
`LOG_WARNING`/`LOG_DEBUG`, and librealsense logging was never enabled, so **there is
currently zero direct evidence either way.** Turning it on (action B1) is cheap and
decisive. Also, Intel quotes "2-4 frames in a row, reaching 7" — an order of magnitude
short of the ~30 needed, so either the mechanism is worse here than Intel measured, or
something amplifies it.

### #2 — CPU starvation of librealsense's V4L2 dequeue thread during UI startup

**Evidence for:**
- Explains "two events clustered near startup" better than anything else. In
  `run/app.py` the ordering is:
  ```
  rig.start_camera(pub)      # line 119 — CameraWorker begins grabbing at 10 Hz
  poses = PoseStreamer(...)  # 30 Hz publisher starts
  threading.Thread(target=pump, ...)
  child = _open_ui(...)      # <- QtWebEngine + Chromium on SOFTWARE Vulkan launches HERE
  sup.run()
  ```
  The camera worker starts grabbing and then, moments later, a software-rasterised
  browser engine spins up. `ui/publisher.py` documents the consequence in its own
  comment: *"streaming 848x480 at 10 Hz froze the 3D panel … QtWebEngine on a software
  Vulkan fallback ('GBM is not supported')"* — the UI is already known to be starved on
  this box.
- The debugfs counters cannot see this: they tick when `uvcvideo` completes a buffer,
  regardless of whether librealsense dequeued it. A stalled *userspace* thread produces
  exactly the observed signature — perfect kernel counters, timed-out `wait_for_frames`.
- The `publish` callback runs **on the CameraWorker thread**, between grabs, adding to
  the same thread's load. (Checked: `PortholeBus.publish` hands the frame to an asyncio
  loop via `_call_on_loop` and does **not** block on a slow subscriber, so the cost is
  the JPEG encode itself, not UI backpressure. `live_stride=2` already quarters it.)

- This is the documented amplifier for cause #1: Intel's known issue explicitly says the
  recurrence rate *"is affected by CPU and resource utilization"*, and librealsense's
  [`doc/frame_lifetime.md`](https://github.com/IntelRealSense/librealsense/blob/master/doc/frame_lifetime.md)
  warns that failing to release a frame within `1000/fps` ms (66.7 ms here) causes drops.
  `grab_aligned` does `align.process()` plus five array copies, and the JPEG encode
  follows on the same thread.

**Evidence against:** load average was 4.11 on 24 threads — no global saturation. A 2 s
stall needs more than ordinary contention. Plausible during a QtWebEngine/lavapipe
cold start, less so for the third, later event. Best read as the *trigger* for #1 rather
than an independent cause.

### #3 — The pipeline aggregator's all-streams-first-frame gate (startup only)

**Evidence for:** verified in source (§5b) — with four streams enabled, `wait_for_frames`
returns nothing until *every* stream has produced its first frame, and one slow imager
start (AE convergence) stalls the lot.

**Evidence against:** this gate is latched permanently once satisfied, so it can only fire
inside `open_camera`'s warm-up loop — and a failure there would have raised out of
`RealRig.__init__` and killed the run before it started, which is not what happened.
Listed because it is a **real startup fragility** worth fixing (action B4), not because
it explains these three events.

### #4 — USB autosuspend on the camera and all three hubs

**Evidence for:** `power/control=auto` on all four devices; camera delay 2000 ms;
everything currently suspended (§3). Autosuspend on a UVC device is a textbook cause of
first-frame stalls, and the "clusters near startup" pattern fits the window between
librealsense's probe and stream start.

**Evidence against:** `uvcvideo` holds a PM reference while streaming, so it cannot fire
mid-run; a three-tier resume is ~100 ms, not 2000 ms; and a failed resume would have
appeared in `dmesg`, which is clean.

**Two research findings worth recording here:**
- Intel does **not** document autosuspend as a cause. The shipped
  `config/99-realsense-libusb.rules` sets only `MODE`/`GROUP` — no `power/control` — and
  there is no `usbcore.autosuspend=-1` recommendation anywhere in librealsense's docs.
  Community reports are contradictory (issues #8694, #8526 — one user's rule explicitly
  did *not* help).
- A common claim is that "the kernel initialises `power/control=on` for all non-hub
  devices, so a UVC camera never autosuspends". **That is not true on this host and I
  verified it directly:** `4-2.4.4.4/power/control = auto`,
  `autosuspend_delay_ms = 2000`, `runtime_status = suspended`. Trust the measurement over
  the generalisation. **Still worth fixing — cheap and risk-free — but expect nothing.**

### #5 — The hub chain, USB3 LPM, cable and connector quality

**Evidence for:** VIA VL81x hubs are documented offenders on the linux-usb list for
device-initiated U1/U2 link power management. A wrist-mounted USB3 cable flexes on every
move and is the highest-risk mechanical element in the system. There is also a **tier
count question worth checking physically**: a "7-port" VL813 box is internally two
cascaded 4-port chips and shows up as `2109:0813` twice, so three VL813 *chips* may be
fewer than three physical boxes — or, if there really are three boxes, the chain could be
up to six tiers, past the USB 5-tier limit.

**Evidence against — this is strong, and the LPM concern is specifically neutralised
here:**
- `errors/empty/invalid = 0` across ~50,000 frames; `over_current_count = 0`; link
  trained and held at 5000 Mbps; 9 % bandwidth utilisation; no resets in any boot's log.
- **U1/U2 LPM is already globally disabled on this controller** by the kernel itself:
  `usb usb4: We don't know the algorithms for LPM for this host, disabling LPM.` So the
  known VL813 LPM defect cannot be in play on this machine, and
  `usbcore.quirks=2109:0813:k` would be a no-op.
- Intel has never said "don't chain" — the USB spec allows five tiers, and Intel's
  multi-camera white paper only stresses that hubs be *externally powered*.

**On this run's evidence the hub chain behaved perfectly.** It remains a latent risk and
the highest-value thing to *simplify*, but it did not cause these three timeouts.

### #6 — Insufficient power

**Evidence against:** everything in §1 — self-powered declarations, zero over-current
events, 5 Gbps link held, no re-enumeration, zero UVC payload errors. **Contingent on
Anton physically confirming the hubs' DC adapters are actually connected** (§1); if they
are not, this jumps near the top. Note this is the *one* thing Intel's own multi-camera
white paper insists on for hub setups.

### #7 — Missing udev rules

The realsense rules present (`99-realsense-cams.rules`) don't even cover this camera, but
they only create cosmetic `cam_*` symlinks. Intel's own `99-realsense-libusb.rules` only
sets `MODE`/`GROUP`, and permissions are evidently fine since the camera streams. Not a
contributor. `usbfs_memory_mb=1000` is already set.

*(The kernel-version mismatch that used to sit here has been promoted into cause #1,
where the evidence actually points.)*

---

## 7. A note on the 2000 ms timeout and the `CameraWorker` retry design

**Is 2000 ms appropriate?** For the steady-state 10 Hz loop, yes — generously so. At
15 fps a frame is due every 66.7 ms, so 2000 ms is **30 frame periods**. It is not a
tight timeout that needs relaxing; anything that trips it is a real stall, not jitter.

The one place it is arguably *too tight* is `open_camera`'s warm-up loop, where the same
2000 ms is applied to the **first** `wait_for_frames` after `pipe.start()` — a moment
that legitimately involves sensor init, auto-exposure convergence, and (given §3) a
possible resume of a four-tier suspended USB chain. A first-frame timeout there raises
out of `RealRig.__init__` and kills the run before it starts.

**Is the retry design sound?** Largely yes, with two gaps:

Sound:
- `CameraWorker.run` catches every `grab()` exception, logs the first with a traceback,
  rate-limits the rest to one line per ~50 ticks, and continues. A transient timeout
  costs one tick.
- It logs recovery (`"camera grab recovered after %d failures"`), which is how these
  three events were noticed at all.
- `fresh_bundle(min_new=3, timeout=3.0)` requires three *new* grabs after the call, so a
  capture can never be served a stale pre-move frame. With a healthy 10 Hz loop three
  grabs take ~300 ms against a 3 s budget — 10x margin.
- `stop()` joins with a timeout and warns rather than hanging if `grab()` is wedged.

Gaps:
1. **A stall is invisible in `fresh_bundle`'s budget accounting.** A single 2 s timeout
   consumes two-thirds of `fresh_bundle`'s 3 s window. Two timeouts inside one capture
   would raise `"camera produced no fresh frames"` — a *capture* failure caused purely by
   a transient. The margin is thinner than it looks.
2. **Nothing is persisted.** `run/app.py:87` is
   `logging.basicConfig(level=logging.INFO, format="%(message)s")` — stdout only, no
   `FileHandler`. There is no log file anywhere under the project or in the journal.
   **I could not recover the exact timestamps of the three events**, which is the single
   biggest obstacle to closing this out: with timestamps I could correlate them against
   `run.json` step boundaries and the UI startup instant and settle #1 vs #2 immediately.

---

## RECOMMENDED ACTIONS

### (a) Physical / hardware

**A1. Plug the D405 directly into a rear-panel root port. — cheapest experiment, but set
expectations**
Use `0000:10:00.3` (bus 4) or the other USB3 controller. This eliminates three tiers of
hub, three tiers of resume latency, and three tiers of power delegation in one move.
*Expected impact:* removes causes #4, #5 and #6 entirely. It does **not** address the
top-ranked cause, so it may well not stop the timeouts — **and that is exactly its
diagnostic value**: if the timeouts survive a direct root-port connection, the USB path is
conclusively exonerated and the whole investigation collapses onto librealsense.
*Verify:* `lsusb -t` shows the camera directly under a root hub; the camera's
`/sys/bus/usb/devices/*/speed` must still read `5000`. Then re-run and re-read the
debugfs counters.

**A1b. While you are at it, count the actual hub tiers.**
A "7-port" VL813 box is internally two cascaded 4-port chips, so the three `2109:0813`
devices in `lsusb -t` may be fewer than three physical boxes — or, if there genuinely are
three boxes, the chain may be up to six tiers, past the USB 5-tier limit. `bcdDevice`
gives a hint here: `4-2` and `4-2.4` both report `90.11` while `4-2.4.4` reports `90.15`,
which is consistent with the first two being one physical two-chip box and the third being
a separate unit.
*Verify:* physically count the hub enclosures against `lsusb -t`.

**A2. Confirm each hub's DC adapter is physically connected.**
The descriptors *claim* self-powered; I cannot verify that from software, and it is the
one hole in the §1 power argument.
*Expected impact:* either closes cause #6 for good, or promotes it to the top.
*Verify:* look at the barrel jacks. If A1 is done, this becomes moot.

**A3. Inspect and, if possible, shorten the wrist cable run.**
A wrist-mounted USB3 cable flexes on every move and is the most likely long-term failure
point, even though it is behaving now.
*Expected impact:* preventative; low probability of changing current behaviour.
*Verify:* `errors`/`invalid` in the uvcvideo debugfs stats must stay at 0 across runs.

### (b) Software / config

**B1. Turn on librealsense debug logging and persist the run log. — do this first; it is
decisive and costs nothing.**
Every mechanism in cause #1 logs, and none of it is currently captured. Set
`LRS_LOG_LEVEL=DEBUG` (or `rs.log_to_file(rs.log_severity.debug, ...)`) for one run, and
add a `FileHandler` writing to `outdir` alongside the stdout handler at `run/app.py:87`
— including a timestamp in the format, which the current `"%(message)s"` lacks entirely.
Then grep the log for:
```
Incomplete frame received                                     -> bytesused mismatch drop
Video frame dropped, video and metadata buffers inconsistency -> verify_vd_md_sync drop
deactivating matcher!                                         -> syncer gave up at ~467 ms
is not longer active, frame dropped!
frame drop? Expecting
```
*Expected impact:* no behaviour change. **This single run distinguishes causes #1, #2 and
#3 from each other**, and turns "three anecdotes" into timestamps correlatable against
`run.json` step `t` values and the `_open_ui` instant.
*Verify:* one of the five strings above appears (or provably does not) at each of the
three failure times.

**B2. Instrument the timeout rather than only counting it.**
In `CameraWorker`, record the wall-clock and the measured duration of each failed
`grab()`, and log the run-relative time. "2003 ms" (a clean librealsense timeout) versus
"2400 ms" (a thread that was also descheduled) separates cause #1 from cause #2.
*Verify:* one line per event carrying a duration.

**B3. Try the RSUSB-backend build — Intel's own documented mitigation for cause #1.**
The installed wheel is a V4L2 build (verified in §5); Intel's remedy for the kernel-5+
consecutive-drop known issue is `cmake -DFORCE_RSUSB_BACKEND=ON`. Their caveat is that
RSUSB *"is not suited for multi-cam scenarios"* — this rig is single-camera, so the
caveat does not apply.
*Expected impact:* if cause #1 is right, this is the fix. Cost is building librealsense
from source rather than `pip install`.
*Verify:* after rebuilding, the `.so` should contain `src/uvc/uvc-streamer.cpp`; then
compare timeout count per 18 min run. Note the kernel debugfs counters in §2 will no
longer be available, because `uvcvideo` leaves the data path.

**B4. Give `open_camera`'s first `wait_for_frames` a longer budget than the steady-state
loop.** The aggregator gate (§5b, cause #3) means the *first* frameset genuinely waits on
the slowest of four streams to start; 2000 ms there is a real startup-fragility risk, and
a failure kills the run inside `RealRig.__init__`.
*Expected impact:* removes a class of hard startup failures. Architecture call — Anton
decides the budget and whether the warm-up loop should retry rather than propagate.
*Verify:* repeated cold starts, especially after the machine has idled long enough for
the USB chain to suspend.

**B5. Disable USB autosuspend for the camera and its hubs.**
A udev rule matching `idVendor=8086`/`idProduct=0b5b` setting `ATTR{power/control}="on"`,
or `usbcore.autosuspend=-1` on the kernel cmdline for a blunter global fix.
*Expected impact:* removes cause #4 and a class of first-frame stalls. Cheap and
zero-risk, but the research says **expect nothing** — Intel doesn't document it and
community reports contradict each other.
*Verify:* `cat /sys/bus/usb/devices/4-2.4.4.4/power/control` reads `on` and
`runtime_status` stays `active` while the app is idle.

**B6. Reconsider the 4-stream pipeline design.** Options for Anton to choose between:
- Bypass both the aggregator and the syncer with `pipe.start(cfg, callback)` or per-stream
  `rs2::frame_queue`s. Note `wait_for_frames` then throws
  `wrong_api_call_sequence_exception`, so this restructures `grab_aligned`.
- Use `poll_for_frames()` in the `CameraWorker` tick, falling back to a blocking wait only
  when the poll is empty — a non-blocking loop cannot time out at all. Intel's own
  guidance (#2422) is that `wait_for_frames` is intended for simple single-camera use.
- Match producer and consumer rates: `FPS = 10` to match the 10 Hz grab loop, instead of
  discarding ~1 frameset in 3 continuously.
- Drop `color` if it isn't needed for the record — it is a fourth stream on the same
  Stereo Module and removing it cuts the aggregator's gate from four streams to three.
*Expected impact:* structural, targets #1/#2/#3 together. Highest design cost — an
architecture decision, not a config tweak.
*Verify:* run with B1/B2 in place and compare timeouts per 18 min run.

**B7. Do not bother with these** — checked, and each is either already correct or
provably irrelevant here:
- `usbcore.usbfs_memory_mb=1000` — already set on the kernel cmdline.
- USB3 U1/U2 LPM — already disabled by the host controller, so `usbcore.quirks=2109:0813:k`
  would be a no-op on this machine.
- `uvcvideo nodrop=0` — `nodrop=1` is the 6.17 kernel default, and librealsense drops the
  bad frame either way; this only moves the drop into the kernel.
- `uvcvideo quirks` — at its default `0xFFFFFFFF` sentinel, not an override.
- Metadata nodes — present and working.
- RT priorities — the app sets none, so nothing preempts the FIFO-50 xHCI IRQ thread.

---

## Verified vs inferred — summary

**Verified by direct measurement on this host:**
USB topology and per-tier speed/power descriptors; camera negotiated 5000 Mbps x1;
`over_current_count = 0`; all hubs declare self-powered with per-port over-current
protection; kernel logs clean across boots 0, -1 and -2; LPM disabled by the host;
autosuspend `auto` on the camera and all three hubs, all currently suspended; uvcvideo
per-stream frame/error counters for the run in question; three bulk streaming interfaces
with `Y8I` interleaved IR; three capture + three metadata V4L2 nodes; librealsense 2.58.3
built with the V4L2 backend; `usbfs_memory_mb=1000`; app enables 4 streams at 848x480@15;
no RT priorities in the app; `_open_ui` launches after `start_camera`; logs are not
persisted; run duration 1100.7 s vs 1105.5 s of streaming.

**Verified from librealsense/kernel sources and Intel's release notes (§5b), not on this
host:** the aggregator's never-cleared `_last_set` gate; the syncer's `7 * gap` ≈ 467 ms
cutout; `verify_vd_md_sync()` dropping frames silently in userspace; librealsense never
checking `V4L2_BUF_FLAG_ERROR`; the D405's colour stream originating from the Stereo
Module rather than a separate RGB imager; kernel 6.17 being absent from 2.58.3's supported
list; `nodrop`'s default flipping to 1 in Dec 2024; Intel's kernel-5+ consecutive-drop
known issue and its RSUSB mitigation.

**Inferred, not proven:**
That the hubs are genuinely externally powered (descriptor claim only — needs eyes on the
barrel jacks). **That `verify_vd_md_sync` or the incomplete-frame path is what fired here
— this is the top-ranked cause but there is currently zero direct evidence, because
librealsense logging was never enabled.** That QtWebEngine startup starves the
librealsense dequeue thread (timing coincidence with `_open_ui`, no direct measurement).
That IF2 carries the IR pair as `Y8I` (strong inference: three streaming interfaces must
carry four logical streams, and `Y8I` is present at 16 bpp on IF2 — but not observed on a
live stream). The mapping of debugfs stream `4-5-1/2/3` to depth/IR/colour (from
descriptor order). That the three timeouts fall where I think they do — **without
persisted logs their exact timestamps are unrecoverable, which is why B1 is the first
recommendation.**

**Explicitly contradicted by measurement on this host:** the common claim that the kernel
pins `power/control=on` for non-hub USB devices so a UVC camera never autosuspends — here
it reads `auto`, delay 2000 ms, currently suspended.
