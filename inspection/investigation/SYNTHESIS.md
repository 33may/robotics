# Why the loop was unstable — synthesis, 2026-08-24

Four independent investigations (`ur_rtde_library.md`, `app_client_audit.md`,
`camera_usb.md`, `network_link.md`) plus main-session code work. They
converged on **one mechanism** behind both the RTDE stream deaths and the
camera frame droughts.

## Root cause: reader threads starved of CPU on this machine

Not a network fault, not a USB fault, not a robot fault. Both subsystems fail
the same way: a thread that must drain a socket or a buffer *promptly* loses
the CPU for longer than the producer tolerates.

### The RTDE half — and why the controller hung up on us

UR's RTDE Guide states the failure verbatim:

> "The client should read data periodically from the socket. **The connection
> is closed by the robot controller when the receive buffer overflows.**"

That is the origin of `asio.misc:2 End of file` — `boost::asio::error::eof`,
i.e. the peer sent FIN. The controller was hanging up on us, deliberately,
because we were not reading fast enough.

The arithmetic explains why our margin was nil:

| | before | after |
|---|---|---|
| variables subscribed | ~30 (ur_rtde's default when `variables` is empty) | 4 |
| frequency | 500 Hz (e-Series default) | 125 Hz |
| wire rate | ~1180 B x 500 Hz ~= **590 KB/s** | ~67 B x 125 Hz ~= **8.4 KB/s** |
| time to overflow a 128 KB socket buffer | **~0.22 s** | **~15 s** |

128 KB is the kernel default and ur_rtde never calls
`setsockopt(SO_RCVBUF)`, so `net.core.rmem_max` is irrelevant — `tcp_rmem`
governs, and a stalled reader never autotunes upward.

**0.22 seconds** was the entire stall budget on a box also running an OMPL
planner, a websocket bus, and Chromium on *software* Vulkan.

Once the FIN arrives, ur_rtde makes it permanent: on any socket exception the
receive thread prints, calls `th_.signalStop()` and **exits forever — there is
no auto-reconnect** (`rtde_receive_interface.cpp:271-278`). `isConnected()`
reports `conn_state_` only and knows nothing about the thread. Getters are
bare cache reads, so they answer with the last good packet indefinitely. That
is the silent staleness we built the freshness guard for.

Independent corroboration from the kernel side: this host advertised a **zero
receive window 213 times** this boot (`TcpExtTCPToZeroWindowAdv`), against a
measured idle background of ~23 — so ~190 events attributable to the runs. A
zero window is by construction "userspace stopped reading".

Everything else about the transport was exonerated with hard numbers: every
`ethtool -S` error counter zero, 0% loss across 605 probes, 0.171 ms average
RTT, MTU clean, a switch (not a hub) with 0 collisions on 1.7M packets, and
**no NetworkManager event inside either run window** — the DHCP-renew theory
is dead.

### The camera half — same disease, different organ

The hub-chain premise I started from was **wrong**, and the evidence is
decisive: `uvcvideo`'s debugfs counters covering exactly this run show
16583/16582/16572 frames on the three streams with **errors 0, empty 0,
invalid 0**, and 16583 / 15 fps = 1105 s against a run of 1100.7 s. The kernel
delivered a complete, error-free 15.00 fps stream. Bandwidth is a non-issue
(293 Mbit/s, 7.3% of line rate) and the D405 uses **bulk, not isochronous**
endpoints, so hub-chain bandwidth reservation cannot fail at all.

So the drops are **above the kernel**, in librealsense's V4L2 backend, which
discards frames in userspace when video and metadata buffer sequences
disagree and logs it only at `LOG_WARNING` — invisible to `dmesg`. Intel
documents this as a known kernel-5+ issue, explicitly "affected by CPU and
resource utilization". A single drop cannot cause a 2000 ms timeout; it needs
~30 consecutive misses. And `run/app.py` launches the software-Vulkan
Chromium child *immediately after* `start_camera()` — which is where two of
the three droughts landed.

## Defects found in our own code

| | defect | effect |
|---|---|---|
| **D1** | `_ts_advancing()` checked its deadline at the TOP of the loop | Two reads legitimately landing on one 2 ms packet, plus a `sleep(0.002)` that overshoots under load, returned `False` **without re-reading** — declaring a HEALTHY stream dead and tearing down a working socket. Misfires only on a loaded machine, which is exactly why 23 isolated reproductions on an idle box all survived while every full run "died". |
| **D2** | no rate limit on reconnect | `PoseStreamer` swallows the RuntimeError and retries at 30 Hz, the safety poll at 20 Hz — an unrecoverable stream meant ~50 reconnect attempts/second at the controller. The guard could manufacture the very failure it exists to survive. Measured: 20 attempts in a 20-call burst. |
| **D3** | `stream_autopsy` forked `ss` (5 s timeout) inside `_assert_fresh`, under `_recv_lock` | During a move, the thread polling `_safety_ok()` **is** the thread watching `stop_event`. STOP could be stalled for seconds with the arm travelling. Introduced earlier the same day; caught by this audit. |

## Fixes applied

1. **Narrow the RTDE subscription** — `RECV_VARIABLES` (4) at
   `RECV_FREQUENCY` 125 Hz. 590 KB/s -> 8.4 KB/s; stall budget 0.22 s -> ~15 s
   (~70x). Directly addresses the documented cause.
   *Validated on the real controller*: all four getters return correct live
   values (robot_mode 7, safety_mode 1) and the observed packet rate is
   exactly 125 Hz. A mis-named variable would RAISE, not silently fake a
   value — verified in `rtde_receive_interface.cpp`, which matters because
   one of the four is the safety mode.
2. **D1** — read after every sleep; the deadline check moved below the read.
   Proven: the old loop returns `False` on a healthy descheduled stream, the
   new one returns `True`.
3. **D2** — `reconnect_backoff = 2.0 s`. Burst of 20 callers now yields 1
   attempt.
4. **D3** — `_assert_fresh(diagnose=False)` on the safety path; the autopsy
   runs only on `q()`, where the arm is parked.

All four covered by regression tests (`tests/test_freshness.py`, 12 tests).
Suite: 90 passed, 1 pre-existing unrelated failure (`test_grid_is_light`,
an import-order artifact that also fails with these changes stashed).

## Not yet done — ranked

1. **`reconnect()` reverts frequency to 500 Hz** (`rtde_receive_interface.cpp:292-295`)
   and spins in an unbounded `while (!getFirstStateReceived())` which, with
   `_recv_lock` held, would wedge the app rather than fail. It keeps
   `variables_`, so a healed stream still gets ~3.8 s of budget (17x better
   than before) — but rebuilding the interface instead would restore the full
   15 s and remove the wedge risk. Unvalidated on this cell; needs its own
   hardware test.
2. **`preflight()` runs twice** per start-up (`app.py:95` and again inside
   `UR5eArm.__init__`), and `RTDEControlInterface` internally opens its own
   RTDE + ScriptClient + DashboardClient. Peak is **5 simultaneous
   connections**. Harmless in isolation (a soak ran a higher count 5x and
   survived) but pointless.
3. **`self.ctrl` is unsynchronised** — touched by the exec worker and by the
   main thread's `arm.stop()`/`close()`. ur_rtde documents RTDEControl as not
   thread-safe. This is a real latent race on the *stop* path.
4. **`stream_autopsy` cannot identify the recv's own socket.** Recording its
   local port at construction would make the next death unambiguous.
5. **Nothing is logged to disk** (`run/app.py:87` is stdout-only, no
   timestamps), so event times are unrecoverable after the fact. One run with
   a `FileHandler` + `LRS_LOG_LEVEL=DEBUG` would separate the camera's
   candidate causes outright.
6. **USB autosuspend is enabled** on the camera and all three hubs
   (`control=auto`, 2000 ms). Worth disabling, but the research says expect
   nothing — it is not the cause.

## Cleared false leads

- Hub chaining / USB bandwidth / power — kernel counters perfect, bulk
  endpoints, 7.3% of line rate, `over_current_count 0`.
- NetworkManager / DHCP / connectivity checks — no event in either run window.
- `Recv-Q 91304` on port 30003 — **normal**. ur_rtde's `ScriptClient` targets
  30003 and never reads it. It was still healthy `ESTAB` at the moment of
  death, so it is not the trigger.
- The 3-RTDE-client folklore limit — our own soak ran a higher client count
  five times and survived.
- Downgrading ur_rtde — issues #235 and #307 are both open and unfixed for
  2-3.5 years; there is no known-good version.
- `rx_dropped 53015` on the NIC — office-subnet broadcast with no handler,
  software-layer only; hardware drops are 0.
