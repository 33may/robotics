# ur_rtde 1.6.5 — why `RTDEReceiveInterface` dies and lies

Investigation date: 2026-08-24. **Offline only** — no RTDE connection was opened,
nothing touched 192.168.2.50. Sources: the C++ source at the exact 1.6.5 release
commit, the Boost.Asio source, UR's official RTDE documentation, UR's own client
library documentation, the ur_rtde GitLab issue tracker, and the UR support forum.

## Provenance and how to read this file

| Marker | Meaning |
|---|---|
| **[V]** | Verified — quoted from source with `file:line`, or quoted from a document/issue with a URL |
| **[I]** | Inference — my reasoning from the verified facts, labelled as such |

**Source under test.** `pip show ur_rtde` → version **1.6.5**, installed at
`/home/anton/miniconda3/envs/robo/lib/python3.11/site-packages/` (binary
`librtde.so.1.6.5`, built 2025-08-14). The wheel is a compiled `.so`; there is no
Python source to read. **[V]**

**Important packaging finding.** The GitLab repo's tags stop at **`v1.6.0`
(2024-10-02)**. Versions 1.6.1–1.6.5 exist only on PyPI and were never tagged or
released on GitLab. **[V]** I therefore worked from a clone pinned to commit
**`3ce20df`** — `Release [1.6.5]`, which is where `CMakeLists.txt:59` reads
`project(ur_rtde VERSION 1.6.5 ...)`. Everything after it on master is
documentation-only, so line numbers below match the shipped 1.6.5 binary. **[V]**
PyPI upload dates: 1.6.0 2025-01-03, 1.6.1 2025-05-07, 1.6.2 2025-09-04,
1.6.3 2026-03-13, 1.6.4 2026-08-06, 1.6.5 2026-08-09. **[V]**

Working clone (disposable): `/tmp/ur_rtde_src`.

---

# Q1 — What the receive thread actually does

### The loop and its exception handler **[V]**

`src/rtde_receive_interface.cpp:225-280`. The handler is the whole story:

```cpp
// rtde_receive_interface.cpp:271-278
catch (const boost::system::system_error& e) { // catch the boost exception
  std::cerr << "RTDEReceiveInterface boost system Exception: (" << e.code() << ") " << e.what() << std::endl;
  if (rtde_->isConnected()) {
    rtde_->disconnect(e.code().value() != boost::asio::error::eof);
  }
  th_.signalStop();
  record_thrd_.signalStop();
}
```

This is the **only** place in the entire library that emits the string you saw.
`RTDEControlInterface` prints a different prefix (`rtde_control_interface.cpp:839`).
So the log line `RTDEReceiveInterface boost system Exception: (asio.misc:2) End of
file` came from a **`RTDEReceiveInterface`'s 30004 socket**, not the control
interface's. **[V]**

Answering your questions precisely:

- **Does it exit the loop?** Yes, permanently. `th_.signalStop()` sets the atomic
  the loop tests (`thread_utility.h:50-53`: `void signalStop() { stop_thread_ = true; }`),
  so `while (!*stop_thread)` at line 227 terminates on the next iteration. **The
  thread returns and is never restarted.** **[V]**
- **Does it set `connected_ = false`?** Effectively yes.
  `rtde_->disconnect(...)` sets `conn_state_ = ConnectionState::DISCONNECTED`
  (`rtde.cpp:100`). **[V]**
- **Is there any auto-reconnect?** **No — and this is the core defect.**
  Compare the two interfaces:

  | | `RTDEReceiveInterface` | `RTDEControlInterface` |
  |---|---|---|
  | loop guard | `while (!*stop_thread)` (`:227`) | `while (!*stop_thread && rtde_->isConnected())` (`:792`) |
  | catch clause | `catch (const boost::system::system_error&)` (`:271`) | `catch (std::exception &e)` (`:836`) — catches everything |
  | on error | `th_.signalStop()` — **give up forever** (`:276`) | `should_reconnect = true` → `reconnect()` (`:840`, `:856`) |
  | on success | — | `"RTDEControlInterface: Successfully reconnected!"` (`:860`) |

  The control interface self-heals. The receive interface does not. Same author,
  same file pair, opposite behaviour. **[V]**
- **What does `isConnected()` report?** `rtde_receive_interface.cpp:409-411`:
  `return rtde_->isConnected();` → `rtde.cpp:105-108`:
  `return conn_state_ == CONNECTED || conn_state_ == STARTED;`.
  **It is a socket-state flag only. It knows nothing about whether the receive
  thread is alive.** After the handler above runs it reports `False` — which is what
  you observed. But there are paths where the thread is dead and it still reports
  `True` (see Q4, issue #235). **`isConnected()` is not, and cannot be made into, a
  freshness check.** **[V]**
- **Why the getters lie.** Every getter is a bare cache read with no connection
  check and no staleness check (`rtde_receive_interface.cpp:415-422`):

  ```cpp
  double RTDEReceiveInterface::getTimestamp()
  {
    double timestamp;
    if (robot_state_->getStateData("timestamp", timestamp))
      return timestamp;
    else
      throw std::runtime_error("unable to get state data for specified key: timestamp");
  }
  ```

  `getStateData` (`include/ur_rtde/robot_state.h:77-93`) takes a mutex and returns
  whatever `state_data_` holds. Once the writer thread is gone, `state_data_` is
  frozen and every getter returns the last packet **forever, successfully**. **[V]**

### What `reconnect()` does **[V]**

`rtde_receive_interface.cpp:282-321`. It is a full re-do of the constructor:
`rtde_->connect()` → `negotiateProtocolVersion()` → `getControllerVersion()` →
recompute frequency → `setupRecipes()` → **new `RobotState`** → `sendStart()` →
`th_.start(receiveCallback)` → wait for first state. Returns `isConnected()`.

Three hazards in it, all verified from source:

1. **`frequency_` is overwritten.** Lines 292-295 unconditionally set
   `frequency_ = 125;` then `= 500` for e-Series, discarding whatever you passed to
   the constructor. This was issue #344, described in the 1.6.3 changelog as
   *"Removed hardcoded frequency in the event of a reconnect this fixes #344"* — but
   the code at 1.6.5 still does it in `RTDEReceiveInterface::reconnect()`. **If you
   adopt the "lower the frequency" fix below, a `reconnect()` silently reverts you to
   500 Hz.** You must reconstruct, or re-assert the frequency. **[V]**
2. **The wait loop has no escape.** Line 313: `while (!robot_state_->getFirstStateReceived())`
   — no `isConnected()` guard. The constructor's equivalent loop *was* given one in
   1.6.0 (*"isConnected() has been added to the while loop, so it does not get stuck"*),
   but `reconnect()` never got the same fix. If the new session dies before the first
   packet, **`reconnect()` hangs forever**, holding the GIL-released call. **[V]**
3. **`robot_state_` is a plain `std::shared_ptr`** (`rtde_receive_interface.h:525`)
   reassigned at line 304 while other threads are dereferencing it in getters. That is
   an unsynchronised pointer swap plus a possible destruction of the object a reader is
   inside → **data race / use-after-free**. Your heal path calls `reconnect()` from a
   worker thread while other threads poll getters, so you are exposed. **[I]** —
   from verified declarations and verified call sites.

### Bonus defect: a latent `std::terminate` **[V]**

Two things line up badly.

`rtde_receive_interface.cpp:242` and `:259` throw **`std::system_error`**, while the
only catch clause at `:271` is **`boost::system::system_error`**. These are unrelated
types — a `std::system_error` thrown there escapes `receiveCallback`, leaves the
thread function, and **aborts the process**. Today this path is unreachable because
`RTDE::async_read_some` throws the *boost* exception first (`rtde.cpp:516`) so
`receiveData`'s `if (ec) return error;` at `:541-542` is dead code. It is a loaded gun
with the safety on. (Side effect worth knowing: **the message
`"RTDEReceiveInterface: Robot closed the connection!"` can never print** — its absence
from your log is not evidence against EOF.) **[V]**

More dangerous, and reachable: for a **non-EOF** error, line 274 passes
`send_pause = true`, so `RTDE::disconnect` calls `sendPause()` (`rtde.cpp:333-339`)
which **writes to and then blocking-reads from the socket that just failed**. Any
throw there happens *inside a catch handler* with nothing above it → `std::terminate`.
This is exactly the crash in open issue **#274**, whose WinDbg stack runs
`receiveCallback → RTDE::disconnect → RTDE::receive → ... → ucrtbase!abort`. **[V]**
Your deaths have been EOF (`send_pause = false`), which is why you have been getting
a silent freeze instead of an abort. **A single non-EOF error while the arm is moving
would abort the process instead.** **[I]**

### Bonus defect: the 2500 ms deadline actor **[V]**

`rtde.cpp:677-699`. A `deadline_timer` armed to `now + 2500 ms` on every read
(`rtde.cpp:484`, `:490`). If it fires it does `socket_->close()`,
`conn_state_ = DISCONNECTED`, `socket_.reset()`. The pending read then completes with
`operation_aborted` → throw → the handler above → thread dead forever. **Any 2.5 s gap
in RTDE data with the socket still open kills the stream permanently**, with an
`asio.system:125 Operation canceled` message rather than `asio.misc:2`. Not what you
saw, but it is a second independent instant-death path.

### Bonus defect: `setRealtimePriority` runs on the *calling* thread **[V]**

`rtde_receive_interface.cpp:26-40` calls `RTDEUtility::setRealtimePriority(rt_priority_)`
whenever `/sys/kernel/realtime` reports true. **You are on `6.17.0-1018-realtime`, so
this branch is taken.** `include/ur_rtde/rtde_utility.h:517-537`:

```cpp
if (priority == 0)
{
  const int thread_priority = sched_get_priority_max(SCHED_FIFO);
  ...
  priority = std::min(90, std::max(0, thread_priority));
}
sched_param thread_param{};
thread_param.sched_priority = priority;
if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &thread_param) != 0)
```

`pthread_self()` — it promotes **the thread that constructed the interface** (your
Python main thread), not the receive thread, and the default `rt_priority = 0` means
**SCHED_FIFO 90**. **[V]**

**[I]** Two outcomes, and you should find out which one you get:
- **It fails** (no `CAP_SYS_NICE` / `RLIMIT_RTPRIO`): it prints
  `ur_rtde: unable to set realtime scheduling: Operation not permitted` plus
  `RTDEReceiveInterface: Warning! Failed to set realtime priority even though a
  realtime kernel is available.` — and the receive thread runs at **plain SCHED_OTHER**,
  fully exposed to CPU contention.
- **It succeeds**: your Python main thread becomes SCHED_FIFO 90. Since
  `pthread_create` defaults to `PTHREAD_INHERIT_SCHED`, **every thread created
  afterwards inherits RT-90**, and SCHED_FIFO survives `fork`/`exec` — so the Chromium
  UI child would be spawned at RT-90 too. On a PREEMPT_RT kernel that outranks the
  threaded NIC IRQ handlers (SCHED_FIFO 50 by default), which would stall network RX
  processing. That is a *worse* failure mode than the first.

Grep your run log for that warning. It is a one-line answer to which world you are in.

---

# Q2 — What produces `asio.misc:2 End of file`

### The error code, decoded **[V]**

From Boost.Asio (`boost/asio/error.hpp`, `impl/error.ipp`):
`enum misc_errors { already_open = 1, eof, not_found, fd_set_failure };` → **`eof = 2`**;
the category name is `"asio.misc"`; the message text is `"End of file"`.

**`asio.misc:2` / `End of file` is unambiguously `boost::asio::error::eof`: the peer
performed an orderly shutdown (sent FIN).** It is not a timeout, not a reset, not a
parse error, not an internal ur_rtde fault. **The UR controller deliberately closed
your RTDE session.**

### The documented reason a UR controller closes an RTDE client **[V]**

From UR's official RTDE Guide
(https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html),
verbatim, in the "Robot Controller Outputs" note:

> **NOTE:** The robot controller requires that the client subscribes to at least one
> output. **The client should read data periodically from the socket. The connection is
> closed by the robot controller when the receive buffer overflows.**

Corroborated verbatim by UR's own C++ client library documentation
(`Universal_Robots_Client_Library/doc/architecture/rtde_client.rst`):

> line 23: "This has to be called with the RTDE control frequency, **as the robot will
> shutdown RTDE communication if data is not read by the client**."
>
> line 69: "data has to be polled regularly, **as the robot will shutdown RTDE
> communication if the receiving side doesn't empty its buffer**."

**This is the single most important fact in this report.** UR documents exactly one
routine cause for an orderly FIN on 30004: **the client did not drain the socket fast
enough, so the controller's buffer for that client overflowed and it hung up.**

### Things that are documented NOT to close the connection **[V]**

All from the same RTDE Guide — worth knowing so you can stop chasing them:

- **Protocol version rejection**: `RTDE_REQUEST_PROTOCOL_VERSION` returns
  *"either 1 (success) or 0 (failed)"*. Negative ack, no close. (And ur_rtde requests
  version 2 — `rtde.cpp:21` `#define RTDE_PROTOCOL_VERSION 2` — which every e-Series
  supports. It also **ignores the accept/reject byte** entirely, `rtde.cpp:371-374`.)
- **Unknown output variable**: type comes back `"NOT_FOUND"`, *"the recipe is
  considered invalid and the RTDE will not output this data"*. No close. ur_rtde turns
  this into a thrown `std::runtime_error` at construction (`rtde.cpp:430-434`), so you
  would have seen it at startup, not 90 s in.
- **Input register conflict** (a fieldbus holding your registers): type comes back
  `"IN_USE"`, *"the recipe is considered invalid (input recipe id = 0)"*. ur_rtde
  throws `"One of the RTDE input registers are already in use!"` (`rtde.cpp:397-400`).
  No close. **Your `RTDEControlInterface` and `RTDEIOInterface` both construct
  successfully, so no EtherNet/IP, PROFINET or MODBUS unit is claiming the low
  registers.** **[I]**
- **Controller CPU starvation**: *"if the controller lacks computational resources, it
  will skip some output packages and only send the most recent data."* It **degrades,
  it does not disconnect.**
- **Pause**: *"The CON will always accept a pause command and return a 1 (success)."*
- **There is no RTDE protocol watchdog.** ur_rtde's `kickWatchdog()` operates inside
  its own uploaded URScript via RTDE registers; it can stop motion, it cannot close a
  socket. **[V]**
- **Local/Remote mode changes** drop 30001/30002/30003, explicitly **not** 30004:
  *"If you need to monitor the robot state both in remote and local modes, you should
  use the ports 30004, 30011, 30012, and 30013."*

**[I] Net: a mid-session orderly FIN on 30004 has essentially three explanations —
(1) the client stopped draining, (2) an RTDE client-slot limit was exceeded, (3)
`URControl` restarted or crashed.** (3) would drop the dashboard (29999) and 30003 at
the same instant; your `ss` capture shows both still `ESTAB`, **so (3) is ruled out.**

### How much slack do you actually have? **[I]**, from **[V]** inputs

`RTDEReceiveInterface(ip)` with no variable list subscribes to the **entire default
recipe** — `rtde_receive_interface.cpp:123-217`, 37 base variables plus
`ft_raw_wrench`, `payload`, `payload_cog`, `payload_inertia`, plus
`output_int_register_2`, int registers 12–19 and **double registers 12–19**. Summing
the wire sizes from the RTDE Guide's type table:

| | bytes/package | at 500 Hz |
|---|---|---|
| ur_rtde default recipe (PolyScope 5.11+) | **~1180 B** | **~590 KB/s** |
| …plus `actual_current_as_torque` on PS ≥ 5.23 | ~1228 B | ~614 KB/s |
| only what your app calls (see Q-fixes) | **116 B** | 58 KB/s |

At ~590 KB/s a default 128 KB socket receive buffer fills in **~0.22 s**. Add the
controller's own per-client buffer and you have **order a few hundred milliseconds to
a couple of seconds of receive-thread stall before UR hangs up on you.** The receive
thread is a plain `boost::thread` running a `sleep_for(100 µs)` poll loop
(`rtde_receive_interface.cpp:264-268`) — at SCHED_OTHER on a saturated box, that margin
is not generous.

---

# Q3 — Is `Recv-Q 91304` on port 30003 normal? Yes — and it is a red herring

### Confirmed: ur_rtde does this to itself, permanently **[V]**

- `include/ur_rtde/script_client.h:27-28` — the default port is **30003**, the
  **500 Hz realtime interface**:
  ```cpp
  RTDE_EXPORT explicit ScriptClient(std::string hostname, uint32_t major_control_version,
                                    uint32_t minor_control_version, int port = 30003, bool verbose = false);
  ```
- `src/rtde_control_interface.cpp:137-141` constructs it **without a port argument**,
  taking that default, and calls `connect()`.
- `src/script_client.cpp` includes only `boost/asio/{connect,socket_base,write}.hpp`.
  The only socket operations are `boost::asio::write()` at lines **68, 239, 264**.
  **There is not a single read anywhere in the file.** No `SO_RCVBUF`, no drain thread.
- It is **not** closed after upload. `script_client_->disconnect()` appears only inside
  `RTDEControlInterface::disconnect()` (`:367-368`). It is reused all session long for
  `sendScriptCommand` (`:979`) and `movePath` injection (`:1099`, `:1129`, `:1196`).
- 30003 is used **only** to push script text. `receiveCallback` reads exclusively from
  the 30004 `RTDE` object; `robot_state_` never touches 30003.

So from the instant `RTDEControlInterface` is constructed until it is disconnected,
your process holds an open connection to a ~500 Hz firehose that nothing ever reads.
`Recv-Q 91304` is a **full** buffer (a 131072-byte `rmem` holds roughly this much
payload once skb overhead is accounted). **This is completely normal for every
ur_rtde `RTDEControlInterface` user on earth.** Your app is not doing anything wrong
here; the library is.

### The case FOR "30003 backpressure stalled the controller and got 30004 dropped"

- The zero window is real, permanent, and starts the moment the control interface is
  built — which is precisely the window in which the deaths occur.
- All the client interfaces are served by the same `URControl` process. If its
  client-comms path were single-threaded and did blocking writes, a permanently
  zero-windowed peer would stall service to every other interface including RTDE.
- There is at least one forum report of an abused 30003 socket taking `URControl` down
  process-wide (UR5 CB3 3.15.8, memory growth → `URControl` crash after 4–6 h, after
  which 29999/30001/30003 all refuse connections). A `URControl` crash *would* FIN your
  30004.

### The case AGAINST — and it is decisive

1. **Your own `ss` capture refutes it.** At the moment of death, 30003 was still
   `ESTAB` with a full `Recv-Q`. **The controller had not closed it and had not
   errored on it** — it was calmly tolerating a zero-window peer. If the controller
   blocked or died on 30003 backpressure, that socket would not have been sitting there
   healthy. **[V]** from your evidence.
2. **UR's documented policy is per-interface.** The RTDE Guide promises a close for a
   non-draining **RTDE** client. There is no such promise for the RT interface, and the
   observed controller behaviour is different: on the UR forum, a Wireshark-traced
   "TCP Window Full" on 30003 produced this line in `URControl.log.0`:
   *"WARNING - Failed to send state to client [IP]. Sending discontinued"* — the
   controller **gives up on that one socket and moves on**. Per-socket abandonment, not
   a shared-thread stall. **[V]** (forum thread 3156.)
3. **The universality argument.** Every `RTDEControlInterface` user in the world has
   this exact undrained socket. At ~0.6 MB/s a default autotuned `rmem` ceiling fills
   within seconds. If it starved `URControl` globally, ur_rtde would be unusable for
   everybody within ten seconds of every session. It plainly is not. **[I]**

**Verdict: `Recv-Q 91304` on 30003 is expected ur_rtde behaviour and is not what
killed your 30004 session. Do not spend time on it as a root cause.**

**But it is still a real latent bug in your setup.** **[I]** Once the controller has
logged "Sending discontinued" for that socket, its state is degraded; a later
`sendScript` / `movePath` write may hit `EPIPE` (`write: Broken pipe`) rather than EOF.
Your app only uses `moveJ` / `stopJ` / `getAsyncOperationProgressEx`, which go through
RTDE input registers on 30004, **so you are not currently exposed** — but you would be
the moment anyone adds `movePath` or a custom script. The clean escape is
`FLAG_USE_EXT_UR_CAP` (`rtde_control_interface.cpp:267`, `:537` set
`ur_cap_port_ = 50002`), where the ExternalControl URCap dials *out* to you and **no
30003 socket is ever opened**.

---

# Q4 — Version history: what upstream knows, and what it has fixed

### Issue #235 — `getting sporadic " Exception: End of file" events since updating to new version`

https://gitlab.com/sdurobotics/ur_rtde/-/issues/235 — **STILL OPEN**, filed
2022-10-06, assigned to the maintainer 2026-03-03. **Three and a half years, no fix,
no MR, no commit ever referenced.** **[V]**

The entire description: *"RTDEReceiveInterface Exception: End of file. Edit - doesn't
happen with 1.5.0"*.

Maintainer's first question, **prier, 2022-10-07** — note what he suspects immediately:

> "How often does it happen, and is it associated with **'heavy load' on the PC**?
> Also do you set a **real-time priority** or not?"

**gabriel.wainmann, 2022-10-07:** *"This happens every run of the code, at different
places. The PC isn't very loaded. **Just a bit of inference of YoloV5**. Nothing crazy."*

**lpfennigschmidt, 2022-11-22 — the concrete workaround:**

> "I ran into the same issue, having the ReceiveInterface throw 'End of file'-errors
> and working very unreliably. … for me it seemed to **stabilise the ReceiveInterface
> when I reduced the frequency … from 500Hz to 100Hz** (or something even lower).
> Maybe as a quick fix, this helps if you don't need such high frequent updates :)"

**lpfennigschmidt, 2023-01-10 — why you cannot catch it:**

> "it does not happen when I call something from the RDTEReceiveInterface, but
> **randomly in between, so I am not able to catch the error** and handle it
> differently. **It does not happen as often with a lower update frequency**, but it
> still happens…"

**amadeuszsz, 2023-01-13:** *"Tested with Python & C++ on Ubuntu 22.04. For me looks
like **downgrading to 1.5.3 solves the problem**."*

**amadeuszsz, 2023-02-06 — this is your bug, described exactly, in the *EOF* thread:**

> "With this code, after random time **`getActualQ()` returns constant variable (during
> robot movement) without any error** (even though `rtde_receive_->isConnected()`
> returns **true**). Same thing with other `rtde_receive_interface` methods
> (**`getTimestamp()` output doesn't change with time** as well). … I assume
> `isConnected()` method was designed to cover that issue."

Never answered. **[V]**

**lpfennigschmidt, 2023-04-05 — the other reporter's actual resolution:**

> "at the moment I am guessing the library has issues **when the connection to the
> robot is lost temporarily**. I used to have the robot arm connected to a Wi-Fi router
> via cable, but my computer was communicating to the router wirelessly. **Tried to
> connect my computer to the router by cable as well and I did not get any exceptions
> thrown yet**, I even upgraded back to version 1.5.5…"

Maintainer's only substantive statement, **prier, 2023-01-16:** *"Sorry my lack of
response here. I am currently on parental leave… Actually fixing this, require some
more investigation on my part."*

### Issue #307 — `UR RTDE Receive Interface does not update TCP Positions and Forces`

https://gitlab.com/sdurobotics/ur_rtde/-/issues/307 — **OPEN**, filed 2024-06-06, no
assignee, dead since 2024-06-21. UR16e, ur_rtde 1.5.7. **[V]**

**Note the reporter's setup — it is your setup:**

```
obj.control = RTDEControlInterface(IP_Robot);
obj.receive = RTDEReceiveInterface(IP_Robot);
obj.io      = RTDEIOInterface(IP_Robot);
```

> "During a test the robot starts doing unusual movements, as it is **always reading
> out the same TCP Position and Forces**; … after a while it does not update the values
> and throws the same one **until it is initialised again**. **No errors pop up or are
> shown** when this happens, and **the control interface still works properly**."

Total maintainer engagement — **prier, 2024-06-18:** *"Please try your code without
matlab if possible, just to rule out any problems there."* Then **joaogariso,
2024-06-21:** *"I'm having a similar problem and can't seem to find a solution."*
End of thread. **[V]**

**[I] #235 and #307 are the same defect at two stages**: a transient EOF kills the
receive thread (#235's symptom if the exception escapes; #307's if it is swallowed),
after which the object survives holding a frozen `RobotState`. #307's *"until it is
initialised again"* is the reporter independently discovering that **reconstruction is
the only reliable recovery**.

### Other issues in the same family **[V]**

| # | Title | State | Outcome |
|---|---|---|---|
| **#155** | `Joint state not updated after Error: End of file exception` | closed 2022-07-06 | **user-side network fix — see below** |
| **#177** | `ur_rtde AND Vision - systems failure - "overload"` | closed | vision node added → `Could not receive data from robot... read: End of file` |
| **#274** | `Crash in RTDEReceiveInterface::receiveCallback exception handling(?)` | **OPEN** | maintainer reply in full: *"Thanks for reporting this."* |
| **#302** | `RTDEControlInterface disconnects randomly during routine…` | closed 2026-03-03 | labelled **"Configuration"**, no code change |
| **#61** | `Return values of the RTDEReceiveInterface (Python) do not reflect the current state` | closed 2021-11-08 | stale-cache class |
| **#133** | `read: End of file error leading to issue with getActualTCPPose` | closed 2023-09-04 | *"The stability should be improved in newer versions of ur_rtde, please update 😉"* |
| **#242** | `RTDE Interfaces stuck forever in constructor` | **OPEN** | relevant to the `reconnect()` hang above |

**#155's closing comment is the most useful data point in the whole tracker**
(ashok930, 2022-03-29):

> "**UPDATE: The issue is resolved now.** We had **other large size and frequency data
> flowing through our network cable(ethernet)** connected to the computer and maybe the
> robot terminated the connection because of insufficient bandwith. **Using a separate
> network interface fixed this.**"

And prier's diagnostic questions in that thread (2022-01-11): *"Can you check if there
is an IP collision on the network…? Also, could it be **related to high CPU load**, do
you have any **CPU intensive tasks running**, when it happens?"*

**#177's reporter** added a VISP AR-tag vision node to a working system and
immediately got `RTDEControlInterface: Could not receive data from robot... read: End
of file`; his own conclusion was that the system *"cannot maintain communication within
the time window required by the UR controller"* — and changing publish rates did not
help. **#302** was closed by the maintainer as a *configuration* problem attributed to
CPU/network load from **image acquisition**, with the advice to use an RT kernel with
high priority or move CV to a separate PC/network. **[V]**

**The pattern across #155, #177, #235, #302 is unmistakable: ur_rtde + a vision
workload on the same host → `End of file`.** Every case that was ever actually resolved
was resolved by **reducing contention** (dedicated NIC, lower frequency, separate
machine) — never by a library fix. **[I]**

### What changed on the receive path between 1.5.0 and 1.6.5 **[V]**

There is **no `CHANGELOG.md`** in the repo; the changelog lives in release commit
messages. Receive-path-relevant entries only:

- **1.4.9** (2022-01) — *"Added support for detection of network connection loss or
  RTDE desynchronization"*. This is where Uwe Kindler added the 2500 ms deadline
  (`aa536070`, *"Added timeout for RDTE::receive_data function to detect connection
  loss"*) and later *"Increased default timeout value to prevent accidental timeouts in
  case of blocked threads"* (`9b0e8ece`).
- **1.5.0** — nothing on the receive path.
- **1.5.3** (`54b8cdc`, 2022-08-11) — *"Important fix for data reception, fixes #204,
  #205, #207, #208, #215, #218"*; *"ur_rtde is now fully realtime capable"*.
- **1.5.4** (`1ef5834`, 2022-09-02) — *"Fix of data reception timing issues, a new
  additional fix for #204, #205, #207, #208, #215 and #218"*; *"Fixed the way the
  real-time priority is specified"*. **This commit introduced the
  `no_bytes_avail_cnt_ > 20` de-synchronisation re-read and the
  `throw std::system_error(ec)` lines.**
- **1.5.6** (2023-04) — *"Fixed application crash in case of network connection loss"*.
- **1.6.0** (`b946a38`, 2024-10-02) — **the commit that created the message you are
  seeing.** Its diff on `rtde_receive_interface.cpp` is:
  ```diff
  -    catch (std::exception& e)
  -    {
  -      std::cerr << "RTDEReceiveInterface Exception: " << e.what() << std::endl;
  -      if (rtde_->isConnected())
  -        rtde_->disconnect();
  +    catch (const boost::system::system_error& e) { // catch the boost exception
  +      std::cerr << "RTDEReceiveInterface boost system Exception: (" << e.code() << ") " << e.what() << std::endl;
  +      if (rtde_->isConnected()) {
  +        rtde_->disconnect(e.code().value() != boost::asio::error::eof);
  +      }
  ```
  Changelog line: *"Fixed managing of eof in RTDEReceiveInterface::receiveCallback()"*.
  **[I] This "fix" narrowed the catch from `std::exception` to
  `boost::system::system_error`** — it stopped the `sendPause`-on-a-dead-socket abort
  for the EOF case (good) but made every other exception type fatal (bad), and **did
  not add reconnection**.
- **1.6.1** (2025-05) — *"Fix segmentation fault due to thread leak"* — introduced
  `ThreadUtility` (`thread_utility.h`) so threads are joined rather than detached.
- **1.6.3** (2026-03) — *"Fixed skipping of data package, this closes #267 and #342"*
  (`5c5ced9`, the `next_message_offset` bug at `rtde.cpp:562-571`); *"Removed hardcoded
  frequency in the event of a reconnect this fixes #344"*.
- **1.6.4** (2026-08) — *"Fix RTDE control callback lifetime during constructor
  failure"*. **`directTorque()` is broken in 1.6.4.**
- **1.6.5** (2026-08-08) — *"Critical fix to directTorque(), arguments and parsing in
  rtde.cpp"*.

**Is there a known-good version?** **No.** **[I]** The two "it was fine before"
datapoints (1.5.0 from the #235 reporter, 1.5.3 from amadeuszsz) contradict each other
and are both unverified anecdotes; 1.4.9 and 1.5.0 were released the same day, and
1.4.9 is where the deadline actor landed. **Neither #235 nor #307 is referenced by any
release. There is no fix, no known-good version, and no open MR touching this code**
(the 7 open MRs are Robotiq framing, Boost 1.89, nanobind, macOS, `setToolVoltage`,
`remove boost thread`, tool analog inputs). **You are already on the newest release;
downgrading buys you nothing but different bugs.**

Maintainer posture, across #235, #302, #133, #274: **environment, not library.** Do not
wait for an upstream fix.

---

# Q5 — How many RTDE clients can a UR controller serve?

**Officially: undocumented.** The RTDE Guide states no connection cap anywhere. The
only *per-client* rule it gives is about variables, not sockets: *"Inputs retain their
last received value, and **only one RTDE client can control a specific variable at any
time**."* UR's client library docs say *"While RTDE allows multiple clients to connect
to the same robot, only one client is allowed to write data to the robot."* **[V]**

**In practice, the widely cited figure is 3**, from the UR support forum
(https://forum.universal-robots.com/t/failures-with-multiple-rtde-connections/2793):

> "It is indeed correct, that the **RTDE currently has a limit of only 3 active
> clients**."
> "**Ethernet/IP and Profinet also counts as a client**, if they are enabled."

And the over-limit *behaviour* matters — from
https://forum.universal-robots.com/t/rtde-outputs-only-two-clients/3673:

> "RTDE outputs subscription is accepted by and working with two clients, the others
> receives an **OK feed back, but the connection is closed bye the UR after maybe 1
> seconde**."

**[V]** — but with a real caveat: **both datapoints are 2018–2019, CB3 / PolyScope 3.x
era.** I found no confirmation or refutation for e-Series PolyScope 5.x, and no UR
statement that the limit was ever raised. Treat 3 as a strong prior, not a fact for
your controller.

**Where your app sits.** Each interface opens exactly one 30004 socket — verified:
`rtde_receive_interface.cpp:51-52`, `rtde_control_interface.cpp:83-85`,
`rtde_io_interface.cpp:40-41`, all `port_ = 30004`. Your `UR5eArm.__init__`
(`motion/execute.py:185-190`) does:

1. `preflight()` → recv + DashboardClient → both disconnected
2. `self.recv = RTDEReceiveInterface(ip)` — client **A**, long-lived
3. `self.ctrl = RTDEControlInterface(ip)` — client **B**, long-lived (+ dashboard + 30003)
4. `RTDEIOInterface(ip).setSpeedSlider(slider)` — client **C**, a Python temporary,
   destroyed immediately → destructor `rtde_->disconnect()` (`rtde_io_interface.cpp:57-63`)

Peak **3 concurrent**, steady state **2**. That is *at* the folklore limit, not over it.
`run/app.py` calls `preflight()` too, so two preflight sessions precede the peak.

**[I] But your own soak data rules this out as the cause.** `motion/rtde_soak.py`
`full_mix` replays that exact sequence — preflight recv + dashboard, disconnect,
long-lived recv, `RTDEControlInterface`, leaked `RTDEIOInterface().setSpeedSlider()` —
**with an extra monitoring recv attached, 5 repeats, 120 s hold each. Zero deaths.**
The monitored variant therefore ran at a *higher* peak client count than the real app
and survived. `threads` (E8) built a real `UR5eArm` (which itself calls `preflight()`)
3× and survived. **The connection topology alone does not kill the session.**

One thing still worth a two-minute check on the pendant: confirm **EtherNet/IP,
PROFINET and MODBUS are all disabled** in the installation, and that no URCap holds an
RTDE session. Each would consume a slot invisibly. Weak evidence says they are already
off — a fieldbus claiming the low input registers would make `RTDEControlInterface`
throw `"One of the RTDE input registers are already in use!"` at construction
(`rtde.cpp:397-400`), and it does not. **[I]**

---

# Ranked root causes, against **all** the evidence

The constraint every hypothesis must satisfy: **7 isolated scenarios / 23 trials, zero
reproductions — yet every one of 5 full application runs dies.**

The critical structural observation about your test matrix **[I]**: reading
`motion/rtde_soak.py`, every scenario **introduces one ingredient into an otherwise
quiet process**, with 5–10 s of idle around each event marker, and **none of them ever
commands motion**. `ui_launch` ran Chromium with *only* a recv. `camera` ran the D405
with *only* a recv. `threads` ran 30 Hz + 20 Hz reads with *only* recv + ctrl. The real
app runs Chromium/software-Vulkan **and** the D405 pipeline **and** point-cloud fusion
**and** the porthole bus **and** the arm actually moving — **simultaneously, during the
startup burst**. A superposition-of-load hypothesis is the only kind that can be
invisible to a one-at-a-time matrix and 100% reproducible in the real app.

### #1 — Receive-thread starvation → controller-side buffer overflow → documented FIN

**Confidence: high.** Mechanism: under combined load the ur_rtde receive thread is
descheduled long enough (a few hundred ms is enough at 590 KB/s) that the socket
receive queue fills, the controller's output buffer for that client overflows, and
**UR closes the session exactly as documented**. ur_rtde then permanently gives up and
its getters lie.

Explains:
- ✅ **`asio.misc:2 End of file`** — the orderly FIN that UR documents for this case,
  and the *only* routine cause UR documents.
- ✅ **Every full run dies; no isolated ingredient does** — the load only superposes in
  the real app.
- ✅ **"Usually within the first ~90 seconds"** — that is the app's peak-transient
  window: Chromium spawn + software-Vulkan init + D405 pipeline start + first fusion +
  the survey move. Your `run.json` shows turn 1 ("survey") completing at **t = 34.8 s**
  and the review notes the death was caught by the **plan-survey** thread, i.e. inside
  that first burst.
- ✅ **Upstream corroboration** — #235 (YOLOv5), #177 (VISP vision node), #302 (image
  acquisition), #155 (network contention, fixed by a dedicated NIC); the maintainer's
  reflex question is *"heavy load on the PC? real-time priority?"*; the one workaround
  that reproducibly helped anyone was **dropping 500 Hz → 100 Hz**.
- ✅ **`isConnected() = False`** — matches the EOF branch at
  `rtde_receive_interface.cpp:273-275` precisely.
- ✅ **30003 still `ESTAB`, 29999 still `ESTAB`** — the controller is alive and healthy;
  it hung up on *one* client, which is per-client policy, not a controller fault.
- ⚠️ Not yet directly measured. You have no evidence of an actual multi-hundred-ms
  stall — that is the gap to close (see fix F0).

### #2 — ur_rtde's non-recovery is the *amplifier*, not the trigger (certain)

**Confidence: certain — this part is proven from source, not inferred.** Whatever
causes the FIN, `rtde_receive_interface.cpp:271-278` guarantees the outcome is
**permanent silent staleness**: no retry, no flag a caller can trust, and getters that
return stale data *successfully*. A single transient network hiccup — one lost link, one
switch reconfiguration — is upgraded into a permanent, invisible failure. This is why
the bug is 100% fatal per run once it fires at all, and it is the part you can fix
outright regardless of #1.

### #3 — RT-priority escalation on your PREEMPT_RT kernel

**Confidence: medium, cheap to test, potentially explains #1's trigger.**
`setRealtimePriority(0)` → `pthread_setschedparam(pthread_self(), SCHED_FIFO, 90)` runs
on the **Python main thread** on your `-realtime` kernel. If it *fails*, the receive
thread runs SCHED_OTHER and is fully exposed to the load in #1. If it *succeeds*, RT-90
is inherited by every later thread **and by the Chromium child across `fork`/`exec`**,
outranking PREEMPT_RT's threaded NIC IRQ handlers (SCHED_FIFO 50) and stalling network
RX — which produces #1 by a different route. Either branch supports #1; they differ in
the fix. **One grep of your log settles it.**

### #4 — RTDE client-slot limit (3) exceeded transiently at startup

**Confidence: low.** The forum signature (accept → OK handshake → FIN ~1 s later) is a
beautiful match for a startup-time EOF, and the app does peak at exactly 3. **But
`full_mix` reproduced that sequence 5× with an *extra* monitoring client and never
died** — which is close to a direct refutation. Keep it alive only as: "is some *other*
consumer (fieldbus / URCap / a leaked socket from a crashed prior run) occupying a
slot?" — cheap to check, unlikely to be the answer.

### #5 — 30003 backpressure stalling the controller

**Confidence: very low — argued in detail in Q3 and refuted by your own `ss` capture**
(30003 was `ESTAB` and healthy at the moment of death) and by the universality
argument. Real latent bug, wrong bug for this failure.

### #6 — Protocol violation / version negotiation / recipe error

**Confidence: near zero.** Every one of these is documented to produce a negative ack
or a `NOT_FOUND`/`IN_USE` recipe rejection, **never a close**, and ur_rtde converts them
into constructor-time exceptions. They cannot produce a mid-session FIN. **Ruled out.**

### #7 — `URControl` restart / crash

**Ruled out** by your own evidence: the dashboard (29999) and 30003 sockets were still
`ESTAB` at the moment of death. A controller restart drops everything at once.

---

# RECOMMENDED FIXES

Ranked by **confidence × cheapness**. F1–F3 are one-line changes with large expected
effect; F0 is free and tells you whether F1–F3 worked for the reason you think.

### F0 — Instrument first (free, do it with F1)

Three things, all in `motion/execute.py`:

1. **Record the recv's local port at construction.** Diff the process's 30004 sockets
   immediately before and after building it. Your own `loop_stability_review.md` §6
   already flags that `stream_autopsy()` cannot tell which socket is the recv's — this
   closes it, and makes the next `ss` capture unambiguous.
2. **Log the first controller timestamp** the recv ever returns, alongside wall-clock.
   Comparing it to the frozen value (`6248.872`) settles in one line whether the stream
   died **at startup** or **90 s in** — currently the single largest ambiguity in the
   evidence, and unresolvable from the run artifacts on disk.
3. **Grep the run log for `unable to set realtime scheduling`** and
   `Failed to set realtime priority`. That decides hypothesis #3 with zero effort.

**Verify:** the next death report names the recv's own socket and says how long the
stream lived.

### F1 — Shrink the receive recipe and drop the frequency (highest value, ~1 line)

Your app calls exactly five receive getters — verified by grep across the whole
codebase: `getTimestamp`, `getActualQ`, `getActualQd`, `getRobotMode`, `getSafetyMode`.
You are currently subscribed to the full ~50-variable default recipe.

```python
# motion/execute.py — UR5eArm.__init__, and preflight()
self.recv = RTDEReceiveInterface(
    ip, 125.0,
    ["timestamp", "actual_q", "actual_qd", "robot_mode", "safety_mode"],
)
```

**~1180 B @ 500 Hz (590 KB/s) → 116 B @ 125 Hz (14.5 KB/s): a ~40× increase in the
stall you can absorb before UR hangs up on you.** The time to fill a 128 KB receive
buffer goes from ~0.22 s to ~9 s.

Confidence: this is the one workaround with an independent report of success (#235,
lpfennigschmidt: *"it seemed to stabilise the ReceiveInterface when I reduced the
frequency … from 500Hz to 100Hz"*), and it attacks hypothesis #1 directly at its
documented mechanism. Cost: one line. You lose nothing — nothing in your app consumes
data faster than 125 Hz, and your freshness guard's `stale_window = 0.10` s still spans
~12 packets at 125 Hz.

⚠️ **Two caveats, both verified from source.**
(a) `reconnect()` **overwrites `frequency_` back to 500** (`rtde_receive_interface.cpp:292-295`)
— so after a heal you silently lose this protection. Prefer **reconstructing** the
interface over `reconnect()` (F2), or re-assert the frequency after healing.
(b) Consider `RTDEControlInterface(ip, 125.0)` as well — its own 30004 subscription is
independent and also runs at 500 Hz. Your app only uses `moveJ`/`stopJ`/
`getAsyncOperationProgressEx` — **no `servoJ`/`speedJ`** — so 125 Hz (the CB3 default)
is safe here. Change it separately from F1 so you can attribute the effect.

**Verify:** `ss -tnop` shows a near-zero `Recv-Q` on the recv's 30004 socket throughout
a run; capture the packet rate and confirm ~125 Hz timestamp advance; then run 5 full
app runs and count deaths.

### F2 — Make recovery reconstruct rather than `reconnect()` (high confidence, small)

`reconnect()` has three verified defects: it resets your frequency (F1 caveat), its
wait loop has **no `isConnected()` guard** (`:313`) so it can **hang forever**, and it
swaps `robot_state_` under concurrent readers (**use-after-free race**) — and your heal
runs on a worker thread while other threads poll getters.

Change the heal path to: quiesce all readers under the existing lock → `recv.disconnect()`
→ drop the object → construct a fresh `RTDEReceiveInterface` with the F1 arguments →
publish it → release. This is also what #307's reporter found to be the only reliable
recovery (*"until it is initialised again"*).

**Verify:** force a death (e.g. `iptables -A INPUT -p tcp --sport 30004 -j DROP` for 5 s
on a bench robot — **not** the production cell) and confirm the heal completes, does not
hang, and comes back at 125 Hz.

### F3 — Give the receive thread scheduling headroom (medium-high confidence, cheap)

Depends on what F0.3 tells you:

- **If RT priority currently *fails***: grant the process `RLIMIT_RTPRIO` /
  `CAP_SYS_NICE` and pass an **explicit modest** priority, e.g.
  `RTDEReceiveInterface(ip, 125.0, vars, False, False, 40)`. Explicit is essential —
  the default `0` means **90**, above PREEMPT_RT's NIC IRQ threads.
- **If it currently *succeeds***: that is the more dangerous branch. Pass
  `rt_priority = -1` to opt out entirely (`rtde_utility.h:510-515` returns early on a
  negative value with *"realtime priority less than 0 specified, realtime priority will
  not be set on purpose!"*), or restructure so the interface is constructed on a
  dedicated thread rather than the Python main thread — otherwise RT-90 leaks into every
  thread you spawn afterwards **and into the Chromium child**.

Independently: `taskset`/`cgroup`-cap the Chromium UI and the point-cloud fusion so
they cannot saturate every core. Upstream's own advice (#302) is exactly this — RT
kernel with high priority for the robot interface, or CV on a separate machine.

**Verify:** `chrt -p <pid>` on the Python main thread and on the Chromium child before
and after; then a full run under `pidstat`/`perf sched` to confirm the receive thread
has no multi-hundred-ms gaps.

### F4 — Isolate the robot on its own NIC (medium confidence, cheap if hardware allows)

The only two upstream cases that were ever definitively *solved* were solved this way:
#155 — *"Using a separate network interface fixed this"* — and #235's lpfennigschmidt,
who eliminated a wireless hop. If the D405, the UI, and the robot share a link or a
congested switch, RTDE is competing with them for the ~590 KB/s (or ~15 KB/s after F1)
it needs without gaps.

**Verify:** dedicated NIC, direct cable to the controller, static route; then `ping -f`
baseline and a 10-minute `recv` soak with `ss -ti` watching for retransmits.

### F5 — Keep the freshness guard; add "the stream is a liar" to its contract (already done, keep)

Your `d6b4e34` freshness guard is **the correct architecture** and this investigation
strengthens the case for it: `isConnected()` is provably not a liveness check
(`rtde_receive_interface.cpp:409-411`), the getters provably cannot fail on a dead
stream, and upstream provably will not fix it (#235 open 3.5 years, #307 open 2 years,
no MR). **Timestamp-advance polling is the only sound detector.** Keep it as the
permanent contract, not a workaround — and note that after F1 the guard should be
retuned: at 125 Hz a packet arrives every 8 ms, so `stale_window = 0.10` s still means
~12 missed packets, which is fine.

### F6 — Eliminate the undrained 30003 socket (low urgency, do it opportunistically)

Not the cause of these deaths (Q3), but a real latent bug: the controller will log
*"Failed to send state to client … Sending discontinued"* and permanently degrade that
socket. You are safe today only because your app never calls `sendScript`/`movePath`.
Two options: switch `RTDEControlInterface` to `FLAG_USE_EXT_UR_CAP` (ExternalControl
URCap, port 50002 — **no 30003 socket is opened at all**), or accept it and add a
comment so nobody adds `movePath` without knowing.

**Verify:** `ss -tnop` shows no connection to `:30003` at all.

### F7 — Do NOT downgrade (explicit non-recommendation)

The anecdotes point at 1.5.0 and 1.5.3 and contradict each other; neither #235 nor #307
is closed by any release; 1.6.1 fixed a thread leak, 1.6.3 fixed real data-package
skipping (#267/#342) and the reconnect frequency (#344), 1.6.4/1.6.5 fixed
`directTorque`. **1.6.5 is the best available build.** Downgrading trades a known bug
for unknown ones.

---

## Appendix — key source coordinates (ur_rtde 1.6.5, commit `3ce20df`)

| What | Where |
|---|---|
| The failing catch handler | `src/rtde_receive_interface.cpp:271-278` |
| Receive loop | `src/rtde_receive_interface.cpp:225-280` |
| `reconnect()` (frequency reset `:292-295`, unguarded wait `:313`) | `src/rtde_receive_interface.cpp:282-321` |
| `isConnected()` — socket state only | `src/rtde_receive_interface.cpp:409-411` → `src/rtde.cpp:105-108` |
| Getters — no freshness check | `src/rtde_receive_interface.cpp:415-…` |
| Default recipe (~50 variables) | `src/rtde_receive_interface.cpp:123-217` |
| `setRealtimePriority` on `pthread_self()`, default = 90 | `include/ur_rtde/rtde_utility.h:479-540` |
| Control interface's **auto-reconnect** (the asymmetry) | `src/rtde_control_interface.cpp:836-874` |
| 2500 ms deadline actor that closes the socket | `src/rtde.cpp:677-699`, `:484`, `:490` |
| `async_read_some` throws rather than returning `ec` | `src/rtde.cpp:514-517` |
| `disconnect(send_pause)` → blocking I/O on a dead socket | `src/rtde.cpp:89-103`, `:333-339`, `:341-357` |
| `ScriptClient` default port **30003**, write-only | `include/ur_rtde/script_client.h:27-28`; `src/script_client.cpp:68,239,264` |
| Three interfaces, three 30004 sockets | `rtde_receive_interface.cpp:51`, `rtde_control_interface.cpp:83`, `rtde_io_interface.cpp:40` |
| `ThreadUtility::signalStop` / `stop` | `include/ur_rtde/thread_utility.h:50-72` |

**External sources**

- UR RTDE Guide — https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
- UR client library, RTDE architecture — https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/master/doc/architecture/rtde_client.rst
- ur_rtde #235 (open) — https://gitlab.com/sdurobotics/ur_rtde/-/issues/235
- ur_rtde #307 (open) — https://gitlab.com/sdurobotics/ur_rtde/-/issues/307
- ur_rtde #155 / #177 / #274 / #302 / #133 / #61 — `.../issues/<n>`
- UR forum, 3-client limit — https://forum.universal-robots.com/t/failures-with-multiple-rtde-connections/2793
- UR forum, FIN ~1 s after an over-limit connect — https://forum.universal-robots.com/t/rtde-outputs-only-two-clients/3673
- UR forum, 30003 TCP-window-full → "Sending discontinued" — https://forum.universal-robots.com/t/realtime-interface-port-30003-not-recovering-from-tcp-window-full/3156

*Tip for future digging: GitLab's REST `issues/<n>/notes` endpoint returns 401 without a
token, but the front-end endpoint `https://gitlab.com/sdurobotics/ur_rtde/-/issues/<n>/discussions.json`
is public and returns every comment.*
