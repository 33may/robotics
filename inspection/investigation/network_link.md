# Network link investigation — PC `vbti-MS-7E66` ↔ UR5e `192.168.2.50`

Measured 2026-08-24, 12:35–12:55 CEST. Read-only: `ip`, `ethtool`, `ss`, `ping`,
`nstat`, `tc`, `sysctl`, `/proc/net/*`, `/sys/class/net/*`, `journalctl`.
**No RTDE (30003/30004) or dashboard (29999) connection was opened. The robot was
not commanded. Nothing was reconfigured.**

Host uptime at time of measurement: booted **2026-08-24 10:14:24**, up 2 h 41 m.
All kernel counters below are cumulative *since that boot*, which contains exactly
the two run windows of interest (~10:27 and ~11:51–12:10). That makes them usable
as evidence about those runs.

---

## VERDICT (short)

**The transport layer is HEALTHY and is exonerated as the cause of the FIN.**

Every physical and IP-level measurement is clean to the point of being boring:
zero NIC errors, zero packet loss across 605 ICMP probes at three sizes and three
rates, 0.171 ms average RTT, 19 data-segment retransmissions in 2.7 M segments,
zero checksum errors, no link flap, no NetworkManager or DHCP event inside either
run window.

**But the same sweep turned up one counter that is not clean and points straight
back at the application:**

```
TcpExtTCPToZeroWindowAdv        213
TcpExtTCPFromZeroWindowAdv      213
TcpExtTCPWantZeroWindowAdv       71
```

This host advertised a **zero TCP receive window 213 times this boot** — i.e. a
socket's receive buffer filled completely because userspace stopped reading it. The
measured idle background rate for this counter is ~0 (flat over a 90 s idle window;
+1 over ~7 min). So the overwhelming majority of those 213 events belong to the
application runs.

A stalled reader on the RTDE stream is the textbook way to get a clean FIN out of a
UR controller: URControl's RTDE server cannot write into a zero-window client, and
drops it. That produces exactly `boost asio.misc:2 End of file`.

So: the network did not break the connection. **The evidence suggests the
application stopped draining a socket and the controller hung up on it.**
See "Confidence and what is NOT proven" for the limits of that claim.

---

## 0. Topology — the setup is not what the brief assumed

**VERIFIED.** There is no dedicated robot NIC. `192.168.2.0/24` and the office
`10.11.100.0/23` are **two addresses on the same physical interface `enp12s0`**:

```
2: enp12s0  inet 192.168.2.130/24  brd 192.168.2.255 scope global noprefixroute
2: enp12s0  inet 10.11.101.240/23  brd 10.11.101.255 scope global dynamic  (DHCP)
```

`wlp13s0` is DOWN (NO-CARRIER), `docker0` is DOWN, `tailscale0` is a TUN with a
/32. One cable, one broadcast domain, both subnets.

Consequences, all confirmed by the ARP table (`10.11.100.1`, `10.11.100.20`,
`10.11.100.124` and the robot all resolve **on `enp12s0`**):

- Office broadcast/multicast noise (SSDP :1900, WS-Discovery :3702) shares the wire
  and the NIC's single RX queue with the 500 Hz RTDE stream.
- The office DHCP lease renews on the same interface that carries robot traffic.
- The robot is reachable from the office LAN, and vice versa. It is not isolated.

This is a real hygiene problem and it is the correct thing to fix long-term, but —
see §3 and §4 — it is measurably **not** what is killing the RTDE session today.

---

## 1. Link health

### `ethtool enp12s0` — VERIFIED

```
Speed:                1000Mb/s
Duplex:               Full
Auto-negotiation:     on
Port:                 Twisted Pair
master-slave status:  slave
Link detected:        yes
Supported link modes:  10/100/1000/2500/5000baseT
Link partner advertised link modes:  10baseT/Half 10baseT/Full
                                     100baseT/Half 100baseT/Full
                                     1000baseT/Full
Link partner advertised pause frame use: No
```

The NIC is 5 GbE-capable (RTL8126) but the link partner tops out at 1000baseT/Full,
so 1 Gb/s is the correct negotiated result — not a fault. Flow control is **off in
both directions** (`ethtool -a`: `RX negotiated: off`, `TX negotiated: off`), which
is the right setting for a latency-sensitive stream — no pause frames can stall it.

### `ethtool -S enp12s0` — VERIFIED, ALL ZERO

Sampled twice, ~20 min apart. Final snapshot:

```
tx_packets:            1716610
rx_packets:            2782525
tx_errors:                   0
rx_errors:                   0
rx_missed:                   0
align_errors:                0
tx_single_collisions:        0
tx_multi_collisions:         0
tx_aborted:                  0
tx_underrun:                 0
unicast:               2671627
broadcast:               68993
multicast:               41905
```

**There is not a single non-zero error counter.** No CRC/align errors, no missed
frames (RX ring never overran), no collisions, no TX underruns. The r8169 driver
does not expose a separate `rx_crc_errors`; `align_errors` and `rx_errors` cover it
and both are 0.

`tc -s qdisc show dev enp12s0` — `fq_codel`, **`dropped 0, overlimits 0`**,
`backlog 0b 0p`, `drop_overlimit 0`, `ecn_mark 0`, 23 requeues. TX path is clean.

### The one non-zero counter: `ip -s link` RX dropped — EXPLAINED, BENIGN

```
RX:  bytes       packets   errors  dropped  missed   mcast
     2113781267  2782525        0    53015       0   41905
TX:  bytes       packets   errors  dropped  carrier  collsns
      583985372  1398057        0        0        0        0
```

53 015 RX dropped (1.91 % of packets) looks alarming and is **not** what it appears.
Measured rate over a 20 s idle window: **109 drops / 1837 packets**, i.e. ~3–5 pps,
steady, with the application not running.

These are **software-layer** drops, not hardware: `rx_errors` and `rx_missed` are 0
(the NIC received them fine) and `/proc/net/softnet_stat` column 2 (backlog drop) is
**0 on all 24 CPUs**. The kernel increments `rx_dropped` when a frame is accepted by
the NIC but no protocol handler wants it. Arithmetic confirms the source: 68 993
broadcast + 41 905 multicast = 110 898 L2 flood frames received, of which only
~22 000 were delivered to IPv4/IPv6 (`IpExtInBcastPkts 5196`,
`IpExtInMcastPkts 15249`, `Ip6InMcastPkts 1537`). The remainder is office-LAN
broadcast noise with no listener.

**This costs a little softirq CPU. It does not drop robot traffic** — unicast to
`192.168.2.130` always has a handler. It is a direct consequence of §0 (shared
broadcast domain) and is hygiene, not the bug.

---

## 2. Interface, driver, ring, offloads, coalescing

```
driver:            r8169
version:           6.17.0-1018-realtime
firmware-version:  rtl8126a-3_0.0.5 08/30/24
bus-info:          0000:0c:00.0
PHY:               Realtek Internal NBASE-T PHY
MTU:               1500
qdisc:             fq_codel  (qlen 1000)
```

Kernel: `6.17.0-1018-realtime`, **PREEMPT_RT**, 24 CPUs.

### Ring buffers — `ethtool -g`

```
Pre-set maximums:   RX 256   TX 256
Current settings:   RX 256   TX 256
```

256 descriptors is small in absolute terms but it is the **hardware maximum** for
this Realtek part and it is already fully allocated. Nothing to tune. `rx_missed: 0`
proves the ring has never overrun — at 500 Hz (~1 KB) the stream needs ~500 KB/s,
about 0.4 % of the link.

### Offloads — `ethtool -k` (notable entries)

```
generic-receive-offload (GRO):   on
tcp-segmentation-offload:        on
large-receive-offload:           off [fixed]
rx-checksumming:                 on
tx-checksumming:                 on
receive-hashing:                 off [fixed]
ntuple-filters:                  off [fixed]
```

**GRO is on.** For a 500 Hz stream of small packets GRO can coalesce consecutive
segments and add up to one NAPI poll of latency (tens of µs here). This is a
*theoretical* jitter source, not a correctness one, and the RTT measurements in §3
show it is not costing anything measurable. **Not worth changing.** LRO — the one
that genuinely breaks things — is off and fixed.

### Interrupt coalescing — `ethtool -c`

```
netlink error: Operation not supported
```

**The r8169 driver does not implement coalescing get/set for this chip.** It cannot
be queried or tuned. Same for `ethtool -l` (channels) — not supported; the NIC runs
a **single RX queue**.

### IRQ / softirq placement — worth noting for an RT box

```
IRQ 99  IR-PCI-MSIX-0000:0c:00.0  enp12s0
  smp_affinity_list: 0-23        (all CPUs permitted)
  actual delivery:   CPU16 only  (3 350 311 interrupts)
irqbalance:          inactive
RPS (rx-0/rps_cpus): 000000      (disabled)
napi_defer_hard_irqs: 0
gro_flush_timeout:    0
net.core.busy_poll:   0
/proc/cmdline:        ...usbcore.usbfs_memory_mb=1000 quiet splash
                      (no isolcpus, no nohz_full, no irqaffinity)
kernel.sched_rt_runtime_us: 950000 / period 1000000  (RT throttled at 95%)
```

All NIC receive processing lands on **CPU16**, single-queue, with RPS off and no CPU
isolation. `time_squeeze` on CPU16 = **23** since boot (softnet column 3) with
**0 drops** — negligible, so softirq is currently keeping up.

**This is the one structural weakness in an otherwise clean stack, and it is a
plausible amplifier rather than a cause:** on a PREEMPT_RT kernel with no
`isolcpus`, a busy application RT thread scheduled onto CPU16 competes with NIC
softirq. That would delay *draining*, not delivery — which is consistent with the
zero-window finding in §4, and consistent with the bug never appearing in isolated
tests where CPU16 is idle. **INFERRED, not verified** — proving it needs a run-time
measurement (see RECOMMENDED ACTIONS #3).

PCIe power management is correctly out of the way: `LnkCtl: ASPM Disabled`,
`L1SubCtl1: ASPM_L1.2- ASPM_L1.1-`. EEE is `enabled - inactive` (link partner does
not advertise EEE), so no LPI wake latency today — but it is advertised, so it is a
latent regression risk if the switch is ever replaced.

---

## 3. Latency and loss to the controller — CLEAN, no caveats

All three runs: **0 % loss**. No outlier above 2 ms in any run.

### Sustained, 300 packets at 5 Hz (62 s)

```
ping -i 0.2 -c 300 192.168.2.50
300 packets transmitted, 300 received, 0% packet loss, time 62145ms
rtt min/avg/max/mdev = 0.130/0.171/1.248/0.085 ms

percentiles:  p50 0.162   p90 0.178   p99 0.289   max 1.248 ms
```

A single 1.248 ms outlier in 300 probes (0.33 %), everything else inside 0.29 ms.
That is a normal scheduler artifact, not a network event.

### Large payload, 100 packets at 5 Hz

```
ping -s 1400 -c 100 -i 0.2 192.168.2.50
100 packets transmitted, 100 received, 0% packet loss, time 20589ms
rtt min/avg/max/mdev = 0.213/0.239/0.506/0.028 ms
```

mdev **0.028 ms** — remarkably tight. Full-size frames are not stressing the path.

### Burst, 200 packets at 100 Hz

```
sudo ping -i 0.01 -c 200 192.168.2.50
200 packets transmitted, 200 received, 0% packet loss, time 2232ms
rtt min/avg/max/mdev = 0.111/0.137/0.190/0.013 ms
```

Under a 100 pps burst the path gets *faster and tighter* (0.137 ms avg, 0.190 ms
max) — the signature of a healthy switched path with warm ARP/cache. No queueing.

### MTU / fragmentation — clean, exactly 1500

```
ping -M do -s 1472 -c 5   →  5 received, 0% loss, rtt 0.212/0.227/0.242
ping -M do -s 1473 -c 2   →  0 received, 100% loss, +2 errors  (correct)
```

1472 + 28 = 1500 passes DF; 1473 fails. Path MTU is exactly 1500 with no
intermediate reduction. `IpReasmFails 0`, `IpFragFails 0`, `IpReasmTimeout 0`. RTDE
packets (~1 KB) never fragment. **No MTU issue exists.**

---

## 4. Kernel network state — one counter matters

### Clean: no memory pressure, no drops, no loss

Every counter that would indicate the *kernel* mishandling the stream is zero:

```
TcpExtPruneCalled                 0      TcpExtTCPAbortOnMemory        0
TcpExtRcvPruned                   0      TcpExtTCPMemoryPressures      0
TcpExtOfoPruned                   0      TcpExtTCPMemoryPressuresChrono 0
TcpExtTCPRcvCollapsed             0      TcpExtTCPRcvQDrop             0
TcpExtListenOverflows             0      TcpExtTCPBacklogDrop          0
TcpExtListenDrops                 0      TcpExtPFMemallocDrop          0
TcpExtTCPOFODrop                  0      TcpExtTCPZeroWindowDrop       0
TcpExtTCPAbortOnTimeout           0      TcpExtTCPRetransFail          0
```

`TCPAbortOnMemory 0` and `PruneCalled 0` together rule out the kernel ever
force-closing a socket for memory reasons. `TCPRcvCollapsed 0` means it never even
had to compact a receive queue.

### Clean: retransmission and loss are effectively zero

```
/proc/net/snmp Tcp:
  ActiveOpens 1044   PassiveOpens 79   AttemptFails 52   EstabResets 189
  InSegs 2719668     OutSegs 1773027   RetransSegs 355
  InErrs 0           OutRsts 414       InCsumErrors 0
```

```
TcpExtTCPSynRetrans      336
TcpExtTCPTimeouts        339
TcpExtTcpTimeoutRehash   339
TcpExtTCPLostRetransmit  294
TcpExtTCPFastRetrans       0
TcpExtTCPSlowStartRetrans  0
TcpExtTCPSACKReorder      40
TcpExtTCPOFOQueue        159
TcpExtTCPDSACKRecv        16
TcpExtTCPSpuriousRtxHostQueues 3
```

The decisive decomposition: **`RetransSegs 355` − `TCPSynRetrans 336` = 19
data-segment retransmissions**, across every TCP connection on this machine, since
boot. Out of `TCPOrigDataSent 382369`. That is a data-loss rate of **0.005 %**, and
it is not even attributable to the robot LAN.

Likewise `TCPTimeouts 339` is almost entirely SYN-phase (`TcpTimeoutRehash 339`
matches, and `TCPSynRetrans` is 336) — these are connection *attempts* to
unreachable hosts (office/Tailscale/internet), not established sessions stalling.
`TCPFastRetrans 0` and `TCPSlowStartRetrans 0` confirm no established connection
ever entered loss recovery.

`InErrs 0` and `InCsumErrors 0` on 2.72 M segments — **zero corrupt segments**.
Combined with `align_errors 0` at the NIC, the cabling is beyond suspicion.

### NOT clean: receive-buffer exhaustion

```
TcpExtTCPToZeroWindowAdv        213
TcpExtTCPFromZeroWindowAdv      213
TcpExtTCPWantZeroWindowAdv       71
TcpExtTCPAbortOnData            202
TcpExtTCPAbortOnClose            32
TcpOutRsts                      414
TcpEstabResets                  189
```

`TCPToZeroWindowAdv` is incremented **when this host advertises a zero receive
window** — the local receive queue is completely full and the kernel is telling the
peer to stop sending. It is, by construction, a **userspace-not-reading** signal.
The kernel cannot cause it; only a socket whose owner has stopped calling `recv()`
can.

`FromZeroWindowAdv` equals `ToZeroWindowAdv` (213 = 213), so every stall did
eventually reopen — these were transient fills, not one permanent wedge.

**Idle-baseline control measurement (VERIFIED).** With the application not running
and no socket to `192.168.2.50` present (`ss -tinmo state all dst 192.168.2.50`
returned empty):

| sample | wall time | `TCPToZeroWindowAdv` |
|---|---|---|
| A | ~12:44 | 212 |
| B | ~12:51 | 213 |
| C | B + 90 s idle | **213 (flat)** |

Over the controlled 90 s idle window the counter did not move at all. The one
increment between A and B (~7 min apart) is attributable to background browser /
Tailscale / agent HTTP traffic. Extrapolating that background rate over the 161 min
of uptime yields **~23 expected background events against 213 observed** — leaving
roughly **190 events unexplained by anything except the application runs**.

`TCPAbortOnData 202` (RST sent with unread data still queued) is the matching
tail-end signature: sockets being torn down while their receive queues were
non-empty.

### Not a factor: firewall and conntrack

```
ufw: active. Default deny (incoming), allow (outgoing).
     Anywhere  ALLOW IN  192.168.2.0/24      ← entire robot subnet permitted
nf_conntrack_count 165 / nf_conntrack_max 262144   (0.06% utilisation)
nf_conntrack_tcp_timeout_established = 432000 s  (5 days)
UFW BLOCK events in window 11:45–12:15:  0
```

Conntrack table is at 0.06 % and the established timeout is 5 days, so no entry can
be evicted from a ~90 s session. Zero UFW blocks during the run window. **The
firewall is exonerated.**

Also checked: `net.ipv4.conf.*.rp_filter = 2` (loose) with
`TcpExtIPReversePathFilter 6` — six packets dropped by reverse-path filtering since
boot. Loose mode cannot drop robot traffic (`192.168.2.50` is directly routable via
`enp12s0`); these are stray office/multicast packets. Worth knowing about given the
dual-subnet setup, but not implicated.

---

## 5. Buffer sizing — and why `rmem_max` is the wrong knob

```
net.core.rmem_max          = 212992      (208 KB)
net.core.rmem_default      = 212992
net.core.wmem_max          = 212992
net.ipv4.tcp_rmem          = 4096  131072  33554432    (min 4K, default 128K, max 32M)
net.ipv4.tcp_wmem          = 4096   16384   4194304
net.ipv4.tcp_mem           = 739014  985353  1478028   (pages; ~2.8G / 3.8G / 5.6G)
net.ipv4.tcp_moderate_rcvbuf = 1                        (autotuning ON)
net.core.netdev_max_backlog  = 1000
net.core.netdev_budget       = 300
net.ipv4.tcp_congestion_control = cubic
net.core.default_qdisc          = fq_codel
```

These are stock Ubuntu 24.04 values.

**Which limit actually applies — VERIFIED from source.** `net.core.rmem_max` only
caps sockets that call `setsockopt(SO_RCVBUF)`. Grepping the ur_rtde sources at
`/tmp/ur_rtde_src`:

```
src/rtde.cpp:66:  boost::asio::ip::tcp::no_delay no_delay_option(true);
src/rtde.cpp:67:  boost::asio::socket_base::reuse_address sol_reuse_option(true);
src/rtde.cpp:68:  socket_->set_option(no_delay_option);
```

`TCP_NODELAY` and `SO_REUSEADDR` — **and nothing else**. There is no `SO_RCVBUF`,
no `receive_buffer_size`, anywhere in `src/` or `include/`. Confirmed independently:
`strings` over `librtde.so` and `rtde_receive.cpython-311-x86_64-linux-gnu.so` in
`/home/anton/miniconda3/envs/robo/lib/python3.11/site-packages/` yields no
`SO_RCVBUF` / `rcvbuf` / `receive_buffer_size` symbols.

**Therefore the RTDE receive socket is fully kernel-autotuned within
`tcp_rmem = 4096 131072 33554432`, and `net.core.rmem_max` is irrelevant to it.**
Raising `rmem_max` — the reflexive fix — would change **nothing**. This is worth
stating plainly because it is the most likely wrong turn from here.

**Are the defaults adequate? For a healthy reader, yes, by a wide margin.**
500 Hz × ~1 KB = ~500 KB/s = 4 Mbit/s, 0.4 % of a 1 Gb link. The 128 KB starting
buffer holds ~250 ms of stream. On a 0.17 ms RTT path the bandwidth-delay product is
~85 bytes, so autotuning has no reason to grow the buffer much beyond its default —
`tcp_rcv_space_adjust` grows the window based on bytes *copied to userspace per
RTT*, so **a reader that has stalled does not get a bigger buffer**. The buffer stays
near 128 KB precisely when you would most want it to be large.

**The arithmetic that matters:**

| receive buffer | time to fill at ~500 KB/s with a stalled reader |
|---|---|
| 128 KB (`tcp_rmem` default) | **~0.26 s** |
| 1 MB (modestly autotuned)   | ~2 s |
| 32 MB (`tcp_rmem` max)      | ~64 s |

So **any application stall longer than about a quarter of a second closes the
window** on this stream. 213 zero-window events is entirely consistent with a
reader that repeatedly stalls for a fraction of a second — and 190 of them are
unexplained by background traffic.

`netdev_max_backlog = 1000` is never approached: `/proc/net/softnet_stat` column 2
is 0 on all 24 CPUs. Not a factor.

---

## 6. Routing, dual-homing, and NetworkManager

### Routing — unambiguous, no asymmetry. VERIFIED.

```
$ ip route
default via 10.11.100.1 dev enp12s0 proto dhcp src 10.11.101.240 metric 100
10.11.100.0/23 dev enp12s0 proto kernel scope link src 10.11.101.240 metric 100
172.17.0.0/16 dev docker0 proto kernel scope link src 172.17.0.1 linkdown
192.168.2.0/24 dev enp12s0 proto kernel scope link src 192.168.2.130 metric 100

$ ip route get 192.168.2.50
192.168.2.50 dev enp12s0 src 192.168.2.130 uid 1002   cache
```

Exactly one route to `192.168.2.0/24`, scope link, source `192.168.2.130`. No
overlap with `10.11.100.0/23` or `172.17.0.0/16`. `docker0` is `linkdown` and cannot
attract traffic. Asymmetric routing is impossible with one interface.

`ip rule` contains only the Tailscale fwmark rules (5210/5230/5250 on
`fwmark 0x80000/0xff0000`) plus `lookup 52` at priority 5270 and the standard
local/main/default. **None of these can match robot traffic**: the fwmark rules
require a mark Tailscale only sets on its own tunnel traffic, and table 52 is
Tailscale's. Robot packets fall through to `main`.

### NetworkManager — profile

Active profile: **`Office+Robot`** (`0ef6d59c-98a5-4130-9de8-a77b165bbb55`) on
`enp12s0`, `autoconnect: yes`.

```
ipv4.method:  auto            (DHCP for 10.11.101.240/23)
ipv4.addresses: 192.168.2.130/24   ← static, layered on the same interface
ipv6.method:  auto
connection.gateway-ping-timeout: 0
802-3-ethernet.auto-negotiate: no      (i.e. NM does not force speed/duplex)
DHCP4 lease_time: 7200 s (2 h), server 10.11.100.1
/etc/NetworkManager/conf.d/: only default-wifi-powersave-on.conf
```

Wi-Fi (`wlp13s0`) is DOWN with NO-CARRIER and `wpa_supplicant` has no association —
**roaming is impossible and is ruled out**.

### The specific hypothesis: did NM reconfigure the interface mid-run? NO.

This was the brief's leading network-side theory. It does not survive contact with
the journal. Full `journalctl -u NetworkManager --since today`, run windows marked:

```
10:14:43  device (enp12s0): carrier: link connected
10:14:45  dhcp4 (enp12s0): state changed new lease, address=10.11.101.240
10:14:45  device (enp12s0): Activation: successful, device activated.
10:14:47  manager: startup complete
10:15:20  agent-manager: agent[.../nm-applet/1002]: agent registered
────────  ~10:27 RUN  ────────  ← no NetworkManager event of any kind
10:44:46  manager: NetworkManager state is now CONNECTED_SITE
10:44:48  manager: NetworkManager state is now CONNECTED_GLOBAL
10:49:48  manager: NetworkManager state is now CONNECTED_SITE
10:49:49  manager: NetworkManager state is now CONNECTED_GLOBAL
11:14:45  dhcp4 (enp12s0): state changed new lease, address=10.11.101.240
11:19:49  manager: NetworkManager state is now CONNECTED_SITE
11:19:50  manager: NetworkManager state is now CONNECTED_GLOBAL
────────  11:51–12:10 RUN  ────────  ← no NetworkManager event of any kind
12:14:45  dhcp4 (enp12s0): state changed new lease, address=10.11.101.240
```

**VERIFIED findings:**

1. **No NM event falls inside either run window.** The 10:27 run sits in a 29-minute
   gap of total silence (10:15:20 → 10:44:46). The 11:51–12:10 run sits in a
   55-minute gap (11:19:50 → 12:14:45). The nearest DHCP renew missed the second run
   by **4 min 45 s on one side and 36 min on the other**.
2. **The DHCP renews did not reconfigure anything.** All three renews returned the
   *same* address `10.11.101.240`. There is no following `ip-config` state change,
   no `Activation:` line, no address flap — NM only reapplies configuration when the
   lease actually changes. Renews land on a 1 h cadence (T1 = 50 % of the 7200 s
   lease): 10:14:45, 11:14:45, 12:14:45.
3. **The `CONNECTED_SITE` ⇄ `CONNECTED_GLOBAL` flaps are the connectivity checker
   and they do not touch the interface.** Six transitions today. These are
   `NMConnectivityState` changes only — a portal-detection HTTP probe failing and
   recovering. No `device (enp12s0)` state line accompanies any of them. And in any
   case none coincides with a run.
4. **No carrier event since boot.** `dmesg` shows exactly one link transition, at
   startup:
   ```
   [Aug 24 10:14:41] r8169 0000:0c:00.0 enp12s0: Link is Down
   [Aug 24 10:14:43] r8169 0000:0c:00.0 enp12s0: Link is Up - 1Gbps/Full - flow control off
   ```
   Nothing afterwards. **No link flap during any run.**
5. **Kernel log during the run windows is empty of network events.** Filtering
   `journalctl -k` for 10:20–12:20 (excluding UFW noise) yields only RealSense D405
   USB enumeration at 10:26:01 and one unrelated AppArmor denial at 12:00:36. `UFW
   BLOCK` count in 11:45–12:15: **0**.

**2026-08-21 (VERIFIED, weaker).** Same pattern, no run times supplied so no precise
correlation is possible. Boot at 12:34:08 (`carrier: link connected`), DHCP renews
at 13:34:11 and 14:34:11 (same address each time), connectivity flaps at 13:14:11
and 14:09:12, clean NM shutdown at 14:46:50 (`caught SIGTERM`, machine going down).
No unexplained interface events.

**Conclusion: NetworkManager, DHCP, IPv6 RA, connectivity checking and
`wpa_supplicant` are all exonerated for the 2026-08-24 runs.** This hypothesis is
dead, and cleanly so.

### Config defect found (cosmetic, but fix it)

```
Aug 24 10:14:39 NetworkManager[2549]:
  /etc/netplan/90-NM-8f9793e3-09be-4a01-a9ef-e6f6a0e10679.yaml:9:7:
  Error in network definition: invalid prefix length in address '192.168.2.130/0'
        - "192.168.2.130/0"
```

This is a **stale, inactive** profile (`Profile 1`, UUID `8f9793e3…`) — note it does
not even appear in `nmcli con show`, because NM failed to load it. The active
`Office+Robot` profile correctly carries `192.168.2.130/24`. So this is not causing
the bug. But `/24` written as `/0` is a live foot-gun: it parses as a default route
covering the entire address space, and `netplan apply` errors out on this file
today. There are also unused `UR Direct`, `Teachbot` and `Wired connection 1`
ethernet profiles with `autoconnect: yes`, any of which could activate unexpectedly.

---

## 7. What else is on this LAN?

### On the robot subnet: only the robot. VERIFIED.

```
$ ip neigh show dev enp12s0 | grep 192.168.2
192.168.2.50 lladdr 00:30:d6:2f:65:2f STALE

$ ip neigh show dev enp12s0 | grep -c 192.168.2
1
```

A broadcast sweep (`ping -b -c 3 192.168.2.255`) drew **0 replies** — expected, as
UR controllers do not answer broadcast ICMP. The ARP cache is the reliable evidence:
**exactly one host on `192.168.2.0/24`.** No second machine is competing for the
UR's RTDE client slots from this subnet.

Caveat (INFERRED): this only proves *this PC* has spoken to one host. A third party
on the office side could still reach `192.168.2.50` — see below — and this PC would
never see it. Ruling that out requires checking the UR's own client list or a
capture on the robot port, neither of which is possible under the read-only,
no-connection constraint.

### Full neighbour table — the shared-segment evidence

```
192.168.2.50    lladdr 00:30:d6:2f:65:2f  STALE      ← UR5e
10.11.100.1     lladdr 90:ec:77:86:f2:03  REACHABLE  ← office gateway
10.11.100.20    lladdr 7c:57:58:53:3c:47  STALE
10.11.100.124   lladdr ec:b5:fa:24:31:7c  STALE
fe80::7e57:58ff:fe53:3c47 lladdr 7c:57:58:53:3c:47 STALE
```

Four hosts, one interface. `10.11.100.20` is a WS-Discovery talker sending 2239-byte
UDP datagrams to us (visible in UFW logs); `10.11.10.100` and `10.11.101.132` are
SSDP sources. This is ordinary office chatter and it is sharing the robot's wire.

### Hub or switch? **SWITCH. VERIFIED, conclusively.**

Three independent proofs:

1. **`Duplex: Full` at 1000 Mb/s.** Hubs are half-duplex by definition — they are a
   shared collision domain. A full-duplex link is a point-to-point link to a
   switching device.
2. **Gigabit hubs do not exist.** 1000BASE-T was never specified for repeaters. The
   link partner advertising `1000baseT/Full` settles it.
3. **Zero collisions.** `tx_single_collisions: 0`, `tx_multi_collisions: 0`,
   `tx_aborted: 0` on 1 716 610 transmitted packets. On a real hub carrying office
   traffic, collisions would be unavoidable.

**So the "hub" in the brief is a switch, collisions are impossible, and the
flooding/contention concern is void.** Broadcast and multicast still flood (that is
what §1's `rx_dropped` is), but unicast RTDE traffic is switched point-to-point and
is not visible to, or contended by, other hosts.

---

## Confidence and what is NOT proven

Stating this precisely matters, because the zero-window finding is the load-bearing
one and it is *not* fully verified.

**VERIFIED (measured directly, this session):**
- Every NIC error counter is zero; no link flap since boot.
- 0 % ICMP loss across 605 probes at 3 sizes / 3 rates; RTT p99 = 0.289 ms.
- Path MTU is exactly 1500; no fragmentation.
- 19 data retransmissions and 0 checksum errors in 2.7 M segments.
- No kernel memory pressure, prune, collapse, or backlog drop of any kind.
- `TCPToZeroWindowAdv = 213` this boot, with a measured flat idle baseline.
- ur_rtde sets only `TCP_NODELAY`/`SO_REUSEADDR`; **no `SO_RCVBUF`** — so
  `net.core.rmem_max` does not apply and `tcp_rmem` governs.
- No NetworkManager/DHCP/carrier/UFW event inside either run window.
- Single host on `192.168.2.0/24`; the link partner is a switch, not a hub.
- Both subnets share one NIC and one broadcast domain.

**INFERRED (consistent with the evidence, not directly proven):**
- **That the 213 zero-window events are on port 30004.** The counter is
  system-wide; Linux does not attribute it per-socket, and it cannot be
  reconstructed after the fact. The attribution rests on the near-zero idle
  baseline plus the ~190-event excess coinciding with the only heavy-stream
  workload on the box. Strong, but circumstantial. **This is the single most
  valuable thing left to confirm** — RECOMMENDED ACTION #1 closes it.
- **That URControl responds to a zero-window client by closing the session.** This
  is the documented and widely-reported behaviour of the UR RTDE server and it
  matches the observed clean FIN, but it was not verified against this controller.
- **That CPU16 softirq contention contributes.** Plausible given single-queue RX,
  no `isolcpus`, and a PREEMPT_RT kernel, but `time_squeeze = 23 / 0 drops` is far
  too low to call it proven.

**Why the 23 isolated trials never reproduced it** — consistent with the above: an
isolated test drains the RTDE socket promptly and leaves CPU16 idle, so the receive
window never closes and the controller never has reason to hang up. The bug needs
the full application's CPU load and its (suspected) undrained second socket. This
network sweep cannot confirm that, but it is the only hypothesis left standing after
the transport layer is excluded.

---

## RECOMMENDED ACTIONS

Ranked. Nothing below has been applied — **no configuration was changed.**
Items 1–3 are diagnostic and safe. Items 4–7 are changes, in decreasing value.

### 1. Prove which socket hits zero-window — do this before changing anything

Highest value by a wide margin: it converts the central inference into fact, and it
is passive, read-only, and needs no code change.

While the app runs, from a second terminal:

```sh
# per-socket receive queue + advertised window, 2 Hz, timestamped
while :; do
  printf '%s ' "$(date +%H:%M:%S.%3N)"
  ss -tinm dst 192.168.2.50 | tr '\n' ' '
  echo
  sleep 0.5
done | tee /tmp/rtde_sockets.log

# system-wide zero-window counter, 1 Hz, in parallel
while :; do
  printf '%s %s\n' "$(date +%H:%M:%S)" \
    "$(nstat -asz TcpExtTCPToZeroWindowAdv | tail -1)"
  sleep 1
done | tee /tmp/zerowin.log
```

**How to verify:** at the moment of the FIN, look at `/tmp/rtde_sockets.log`.
- `Recv-Q` on the **:30004** socket climbing toward `rcv_ssthresh`/`skmem` limits,
  with `rcv_space` shrinking → **application-side stall confirmed**; the network is
  done as a suspect and the fix belongs in the app's read loop.
- `Recv-Q` at ~0 and `TCPToZeroWindowAdv` flat in `/tmp/zerowin.log` while the FIN
  still arrives → my inference is wrong; reopen the investigation on the controller
  side (RTDE recipe/protocol error, watchdog, or a controller-side timeout).

`ss -tinm` also prints `retrans:`, `rto:` and `lastrcv:` per socket, which will
independently corroborate §4's finding that there is no retransmission on this path.

### 2. Log the receive socket's local port at construction

Directly closes the instrumentation gap already flagged in
`loop_stability_review.md` §6: `stream_autopsy()` cannot currently tell which of the
process's two :30004 sockets belongs to `RTDEReceiveInterface` (the other is
`RTDEControlInterface` — ur_rtde opens :30004 for both), so the same autopsy
evidence has been read two contradictory ways.

Diff the process's :30004 sockets immediately before and after constructing the
receive interface; the new one is its own. Record that port.

**How to verify:** on the next death, the autopsy names a specific local port and
its state (`ESTABLISHED` / `CLOSE_WAIT` / `TIME-WAIT`) unambiguously, and the
`ss` log from #1 can be filtered to exactly that socket.

### 3. Check whether NIC softirq and the app's RT threads collide on CPU16

All RX lands on CPU16 (single queue, RPS off, `irqbalance` inactive, no `isolcpus`).

```sh
# during a run:
watch -n1 'grep -E "^ *99:" /proc/interrupts'
awk '{print NR-1, $3}' /proc/net/softnet_stat   # col 3 = time_squeeze, per CPU
ps -eLo psr,pri,cls,comm | awk '$1==16'         # what else is pinned to CPU16?
```

**How to verify:** `time_squeeze` on CPU16 climbing during a run (baseline is 23
since boot, ~0/min at idle), or an RT-class app thread showing `psr == 16`. If
either is true, move NIC IRQ 99 to a housekeeping CPU
(`echo 2 > /proc/irq/99/smp_affinity_list`) and keep the app off that CPU. If
`time_squeeze` stays flat, drop this line of enquiry.

### 4. Fix the read loop — the actual fix, if #1 confirms

If #1 confirms a stalled reader, no sysctl will fix it; buffer tuning only moves the
deadline. The reader must never block. Standard shape: a dedicated thread that does
nothing but `recv()` into a bounded ring buffer and drops oldest on overflow, with
consumers reading the ring — never the socket. For a 500 Hz state stream, dropping a
stale sample is always better than stalling the socket.

**How to verify:** re-run with the #1 monitors attached. `Recv-Q` on :30004 stays
near zero for the whole run and `TCPToZeroWindowAdv` does not advance. Then run the
full app past the ~90 s mark repeatedly — the bug is fixed only when it survives
several runs well beyond its usual failure point, not one.

### 5. Give the stream more headroom — mitigation only, not a fix

Buys time for a briefly-stalled reader; does **not** fix a permanently stalled one.
Apply only *after* #1, and only as a safety margin alongside #4.

```sh
# WOULD change (not applied). Raises the autotuned default 128K → 1M.
sudo sysctl -w net.ipv4.tcp_rmem="4096 1048576 33554432"
```

Note carefully: **do not raise `net.core.rmem_max`.** Per §5 it is verified to be
irrelevant here — ur_rtde never calls `setsockopt(SO_RCVBUF)`, so `rmem_max` never
applies to this socket. Raising it is a placebo.

At ~500 KB/s this moves the zero-window deadline from ~0.26 s to ~2 s.

**How to verify:** `ss -tinm dst 192.168.2.50` shows a larger `skmem:(r…)` ceiling
during a run, and `TCPToZeroWindowAdv` accumulates more slowly for the same
workload. If the FIN still arrives on schedule, the stall is longer than 2 s and
#4 is mandatory.

### 6. Separate the robot from the office LAN

Addresses §0. The robot LAN is not currently a dedicated LAN — it is a second subnet
on the office broadcast domain, which is why ~3–5 pps of foreign broadcast is
landing on the NIC that carries the 500 Hz stream, and why the robot is reachable
from the office.

Two options, both requiring a decision you should make rather than one I pick:
- **Second NIC** for `192.168.2.0/24`, robot on its own physical segment. Cleanest;
  needs hardware. The unused `UR Direct` profile (`2c423562-…`) suggests this was
  the original intent.
- **VLAN-separate** the robot port on the switch, keeping one NIC. No hardware; needs
  switch access.

**How to verify:** after separation, `ip neigh show dev <robot-iface>` lists only
`192.168.2.50`, and `cat /sys/class/net/<robot-iface>/statistics/rx_dropped` stays
at ~0 over several minutes instead of climbing 3–5 pps.

This is worth doing on general principle. **It will not fix the FIN** — the evidence
in §1–§4 is unambiguous that the transport is healthy — so do not sequence it ahead
of #1 or #4.

### 7. Clean up the NetworkManager / netplan profiles

Low urgency; not implicated in the bug. Prevents a future surprise:

- `/etc/netplan/90-NM-8f9793e3-09be-4a01-a9ef-e6f6a0e10679.yaml` contains
  `192.168.2.130/0` and fails to parse. The owning profile (`Profile 1`) is dead —
  delete it, or correct `/0` → `/24`.
- `UR Direct`, `Teachbot`, `Wired connection 1` are unused ethernet profiles with
  `autoconnect: yes` and could activate unexpectedly on `enp12s0`. Set
  `autoconnect no` on the ones you are not using.
- Consider `ethtool --set-eee enp12s0 eee off` **only if** the switch is ever
  replaced with one that advertises EEE. Today it reads `enabled - inactive`, so
  there is nothing to gain — noted purely as a latent risk.

**How to verify:** `sudo netplan generate` completes without error, and
`nmcli con show` lists no autoconnect-enabled ethernet profile bound to `enp12s0`
other than `Office+Robot`.
