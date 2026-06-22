---
name: limx-sdk-role-gating
description: LimX MROS bus is role-gated bidirectionally — sim peers publish state/IMU and subscribe to RobotCmdForSim; policy peers publish RobotCmd and subscribe to state/IMU. Cross-role subscriptions silently deliver zero.
metadata:
  type: reference
---

The LimX SDK splits each peer's bus view by **role**, chosen at `Robot()` construction time. Role gating runs in both directions — get it wrong and the SDK silently delivers zero samples instead of raising.

## Role taxonomy

`Robot(RobotType.Humanoid, is_sim)` — second positional arg picks the role.

|  | `is_sim=False` (default) | `is_sim=True` |
|---|---|---|
| Role | policy peer | sim peer |
| Publishes | `RobotCmd` | `RobotState`, `ImuData` |
| Receives `RobotState` | yes (via `subscribeRobotState`) | **no** — sim is the publisher |
| Receives `ImuData` | yes (via `subscribeImuData`) | **no** — sim is the publisher |
| Receives `RobotCmd` | **self-loopback only** (`subscribeRobotCmd` — sees only what *we* published) | yes (via `subscribeRobotCmdForSim`) |

## Failure mode

A naive probe constructed as `Robot(type)` with `subscribeRobotCmd` will see RobotState + ImuData fine but **zero RobotCmd**, because the policy-side `subscribeRobotCmd` is a self-loopback that only echoes packets the same peer published. The cmd packets from other policy peers *do* exist on the wire — they're just filtered out by the role layer.

Flip to `Robot(type, True)` + `subscribeRobotCmdForSim` and the mirror happens: RobotCmd arrives, state/IMU stop.

This was the trap on Friday 2026-06-19 (got state+IMU, zero cmd) and again Monday 2026-06-22 (sim-role flip → got cmd, zero state). The full wire contract requires **two passes** of the probe, one per role.

## Probe pattern

`humanoid/logic/simulation/mujoco/probe_contract.py` encodes this as `--role both` (default) — spawns two sequential subprocesses for clean SDK state isolation per role. See its module docstring for the asymmetry table.

## Implications for the Isaac sim peer (MAY-147)

When wiring the Isaac driver to the bus:

- Construct as `Robot(RobotType.Humanoid, True)` — Isaac is a sim peer.
- Subscribe to `subscribeRobotCmdForSim` for incoming cmds. **Not** `subscribeRobotCmd`.
- Implement `publishRobotState(...)` and `publishImuData(...)` at the physics tick (1 kHz target).
- Do NOT also subscribe to `RobotState`/`ImuData` — sim peers don't receive those, and a subscribe attempt would be silently dead.

## Why it exists

Likely an anti-replay safeguard: a misconfigured policy peer cannot accidentally consume its own teammate's cmd packets and treat them as fresh commands. Common pattern in role-gated middleware (DDS QoS partitions, ROS2 contextual policies). The cost is the dual-pass requirement for any external sniffer that's neither sim nor policy.

Related: [[vendor-humanoid-mujoco-sim]], [[reference-oli-corpus-mcp]], [[isaac-oli-smoke-loader]].
