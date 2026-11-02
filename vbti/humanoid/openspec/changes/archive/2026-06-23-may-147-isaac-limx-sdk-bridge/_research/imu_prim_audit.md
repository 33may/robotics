# IMU prim audit — `HU_D04_01.usd` at `/World/Oli` (task 2.5)

Date: 2026-06-22
Source dump: `_research/imu_prim_audit.txt` (raw walk of `Usd.PrimRange(/World/Oli)`).

## Result

**No IMU sensor prim ships with `HU_D04_01.usd`.**

Walked every prim under `/World/Oli` looking for:
- prim types containing `Imu` or `Sensor` — none found
- applied schemas containing `Imu` — none found
- `IsaacImuSensor` type — none found

The USD layers (including the speculated `_sensor.usd` layer referenced in the vendor doc) do not currently define an IMU. The Oli main software 2.2.12 ships a `hi13_imu_driver` package (`oli-corpus://oli-main-2.2.12#install/share/hi13_imu_driver/package.xml`), confirming the real robot has an HI13 IMU — but the asset bundle for sim doesn't model it as a prim.

## Conclusion for `Oli` class implementation

`Oli.__init__` MUST attach an IMU sensor at runtime. Per design D8:

- Mount point: `base_link` (the only prim under `/World/Oli` with `PhysicsArticulationRootAPI` — confirmed in this audit).
- Schema: `omni.isaac.sensor.IMUSensor`.
- Local pose offset: identity for v1; `imu_offset_pitch`/`imu_offset_roll` from `oli-corpus://oli-main-2.2.12#install/etc/mission_engine/imuoffset.yaml` are both `0.0` in the on-robot default, so identity is faithful.
- Sample rate: matches physics dt (1 kHz target).
- Output convention: `acc` (m/s², body frame), `gyro` (rad/s, body frame), `quat` (`(w, x, y, z)` first-w). Matches both Isaac's native quat ordering and the LimX `ImuData` struct.

## Confirmed details about `base_link` from the audit

- Prim path: `/World/Oli/base_link`
- Type: `Xform`
- Applied APIs: `PhysxArticulationAPI`, `PhysicsRigidBodyAPI`, `PhysicsMassAPI`, `PhysicsArticulationRootAPI`, `IsaacLinkAPI`
- Articulation root: ✓ (per the `PhysicsArticulationRootAPI`)
- This is the root link of the articulation chain — every hip joint has it as parent.

`base_link` is unambiguously the right mount point.
