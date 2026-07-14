## ADDED Requirements

### Requirement: Robot-mounted RGBD cameras baked into the USD

The robot USD SHALL define two RGBD cameras as part of Oli's embodiment: a chest camera parented to `waist_pitch_link` (35° pitched down) and a head camera parented to `head_pitch_link` (horizontal), at the mounts documented in `oli-corpus://user-manual#1.4.1`, with Intel RealSense D435i intrinsics. The cameras SHALL be children of the moving links so they track body motion. They SHALL be authored by an idempotent build script operating on the project asset copy; vendor USD layers SHALL remain untouched.

#### Scenario: Cameras track body motion

- **WHEN** a link between `base_link` and a camera's parent rotates (e.g. the waist pitches or the head yaws)
- **THEN** that camera's world pose rotates with the link
- **AND** the camera is not fixed to a static world pose

#### Scenario: Cameras placed at the documented mounts

- **WHEN** the robot USD is loaded at the nominal (zero) joint pose
- **THEN** the chest camera sits at `[0.092, 0.0175, 0.4336]` in `base_link` pitched 35° down
- **AND** the head camera sits at `[0.0615, 0.0175, 0.652]` in `base_link` horizontal (within tolerance)

#### Scenario: Build script is idempotent

- **WHEN** the camera build script runs against a USD that already has the cameras
- **THEN** the sensor layer still contains exactly two camera prims (no duplication)

### Requirement: World renders RGB and depth per camera

For each camera the World SHALL render an RGB image (uint8, H×W×3) and a depth image (planar distance-to-image-plane, in meters) at a configurable resolution and rate, defaulting to 1280×720 at 30 Hz to match the real `realsense_mros` configuration. Frames SHALL be exposed in-process as numpy arrays. Depth SHALL be planar Z, not radial distance.

#### Scenario: RGB and depth produced

- **WHEN** the World steps with cameras enabled
- **THEN** each camera yields an RGB uint8 array and a finite depth array at the configured resolution
- **AND** depth values are planar distance-to-image-plane in meters

#### Scenario: Resolution is configurable

- **WHEN** the camera resolution is configured to 640×360
- **THEN** the rendered RGB and depth arrays are 640×360

### Requirement: Invariant CameraFrame contract

Camera data crossing to the brain SHALL be expressed as a `CameraFrame` — stamp, camera name, RGB, depth, and intrinsics — in a world-order-free, unscaled form. A `CameraFrame` SHALL carry camera intrinsics (fx, fy, cx, cy, width, height) but SHALL NOT carry extrinsics; the camera pose SHALL be derivable brain-side by forward kinematics from an `Observation` plus the static mount table, a derivation identical in sim and real. The brain SHALL receive a byte-identical `CameraFrame` structure whether the World is Isaac or the physical robot.

#### Scenario: Frame carries intrinsics, not extrinsics

- **WHEN** a `CameraFrame` is produced
- **THEN** it includes fx/fy/cx/cy and the image resolution
- **AND** it includes no camera-in-world pose

#### Scenario: Identical structure sim vs real

- **WHEN** the brain consumes a `CameraFrame` from `SimComm` and (later) from `RealComm`
- **THEN** the contract type and field layout are identical between the two

### Requirement: Frame transport never stalls the control loop

Camera frames SHALL travel a transport separate from the proprioceptive control channel. The frame transport SHALL NOT block or slow the World's control loop, and a slow or absent brain consumer SHALL NOT back up the World. The brain's camera read SHALL be latest-wins.

#### Scenario: Control loop is unaffected by frame publishing

- **WHEN** camera publishing is active
- **THEN** the World's control-step cadence is unchanged
- **AND** the control loop never blocks waiting on a frame send

#### Scenario: Latest-wins read

- **WHEN** multiple frames queue while the brain is busy
- **THEN** the brain's next camera read returns the newest frame
- **AND** stale intermediate frames are dropped

#### Scenario: Absent consumer

- **WHEN** no brain is connected to the frame channel
- **THEN** the World keeps rendering and stepping without error

### Requirement: Communication is the only camera-aware adapter

Communication SHALL be the sole layer that reads a world-native camera and produces a `CameraFrame`: `SimComm` reads the Isaac `Camera` sensors; `RealComm` (deferred) reads the real RealSense. No perception, policy-encoding, or reconstruction logic SHALL live in Communication — only capture and packaging. The brain SHALL import neither `isaacsim` nor `limxsdk` to consume frames.

#### Scenario: SimComm packages Isaac frames

- **WHEN** `SimComm` reads the Isaac cameras
- **THEN** it emits `CameraFrame`s
- **AND** the brain never touches an Isaac sensor handle or DOF index

#### Scenario: No perception logic in Communication

- **WHEN** Communication processes a camera frame
- **THEN** it only captures and packages the frame
- **AND** it does not run detection, segmentation, or reconstruction

### Requirement: Camera pose and field of view are verifiable

The simulated cameras SHALL reproduce the documented mounts and the D435i field of view closely enough that a rendered scene matches the expected viewpoint. A smoke test SHALL render each camera, save an RGB and a depth frame, and verify camera pose and FOV.

#### Scenario: Smoke test renders and verifies

- **WHEN** the camera smoke test runs against a scene with a known target
- **THEN** it saves an RGB frame and a depth frame per camera
- **AND** it asserts each camera's world pose matches the manual mount within tolerance
- **AND** it asserts each camera's horizontal field of view matches the D435i value
