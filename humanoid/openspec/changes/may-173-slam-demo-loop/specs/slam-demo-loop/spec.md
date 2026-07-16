## ADDED Requirements

### Requirement: No privileged information in the demo loop

The demo loop SHALL consume only what a physical robot would have: its own camera frames, operator hints, and artifacts baked from its own recordings. Ground truth SHALL NOT be read by any component of the loop (localizer, planner, follower, displayed map). The occupancy grid SHALL be built from cuVSLAM-estimated poses, making the map frame the world frame of the entire demo; no GT-derived registration is consumed at runtime except by display-only validation instruments.

#### Scenario: GT absent from the runtime dependency graph

- **WHEN** the deployment stack runs in LOC MODE with validation instruments disabled
- **THEN** no process in the loop reads sim GT poses, the Isaac-GUI occupancy export, or `registration.json`
- **AND** the robot's displayed position, planned path, and steering derive solely from camera frames and baked artifacts

### Requirement: Three-phase demo scenario

The system SHALL support the full explore→bake→deploy loop as an operator-driven scenario: (P1) teleoperated exploration recording stereo at 30 Hz per the locked capture recipe; (P2) offline bake of the recording into a cuVSLAM keyframe map, a cuVGL BoW index, and an occupancy grid; (P3) deployment from a known start where the operator clicks a goal on the SLAM-built map and the robot navigates there while localizing against the baked artifacts.

#### Scenario: End-to-end zone demo

- **WHEN** the operator records a teleop walk of the demo zone, runs the P2 bake, then relaunches the stack in deployment mode and clicks a reachable goal on the occupancy map
- **THEN** the robot plans a path on the SLAM-built grid and reaches the goal steering on its estimated pose
- **AND** no tracking-loss teleport (>0.5 m step) occurs while inside the mapped zone

### Requirement: Live localizer process

Localization SHALL run as its own launcher-supervised process in its own environment, consuming the camera stream and publishing pose plus tracking state (OK, DEGRADED, LOST) at camera rate. The brain SHALL NOT import the localization library; a localizer crash SHALL degrade the stack to LOST rather than terminate it.

#### Scenario: Map-relative pose after snap

- **WHEN** the localizer receives a pose hint and `localize_in_map` succeeds against the loaded map
- **THEN** subsequent published poses are map-anchored (matched against baked keyframes, not dead-reckoned from the hint)
- **AND** each pose carries a tracking state and a map-frame coordinate

### Requirement: LOC MODE pose-source switch

dev_app SHALL provide a LOC MODE toggle that switches the displayed and control-consumed pose source from sim GT to the localizer. On enable, the localization hint SHALL come from an operator "you are here" click on the occupancy map (demo) or the current GT pose (validation rig only). Teleoperation SHALL remain fully functional in LOC MODE.

#### Scenario: Operator flips LOC MODE inside a mapped corridor

- **WHEN** the operator teleoperates into a mapped corridor and enables LOC MODE with a hint
- **THEN** the MapPanel pose switches to the localizer estimate and GT injection into the displayed pose stops
- **AND** localization success or failure of the snap is reported explicitly to the operator

### Requirement: SLAM-built occupancy artifact

The P2 bake SHALL produce a 2D occupancy grid computed from recorded depth projected through cuVSLAM-estimated poses, expressed in the map frame, loadable by the existing `OccupancyGrid`/MapPanel machinery. The Isaac-GUI occupancy export SHALL NOT be part of the demo artifact set.

#### Scenario: Occupancy grid from robot's own data

- **WHEN** the P2 bake runs on a recorded walk
- **THEN** an occupancy grid file is produced from depth + estimated poses only
- **AND** free space along the driven route is marked traversable and rack/wall structure appears occupied

### Requirement: Goal-click autonomy on estimated pose

In deployment mode, a goal click on the occupancy map SHALL trigger path planning on the SLAM-built grid and autonomous following via GLIDE_CMD, where the follower's pose feedback is the localizer estimate. GT SHALL NOT feed the follower.

#### Scenario: Localization error feeds back into control

- **WHEN** the robot follows a planned path in LOC MODE
- **THEN** steering corrections derive from the estimated pose alone
- **AND** the run is scored afterwards against GT without GT having influenced it

### Requirement: Validation oracle instruments

For sim validation only, dev_app SHALL render a GT ghost trail alongside the estimate, a live |est−GT| error readout, and loss/reloc event markers, using a bake-time `registration.json` (map↔sim transform surveyed from bake keyframes vs recorded GT). The system SHALL log est pose, GT pose, and tracking state per frame to disk for post-session autopsy. These instruments SHALL be display/log-only and removable without affecting the demo loop.

#### Scenario: Post-session autopsy

- **WHEN** a LOC MODE session ends
- **THEN** a session log exists with per-frame est, GT, and tracking state
- **AND** existing audit tooling can plot trajectories and error from it without re-running the session
