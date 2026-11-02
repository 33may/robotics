# Oli Corpus Source Map

Seed questions from MAY-137 are mapped to official source citations. If no official source was found, the entry points to `gaps.md`.

## 1. What SDK entry points does LimX expose for Oli?

- `oli-corpus://sdk-guide#3` — MCP Server Interface.
- `oli-corpus://sdk-guide#4` — HTTP Interface.
- `oli-corpus://sdk-guide#5` — Motion Control Development Library Interface.

## 2. What high-level control commands can we send to Oli?

- `oli-corpus://sdk-guide#4.4` — Basic Function Protocol Interfaces.
- `oli-corpus://sdk-guide#4.4.5` — Control Robot Walking.
- `oli-corpus://sdk-guide#4.4.13` — Robot Dance.
- `oli-corpus://sdk-guide#4.4.15` — Robot Action Library.
- `oli-corpus://sdk-guide#4.4.16` — Motion Control.
- `oli-corpus://user-manual#3.5.4` — Voice-Based Motion Control.

## 3. What low-level control interfaces are available?

- `oli-corpus://sdk-guide#5.1.6` — Python `publishRobotCmd` interface.
- `oli-corpus://sdk-guide#5.2.7` — C++ `publishRobotCmd` interface.
- `oli-corpus://sdk-guide#5` — Motion Control Development Library Interface.

## 4. What does the PlayStation controller currently send into the locomotion policy?

- no source — see `gaps.md#playstation-controller-locomotion-policy`.

## 5. What robot state outputs can we read?

- `oli-corpus://sdk-guide#4.4.20` — Robot Joints State.
- `oli-corpus://sdk-guide#4.8.1` — Robot Status Information.
- `oli-corpus://sdk-guide#5.1.5` — Python `subscribeRobotState` interface.
- `oli-corpus://sdk-guide#5.2.6` — C++ `subscribeRobotState` interface.
- `oli-corpus://user-manual#4.1` — Robot Status List.

## 6. What sensor outputs can we access?

- `oli-corpus://sdk-guide#5.1.7` — Python `subscribeSensorJoy` interface.
- `oli-corpus://sdk-guide#5.2.8` — C++ `subscribeSensorJoy` interface.
- `oli-corpus://sdk-guide#12` — RealSense Camera Data Acquisition.
- `oli-corpus://user-manual#1.4` — Field of View of Cameras.
- `oli-corpus://user-manual#1.4.1` — Camera Position Coordinates.

## 7. What is the startup and connection flow from laptop/controller to robot?

- `oli-corpus://quick-start#section-1` — Quick Start scanned-page images.
- `oli-corpus://user-manual#2` — Remote Controller Description.
- `oli-corpus://sdk-guide#3.2` — Connecting to the MCP Service.
- `oli-corpus://sdk-guide#4.4.1` — Connect to Wi-Fi Hotspot.
- `oli-corpus://sdk-guide#4.4.2` — Query Wi-Fi Connection Status.

## 8. What simulation or deployment tools exist for testing commands safely?

- `oli-corpus://sdk-guide#6` — Check and Set the Robot Model.
- `oli-corpus://sdk-guide#7` — Robot Simulator.
- `oli-corpus://sdk-guide#7.1` — Running the Simulator.

## 9. What is still blocked, unclear, or needs confirmation from LimX?

- no source — see `gaps.md#open-confirmations-from-limx`.
