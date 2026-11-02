"""teleop_panel.py — launch + monitor the VENDOR joystick pad from the dev app.

Instead of our stripped keyboard port, this launches LimX's exact `robot-joystick` binary
(the full gamepad) plus the `sensorjoy_bridge.py` that relays its `SensorJoy` (limxsdk bus)
to our `JoyPacket` UDP :9001 — the same pair `run_oli_mujoco.py` uses. Verified to run
standalone (no sim / MROS edge). The brain listens on :9001 (`--joystick socket`), so once
the pad has focus its keys drive Oli. This panel owns none of the joystick logic — it only
starts/stops the two processes and shows their status.

    pad     : vendor/humanoid-mujoco-sim/robot-joystick/robot-joystick   (self-contained)
    bridge  : conda env `limx` → logic/simulation/mujoco/sensorjoy_bridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from imgui_bundle import imgui

from ..launcher import ProcessLauncher
from ..panel import Panel
from ..state import AppState

_REPO_ROOT = Path(__file__).resolve().parents[5]
_JOYSTICK_BIN = (_REPO_ROOT / "humanoid" / "vendor" / "humanoid-mujoco-sim"
                 / "robot-joystick" / "robot-joystick")
_BRIDGE = _REPO_ROOT / "humanoid" / "logic" / "simulation" / "mujoco" / "sensorjoy_bridge.py"
# The bridge needs limxsdk (py3.8) → the `limx` env, a sibling of the running brain env.
_LIMX_PY = Path(sys.executable).parents[2] / "limx" / "bin" / "python"


class TeleopPanel(Panel):
    title = "Teleop"
    dock_space = "LeftSpace"

    def __init__(self, host: str = "127.0.0.1", port: int = 9001) -> None:
        self._host = host
        self._port = int(port)
        self._pad = ProcessLauncher(
            [str(_JOYSTICK_BIN)], cwd=str(_JOYSTICK_BIN.parent), name="pad")
        self._bridge = ProcessLauncher(
            [str(_LIMX_PY), "-u", str(_BRIDGE),
             "--host", host, "--joy-port", str(port), "--robot-ip", "127.0.0.1"],
            cwd=str(_REPO_ROOT), name="bridge")

    def _launch(self) -> None:
        self._pad.start()      # publisher first
        self._bridge.start()   # then the SensorJoy → JoyPacket relay

    def _stop(self) -> None:
        self._bridge.stop()
        self._pad.stop()

    def draw(self, state: AppState) -> None:
        imgui.text("Vendor joystick")
        imgui.text_disabled(f"SensorJoy -> UDP {self._host}:{self._port}")
        imgui.separator()

        if not _JOYSTICK_BIN.exists():
            imgui.text_colored(imgui.ImVec4(0.95, 0.5, 0.4, 1.0), "vendor pad not found:")
            imgui.text_disabled(str(_JOYSTICK_BIN))
            return

        running = self._pad.is_running()
        if running:
            if imgui.button("Stop"):
                self._stop()
        else:
            if imgui.button("Launch"):
                self._launch()
        imgui.same_line()
        imgui.text_disabled("focus the pad window to drive")

        imgui.spacing()
        _status_line("pad", self._pad.is_running(), self._pad.status())
        _status_line("bridge", self._bridge.is_running(), self._bridge.status())

        # Live signal chain: raw stick axes → brain intent → glide command sent to the World.
        # If axes stay ~0 while you press the arrows, the pad isn't focused / not sending; if
        # axes move but glide is 0, the problem is brain-side. (Arrow keys drive it — not U+J.)
        imgui.spacing()
        imgui.separator()
        imgui.text("joystick -> intent -> robot")
        intent, joy = state.teleop_snapshot()
        _, _obs, policy_out, _mode = state.brain_snapshot()
        if joy is not None:
            axes = list(joy.axes)[:4]
            imgui.text("axes:   " + "  ".join(f"{a:+.2f}" for a in axes))
        else:
            imgui.text_disabled("axes:   (no joystick packet yet)")
        if intent is not None:
            imgui.text(f"intent: vx {intent.v_x:+.2f}  vy {intent.v_y:+.2f}  wz {intent.w_z:+.2f}")
        else:
            imgui.text_disabled("intent: (brain not attached)")
        if policy_out is not None and hasattr(policy_out, "v_x"):
            imgui.text_colored(
                imgui.ImVec4(0.4, 0.9, 0.4, 1.0),
                f"glide:  vx {policy_out.v_x:+.2f}  vy {policy_out.v_y:+.2f}  wz {policy_out.w_z:+.2f}")

    def teardown(self) -> None:
        self._stop()


def _status_line(label: str, running: bool, status: str) -> None:
    imgui.text(f"{label}:")
    imgui.same_line()
    color = imgui.ImVec4(0.4, 0.9, 0.4, 1.0) if running else imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
    imgui.text_colored(color, status)
