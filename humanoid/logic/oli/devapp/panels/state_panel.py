"""state_panel.py — live readout of the attached brain (or a hint when UI-only).

When a BrainLink is attached, shows the current mode, sim stamp, a sample joint's command
tracking, and the base tilt derived from the IMU quaternion — enough to see at a glance
that the brain is stepping and the robot is being driven.
"""

from __future__ import annotations

import math

from imgui_bundle import imgui

from ..panel import Panel
from ..state import AppState


def _base_roll_pitch_deg(quat_wxyz) -> tuple:
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


class StatePanel(Panel):
    title = "State"
    dock_space = "BottomSpace"

    def draw(self, state: AppState) -> None:
        attached, obs, policy_out, mode = state.brain_snapshot()
        imgui.text("Oli — Robot Brain")
        imgui.same_line()
        imgui.text_disabled(f"· UI frame {state.frame_index}")
        imgui.separator()

        if not attached or obs is None:
            imgui.text_disabled("brain: not attached (UI-only)")
            imgui.text_disabled("pass --socket /tmp/oli-world.sock to drive a running World")
            return

        imgui.text_colored(imgui.ImVec4(0.4, 0.9, 0.4, 1.0), f"attached · mode {mode}")
        imgui.text(f"sim stamp: {obs.stamp_ns / 1e9:8.3f} s")
        roll, pitch = _base_roll_pitch_deg(obs.quat_wxyz)
        imgui.text(f"base tilt: roll {roll:+6.1f}°  pitch {pitch:+6.1f}°")
        if policy_out is not None and hasattr(policy_out, "q_des"):
            err = float(max(abs(policy_out.q_des[i] - obs.q[i]) for i in range(len(obs.q))))
            imgui.text(f"max |q_des - q|: {err:.4f} rad")
        elif policy_out is not None and hasattr(policy_out, "v_x"):
            imgui.text(
                f"glide cmd: vx {policy_out.v_x:+.2f}  vy {policy_out.v_y:+.2f}  "
                f"wz {policy_out.w_z:+.2f}"
            )
