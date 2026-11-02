"""panels/loc_panel.py — the Localization panel (slam-demo-loop, designed with Anton 17-07).

The demo's localization cockpit, two columns:
  left   "what the robot sees" — head_left + cuVSLAM tracked-feature dots (id-colored,
         NVIDIA figure-3 style), the money shot;
  right  health header (host state + status + fix evidence) · |est−GT| now/mean/max ·
         error + feature-count sparklines · est/GT pose rows · candidate/map line ·
         [Re-localize (GT hint)] [Stop] · last few events.

Pure I/O: every number comes from AppState (brain thread publishes) or the shared camera
source; the two buttons write one pending command that the brain loop consumes
(`AppState.set_loc_command` — the nav-goal UI/brain split). Feature dots and the displayed
frame come from separate consumers, so they may be ±1 frame apart — invisible live.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from imgui_bundle import imgui, immvision

from ..imaging import bake_feature_dots, fit_within
from ..panel import Panel
from ..state import AppState

_HISTORY = 1200          # ~40 s of UI frames in the sparklines
_GREEN = imgui.ImVec4(0.25, 0.9, 0.35, 1.0)
_YELLOW = imgui.ImVec4(0.95, 0.8, 0.25, 1.0)
_RED = imgui.ImVec4(0.95, 0.3, 0.25, 1.0)
_DIM = imgui.ImVec4(0.6, 0.6, 0.68, 1.0)


class LocPanel(Panel):
    title = "Localization"
    dock_space = "MainDockSpace"

    def __init__(self, camera_source, candidate: str, map_name: str,
                 feature_camera: str = "head_left") -> None:
        self._source = camera_source
        self._candidate = candidate
        self._map_name = map_name
        self._feature_camera = feature_camera
        self._err_hist: deque = deque(maxlen=_HISTORY)
        self._feat_hist: deque = deque(maxlen=_HISTORY)
        self._err_max = 0.0
        self._err_sum = 0.0
        self._err_n = 0
        self._events: deque = deque(maxlen=6)
        self._last_state: str | None = None
        self._last_lc_events: int | None = None
        self._last_reloc_ok: int | None = None
        self._last_reloc_fail: int | None = None

    def setup(self) -> None:
        immvision.use_rgb_color_order()

    def teardown(self) -> None:
        immvision.clear_texture_cache()

    # ── draw ─────────────────────────────────────────────────────────────────────

    def draw(self, state: AppState) -> None:
        pose, _, _ = state.nav_snapshot()
        gt_pose, loc_state = state.loc_snapshot()
        diag = state.get_loc_diag() or {}
        self._ingest(pose, gt_pose, loc_state, diag)

        avail = imgui.get_content_region_avail()
        left_w = max(avail.x * 0.58, 200.0)
        if imgui.begin_child("##loc_left", imgui.ImVec2(left_w, 0)):
            self._draw_feature_view(diag)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##loc_right", imgui.ImVec2(0, 0)):
            self._draw_readout(state, pose, gt_pose, loc_state, diag)
        imgui.end_child()

    def _draw_feature_view(self, diag: dict) -> None:
        obs = diag.get("observations") or []
        rgb = diag.get("rgb")
        if rgb is not None:
            # the tracker's OWN frame (module diagnostics) — dots and pixels are the same
            # instant, so the overlay stays glued to features between module steps
            scale = float(diag.get("rgb_scale", 1.0))
            img = bake_feature_dots(rgb, [(u * scale, v * scale, i) for u, v, i in obs],
                                    radius=2)
        else:
            # fallback pre-first-step: the live stream, no dots to align anyway
            frame = self._source.read(self._feature_camera)
            if frame is None or frame.rgb is None:
                imgui.text_disabled(f"no {self._feature_camera} frames yet")
                return
            img = bake_feature_dots(frame.rgb, obs)
        h, w = img.shape[:2]
        avail = imgui.get_content_region_avail()
        disp = fit_within(w, h, avail.x, avail.y - imgui.get_frame_height_with_spacing())
        immvision.image_display("##loc_features", img, disp, True)
        imgui.text_disabled(f"{len(obs)} features · {self._feature_camera}")

    def _draw_readout(self, state: AppState, pose, gt_pose, loc_state, diag) -> None:
        # header: one-glance health
        color, label = self._health(loc_state)
        imgui.text_colored(color, f"● {label}")
        lm = diag.get("lc_good_landmarks")
        if lm:
            imgui.same_line()
            imgui.text_disabled(f"· {lm} map landmarks confirmed")
        lc_n = diag.get("lc_events")
        if lc_n is not None:
            imgui.same_line()
            imgui.text_disabled(f"· LC {lc_n}")

        # accuracy block (D8 oracle)
        if self._err_hist:
            err = self._err_hist[-1]
            mean = self._err_sum / max(self._err_n, 1)
            imgui.text(f"|est−GT|  {err:.3f} m")
            imgui.same_line()
            imgui.text_disabled(f"mean {mean:.3f} · max {self._err_max:.3f}")
            imgui.plot_lines("##err", np.array(self._err_hist, np.float32),
                             graph_size=imgui.ImVec2(-1, 46),
                             scale_min=0.0, scale_max=self._err_max * 1.2 or 0.1,
                             overlay_text="err [m]")
        else:
            imgui.text_disabled("|est−GT|  —")
        if self._feat_hist:
            imgui.plot_lines("##feat", np.array(self._feat_hist, np.float32),
                             graph_size=imgui.ImVec2(-1, 34),
                             scale_min=0.0,
                             overlay_text="features")

        # pose rows
        if pose is not None:
            imgui.text(f"est  x {pose.x:+.2f}  y {pose.y:+.2f}  yaw {pose.yaw:+.2f}")
        if gt_pose is not None:
            imgui.text_disabled(
                f"GT   x {gt_pose.x:+.2f}  y {gt_pose.y:+.2f}  yaw {gt_pose.yaw:+.2f}")
        imgui.text_disabled(f"{self._candidate} · {self._map_name}")

        # controls → one pending command, brain loop consumes (never blocks the UI)
        if imgui.button("Re-localize"):
            state.set_loc_command("rehint")
            self._event("re-hint requested")
        imgui.same_line()
        if imgui.button("Stop"):
            state.set_loc_command("stop")
            self._event("stop requested")

        for stamp, text in reversed(self._events):
            imgui.text_disabled(f"{stamp}  {text}")

    # ── bookkeeping ──────────────────────────────────────────────────────────────

    def _ingest(self, pose, gt_pose, loc_state, diag: dict) -> None:
        if pose is not None and gt_pose is not None:
            err = float(np.hypot(pose.x - gt_pose.x, pose.y - gt_pose.y))
            self._err_hist.append(err)
            self._err_max = max(self._err_max, err)
            self._err_sum += err
            self._err_n += 1
        obs = diag.get("observations")
        if obs is not None:
            self._feat_hist.append(float(len(obs)))
        if loc_state != self._last_state and loc_state is not None:
            self._event(f"state → {loc_state}")
            self._last_state = loc_state
        lc_n = diag.get("lc_events")
        if lc_n is not None:
            if self._last_lc_events is not None and lc_n > self._last_lc_events:
                self._event(f"LC fix · {diag.get('lc_good_landmarks', 0)} lm")
            self._last_lc_events = lc_n
        r_ok = diag.get("reloc_ok")
        if r_ok is not None:
            if self._last_reloc_ok is not None and r_ok > self._last_reloc_ok:
                self._event("reloc · fix")
            self._last_reloc_ok = r_ok
        r_fail = diag.get("reloc_fail")
        if r_fail is not None:
            if self._last_reloc_fail is not None and r_fail > self._last_reloc_fail:
                self._event("reloc · refused")
            self._last_reloc_fail = r_fail

    def _event(self, text: str) -> None:
        self._events.append((time.strftime("%H:%M:%S"), text))

    @staticmethod
    def _health(loc_state):
        s = (loc_state or "").lower()
        if "crashed" in s or "lost" in s:
            return _RED, loc_state or "no localizer"
        if "running" in s and "tracking" in s:
            return _GREEN, loc_state
        if s:
            return _YELLOW, loc_state
        return _DIM, "no localizer attached"
