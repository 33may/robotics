"""camera_panel.py — display live RGBD streams from a CameraSource.

Renders each stream as an RGB image beside a colourized depth image, using ImmVision
(zoom, pan, pixel-value peek — handy for a camera debug view). Depends only on the
CameraSource protocol, so it renders identically against the synthetic test source or the
real Isaac / robot source.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, immvision

from ..imaging import colorize_depth, fit_within
from ..panel import Panel
from ..sources.camera_source import CameraSource
from ..state import AppState


class CameraPanel(Panel):
    title = "Cameras"
    dock_space = "MainDockSpace"

    def __init__(self, source: CameraSource) -> None:
        self._src = source

    def setup(self) -> None:
        # ImmVision requires an explicit colour-order choice once at startup
        # (breaking change, Oct 2024). Our frames are RGB.
        immvision.use_rgb_color_order()

    def teardown(self) -> None:
        # Release ImmVision's GL texture cache while the GL context is still alive;
        # otherwise its textures are freed after context destruction → segfault on exit.
        immvision.clear_texture_cache()
        self._src.close()

    def draw(self, state: AppState) -> None:
        names = self._src.stream_names()
        if not names:
            imgui.text_disabled("no camera streams")
            return
        # Fit to the live panel: two images across (rgb | depth), streams stacked. Split the
        # content region in half horizontally and share the height across streams (reserving
        # ~2 text lines + separator per stream) so the whole grid stays visible when docked.
        avail = imgui.get_content_region_avail()
        line = imgui.get_frame_height_with_spacing()
        box_w = (avail.x - imgui.get_style().item_spacing.x) / 2.0
        box_h = avail.y / max(1, len(names)) - 2.0 * line
        for name in names:
            frame = self._src.read(name)
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), name)
            if frame is None:
                imgui.same_line()
                imgui.text_disabled("(no frame)")
                imgui.separator()
                continue

            h, w = frame.rgb.shape[:2]
            disp = fit_within(w, h, box_w, box_h)
            # ImmVision's ImageBuffer binding requires a WRITEABLE array, but
            # `decode_camera_frame` returns a read-only np.frombuffer array — and
            # np.ascontiguousarray keeps it read-only when it's already contiguous. Force a
            # writeable copy (copy=True) or ImmVision raises "incompatible function arguments".
            rgb = np.array(frame.rgb, dtype=np.uint8, copy=True)
            immvision.image_display(f"{name} rgb##rgb_{name}", rgb, disp, True)
            imgui.same_line()
            depth_rgb = np.array(colorize_depth(frame.depth), dtype=np.uint8, copy=True)
            immvision.image_display(f"{name} depth##depth_{name}", depth_rgb, disp, True)

            if frame.intrinsics is not None:
                k = frame.intrinsics
                dmin = float(np.nanmin(frame.depth))
                dmax = float(np.nanmax(frame.depth))
                imgui.text_disabled(
                    f"{w}x{h}  fx={k.fx:.0f} fy={k.fy:.0f}  depth[{dmin:.2f},{dmax:.2f}]m"
                )
            imgui.separator()
