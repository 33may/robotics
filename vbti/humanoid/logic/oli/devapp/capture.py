"""capture.py — headless screenshot of the app, for self-validation.

`enable_screenshot(runner_params, out_path, n_frames)` wires Hello ImGui callbacks so the
app renders `n_frames`, grabs the final framebuffer via
`hello_imgui.final_app_window_screenshot()`, saves it as a PNG, and exits. Run under a
virtual display for a truly headless capture (no monitor needed):

    xvfb-run -a -s "-screen 0 1600x1000x24" \\
        /home/may33/miniconda3/envs/brain/bin/python \\
        -m humanoid.logic.oli.devapp --screenshot /tmp/app.png

It chains (does not clobber) any existing pre_new_frame / before_exit callbacks, and
captures BEFORE panel teardown so textures are still valid.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import hello_imgui
from PIL import Image


def enable_screenshot(runner_params, out_path: str, n_frames: int = 20) -> None:
    """Make the app auto-screenshot to `out_path` after `n_frames`, then exit."""
    rp = runner_params
    counter = {"n": 0}
    prev_pre = rp.callbacks.pre_new_frame
    prev_exit = rp.callbacks.before_exit

    def pre_new_frame() -> None:
        if prev_pre is not None:
            prev_pre()
        counter["n"] += 1
        if counter["n"] >= n_frames:
            rp.app_shall_exit = True

    def before_exit() -> None:
        try:
            img = np.asarray(hello_imgui.final_app_window_screenshot())
            if img.size:
                Image.fromarray(img).save(out_path)
                print(f"[devapp] screenshot saved: {out_path} shape={img.shape}", flush=True)
            else:
                print("[devapp] screenshot FAILED — empty framebuffer", flush=True)
        finally:
            if prev_exit is not None:
                prev_exit()

    rp.callbacks.pre_new_frame = pre_new_frame
    rp.callbacks.before_exit = before_exit
