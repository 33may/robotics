"""record_panel.py — start / monitor / stop a live mapping recording (MAY-173 1.5).

The dev_app is the REMOTE CONTROL, never the writer: Start spawns the standalone
`recording.recorder_main` process (ProcessLauncher, same pattern as the teleop
pad) which drains the World's already-flowing camera + debug-pose channels to
disk. That split is the crash-safety story — this panel (or the whole dev_app)
can die and the recording keeps going and finalizes safely (idle-timeout or a
later manual stop); every frame/row is on disk the moment it's written.

Monitor = the recorder's `<out>/status.json` heartbeat (atomic rewrite ~1 Hz):
state, frames, distinct stamps, cadence gaps, skipped-no-pose. Read here with a
0.5 s throttle — no socket coupling between panel and recorder.

Stop = SIGTERM → the recorder drains its encoder queue and exits 0 ("saved").
Panel teardown does the same, so closing the dev_app window saves the take.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from imgui_bundle import imgui

from ..launcher import ProcessLauncher
from ..panel import Panel
from ..state import AppState

_REPO_ROOT = Path(__file__).resolve().parents[5]
_HUMANOID = _REPO_ROOT / "humanoid"
_DEFAULT_PARENT = "data/coverage_drives"
_CODECS = ["jpeg", "png"]
_GREEN = imgui.ImVec4(0.4, 0.9, 0.4, 1.0)
_RED = imgui.ImVec4(0.95, 0.5, 0.4, 1.0)
_GREY = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)


class RecordPanel(Panel):
    title = "Recording"
    dock_space = "LeftSpace"

    def __init__(self, camera_socket: str = "/tmp/oli-world-frames.sock",
                 pose_socket: str = "/tmp/oli-record-pose.sock") -> None:
        self._camera_socket = camera_socket
        self._pose_socket = pose_socket
        self._out = f"{_DEFAULT_PARENT}/teleop_{time.strftime('%d%m_%H%M')}"
        self._codec_idx = 0
        self._proc: ProcessLauncher | None = None
        self._launched_out: Path | None = None
        self._status_cache: dict = {}
        self._status_read_wall = 0.0

    # ── control ──────────────────────────────────────────────────────────────

    def _start(self) -> None:
        out = (_HUMANOID / self._out).resolve()
        cmd = [sys.executable, "-m", "humanoid.logic.oli.recording.recorder_main",
               "--out", str(out), "--codec", _CODECS[self._codec_idx],
               "--camera-socket", self._camera_socket,
               "--pose-socket", self._pose_socket,
               "--connect-timeout", "240"]  # tolerate a still-booting Isaac World
        self._proc = ProcessLauncher(cmd, cwd=str(_REPO_ROOT), name="recorder")
        self._launched_out = out
        self._status_cache = {}
        self._proc.start()

    def _stop(self) -> None:
        if self._proc is not None:
            # generous timeout: SIGTERM → the recorder drains its queue first
            self._proc.stop(timeout=20.0)

    def _status(self) -> dict:
        """Throttled read of the recorder's heartbeat file (atomic on its side)."""
        now = time.monotonic()
        if now - self._status_read_wall < 0.5:
            return self._status_cache
        self._status_read_wall = now
        if self._launched_out is not None:
            try:
                self._status_cache = json.loads(
                    (self._launched_out / "status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # not written yet / mid-rename — keep the last snapshot
        return self._status_cache

    # ── ui ───────────────────────────────────────────────────────────────────

    def draw(self, state: AppState) -> None:
        running = self._proc is not None and self._proc.is_running()

        imgui.text("Mapping recording")
        imgui.text_disabled(f"cameras: {self._camera_socket}")
        imgui.separator()

        imgui.begin_disabled(running)
        _, self._out = imgui.input_text("out dir", self._out)
        _, self._codec_idx = imgui.combo("codec", self._codec_idx, _CODECS)
        imgui.end_disabled()

        if running:
            if imgui.button("Stop + save"):
                self._stop()
        else:
            if imgui.button("Start recording"):
                self._start()
        imgui.same_line()
        proc_status = self._proc.status() if self._proc is not None else "idle"
        rc = self._proc.returncode if self._proc is not None else None
        color = _GREEN if running else (_RED if rc not in (None, 0) else _GREY)
        imgui.text_colored(color, proc_status)

        st = self._status()
        if not st:
            imgui.text_disabled("(no status yet — is the World up with --cameras "
                                "--stereo-cameras --debug-pose?)")
            return
        state_str = st.get("state", "?")
        imgui.text_colored(
            _GREEN if state_str in ("recording", "saved") else _GREY,
            f"state: {state_str}")
        if st.get("frames") is not None:
            gaps = st.get("gaps", 0)
            imgui.text(f"frames: {st.get('frames', 0)}   stamps: {st.get('stamps', 0)}")
            imgui.text(f"persisted: {st.get('queued', 0)}   elapsed: {st.get('elapsed_s', 0)}s")
            imgui.text_colored(
                _RED if gaps else _GREY,
                f"gaps: {gaps} (missed ~{st.get('missed_frames', 0)})")
            skipped = st.get("skipped_no_pose", 0)
            if skipped:
                imgui.text_colored(_RED, f"skipped (no GT pose): {skipped}")
        per = st.get("per_stream") or {}
        if per:
            imgui.text_disabled("  ".join(f"{k}:{v}" for k, v in sorted(per.items())))

    def teardown(self) -> None:
        self._stop()  # graceful: dev_app close = save the take
