"""
Evaluation engine — protocol-driven real-robot eval.

Usage:
    python vbti/logic/inference/eval_engine.py run \
        --checkpoint=v001/checkpoints/best \
        --protocol=checkpoint_sweep

Flow per trial:
    1. Live camera + trial guidance overlay (duck/cup markers)
    2. Operator places objects → presses SPACE to start inference
    3. Inference runs live — press S (success) or F (failure) to stop
    4. Optional: press V to save video of the run
    5. Robot returns to rest → next trial

Keys:
    SPACE     start inference
    S         stop + mark success
    F         stop + mark failure
    V         save video (after S/F)
    Q / Esc   quit session early
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import re
import numpy as np
import torch
import cv2
from datetime import datetime
from pathlib import Path

from vbti.logic.inference.run_real_inference import (
    _load_policy,
    _init_robot,
    _init_cameras,
    _capture_frames,
    _stop_cameras,
    _build_observation,
    _build_grid_frame,
    _save_video_ffmpeg,
    _init_detector,
    _run_detection_overlay,
    DetectionStateHolder,
    JOINT_NAMES,
    CAMERA_PRESETS,
)
from vbti.logic.cameras.cameras import get_latest_depth
from vbti.logic.depth.realtime_prepare import depth_uint16_to_turbo_rgb
from vbti.logic.inference.async_chunk_runner import AsyncChunkRunner
from vbti.logic.servos.rest import move_to_rest
from vbti.logic.train.experiment_utils import (
    _resolve_experiment,
    _resolve_version,
    _version_dir,
)

PROTO_DIR  = Path(__file__).parent / "protocols"
WINDOW     = "Eval Engine"


# ── Checkpoint resolver ───────────────────────────────────────────────────────

def _resolve_ckpt_path(spec: str | int, experiment: str) -> Path:
    """Smart checkpoint resolution.

    Formats accepted (examples for experiment duck_cup_smolvla):
        v13-30000   → v013/lerobot_output_r1/checkpoints/030000
        v13-30k     → same (k = ×1000)
        v13-last    → latest step in v013
        v13         → latest step in v013 (no step = last)
        raw path    → used directly if it exists
        named       → falls back to existing resolve_checkpoint()
    """
    from vbti.logic.train.experiment_utils import EXPERIMENTS_ROOT, resolve_checkpoint

    # Fire parses bare numeric CLI values like ``--checkpoint=111960`` as int.
    # Treat every checkpoint spec as text before path/shorthand resolution.
    spec = str(spec)

    # 1. Direct path
    direct = Path(spec).expanduser()
    if direct.exists():
        return _pick_pretrained(direct)

    # 2. Shorthand: v13-30000 / v13-30k / v13-last / v13 / 30k / 30000 / last
    import re

    def _parse_step(s: str) -> int:
        s = s.lower()
        return int(s[:-1]) * 1000 if s.endswith("k") else int(s)

    def _resolve_in_ver(ver_str: str, step_raw: str | None) -> Path:
        exp_dir   = EXPERIMENTS_ROOT / experiment / ver_str
        ckpt_roots = sorted(exp_dir.glob("*/checkpoints"))
        if not ckpt_roots:
            raise FileNotFoundError(f"No checkpoints dir found under {exp_dir}")
        ckpts_dir = ckpt_roots[0]
        if step_raw is None or step_raw.lower() == "last":
            step_dirs = sorted(d for d in ckpts_dir.iterdir() if d.is_dir())
            if not step_dirs:
                raise FileNotFoundError(f"No checkpoints in {ckpts_dir}")
            return _pick_pretrained(step_dirs[-1])
        step_n   = _parse_step(step_raw)
        step_dir = ckpts_dir / f"{step_n:06d}"
        if not step_dir.exists():
            matches = sorted(ckpts_dir.glob(f"*{step_n}*"))
            if not matches:
                raise FileNotFoundError(f"Step {step_n} not found in {ckpts_dir}")
            step_dir = matches[0]
        return _pick_pretrained(step_dir)

    # v13-30k  /  v13-last  /  v13
    m = re.fullmatch(r"v(\d+)(?:-(\d+k?|last))?", spec, re.IGNORECASE)
    if m:
        ver_str  = f"v{int(m.group(1)):03d}"
        return _resolve_in_ver(ver_str, m.group(2))

    # 30k  /  30000  /  last  (uses active version)
    if re.fullmatch(r"\d+k?|last", spec, re.IGNORECASE):
        from vbti.logic.train.experiment_utils import _resolve_version
        active_ver = _resolve_version(experiment, None)
        return _resolve_in_ver(active_ver, spec)

    # 3. Named specifier (best / final / step_030000)
    from vbti.logic.train.experiment_utils import _resolve_experiment, _resolve_version
    exp  = _resolve_experiment(experiment)
    ver  = _resolve_version(exp, None)
    paths = resolve_checkpoint(spec, exp, ver)
    if not paths:
        raise FileNotFoundError(f"Checkpoint not found: {spec}")
    return paths[0]


def _pick_pretrained(step_dir: Path) -> Path:
    """Return pretrained_model subfolder if it exists, else step_dir itself."""
    pm = step_dir / "pretrained_model"
    return pm if pm.exists() else step_dir


def _ckpt_label(ckpt_path: Path) -> str:
    if ckpt_path.name == "pretrained_model":
        parent = ckpt_path.parent.name
        label = f"step_{int(parent):06d}" if parent.isdigit() else parent
    else:
        label = ckpt_path.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "checkpoint"


def _find_latest_session(version_d: Path, ckpt_label: str,
                         action_horizon: int, protocol: str) -> Path | None:
    """Find the most recent eval_session matching this checkpoint+ah+protocol.

    Naming convention (from `run()`):
        chkpt_<ckpt_label>_ah_<ah>_pr_<protocol>_<YYYYMMDD_HHMMSS>

    Returns the latest matching directory by timestamp suffix, or None.
    """
    sessions_dir = version_d / "eval_sessions"
    if not sessions_dir.exists():
        return None
    pattern = f"chkpt_{ckpt_label}_ah_{action_horizon}_pr_{protocol}_*"
    matches = sorted(d for d in sessions_dir.glob(pattern) if d.is_dir())
    return matches[-1] if matches else None
CUP_COLORS = [
    (0,200,255),(60,180,255),(120,160,255),
    (180,140,255),(220,120,255),(255,100,200),
]

# Entity-schema BGR fill colors (cv2 is BGR, not RGB)
ENTITY_BGR_COLORS = {
    "red":    ( 32,  32, 200),
    "black":  ( 32,  32,  32),
    "blue":   (200,  32,  32),
    "green":  ( 32, 180,  32),
    "yellow": (  0, 210, 255),
}
DUCK_BGR = (0, 210, 255)


def _is_entities_trial(t: dict) -> bool:
    """Detect entity-schema trial (has the `entities` list field)."""
    return "entities" in t


# ── Protocol ──────────────────────────────────────────────────────────────────

def load_protocol(name: str) -> dict:
    path = PROTO_DIR / (name if name.endswith(".json") else name + ".json")
    with open(path) as f:
        return json.load(f)


# ── Overlay rendering ─────────────────────────────────────────────────────────
# Frames from cameras are RGB. Draw on BGR copy, convert back.

def _draw_arrow_bgr(img, cx, cy, deg, length=20, color=(255,255,255), thickness=2):
    rad = np.radians(deg)
    ex, ey = int(cx + length*np.cos(rad)), int(cy + length*np.sin(rad))
    cv2.arrowedLine(img, (cx,cy), (ex,ey), color, thickness, tipLength=0.38)




def _mark_top_camera(frame_rgb: np.ndarray, trial: dict,
                     cup_positions: list) -> np.ndarray:
    """Draw duck + cup markers on the top camera RGB frame. Returns RGB."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = img.copy()

    dx, dy = trial["duck_px"]
    cx, cy = trial["cup_px"]
    d      = trial["duck_dir_deg"]
    g      = trial["cup_group"]
    zone   = trial["zone"]
    ccol = CUP_COLORS[g % len(CUP_COLORS)]
    dcol = (255,210,0) if zone == "ID" else (255,80,0)

    cv2.rectangle(overlay, (270,191), (424,313), (60,180,0), 1)

    for gi, (cpx, cpy) in enumerate(cup_positions):
        if gi != g:
            cv2.circle(overlay, (cpx,cpy), 8, (60,60,60), -1)

    cv2.circle(overlay, (cx,cy), 16, ccol, -1)
    cv2.circle(overlay, (cx,cy), 16, (255,255,255), 2)
    cv2.putText(overlay, f"C{g}", (cx-10,cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,20), 2)

    cv2.circle(overlay, (dx,dy), 15, dcol, -1)
    cv2.circle(overlay, (dx,dy), 15, (255,255,255), 2)
    _draw_arrow_bgr(overlay, dx, dy, d, length=22)

    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _mark_top_camera_entities(
    frame_rgb: np.ndarray,
    trial:      dict,
    nogo_bbox:  list | tuple | None = None,
    workspace_bbox: list | tuple | None = None,
    show_zones: bool = False,
) -> np.ndarray:
    """Draw entity overlays for entity-schema protocols.

    By default at eval time, only entity markers are drawn — bboxes are
    omitted (operator knows the table boundaries; rectangles clutter the live
    camera feed). Pass `show_zones=True` for debug rendering that overlays
    the workspace bbox (green) and robot no-go zone (gray).
    """
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = img.copy()

    if show_zones:
        if workspace_bbox is not None:
            x0, y0, x1, y1 = (int(v) for v in workspace_bbox)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (60, 180, 0), 1)
        if nogo_bbox is not None:
            rx0, ry0, rx1, ry1 = (int(v) for v in nogo_bbox)
            nogo_fill = overlay.copy()
            cv2.rectangle(nogo_fill, (rx0, ry0), (rx1, ry1), (128, 128, 128), -1)
            cv2.addWeighted(nogo_fill, 0.35, overlay, 0.65, 0, overlay)
            cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), (40, 80, 220), 2)

    target_name = trial.get("target")
    for ent in trial.get("entities", []):
        x, y = int(ent["px"][0]), int(ent["px"][1])
        kind = ent.get("kind", "")
        is_target = ent.get("name") == target_name

        if kind == "cup":
            color_name = ent.get("color", "default")
            bgr = ENTITY_BGR_COLORS.get(color_name, (180, 180, 180))
            r   = 18 if is_target else 14
            cv2.circle(overlay, (x, y), r, bgr, -1)
            cv2.circle(overlay, (x, y), r, (255, 255, 255), 3 if is_target else 1)
            # Letter labels omitted — color + size + outline encode target

        elif kind == "duck":
            cv2.circle(overlay, (x, y), 15, DUCK_BGR, -1)
            cv2.circle(overlay, (x, y), 15, (255, 255, 255), 2)
            if ent.get("dir_deg") is not None:
                # Same call as legacy `_mark_top_camera`: white arrow, thickness=2,
                # length=22, drawn on the main overlay (single addWeighted at end).
                _draw_arrow_bgr(overlay, x, y, ent["dir_deg"], length=22)

    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _grid_hud(grid: np.ndarray, top: str, top_col: tuple,
              bottom: str) -> np.ndarray:
    """Draw status bars on the full camera grid (BGR)."""
    h, w = grid.shape[:2]
    cv2.rectangle(grid, (0,0), (w,30), (15,15,15), -1)
    cv2.putText(grid, top, (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, top_col, 2)
    cv2.rectangle(grid, (0,h-24), (w,h), (15,15,15), -1)
    cv2.putText(grid, bottom, (8,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    return grid


def _text_input(window: str, base_frame: np.ndarray, prompt: str) -> str:
    """Capture a line of text typed into the OpenCV window.

    Returns the typed string, or "" if cancelled (Esc).
    Enter confirms, Backspace deletes, Esc cancels.
    """
    text = ""
    while True:
        display = base_frame.copy()
        h, w = display.shape[:2]
        # Input bar background
        cv2.rectangle(display, (0, h//2 - 36), (w, h//2 + 36), (20, 20, 20), -1)
        cv2.rectangle(display, (0, h//2 - 36), (w, h//2 + 36), (80, 80, 80), 1)
        # Prompt
        cv2.putText(display, prompt, (12, h//2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        # Typed text + blinking cursor
        cursor = "_" if (int(time.time() * 2) % 2 == 0) else " "
        cv2.putText(display, text + cursor, (12, h//2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # Hint
        cv2.putText(display, "Enter = confirm    Esc = cancel", (12, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
        cv2.imshow(window, display)

        key = cv2.waitKey(50) & 0xFF
        if key == 13:          # Enter
            return text.strip()
        elif key == 27:        # Esc
            return ""
        elif key == 8:         # Backspace
            text = text[:-1]
        elif 32 <= key <= 126: # Printable ASCII
            text += chr(key)


def _progress_bar(grid: np.ndarray, step: int, max_steps: int):
    w = grid.shape[1]
    bar_w = int((w - 40) * step / max_steps)
    cv2.rectangle(grid, (w-bar_w-20, 12), (w-20, 24), (80,200,80), -1)


# ── Results ───────────────────────────────────────────────────────────────────

class EvalSession:
    def __init__(self, session_dir: Path, checkpoint: str, protocol_name: str,
                 resume: bool = False):
        """Create or resume an eval session.

        If `resume=True` and `session_dir/session.json` exists, load the
        existing data dict (preserving previously-recorded trials) and append
        new trials to it. Otherwise start a fresh session.
        """
        self.dir = session_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "videos").mkdir(exist_ok=True)
        self.path = self.dir / "session.json"

        if resume and self.path.exists():
            self.data = json.loads(self.path.read_text())
            self.data.setdefault("trials", [])
            self.data["resumed_at"] = datetime.now().isoformat(timespec="seconds")
        else:
            self.data = {
                "checkpoint":    checkpoint,
                "protocol":      protocol_name,
                "date":          datetime.now().isoformat(timespec="seconds"),
                "trials":        [],
            }
        self._save()

    @property
    def n_done(self) -> int:
        return len(self.data.get("trials", []))

    def record(self, trial: dict, result: str, steps: int,
               video: str | None = None, note: str = ""):
        """Record a trial outcome. Schema-agnostic: dumps the entire trial dict
        (entity-style or legacy) plus run-time fields."""
        rec = dict(trial)   # shallow copy of trial fields (entities/tags/legacy fields)
        rec.update({
            "result": result,
            "steps":  steps,
            "video":  video,
            "note":   note,
        })
        self.data["trials"].append(rec)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def write_markdown(self):
        """Generate session.md — human-readable summary the operator can edit afterwards.

        Sections:
          - Config (from session.data["config"])
          - Overall + per-zone success rate
          - Per-cup-group breakdown
          - Failure list with pixel coords + direction + steps + note
          - Operator notes (any non-empty trial notes)
          - Insights (blank — for operator/copilot to fill in later)
        """
        from collections import defaultdict
        trials = self.data["trials"]
        cfg    = self.data.get("config", {})
        total  = len(trials)
        out    = []

        out.append(f"# Eval Session — {self.dir.name}\n")
        out.append(f"**Date:** {self.data['date']}  ")
        out.append(f"**Checkpoint:** `{cfg.get('checkpoint_label', '?')}`  ")
        out.append(f"**Protocol:** `{self.data['protocol']}`  \n")

        # Config
        out.append("## Config")
        for k in ("experiment", "version", "action_horizon", "max_steps", "fps",
                  "delta_actions", "detection", "enable_rtc", "record"):
            if k in cfg:
                out.append(f"- {k}: `{cfg[k]}`")
        out.append("")

        if total == 0:
            out.append("_No trials completed._")
        else:
            is_entity = any(_is_entities_trial(t) for t in trials)

            success = sum(1 for t in trials if t["result"] == "success")
            sr_pct  = success / total
            out.append("## Overall")
            out.append(f"- **Success rate:** {success}/{total} ({sr_pct:.0%})")

            if is_entity:
                # Per-scene breakdown (entity schema)
                by_scene = defaultdict(list)
                for t in trials:
                    by_scene[t.get("tags", {}).get("scene", "?")].append(t)
                if len(by_scene) > 1:
                    out.append("\n## Per scene")
                    for sc in sorted(by_scene):
                        grp = by_scene[sc]
                        s   = sum(1 for t in grp if t["result"] == "success")
                        out.append(f"- {sc}: {s}/{len(grp)} ({s/len(grp):.0%})")

                # Per-target-color
                by_color = defaultdict(list)
                for t in trials:
                    by_color[t.get("tags", {}).get("target_color", "?")].append(t)
                if len(by_color) > 1:
                    out.append("\n## Per target color")
                    for c in sorted(by_color):
                        grp = by_color[c]
                        s   = sum(1 for t in grp if t["result"] == "success")
                        out.append(f"- {c}: {s}/{len(grp)} ({s/len(grp):.0%})")

                # Dual-cup language disambiguation cells
                duals = [t for t in trials
                         if t.get("tags", {}).get("scene") == "both"]
                if duals:
                    out.append("\n## Dual-cup cells (target_color × side × closer)")
                    by_cell = defaultdict(list)
                    for t in duals:
                        tg = t.get("tags", {})
                        by_cell[(tg.get("target_color"), tg.get("target_side"),
                                 tg.get("target_closer"))].append(t)
                    for cell in sorted(by_cell, key=lambda c: (c[0] or "", c[1] or "", str(c[2]))):
                        grp = by_cell[cell]
                        s   = sum(1 for t in grp if t["result"] == "success")
                        out.append(f"- {cell}: {s}/{len(grp)}")
                out.append("")

                # Failures with entity positions + prompt
                failures = [t for t in trials if t["result"] == "failure"]
                if failures:
                    out.append("## Failures")
                    for t in failures:
                        note = f"  — _{t['note']}_" if t.get("note") else ""
                        scene = t.get("tags", {}).get("scene", "?")
                        prompt = t.get("task", "")
                        out.append(
                            f"- Trial {t['trial_id']:02d} [{scene}] steps:{t['steps']}  "
                            f"prompt: \"{prompt}\"{note}"
                        )
                    out.append("")
            else:
                # ── Legacy schema ──
                id_t  = [t for t in trials if t.get("zone") == "ID"]
                ood_t = [t for t in trials if t.get("zone") == "OOD"]
                if id_t:
                    s = sum(1 for t in id_t if t["result"] == "success")
                    out.append(f"- ID:  {s}/{len(id_t)} ({s/len(id_t):.0%})")
                if ood_t:
                    s = sum(1 for t in ood_t if t["result"] == "success")
                    out.append(f"- OOD: {s}/{len(ood_t)} ({s/len(ood_t):.0%})")
                out.append("")

                # Per-cup-group
                by_cup = defaultdict(list)
                for t in trials:
                    if "cup_group" in t:
                        by_cup[t["cup_group"]].append(t)
                if len(by_cup) > 1:
                    out.append("## Per cup group")
                    for g in sorted(by_cup):
                        grp = by_cup[g]
                        s   = sum(1 for t in grp if t["result"] == "success")
                        out.append(f"- C{g}: {s}/{len(grp)} ({s/len(grp):.0%})")
                    out.append("")

                # Failures (legacy)
                failures = [t for t in trials if t["result"] == "failure"]
                if failures:
                    out.append("## Failures")
                    for t in failures:
                        note = f"  — _{t['note']}_" if t.get("note") else ""
                        out.append(
                            f"- Trial {t['trial_id']:02d} [{t.get('zone','?')}] "
                            f"duck:{tuple(t['duck_px'])} cup:{tuple(t['cup_px'])} "
                            f"dir:{t['duck_dir_deg']:.0f}° steps:{t['steps']}{note}"
                        )
                    out.append("")

            # Notes
            noted = [t for t in trials if t.get("note")]
            if noted:
                out.append("## Operator notes")
                for t in noted:
                    out.append(f"- Trial {t['trial_id']:02d} ({t['result']}): {t['note']}")
                out.append("")

            # Videos
            with_video = [t for t in trials if t.get("video")]
            if with_video:
                out.append("## Videos")
                for t in with_video:
                    out.append(f"- Trial {t['trial_id']:02d} ({t['result']}): "
                               f"`{Path(t['video']).name}`")
                out.append("")

        out.append("## Insights\n")
        out.append("_Fill in observations after reviewing videos / running eval-copilot._\n")

        md_path = self.dir / "session.md"
        md_path.write_text("\n".join(out))
        print(f"  session.md written: {md_path}")

    def print_summary(self):
        trials  = self.data["trials"]
        total   = len(trials)
        if total == 0:
            print("No trials completed.")
            return
        success  = sum(1 for t in trials if t["result"] == "success")
        id_t     = [t for t in trials if t["zone"] == "ID"]
        ood_t    = [t for t in trials if t["zone"] == "OOD"]
        id_sr    = sum(1 for t in id_t  if t["result"] == "success") / max(len(id_t),  1)
        ood_sr   = sum(1 for t in ood_t if t["result"] == "success") / max(len(ood_t), 1)

        print(f"\n{'='*55}")
        print(f"EVAL SUMMARY — {self.data['checkpoint']}")
        print(f"Protocol: {self.data['protocol']}   Date: {self.data['date']}")
        print(f"{'='*55}")
        print(f"  Total:    {total}  |  Success: {success}/{total}  ({success/total:.0%})")
        print(f"  ID:       {len(id_t)}  |  {id_sr:.0%} success rate")
        print(f"  OOD:      {len(ood_t)}  |  {ood_sr:.0%} success rate")
        print(f"  Results:  {self.path}")
        print(f"{'='*55}\n")


# ── Main eval loop ────────────────────────────────────────────────────────────

def run(
    checkpoint:          str,
    protocol:            str   = "checkpoint_sweep",
    port:                str   = "/dev/ttyACM1",
    cameras:             str   = "realsense",
    experiment:          str | None = None,
    version:             str | None = None,
    robot_id:            str | None = None,
    task:                str | None = None,   # explicit override; if None, use per-trial (entity) or proto-level (legacy)
    action_horizon:      int   = 10,
    max_steps:           int   = 600,
    fps:                 int   = 30,
    max_relative_target: float = 10.0,
    delta_actions:       bool       = False,
    detection:           bool       = False,
    camera_name_map:     dict | None = None,
    device:              str        = "auto",
    start_trial:         int   = 0,
    enable_rtc:          bool       = True,
    record:              bool       = False,
    resume:              bool       = False,
    depth:               bool       = False,
    gato:                bool       = False,
):
    """Run a protocol-driven evaluation session on the real robot.

    When `resume=True`, the latest eval_session matching this checkpoint+
    action_horizon+protocol is found, loaded, and continued from the next
    un-recorded trial. New trials are appended to its `session.json`.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    experiment = _resolve_experiment(experiment)
    version    = _resolve_version(experiment, version)
    version_d  = _version_dir(experiment, version)

    ckpt_path  = _resolve_ckpt_path(checkpoint, experiment)
    ckpt_label = _ckpt_label(ckpt_path)

    # ── Session dir: resume vs fresh ──────────────────────────────────────────
    latest = (_find_latest_session(version_d, ckpt_label, action_horizon, protocol)
              if resume else None)
    if latest is not None:
        session_dir = latest
        session_id  = session_dir.name
        session     = EvalSession(session_dir, str(ckpt_path), protocol, resume=True)
        done_n      = session.n_done
        print(f"--resume: continuing {session_id} ({done_n} trials already recorded)")
        if start_trial == 0:
            start_trial = done_n          # auto-skip already-done trials
    else:
        if resume:
            print(f"--resume: no prior session for {ckpt_label}+ah{action_horizon}+{protocol}; starting fresh.")
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id  = f"chkpt_{ckpt_label}_ah_{action_horizon}_pr_{protocol}_{ts}"
        session_dir = version_d / "eval_sessions" / session_id
        session     = EvalSession(session_dir, str(ckpt_path), protocol)

    # Save full config — but on a resumed session preserve the original config
    # (only update mutable runtime fields like task/cameras/record) so the
    # provenance of trials 0..N-1 stays accurate.
    config_now = {
        "checkpoint":          str(ckpt_path),
        "checkpoint_label":    ckpt_label,
        "experiment":          experiment,
        "version":             version,
        "protocol":            protocol,
        "task":                task,
        "cameras":             cameras,
        "port":                port,
        "robot_id":            robot_id,
        "action_horizon":      action_horizon,
        "max_steps":           max_steps,
        "fps":                 fps,
        "max_relative_target": max_relative_target,
        "delta_actions":       delta_actions,
        "detection":           detection,
        "device":              device,
        "enable_rtc":          enable_rtc,
        "record":              record,
        "depth":               depth,
        "gato":                gato,
    }
    if latest is not None and "config" in session.data:
        session.data.setdefault("config_history", []).append(config_now)
    else:
        session.data["config"] = config_now
    session._save()

    proto          = load_protocol(protocol)
    trials         = proto["trials"]
    cup_positions  = proto.get("cup_positions", [])
    workspace_bbox = proto.get("workspace_bbox")
    nogo_bbox      = proto.get("robot_nogo_bbox")
    proto_schema   = proto.get("schema", "legacy")
    total          = len(trials)
    proto_task     = proto.get("task")   # protocol-level fallback
    DEFAULT_TASK   = "pick up the duck and place it in the cup"

    if task is not None:
        task_mode = "explicit-override"
        print(f"Task: GLOBAL OVERRIDE → {task!r} (--task flag, applied to every trial)")
    elif proto_schema == "entities":
        task_mode = "per-trial"
        print(f"Task: PER-TRIAL (entity protocol). Each trial supplies its own prompt.")
    elif proto_task:
        task_mode = "proto-task"
        print(f"Task: PROTO-LEVEL → {proto_task!r}")
    else:
        task_mode = "default"
        print(f"Task: DEFAULT → {DEFAULT_TASK!r}")

    print(f"Protocol schema: {proto_schema}"
          + (f"  workspace_bbox={workspace_bbox}" if workspace_bbox else "")
          + (f"  nogo_bbox={nogo_bbox}" if nogo_bbox else ""))

    # Persist task resolution into session config so downstream analysis
    # can tell whether the prompt came from --task / per-trial / proto / default.
    session.data["config"]["task"]      = task
    session.data["config"]["task_mode"] = task_mode
    session._save()

    # Early-exit guard for resume: nothing left to run
    if start_trial >= total:
        print(f"\nSession already complete: {session.n_done}/{total} trials recorded.")
        print(f"  Path: {session.path}")
        session.write_markdown()
        session.print_summary()
        return

    # ── Hardware init ─────────────────────────────────────────────────────────
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    cam_config  = CAMERA_PRESETS.get(cameras, CAMERA_PRESETS["realsense"])

    # ── Depth wiring ──────────────────────────────────────────────────────────
    # When ``depth=True``, the gripper D405 is reconfigured to also stream
    # aligned depth. Each capture iteration we read the uint16 depth, run it
    # through the SAME turbo-RGB bake the dataset prep used (clip [0.05, 0.20]
    # m → COLORMAP_TURBO via colorize_fixed_clip), and inject it as a virtual
    # camera ``gripper_depth`` so it flows through the existing
    # ``_build_observation`` / ``_build_grid_frame`` machinery without any
    # special-casing — the policy gets ``observation.images.gripper_depth`` and
    # the GUI/recording shows it as another tile.
    if depth:
        if "gripper" not in cam_config:
            raise ValueError(
                f"--depth=true requires a 'gripper' camera in preset '{cameras}'; "
                f"got: {list(cam_config.keys())}"
            )
        cam_config = dict(cam_config)
        cam_config["gripper"] = {**cam_config["gripper"], "depth": True}
        # OpenCV V4L2 can only see the D405's color stream — depth lives only
        # in librealsense's z16 path. Auto-swap the gripper to its canonical
        # librealsense entry (other cameras stay on whatever backend the user
        # picked) so ``--cameras=opencv --depth=true`` just works.
        if cam_config["gripper"].get("type", "realsense") != "realsense":
            ref = CAMERA_PRESETS["realsense_depth"]["gripper"]
            print(
                f"[depth] gripper was on '{cam_config['gripper'].get('type')}' "
                f"(no z16 stream); swapping to RealSense serial={ref['serial']} "
                f"so depth can be captured. top/left/right stay on the original backend."
            )
            cam_config["gripper"] = dict(ref)

    cam_devices = _init_cameras(cam_config, fps=fps)
    cam_names   = list(cam_config.keys())
    if depth:
        cam_names.append("gripper_depth")

    robot = _init_robot(port, robot_id=robot_id, max_relative_target=max_relative_target)
    move_to_rest(robot, fps=fps)

    policy, preprocessor, postprocessor = _load_policy(str(ckpt_path), dev)

    detector = None
    state_holder = None
    if detection:
        detector = _init_detector(device=str(dev))
        state_holder = DetectionStateHolder()
        print(f"Detection: ENABLED — overlay + state-aug (hold-last-good) active")

    runner = AsyncChunkRunner(
        policy,
        postprocessor,
        execution_horizon=action_horizon,
        enable_rtc=enable_rtc,
    )
    runner.start()
    print(f"Async chunk runner: ah={action_horizon}, RTC={'on' if enable_rtc else 'off'}")

    # ── Depth bake closure ────────────────────────────────────────────────────
    # When depth is enabled, every capture site needs the same post-step:
    # pull aligned uint16 depth → run the bake → splice into the frames dict
    # under the virtual ``gripper_depth`` key. Wrapped here once so the three
    # capture call sites stay readable.
    def _capture(cam_devices):
        frames = _capture_frames(cam_devices)
        if depth:
            depths = get_latest_depth(cam_devices)
            d = depths.get("gripper")
            if d is not None:
                frames = dict(frames)
                # Use the device-reported scale so we don't rely on the
                # 1e-4 hardcode if a different D405 firmware ever ships
                # a different default. Falls back to the bake's default.
                scale = cam_devices["gripper"].get("depth_scale", None)
                if scale is None or scale <= 0:
                    frames["gripper_depth"] = depth_uint16_to_turbo_rgb(d)
                else:
                    frames["gripper_depth"] = depth_uint16_to_turbo_rgb(
                        d, depth_scale_m=float(scale)
                    )
        return frames

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print(f"\nEval session: {session_id}")
    print(f"Checkpoint:   {ckpt_label}")
    print(f"Protocol:     {protocol}  ({total} trials)")
    print(f"Starting at trial {start_trial + 1}")
    print("="*55)

    step_dt = 1.0 / fps

    try:
        for trial_idx in range(start_trial, total):
            trial     = trials[trial_idx]
            trial_num = trial_idx + 1
            is_entity = _is_entities_trial(trial)

            # Resolve per-trial task with precedence:
            #   1. --task explicit override (applies to every trial)
            #   2. trial.task           (entity protocols)
            #   3. proto.task           (legacy protocols' top-level task)
            #   4. DEFAULT_TASK
            if task is not None:
                trial_task = task
            else:
                trial_task = (
                    trial.get("task")
                    or proto_task
                    or DEFAULT_TASK
                )

            # ── Build hud strings + colours per schema ─────────────────────
            if is_entity:
                tags        = trial.get("tags", {})
                scene       = tags.get("scene", "")
                target_color = tags.get("target_color", "?")
                # Header colour by scene type
                zone_col = (
                    (60, 200, 60)  if scene.startswith("single") else
                    (50, 120, 255) if scene == "both" else
                    (200, 200, 200)
                )
                hud_top = (
                    f"Trial {trial_num}/{total}  [{scene}]  "
                    f"target={target_color.upper()}"
                )
                hud_bottom_inference = f'"{trial_task}"'
                print(f"\nTrial {trial_num}/{total}  [{scene}]  "
                      f"target={target_color}  task={trial_task!r}")
            else:
                zone_col = (0, 220, 80) if trial["zone"] == "ID" else (50, 120, 255)
                hud_top = (
                    f"Trial {trial_num}/{total}  [{trial['zone']}]  "
                    f"duck:{trial['duck_px']}  dir:{trial['duck_dir_deg']:.0f}deg  "
                    f"cup-grp:{trial['cup_group']}"
                )
                hud_bottom_inference = (
                    "S = success    F = failure    (stops inference)"
                )
                print(f"\nTrial {trial_num}/{total}  [{trial['zone']}]  "
                      f"duck:{trial['duck_px']}  cup:{trial['cup_px']}  "
                      f"dir:{trial['duck_dir_deg']:.0f}deg")
            print("  Place objects then press SPACE in window to start.")

            # ── Guidance phase — 4 cameras, markers on top cam, wait SPACE ──
            while True:
                frames = _capture(cam_devices)
                top_key = "top" if "top" in frames else "top_cam" if "top_cam" in frames else None
                if top_key is not None:
                    frames = dict(frames)
                    if is_entity:
                        frames[top_key] = _mark_top_camera_entities(
                            frames[top_key], trial,
                            nogo_bbox=nogo_bbox,
                            workspace_bbox=workspace_bbox,
                        )
                    else:
                        frames[top_key] = _mark_top_camera(
                            frames[top_key], trial, cup_positions)
                grid = _build_grid_frame(frames, cam_names, 0, None,
                                         right_column=(["gripper_depth"] if depth else None),
                                         gato=gato)
                _grid_hud(grid,
                          top=hud_top, top_col=zone_col,
                          bottom="Place objects as shown  →  SPACE to start inference   Q to quit")
                cv2.imshow(WINDOW, grid)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    break
                if key in (ord('q'), 27):
                    print("Session quit by user.")
                    raise KeyboardInterrupt

            # ── Inference phase ───────────────────────────────────────────────
            runner.reset()   # clear chunk state from previous trial
            if state_holder is not None:
                state_holder.reset()  # mirror per-episode reset of dataset's apply_confidence_hold
            recorded_frames = []
            step            = 0
            result          = None

            try:
                while step < max_steps and result is None:
                    t0 = time.perf_counter()

                    obs_state = np.array([
                        robot.get_observation()[f"{n}.pos"] for n in JOINT_NAMES
                    ])
                    frames = _capture(cam_devices)

                    det_results = None
                    if detector is not None:
                        det_results = _run_detection_overlay(detector, frames, cam_names)

                    grid = _build_grid_frame(frames, cam_names, step, None,
                                             right_column=(["gripper_depth"] if depth else None),
                                             gato=gato)
                    _grid_hud(grid,
                              top=f"Inference  step {step}/{max_steps}",
                              top_col=(200,200,200),
                              bottom=hud_bottom_inference)
                    _progress_bar(grid, step, max_steps)
                    cv2.imshow(WINDOW, grid)
                    recorded_frames.append(grid.copy())

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        result = "success"
                        break
                    elif key == ord('f'):
                        result = "failure"
                        break
                    elif key in (ord('q'), 27):
                        print("  Quit during inference.")
                        raise KeyboardInterrupt

                    state_aug = (state_holder.update_and_vector(det_results)
                                 if state_holder is not None else None)
                    obs    = _build_observation(obs_state, frames, cam_names,
                                                trial_task, preprocessor, dev,
                                                state_aug=state_aug)
                    action = runner.step(obs)

                    if delta_actions:
                        target = obs_state + action
                        target[5] = action[5]  # gripper absolute
                        action = target

                    robot.send_action({f"{n}.pos": float(action[j])
                                       for j, n in enumerate(JOINT_NAMES)})
                    step += 1

                    elapsed = time.perf_counter() - t0
                    if elapsed < step_dt:
                        time.sleep(step_dt - elapsed)

                if result is None:
                    result = "failure"
                    print(f"  Timeout at {max_steps} steps → failure")

            except KeyboardInterrupt:
                raise

            print(f"  Result: {result.upper()}  ({step} steps)")

            # ── Result screen — V to save video, SPACE for next ───────────────
            res_col    = (0,220,80) if result == "success" else (50,80,255)
            res_label  = "SUCCESS" if result == "success" else "FAILURE"
            video_path = None
            note       = ""

            # Auto-record: save video immediately if record=True
            if record and recorded_frames:
                fname      = f"trial_{trial['trial_id']:02d}_{result}.mp4"
                video_path = str(session.dir / "videos" / fname)
                _save_video_ffmpeg(recorded_frames, Path(video_path), fps)
                print(f"  Video auto-saved: {fname}")
            last_grid  = recorded_frames[-1] if recorded_frames else \
                          _build_grid_frame(_capture(cam_devices), cam_names, step, None,
                                            right_column=(["gripper_depth"] if depth else None),
                                            gato=gato)
            while True:
                display = last_grid.copy()
                note_indicator = f"  [note: {note[:40]}]" if note else ""
                _grid_hud(display,
                          top=f"{res_label}  ({step} steps){note_indicator}",
                          top_col=res_col,
                          bottom="V=video   N=add note   SPACE=next trial   Q=quit")
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('v') and recorded_frames and video_path is None:
                    fname      = f"trial_{trial['trial_id']:02d}_{result}.mp4"
                    video_path = str(session.dir / "videos" / fname)
                    _save_video_ffmpeg(recorded_frames, Path(video_path), fps)
                    print(f"  Video saved: {fname}")
                elif key == ord('n'):
                    typed = _text_input(WINDOW, last_grid,
                                        f"Note for trial {trial_num}/{total}:")
                    if typed:
                        note = typed
                elif key == ord(' '):
                    break
                elif key in (ord('q'), 27):
                    raise KeyboardInterrupt

            session.record(trial, result, step, video=video_path, note=note)

            # ── Return to rest ────────────────────────────────────────────────
            move_to_rest(robot, fps=fps)

    except KeyboardInterrupt:
        print("\nSession ended early.")

    finally:
        runner.stop()
        _stop_cameras(cam_devices)
        try:
            move_to_rest(robot, fps=fps)
        except Exception:
            pass
        robot.disconnect()
        cv2.destroyAllWindows()

    session.write_markdown()
    session.print_summary()


if __name__ == "__main__":
    import fire
    fire.Fire(run)
