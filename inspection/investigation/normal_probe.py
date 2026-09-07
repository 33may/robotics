#!/usr/bin/env python3
"""Can ONE crop box name the viewsphere cell where that feature reads best?

The subagent already found the feature and drew a box round it. Today it only
reports a DIRECTION ("turning away to the left") with no magnitude, so the
orchestrator single-steps: run 2508-aiduck2 burned 8 views on an answer that
needed about 2. This probe asks whether the box alone is enough to name the
cell: back-project the crop through THAT view's own aligned depth, anchor a
radius-swept PCA patch on it, and score every cell by how squarely its
viewing ray opposes the local surface normal.

What it may look at, per case, is exactly `(run, cap, box_up)` plus that
capture's own files — never the visit order, never `store["findings"]`, never
a later capture (trap T3). The one exception is the viewsphere CENTRE, which
is averaged over the run's cell poses because a single pose fixes it to
15.6 mm anyway (measured below); the source-view-only variant is computed and
reported alongside as `predicted_cell_source_centre`.

Hazards this file is built around, all previously measured:
  * The FUSED cloud sits ~4.2 cm from what a single view's own aligned depth
    says, so `fused_cloud.npy` is never a probe input — only a diagnostic.
  * `rgb.png`/`depth_aligned.npy` are stored UPRIGHT since 2026-08-24
    (perception/capture.py:55); `depth_raw`, the IR pair and `T_base_cam`
    stay RAW. Crop boxes in transcripts are in the UPRIGHT frame the VLM saw
    (eyes/tools.py:127 crops `img.rgb`, which went through `upright()`).
    Two INDEPENDENT flags govern this and must not be conflated:
    `is_half_turn(T)` governs the BOX, `meta["rgb_rotation_deg"]` governs the
    ARRAYS. They disagree on the legacy runs (2408-cup1: half=True,
    disk_up=False).
  * Intrinsics are `session["intrinsics"]["color"]` because the array is
    `depth_aligned`. `ir_left` is 5-8 px off in x — the whole hand-eye budget.
  * Depth scale is 1e-4 m/unit (session["depth_scale_m_per_unit"]), not 1e-3.
  * The D405 is passive stereo with no projector: dark, glossy or textureless
    surfaces return holes exactly where a printed logo lives, so depth
    legibility is a PER-VIEW property and is reported per case, not assumed.

DATA CONTAMINATION (verified 2026-08-26). Every recorded run ran with the
vision verbs on a STUB backend: 195 `detect` calls all returned the same
hardcoded (10,10,100,100)/score 0.9/label "stub", 22 `read_text` calls
returned nothing, the 1 `segment` call filled its prompt box. 184 turn
results contain the word "stub". `crop` (139 calls) needs no model backend,
so those boxes are genuine VLM-chosen regions and are the ONLY tool record
this probe reads. `<cap>/mask.png` is real (SAM 3, a different code path) and
is used as a validity filter only, never as a probe input.

READ-ONLY. Nothing under data/runs/ is ever opened for writing. The old
`inspection.eyes.replay.load_run()` was FORBIDDEN here for exactly this
reason — it went through `FactWriter.add_view -> RunStore._flush()` and
rewrote `eyes/store.json` in place; that module is deleted (task-5,
2026-09-07), and its reading role moved to `inspection.record.run.Run`,
which is genuinely read-only (own docstring: "never writes a byte into the
run"). This probe still avoids it and reads with plain `json.loads`
throughout, to keep the guarantee visible at every call site.

    p inspection/investigation/normal_probe.py
    p inspection/investigation/normal_probe.py --out /tmp/normal_probe
"""
import argparse
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from inspection.cell.geometry import (BOX_PCT, MAX_RANGE_M, MIN_RANGE_M,
                                      above_table, cam_to_base, deproject,
                                      is_half_turn, object_in_base, rotate180)
from inspection.eyes.models import GeminiVlm
from inspection.view.grid import moves_from, step_delta
from inspection.view.viewsphere import ViewSphere

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"
FRAME_W, FRAME_H = 848, 480

# ---- pre-registered configuration (section 5 "config") ---------------------
RHO_SWEEP = (0.006, 0.008, 0.012, 0.016, 0.020)
MIN_PATCH_POINTS = 200          # -> ABSTAIN (gate_thin)
OFFOBJECT_R95_MULT = 1.5        # -> ABSTAIN (gate_offobject)
MIN_DEPTH_VALID_FRAC = 0.30     # -> FLAG only
MAX_PLANE_RMS_M = 0.005         # -> FLAG only
MAX_FAN_WIDTH_DEG = 15.0        # -> FLAG only
MIN_PCA_POINTS = 20             # a radius with fewer points is not accepted
FEATURE_CROP_MAX_AREA = 12000   # admission (iii): a feature crop, not the object
RANDOM_SEED = 0
B1_MAPPING = {"right": [1, 0], "left": [-1, 0], "up": [0, 1], "down": [0, -1],
              "closer": [0, 0], "none": [0, 0]}
B1_CALIBRATED_N = 2             # the right->h+1 mapping is calibrated on 2508-aiduck2 only

#: The FROZEN corpus. Every corpus-level count in this file (139 crops, 195
#: detect, 22 read_text, 1 segment, 184 stub results, 2 box-convention
#: mismatches) was audited against exactly these ten runs on 2026-08-26. The
#: list is pinned rather than globbed because data/runs/ is a LIVE directory:
#: `2608-aicam` appeared in it at 14:36 while this file was being written and
#: silently pushed the crop count to 146 and the cell round-trip to 68/68.
#: A corpus that changes under the probe cannot be re-audited, and a run that
#: is still being written is not an artifact.
CORPUS_RUNS = ("2408-cup1", "2408-cup2", "2408-seeded", "2508-ai1", "2508-ai2",
               "2508-ai3", "2508-ai4", "2508-ai5", "2508-aiduck", "2508-aiduck2")

OBJECTS = {
    "2408-cup1": ("ceramic mug", "VBTI logo", "VBTI"),
    "2508-ai1": ("black mug", "VBTI logo", "VBTI"),
    "2508-ai2": ("black mug", "VBTI logo", "VBTI"),
    "2508-ai3": ("black mug", "VBTI logo", "VBTI"),
    "2508-ai4": ("black mug", "VBTI logo", "VBTI"),
    "2508-ai5": ("black mug", "VBTI logo", "VBTI"),
    "2508-aiduck": ("blue rubber duck", "WeAreDevelopers wordmark", "WeAreDevelopers"),
    "2508-aiduck2": ("blue rubber duck", "WeAreDevelopers wordmark", "WeAreDevelopers"),
}

# ---- THE TEST CASES --------------------------------------------------------
# Boxes are the EXECUTED pixel boxes (x0,y0,x1,y1) in the UPRIGHT frame the VLM
# saw, parsed from the result string of the turn AFTER the crop call. Every one
# was re-verified against the transcripts before this table was frozen; the
# script re-verifies each of them again at run time and aborts the case on a
# mismatch. `turn` indexes the crop CALL; the result lives at turn+1.
#
# Tier A (homing) admission, pre-registered, all three had to hold:
#  (i) the source view's prose read of the feature is PARTIAL; (ii) some OTHER
#  captured cell in the same run produced a FULL correct read of the same
#  feature; (iii) the executed box area is < 12 000 px2 (a feature crop, not
#  the whole object).
CASES = [
    # id    tier run             cell     cap    transcript turn box                        labels                strength  gap
    ("A01", "A", "2508-ai3",     (2, 0),  "001", "f001",  0, (466, 253, 505, 299), [(4, 0)],          "STRONG", 2),
    ("A02", "A", "2508-ai3",     (3, 0),  "002", "f003",  0, (433, 226, 506, 306), [(4, 0)],          "weak",   1),
    ("A03", "A", "2508-ai4",     (9, 0),  "003", "f004",  6, (342, 187, 410, 308), [(8, 0)],          "weak",   1),
    ("A04", "A", "2508-ai4",     (8, 1),  "004", "f005",  0, (357, 272, 431, 319), [(8, 0)],          "weak",   1),
    ("A05", "A", "2508-ai4",     (7, 1),  "005", "f006",  0, (375, 260, 451, 329), [(8, 0)],          "STRONG", 2),
    ("A06", "A", "2508-aiduck",  (9, 1),  "003", "f003",  4, (333, 286, 383, 343), [(7, 0), (8, 0)],  "STRONG", 2),
    ("A07", "A", "2508-aiduck",  (9, 0),  "004", "f005",  0, (342, 259, 393, 310), [(7, 0), (8, 0)],  "STRONG", 1),
    ("A08", "A", "2508-aiduck",  (10, 0), "005", "f006",  0, (343, 240, 372, 286), [(7, 0), (8, 0)],  "STRONG", 2),
    ("A09", "A", "2508-aiduck2", (8, 0),  "006", "f006",  0, (424, 230, 500, 288), [(9, 0)],          "STRONG", 1),
    ("A10", "A", "2408-cup1",    (8, 0),  "003", "f022",  6, (416, 230, 483, 336), [(9, 0)],          "weak",   1),
    # Tier B — "already there" control. Correct answer is STAY PUT. Scored
    # separately and never mixed into the Tier A numbers: a probe that always
    # jumps is broken too, and this control costs nothing.
    ("B01", "B", "2508-ai3",     (4, 0),  "003", "f005",  0, (381, 234, 469, 308), [(4, 0)],          None, 0),
    ("B02", "B", "2508-ai4",     (8, 0),  "002", "f002",  0, (356, 240, 458, 326), [(8, 0)],          None, 0),
    ("B03", "B", "2508-ai5",     (9, 0),  "002", "f002",  0, (332, 261, 409, 326), [(9, 0)],          None, 0),
    ("B04", "B", "2508-ai2",     (3, 0),  "002", "f002",  0, (441, 245, 505, 308), [(3, 0)],          None, 0),
    ("B05", "B", "2508-aiduck",  (8, 0),  "006", "f007",  0, (356, 259, 437, 306), [(8, 0), (7, 0)],  None, 0),
    ("B06", "B", "2508-aiduck",  (7, 0),  "007", "f008",  0, (398, 262, 475, 302), [(7, 0), (8, 0)],  None, 0),
    ("B07", "B", "2508-aiduck2", (9, 0),  "007", "f007",  0, (411, 247, 481, 292), [(9, 0)],          None, 0),
    ("B08", "B", "2408-cup1",    (9, 0),  "002", "f023",  0, (386, 278, 479, 350), [(9, 0)],          None, 0),
    # Tier C — RUN BUT DO NOT SCORE. Declared exclusions, printed with their
    # predictions so nothing is hidden.
    ("C01", "C", "2508-aiduck2", (7, 0),  "005", "f005", 24, (407, 211, 509, 269), [],                None, 0),
    ("C02", "C", "2508-aiduck2", (1, 1),  "003", "f003",  0, (339, 216, 479, 331), [],                None, 0),
    ("C03", "C", "2508-ai1",     (2, 0),  "001", "f002",  2, (449, 254, 485, 312), [],                None, 0),
]
#: Pre-registered ALTERNATIVE label cells, declared in the frozen case table
#: before this file was run and deliberately NOT scored. Recorded so a figure
#: can show them; the headline counts only `label_cells`.
LABEL_ALT = {"A01": [(3, 0)]}
TIER_C_REASON = {
    "C01": "crop contains no logo (verified visually); the VLM claimed one — "
           "upstream failure, not a probe failure",
    "C02": "the VLM cropped the tape measure's belt clip, not the duck — the "
           "off-object gate MUST reject this",
    "C03": "textbook clipped-'va' source but the run captured ONE cell — "
           "prediction possible, no held-out image",
}
EXCLUDED = [
    {"run": "2508-ai2", "cell": [2, 0],
     "reason": "source crop is the whole object (26100 px2), not a feature crop"},
    {"run": "2508-ai5", "cell": [10, 0],
     "reason": "source crop is the whole object (31842 px2), not a feature crop"},
    {"run": "2408-cup2", "cell": None,
     "reason": "feature never found from any captured cell — a coverage "
               "question, not a homing question"},
    {"run": "2408-seeded", "cell": None,
     "reason": "frames tilted +-90 uncorrected; f001 boxes predate to_pixel_box"},
    {"run": "2108-*", "cell": None, "reason": "no feature-crop episodes"},
    {"run": "2408-cup3/4/5", "cell": None, "reason": "no feature-crop episodes"},
    {"run": "2408-geomtest", "cell": None, "reason": "no feature-crop episodes"},
    {"run": "*", "cell": None,
     "reason": "detect (195), read_text (22) and segment (1) tool output "
               "discarded corpus-wide: stub backend, zero information. crop "
               "(139) is the only trustworthy tool record."},
]

# ---- prose read classifier -------------------------------------------------
# The MACHINE label is prose-derived FULL/PARTIAL/NONE: auditable, corpus-wide
# and independent of depth. Metric crop width was dropped — it inverts on the
# flagship case. The human eye remains the verdict (Figure 1).
#
# HEDGE is deliberately small. "clipped" and "cut off" were tried and removed:
# the Tier-B transcripts say "fully visible, NOT clipped" and a substring test
# cannot see the negation (ai4/f002, ai5/f002, ai2/f002 all mis-classified).
HEDGE = ("not face-on", "closer to the handle", "blurry", "illegible",
         "lacks any legible", "not legible", "pixelat", "garbled",
         "hard to read", "difficult to read", "obliqu")
NEGATION = ("no logo", "no logos", "no text", "no visible logo", "not visible",
            "no printed", "no printing", "no readable text", "none visible",
            "no brand", "completely plain", "no signs of", "no visible text",
            "no marks", "no other printed")


def final_text(transcript):
    """The subagent's own words: evidence + reasoning + answer, last block."""
    for turn in reversed(transcript.get("turns", [])):
        if "answer" in turn:
            ev = " ".join(turn.get("evidence") or [])
            return f"{ev} {turn.get('reasoning', '')} {turn.get('answer', '')}"
    return ""


def read_class(text, wordmark):
    """FULL / PARTIAL / NONE from prose alone. Case-SENSITIVE on the wordmark:
    ai4/f005 read 'VBTi', which is a garbled read of 'VBTI' and must not count
    as a full one."""
    if not text.strip():
        return "NONE"
    low = text.lower()
    if wordmark in text:
        return "PARTIAL" if any(h in low for h in HEDGE) else "FULL"
    if any(n in low for n in NEGATION):
        return "NONE"
    if low.strip() in ("unknown", "none"):
        return "NONE"
    return "PARTIAL"


# ---- read-only loading -----------------------------------------------------
def load_store(run):
    return json.loads((RUNS / run / "eyes" / "store.json").read_text())


def load_session(run):
    return json.loads((RUNS / run / "session.json").read_text())


def load_meta(run, cap):
    return json.loads((RUNS / run / cap / "meta.json").read_text())


def load_transcript(run, stem):
    return json.loads((RUNS / run / "eyes" / "transcripts" / f"{stem}.json").read_text())


def executed_box(transcript, turn_index):
    """The box the crop tool ACTUALLY executed, parsed from the result string.

    NOT `turns[i]["args"]["box"]`: that is Gemini ER-2's [ymin,xmin,ymax,xmax]
    normalised 0-1000 (eyes/models.py:76 to_pixel_box). Reading those four
    numbers as pixels puts 111 of 139 boxes outside the frame and still
    returns a picture — trap T1.
    """
    turns = transcript["turns"]
    if turn_index + 1 >= len(turns):
        return None
    m = re.search(r"crop (\d+),(\d+)-(\d+),(\d+)",
                  str(turns[turn_index + 1].get("result", "")))
    return tuple(int(g) for g in m.groups()) if m else None


def cap_of_transcript(run, transcript, store):
    """Which capture dir a transcript looked at. `image` is authoritative when
    present (eyes/inspect_agent.py:250 writes eyes/frames/<cap_dir>.png); the
    legacy 2408 runs predate that key, so fall back to cell -> store view."""
    img = transcript.get("image") or ""
    m = re.match(r"eyes/frames/(\d+)\.png", img) or re.match(r"^(\d{3})/", img)
    if m:
        return m.group(1)
    cell = transcript.get("cell")
    if cell is None:
        return None
    for v in store["views"]:
        if v["cell"] is not None and tuple(v["cell"]) == tuple(cell):
            return v["cap_dir"]
    return None


# ---- geometry --------------------------------------------------------------
class Scene:
    """One capture, in the RAW sensor grid — everything the probe may see.

    `depth_aligned` and `mask.png` are rotated back to RAW when the writer
    stored them upright, because `ppx/ppy` are raw-sensor quantities and
    `T_base_cam` is a raw-frame pose. Rotate the ARRAY and the BOX; never
    rotate the intrinsics.
    """

    def __init__(self, run, cap, disk_up_override=None):
        session = load_session(run)
        meta = load_meta(run, cap)
        self.run, self.cap, self.meta = run, cap, meta
        self.T = np.asarray(meta["T_base_cam"], float)
        self.half = bool(is_half_turn(self.T))          # governs the BOX
        stored = bool(meta.get("rgb_rotation_deg"))     # governs the ARRAYS
        self.disk_up = stored if disk_up_override is None else disk_up_override
        self.intr = session["intrinsics"]["color"]
        self.scale = float(session["depth_scale_m_per_unit"])

        depth = np.load(RUNS / run / cap / "depth_aligned.npy")
        self.depth = rotate180(depth) if self.disk_up else depth
        self.H, self.W = self.depth.shape
        mp = RUNS / run / cap / "mask.png"
        self.mask = None
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            self.mask = rotate180(m) if self.disk_up else m
        self.mask_score = (meta.get("mask") or {}).get("score")
        self.mask_px = int(self.mask.sum()) if self.mask is not None else None

        # The view's OWN object cloud. The table plane is fitted on the FULL
        # view (cell/geometry.py:376) because the plane needs the table and the
        # mask deliberately removes it.
        full = object_in_base(self.depth, self.intr, self.scale, self.T,
                              mask=self.mask)
        self.plane = full["plane"]
        self.obj = full["points"]
        self.centroid = full["centroid"]
        # WHICH rule produced the reference centroid the off-object gate is
        # measured against. Without a mask `object_in_base` bootstraps to the
        # largest above-table cluster (cell/geometry.py:390) — the rule the
        # module's own docstring blames for run 2408-geomtest, where a cable
        # out-voted a black mug 568 to 392. Recorded so a gate trip can be
        # read as "the anchor is off the object" or "the reference is".
        self.object_source = ("mask" if self.mask is not None
                              else "largest_cluster_bootstrap")
        self.r95 = (float(np.percentile(
            np.linalg.norm(self.obj - self.centroid, axis=1), 95))
            if self.centroid is not None and len(self.obj) else None)

    def box_raw(self, box_up, half=None):
        """UPRIGHT crop box -> RAW sensor box. Same map as run/segmenter.py:98."""
        half = self.half if half is None else half
        x0, y0, x1, y1 = box_up
        if half:
            x0, y0, x1, y1 = self.W - x1, self.H - y1, self.W - x0, self.H - y0
        return (max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1))

    def patch(self, box_up, half=None, use_mask=True):
        """The crop's 3D points, base frame, table dropped.

        stride=1 on purpose. `object_in_base` deprojects at the module default
        stride=2 (cell/geometry.py:123), which quarters a 29x46 crop (A08) —
        1334 px becomes ~330. The patch is built directly instead.
        """
        bx = self.box_raw(box_up, half)
        bm = np.zeros((self.H, self.W), bool)
        bm[bx[1]:bx[3], bx[0]:bx[2]] = True
        if use_mask and self.mask is not None:
            bm &= self.mask
        # n_box is counted AFTER the mask gate: it is the denominator of
        # depth_valid_frac, and a hole in the object is what that fraction is
        # about, not the background fringe the mask already removed.
        n_box = int(bm.sum())
        if n_box == 0:
            return bx, 0, np.empty((0, 3)), np.zeros((self.H, self.W), bool)
        pts_cam = deproject(np.where(bm, self.depth, 0), self.intr, self.scale,
                            stride=1)
        pts = above_table(cam_to_base(pts_cam, self.T), self.plane)
        return bx, n_box, pts, self._valid_map(bm)

    def _valid_map(self, bm):
        """Per-pixel: did this pixel contribute a point? Same three tests
        `deproject` + `above_table` apply, kept per-pixel for the overlay."""
        z = self.depth.astype(np.float64) * self.scale
        ok = bm & (z > MIN_RANGE_M) & (z < MAX_RANGE_M)
        vs, us = np.nonzero(ok)
        if not len(us):
            return ok
        zz = z[vs, us]
        x = (us - self.intr["ppx"]) / self.intr["fx"] * zz
        y = (vs - self.intr["ppy"]) / self.intr["fy"] * zz
        p = cam_to_base(np.column_stack([x, y, zz]), self.T)
        keep = (p @ self.plane[:3] + self.plane[3]) > 0.008   # ABOVE_TABLE_M
        out = np.zeros_like(ok)
        out[vs[keep], us[keep]] = True
        return out


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def normal_sweep(pts, anchor, cam_pos):
    """Anchored, radius-swept PCA. Returns (per-radius dict, headline, fan).

    Anchored rather than a whole-box plane fit: whole-box fits give 12-25 mm
    off-plane rms on these crops, the anchored patch gives 0.2-3.9 mm at
    rho = 6-20 mm. NOT open3d's estimate_normals with KDTreeSearchParamHybrid
    — max_nn=30 silently shrinks a 20 mm ball (~1800 pts here) to ~4 mm. Direct
    SVD is four lines and has no such trap.

    The sweep is a CURVATURE-BIAS study, not a noise study: the 3-6 mm scene
    pose error is a rigid translation and does not tilt a normal at all.
    """
    d = np.linalg.norm(pts - anchor, axis=1)
    per = {}
    for rho in RHO_SWEEP:
        P = pts[d < rho]
        if len(P) < MIN_PCA_POINTS:
            per[rho] = None
            continue
        Q = P - P.mean(0)
        _, S, Vt = np.linalg.svd(Q, full_matrices=False)
        n = Vt[2]
        if n @ (cam_pos - P.mean(0)) < 0:      # orient toward THIS camera
            n = -n
        per[rho] = {"n": n, "k": int(len(P)),
                    "rms": float(np.sqrt(((Q @ n) ** 2).mean())),
                    "planarity": float(S[2] / S[0]) if S[0] > 0 else 0.0,
                    "mean": P.mean(0)}
    acc = [v["n"] for v in per.values() if v is not None]
    head = unit(np.sum(acc, axis=0)) if acc else None   # no radius cherry-picked
    fan = 0.0
    for i in range(len(acc)):
        for j in range(i + 1, len(acc)):
            fan = max(fan, math.degrees(
                math.acos(float(np.clip(acc[i] @ acc[j], -1, 1)))))
    return per, head, fan


def rms_against(pts, anchor, normal, rho):
    P = pts[np.linalg.norm(pts - anchor, axis=1) < rho]
    if len(P) < MIN_PCA_POINTS:
        return None
    return float(np.sqrt((((P - P.mean(0)) @ normal) ** 2).mean()))


def sphere_centre(store, views=None):
    """Viewsphere centre from the recorded poses: camera position pushed r
    along its own CV boresight (+Z, cell/geometry.py:32).

    Verified before this file was written: this recovers 32/32 recorded cells
    across the 7 multi-cell runs, max angular residual 3.7 deg, per-run centre
    spread <= 15.6 mm. The fused-cloud centroid is NOT used — it drifts ~4.2 cm.
    """
    r = store["grid"]["r"]
    vs = views if views is not None else [v for v in store["views"]
                                          if v["cell"] is not None]
    C = np.array([np.asarray(v["T_base_cam"], float)[:3, 3]
                  + r * np.asarray(v["T_base_cam"], float)[:3, 2] for v in vs])
    return C.mean(0), float(np.linalg.norm(C - C.mean(0), axis=1).max() * 1000)


def rank_cells(vs, centre, r, anchor, normal):
    """Score every cell by how squarely it faces the FEATURE.

    The ray is taken from the ANCHOR, not from the sphere centre: the feature
    sits 40-60 mm off centre, which is up to 12 deg at r=0.24 — nearly half a
    cell. `predicted_cell_centre_ray` reports the naive variant for sensitivity.
    """
    out = []
    for h, v in vs.cells():
        ray = unit((centre + r * vs.cell_dir(h, v)) - anchor)
        out.append(((h, v), float(ray @ normal)))
    out.sort(key=lambda x: -x[1])
    return out


def cells_err(pred, labels):
    """Project's own metric: |signed azimuth steps| + |elevation steps|,
    minimum over the label set (view/grid.py:90 step_delta, h wraps)."""
    best, which = None, None
    for lab in labels:
        e = abs(step_delta(pred[0], lab[0])) + abs(pred[1] - lab[1])
        if best is None or e < best:
            best, which = e, lab
    return best, which


def deg_between(vs, centre, r, anchor, a, b):
    ra = unit((centre + r * vs.cell_dir(*a)) - anchor)
    rb = unit((centre + r * vs.cell_dir(*b)) - anchor)
    return float(math.degrees(math.acos(float(np.clip(ra @ rb, -1, 1)))))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - hw), 4), round(min(1.0, c + hw), 4)]


def circular_spread(degs):
    """Width of the smallest arc containing every azimuth."""
    if len(degs) < 2:
        return 0.0
    a = sorted(d % 360 for d in degs)
    gaps = [(a[(i + 1) % len(a)] - a[i]) % 360 for i in range(len(a))]
    return float(360.0 - max(gaps))


# ---- reachability (supporting field only) ----------------------------------
def reachable_cells(run, store, centre):
    """Which cells the arm can actually reach with the object as an obstacle.

    Supporting only — `predicted_reachable` never enters the headline. The
    object is a HARD obstacle (view/viewsphere.py:13), sized from the run's
    fused cloud with the same percentile trim the collision box uses
    (cell/geometry.py BOX_PCT). Pure CPU: no robot, no network.
    """
    try:
        from inspection.cell.world import RobotCell
        from inspection.motion.ik import UR5eIK
        fc = RUNS / run / "fused_cloud.npy"
        world, ik = RobotCell(), UR5eIK()
        if fc.exists():
            p = np.load(fc)
            lo = np.percentile(p, BOX_PCT, axis=0)
            hi = np.percentile(p, 100 - BOX_PCT, axis=0)
            world.set_object("object", (hi - lo).tolist(),
                             [*((lo + hi) / 2), 0, 0, 0], parent="base")
        vs = ViewSphere(centre, r=store["grid"]["r"],
                        elevations=tuple(store["grid"]["v_elevs"]))
        reach = vs.reachability(world, ik)
        return {c for c, roll in reach.items() if roll is not None}
    except Exception as e:                                    # noqa: BLE001
        print(f"  ! reachability unavailable for {run}: {e}")
        return None


# ---- corpus audit ----------------------------------------------------------
def all_crops():
    """Every crop call in the corpus, with its capture. 139 of them."""
    out = []
    for run_dir in sorted(RUNS / r for r in CORPUS_RUNS):
        td = run_dir / "eyes" / "transcripts"
        if not td.is_dir():
            continue
        store = load_store(run_dir.name)
        for f in sorted(td.glob("f*.json")):
            t = json.loads(f.read_text())
            cap = cap_of_transcript(run_dir.name, t, store)
            for i, turn in enumerate(t["turns"]):
                if turn.get("tool") != "crop":
                    continue
                out.append({"run": run_dir.name, "stem": f.stem, "turn": i,
                            "cap": cap, "args_box": turn.get("args", {}).get("box"),
                            "box": executed_box(t, i)})
    return out


def audit_tool_calls():
    """Count what the recorded subagents actually called, and how many turn
    results came back from the STUB backend. Measured here rather than quoted:
    every detect/read_text/segment result in this corpus is stub output and
    must be reported as discarded, so the count has to be a fact on disk."""
    counts, stub = {}, 0
    for run in CORPUS_RUNS:
        td = RUNS / run / "eyes" / "transcripts"
        if not td.is_dir():
            continue
        for f in sorted(td.glob("f*.json")):
            for turn in json.loads(f.read_text())["turns"]:
                if "tool" in turn:
                    counts[turn["tool"]] = counts.get(turn["tool"], 0) + 1
                if "stub" in str(turn.get("result", "")).lower():
                    stub += 1
    counts["turn_results_saying_stub"] = stub
    return counts


def audit_box_convention(crops):
    """T1. `to_pixel_box(args.box)` must reproduce the result-string box in
    137/139; the 2 mismatches are 2408-seeded/f001 turns 8 and 10, recorded
    before to_pixel_box existed and executed as raw pixels."""
    bad = []
    for c in crops:
        if c["box"] is None or c["args_box"] is None:
            bad.append(f"{c['run']}/{c['stem']}#t{c['turn']:02d}")
            continue
        dec = GeminiVlm.to_pixel_box(c["args_box"], FRAME_W, FRAME_H)
        if tuple(dec) != tuple(c["box"]):
            bad.append(f"{c['run']}/{c['stem']}#t{c['turn']:02d}")
    return bad


def audit_orientation_ab(crops, scenes):
    """T2a. The whole corpus, run twice, with `half` inverted the second time.

    Per-view checks cannot settle orientation (Scout 4's F5): the camera is
    aimed AT the object, so a box near the frame centre maps to another box
    near the frame centre under a half turn and both land on the object. This
    is the corpus-level version of the same question.
    """
    stats = {}
    for name, flip in (("correct", False), ("inverted", True)):
        off = tot = 0
        disp = []
        for c in crops:
            if c["box"] is None or c["cap"] is None:
                continue
            s = scenes.get((c["run"], c["cap"]))
            if s is None or s.centroid is None:
                continue
            tot += 1
            half = (not s.half) if flip else s.half
            _, _, pts, _ = s.patch(c["box"], half=half)
            _, _, pts0, _ = s.patch(c["box"])
            if len(pts) < MIN_PCA_POINTS:
                off += 1
                continue
            a = np.median(pts, axis=0)
            if np.linalg.norm(a - s.centroid) > OFFOBJECT_R95_MULT * s.r95:
                off += 1
            if len(pts0) >= MIN_PCA_POINTS:
                disp.append(float(np.linalg.norm(a - np.median(pts0, axis=0)) * 1000))
        stats[name] = {"off": off, "tot": tot, "frac": off / max(tot, 1),
                       "median_disp_mm": float(np.median(disp)) if disp else 0.0}
    return stats


def audit_survey_anchor(scenes):
    """T2b. Same A/B on captures whose object is OFF-CENTRE (the survey pose,
    cap 000). With no crop box on a survey view, the SAM 3 prompt box from
    meta["mask"]["box"] (RAW frame, perception/capture.py:100) stands in for
    one, mapped to the upright frame the same way a VLM box would be."""
    out = []
    for (run, cap), s in sorted(scenes.items()):
        mb = (s.meta.get("mask") or {}).get("box")
        if mb is None or s.centroid is None:
            continue
        if cap != "000" and s.half:
            continue                       # only off-centre / non-half captures
        box_up = ((s.W - mb[2], s.H - mb[3], s.W - mb[0], s.H - mb[1])
                  if s.half else tuple(mb))
        row = {"run": run, "cap": cap, "correct_mm": None, "inverted_mm": None,
               "correct_points": 0, "inverted_points": 0}
        for key, half in (("correct", s.half), ("inverted", not s.half)):
            _, _, pts, _ = s.patch(box_up, half=half)
            row[f"{key}_points"] = int(len(pts))
            if len(pts) >= MIN_PCA_POINTS:
                row[f"{key}_mm"] = round(float(np.linalg.norm(
                    np.median(pts, axis=0) - s.centroid) * 1000), 1)
        out.append(row)
    return out


def audit_mask_orientation():
    """T2c. `mask.png` is stored in the SAME orientation as `rgb.png`
    (perception/capture.py:88), while `meta["mask"]["box"]` is the prompt box
    in the RAW frame. So for a half-turn capture the mask must be rotated back
    before it lands inside its own box. This is an orientation proof that does
    not touch depth at all."""
    rows = []
    for run_dir in sorted(RUNS / r for r in CORPUS_RUNS):
        for cap_dir in sorted(p for p in run_dir.iterdir()
                              if p.is_dir() and p.name.isdigit()):
            mp = cap_dir / "mask.png"
            if not mp.exists():
                continue
            meta = json.loads((cap_dir / "meta.json").read_text())
            mb = (meta.get("mask") or {}).get("box")
            if mb is None or not meta.get("rgb_rotation_deg"):
                continue                    # only the half-turn captures prove it
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            x0, y0, x1, y1 = mb
            tot = m.sum()
            if not tot:
                continue
            rows.append({"run": run_dir.name, "cap": cap_dir.name,
                         "raw_mapped_containment": round(float(
                             rotate180(m)[y0:y1, x0:x1].sum() / tot), 4),
                         "unrotated_containment": round(float(
                             m[y0:y1, x0:x1].sum() / tot), 4)})
    return rows


# ---- picture side-cars -----------------------------------------------------
def frame_path(run, cap):
    return RUNS / run / "eyes" / "frames" / f"{cap}.png"


def save_crop(out, name, run, cap, box_up):
    """Re-cut from eyes/frames/<cap>.png. NEVER eyes/crops/*.png — trap T6:
    that filename is {cap_dir}_{turn:02d} and collides across transcripts
    (36 of 56 overwritten in 2408-cup1; 2508-aiduck2/eyes/crops/007_01.png is
    f008's 72x45 box, not f007's 70x45)."""
    src = frame_path(run, cap)
    if not src.exists():
        return None
    img = cv2.imread(str(src))
    if img is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box_up]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    d = out / "crops"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), img[y0:y1, x0:x1])
    return f"crops/{name}.png"


def save_validity(out, name, run, cap, box_up, scene, valid_raw, frac, npts):
    """The crop with valid-depth pixels tinted green and the hole fraction
    burned in. This is the passive-stereo hazard made visible."""
    src = frame_path(run, cap)
    if not src.exists():
        return None
    img = cv2.imread(str(src))
    if img is None:
        return None
    vis = rotate180(valid_raw) if scene.half else valid_raw   # RAW -> upright
    x0, y0, x1, y1 = [int(v) for v in box_up]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    tile = img[y0:y1, x0:x1].astype(np.float64).copy()
    sub = vis[y0:y1, x0:x1]
    tile[sub] = 0.45 * tile[sub] + 0.55 * np.array([0, 255, 0])
    tile = tile.astype(np.uint8)
    k = max(1, int(round(200 / max(tile.shape[0], 1))))
    tile = cv2.resize(tile, (tile.shape[1] * k, tile.shape[0] * k),
                      interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, f"valid {frac:.2f}  n={npts}", (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    d = out / "validity"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), tile)
    return f"validity/{name}.png"


def mirror_frames(out, wanted):
    """Copy every frame a figure will open into <out>/frames/, keeping the
    runs_root-relative path so both roots resolve the same string."""
    for rel in sorted(wanted):
        src = RUNS / rel
        if not src.exists():
            continue
        dst = out / "frames" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


# ---- the probe itself ------------------------------------------------------
def probe_case(scene, box_up, centre, r, elevs, mirror=False, half=None,
               use_mask=True):
    """(run, cap, box_up) -> anchor, normal, ranked cells. Nothing else enters.

    `mirror` is the T2d control: the box is mirrored horizontally in the
    upright frame. If accuracy survives that, the probe is reading the pose
    and not the image, and Figures 1 and 2 are void.
    """
    if mirror:
        x0, y0, x1, y1 = box_up
        box_up = (FRAME_W - x1, y0, FRAME_W - x0, y1)
    bx, n_box, pts, valid = scene.patch(box_up, half=half, use_mask=use_mask)
    res = {"box_raw": list(bx), "n_box": n_box, "n_pts": int(len(pts)),
           "valid_frac": (len(pts) / n_box) if n_box else 0.0,
           "valid_map": valid, "pts": pts, "anchor": None, "per": {},
           "normal": None, "fan": 0.0, "ranked": None, "rms": None}
    if len(pts) == 0:
        return res
    res["anchor"] = np.median(pts, axis=0)     # robust to the crop's fringe
    per, head, fan = normal_sweep(pts, res["anchor"], scene.T[:3, 3])
    res["per"], res["normal"], res["fan"] = per, head, fan
    if head is None:
        return res
    acc = [rho for rho, v in per.items() if v is not None]
    rho_mid = acc[len(acc) // 2]
    res["rho_rms"] = rho_mid
    res["rms"] = rms_against(pts, res["anchor"], head, rho_mid)
    vs = ViewSphere(centre, r=r, elevations=elevs)
    res["vs"] = vs
    res["ranked"] = rank_cells(vs, centre, r, res["anchor"], head)
    for rho, v in per.items():
        if v is not None:
            v["cell"] = rank_cells(vs, centre, r, res["anchor"], v["n"])[0][0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/normal_probe")
    ap.add_argument("--skip-reach", action="store_true",
                    help="skip the IK reachability field (supporting only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    t_start = time.time()
    say("=" * 100)
    say("NORMAL PROBE — can ONE crop box name the cell where the feature reads best?")
    say(f"{sum(1 for c in CASES if c[1] == 'A')} scored homing cases · "
        f"{len({c[2] for c in CASES if c[1] == 'A'})} runs · 2 physical "
        f"feature instances. A DEMONSTRATION, NOT A PROOF.")
    say("=" * 100)

    # ---------------------------------------------------------------- T1
    say("\n[T1] box convention audit over every crop in the corpus")
    crops = all_crops()
    bad = audit_box_convention(crops)
    say(f"     {len(crops)} crop calls · to_pixel_box reproduces the executed "
        f"box in {len(crops) - len(bad)}/{len(crops)}")
    for b in bad:
        say(f"     mismatch: {b}")
    tool_counts = audit_tool_calls()
    say("     tool calls in the corpus: " + ", ".join(
        f"{k}={v}" for k, v in sorted(tool_counts.items())))
    say("     detect / read_text / segment output is STUB and is DISCARDED "
        "corpus-wide; crop is the only tool record this probe reads.")
    t1_ok = len(bad) == 2 and all("2408-seeded/f001" in b for b in bad)
    say(f"     PASS={t1_ok}  (expected exactly 2, both 2408-seeded/f001, both discarded)")
    if not t1_ok:
        say("     ABORT: box bookkeeping is wrong, every downstream number is void")
        (out / "console.txt").write_text("\n".join(log) + "\n")
        return 1

    # -------------------------------------------------- scenes (cached once)
    scenes = {}
    caps_needed = {(c["run"], c["cap"]) for c in crops if c["cap"]}
    caps_needed |= {(c[2], c[4]) for c in CASES}
    for run_dir in sorted(RUNS / r for r in CORPUS_RUNS):
        for cap_dir in sorted(p for p in run_dir.iterdir()
                              if p.is_dir() and p.name.isdigit()):
            caps_needed.add((run_dir.name, cap_dir.name))
    skipped = []
    for run, cap in sorted(caps_needed):
        try:
            scenes[(run, cap)] = Scene(run, cap)
        except Exception as e:                                # noqa: BLE001
            skipped.append(f"{run}/{cap}: {e}")
    say(f"\n     {len(scenes)} captures reconstructed, {len(skipped)} unusable")
    for s in skipped:
        say(f"     unusable: {s}")

    # ---------------------------------------------------------------- T2a-c
    say("\n[T2a] orientation A/B over the whole corpus (`half` inverted)")
    ab = audit_orientation_ab(crops, scenes)
    say(f"     correct  map: off-object {ab['correct']['off']}/{ab['correct']['tot']}"
        f" = {ab['correct']['frac']:.3f}")
    say(f"     inverted map: off-object {ab['inverted']['off']}/{ab['inverted']['tot']}"
        f" = {ab['inverted']['frac']:.3f}   median 3D displacement "
        f"{ab['inverted']['median_disp_mm']:.1f} mm")
    discriminates = (ab["inverted"]["frac"] - ab["correct"]["frac"]) > 0.10
    say(f"     discriminates={discriminates}  (the spec predicted ~0.26 vs ~0.64 "
        f"and ~379 mm; MEASURED values are above and do not reproduce that —")
    say("      reported as measured, not repaired. Scout 4's F5 explains it: the "
        "camera is aimed AT the object, so a")
    say("      half-turned box near the frame centre still lands on the object. "
        "T2b/T2c below carry the orientation evidence.")

    say("\n[T2b] off-centre anchor A/B (survey pose cap 000 + non-half captures)")
    sab = audit_survey_anchor(scenes)
    for row in sab:
        say(f"     {row['run']}/{row['cap']}  correct {row['correct_mm']} mm "
            f"({row['correct_points']} pts)   inverted {row['inverted_mm']} mm "
            f"({row['inverted_points']} pts)")

    say("\n[T2c] mask-box orientation proof (no depth involved)")
    mo = audit_mask_orientation()
    head = next((r for r in mo if r["run"] == "2508-aiduck2" and r["cap"] == "001"),
                mo[0] if mo else None)
    for row in mo:
        say(f"     {row['run']}/{row['cap']}  RAW-mapped {row['raw_mapped_containment']:.3f}"
            f"   un-rotated {row['unrotated_containment']:.3f}")

    # ---------------------------------------------------- self-tests (2.10)
    say("\n[SELF-TEST] the four numbers this run must reproduce")
    st = {}
    s1 = scenes[("2508-aiduck2", "007")]
    bx1, nb1, p1, _ = s1.patch((411, 247, 481, 292))
    a1 = np.median(p1, axis=0)
    ok1 = (s1.half and s1.disk_up and tuple(bx1) == (367, 188, 437, 233)
           and nb1 == 2954 and len(p1) == 2802
           and abs(len(p1) / nb1 - 0.949) < 5e-4
           and np.allclose(a1, [0.1346, -0.3420, 0.0587], atol=5e-4))
    say(f"  1. 2508-aiduck2/007 box (411,247,481,292): half={s1.half} "
        f"disk_up={s1.disk_up} box_raw={tuple(bx1)} n_box={nb1} "
        f"n_pts={len(p1)} valid={len(p1)/nb1:.3f} anchor={np.round(a1, 4).tolist()} -> {ok1}")
    st["aiduck2_f007_n_pts"] = int(len(p1))

    say(f"  2. box-decode audit: {len(bad)} mismatches, both 2408-seeded/f001 "
        f"-> {t1_ok}")

    # Restricted to the runs the probe actually predicts on (OBJECTS): those
    # are the eight the sphere-centre recovery was verified against, and ai1
    # drops out for having a single cell view -> 7 runs, 32 cells.
    tot = okc = 0
    maxres = 0.0
    centres, spreads = {}, {}
    for run_dir in sorted(RUNS / r for r in OBJECTS):
        sp = run_dir / "eyes" / "store.json"
        if not sp.exists():
            continue
        store = json.loads(sp.read_text())
        cv = [v for v in store["views"] if v["cell"] is not None]
        if len(cv) < 2:
            continue
        c, spread = sphere_centre(store)
        centres[run_dir.name], spreads[run_dir.name] = c, spread
        vs = ViewSphere(c, r=store["grid"]["r"],
                        elevations=tuple(store["grid"]["v_elevs"]))
        for v in cv:
            d = unit(np.asarray(v["T_base_cam"], float)[:3, 3] - c)
            best = max(vs.cells(), key=lambda k: vs.cell_dir(*k) @ d)
            ang = math.degrees(math.acos(float(np.clip(
                vs.cell_dir(*best) @ d, -1, 1))))
            maxres = max(maxres, ang)
            tot += 1
            okc += int(list(best) == list(v["cell"]))
    ok3 = (okc == tot == 32) and maxres <= 3.8
    say(f"  3. cell round-trip: {okc}/{tot} recovered, max residual "
        f"{maxres:.1f} deg -> {ok3}")
    st["cell_roundtrip"] = f"{okc}/{tot}"
    st["cell_roundtrip_max_residual_deg"] = round(maxres, 2)

    s4 = scenes[("2508-ai3", "003")]
    _, nb4, p4, _ = s4.patch((381, 234, 469, 308))
    st4 = load_store("2508-ai3")
    c4, _ = sphere_centre(st4)
    r4 = probe_case(s4, (381, 234, 469, 308), c4, st4["grid"]["r"],
                    tuple(st4["grid"]["v_elevs"]))
    pred4 = r4["ranked"][0][0] if r4["ranked"] else None
    ok4 = (len(p4) == 3243 and abs(len(p4) / nb4 - 0.498) < 5e-4
           and pred4 == (4, 0))
    say(f"  4. 2508-ai3/003 box (381,234,469,308): n_pts={len(p4)} "
        f"valid={len(p4)/nb4:.3f} prediction={pred4} -> {ok4}")
    self_ok = bool(ok1 and t1_ok and ok3 and ok4)
    say(f"  SELF-TEST PASSED = {self_ok}")
    if not self_ok:
        say("  ABORT: bookkeeping differs from the pre-registered numbers, "
            "the run is VOID")
        (out / "console.txt").write_text("\n".join(log) + "\n")
        return 1

    # ------------------------------------------------------- per-run context
    runs_used = sorted({c[2] for c in CASES})
    stores = {r: load_store(r) for r in runs_used}
    reach = {}
    for r in runs_used:
        if r not in centres:                     # 2508-ai1 has one cell view
            centres[r], spreads[r] = sphere_centre(stores[r])
        reach[r] = None if args.skip_reach else reachable_cells(
            r, stores[r], centres[r])

    run_ctx = {}
    for r in runs_used:
        store = stores[r]
        cellviews = [v for v in store["views"] if v["cell"] is not None]
        srcs = [c for c in CASES if c[2] == r and c[1] in ("A", "B")]
        caps = [v["cap_dir"] for v in cellviews]
        first = min((caps.index(c[4]) for c in srcs if c[4] in caps),
                    default=len(caps) - 1)
        obj, feat, _ = OBJECTS[r]
        run_ctx[r] = {
            "object": obj, "feature": feat, "r_m": store["grid"]["r"],
            "v_elevs_deg": list(store["grid"]["v_elevs"]),
            "h_bins": store["grid"]["h_bins"],
            "sphere_centre_base_m": np.round(centres[r], 6).tolist(),
            "sphere_centre_spread_mm": round(spreads[r], 2),
            "captured_cells": [list(v["cell"]) for v in cellviews],
            "captured_caps": caps,
            "captured_frames_rel": [f"{r}/eyes/frames/{c}.png" for c in caps],
            "survey_cap": next((v["cap_dir"] for v in store["views"]
                                if v["cell"] is None), None),
            "views_total": len(cellviews),
            "views_before_first_feature_crop": int(first),
            "max_possible_saving_views": int(len(caps) - first - 1),
            "reachable_cells": (None if reach[r] is None
                                else [list(c) for c in sorted(reach[r])]),
        }

    # label bookkeeping: earliest FULL-class transcript per (run, cell)
    label_info = {}
    strip_prose = {}
    for r in runs_used:
        store = stores[r]
        wordmark = OBJECTS[r][2]
        for fnd in store["findings"]:
            cell = tuple(fnd["cell"])
            stem = Path(fnd["transcript"]).stem
            t = load_transcript(r, stem)
            txt = final_text(t)
            rc = read_class(txt, wordmark)
            strip_prose[(r, cell)] = (rc, fnd["summary"], stem)
            if rc == "FULL" and (r, cell) not in label_info:
                boxes = [executed_box(t, i) for i, tu in enumerate(t["turns"])
                         if tu.get("tool") == "crop"]
                boxes = [b for b in boxes if b]
                feat_boxes = [b for b in boxes
                              if (b[2] - b[0]) * (b[3] - b[1]) < FEATURE_CROP_MAX_AREA]
                label_info[(r, cell)] = {
                    "stem": stem, "prose": fnd["summary"],
                    "box": list(feat_boxes[0]) if feat_boxes
                    else (list(boxes[0]) if boxes else None)}

    # ------------------------------------------------------------ the cases
    say("\n" + "=" * 100)
    say("PER-CASE RESULTS")
    say("=" * 100)
    hdr = (f"{'id':>4} {'run':>13} {'src':>7} {'label':>12} {'pred':>7} "
           f"{'top3':>24} {'err':>4} {'deg':>6} {'valid':>6} {'npts':>6} "
           f"{'rms':>6} {'fan':>6} {'bucket':>19} {'B0':>6} {'B1':>6} "
           f"{'B4':>6} {'B5':>6} {'mirror':>7}")
    say(hdr)
    say("-" * len(hdr))

    rng = np.random.default_rng(RANDOM_SEED)
    cases_out, rejections, failures = [], [], []
    wanted_frames = set()

    for (cid, tier, run, src_cell, cap, stem, turn, box_tbl, labels,
         strength, gap) in CASES:
        store = stores[run]
        r = store["grid"]["r"]
        elevs = tuple(store["grid"]["v_elevs"])
        centre = centres[run]
        obj, feat, wordmark = OBJECTS[run]
        cellviews = [v for v in store["views"] if v["cell"] is not None]
        caps = [v["cap_dir"] for v in cellviews]
        captured = [tuple(v["cell"]) for v in cellviews]

        rec = {
            "case_id": cid, "tier": tier,
            "admissible": tier in ("A", "B"),
            "exclusion_reason": TIER_C_REASON.get(cid),
            "strength": strength,
            "run": run, "feature": feat, "object": obj,
            "source_cell": list(src_cell), "source_cap": cap,
            "transcript_rel": f"{run}/eyes/transcripts/{stem}.json",
            "turn_index": turn,
            "error": None,
        }

        t = load_transcript(run, stem)
        box_up = executed_box(t, turn)
        if box_up is None or tuple(box_up) != tuple(box_tbl):
            rec["error"] = (f"executed box {box_up} does not match the frozen "
                            f"table entry {box_tbl}")
            failures.append((cid, rec["error"]))
            cases_out.append(rec)
            say(f"{cid:>4} {run:>13}  FAILED: {rec['error']}")
            continue

        # orientation cross-check: the view_text suffix is authoritative
        # provenance (eyes/tools.py:25 upright()).
        scene = scenes.get((run, cap))
        if scene is None:
            rec["error"] = f"capture {run}/{cap} unusable"
            failures.append((cid, rec["error"]))
            cases_out.append(rec)
            say(f"{cid:>4} {run:>13}  FAILED: {rec['error']}")
            continue
        vt = t.get("view_text")
        token = None
        if vt:
            for tok in (" · upright on disk", " · flipped upright"):
                if vt.endswith(tok):
                    token = tok
            if token is None:
                token = " · TILTED" if " · TILTED" in vt else ""
        if token == " · upright on disk":
            agree = scene.half and scene.disk_up
        elif token == " · flipped upright":
            agree = scene.half and not scene.disk_up
        elif token == "":
            agree = not scene.half
        else:
            # legacy transcripts (2408-cup1 f002-f007) carry view_text=None —
            # fall back to the flags, which are derived from the pose itself.
            agree = True
        rec.update({
            "crop_box_upright": list(box_up),
            "crop_wh": [box_up[2] - box_up[0], box_up[3] - box_up[1]],
            "crop_area_px": (box_up[2] - box_up[0]) * (box_up[3] - box_up[1]),
            "args_box_raw_model": list(t["turns"][turn]["args"]["box"]),
            "box_decode_agrees": tuple(GeminiVlm.to_pixel_box(
                t["turns"][turn]["args"]["box"], FRAME_W, FRAME_H)) == tuple(box_up),
            "half_turn": bool(scene.half), "disk_upright": bool(scene.disk_up),
            "view_text_token": token, "orientation_flags_agree": bool(agree),
        })
        if not agree:
            rec["error"] = (f"orientation flags disagree with view_text "
                            f"{token!r}: half={scene.half} disk_up={scene.disk_up}")
            failures.append((cid, rec["error"]))
            cases_out.append(rec)
            say(f"{cid:>4} {run:>13}  FAILED: {rec['error']}")
            continue

        src_txt = final_text(t)
        rec["source_prose"] = (strip_prose.get((run, src_cell), (None, "", None))[1]
                               if (run, src_cell) in strip_prose else src_txt[:300])
        rec["source_framing"] = t.get("framing")
        rec["source_read_class"] = read_class(src_txt, wordmark)

        # ---- geometry
        res = probe_case(scene, box_up, centre, r, elevs)
        rec.update({
            "crop_box_raw": res["box_raw"],
            "T_base_cam": np.asarray(scene.T).tolist(),
            "camera_pos_base_m": np.round(scene.T[:3, 3], 6).tolist(),
            "intrinsics_color": {k: scene.intr[k] for k in ("fx", "fy", "ppx", "ppy")},
            "depth_scale_m_per_unit": scene.scale,
            "n_box_px": res["n_box"], "n_patch_points": res["n_pts"],
            "depth_valid_frac": round(res["valid_frac"], 4),
            "mask_available": scene.mask is not None,
            "mask_score": scene.mask_score,
            "mask_px": scene.mask_px,
            "view_object_source": scene.object_source,
            "view_object_centroid_base_m": (None if scene.centroid is None
                                            else np.round(scene.centroid, 6).tolist()),
            "view_object_r95_m": None if scene.r95 is None else round(scene.r95, 5),
        })
        # How much of the crop box the SAM 3 mask actually admits. On
        # 2508-ai5/002 this is 0.000: the mask there is the rim and the handle
        # only — the black matte body returned nothing, so the object-only gate
        # empties a box that sits squarely on the logo. A per-view mask failure,
        # not a probe failure, but it must be visible as a number.
        _bxn, n_nomask, pts_nomask, _ = scene.patch(box_up, use_mask=False)
        rec["mask_box_overlap_frac"] = (
            round(res["n_box"] / n_nomask, 4) if n_nomask else None)
        rec["n_patch_points_no_mask"] = int(len(pts_nomask))
        fused = RUNS / run / "fused_cloud.npy"
        rec["fused_offset_mm"] = None
        if fused.exists() and res["anchor"] is not None:
            fc = np.load(fused)
            rec["fused_offset_mm"] = round(float(np.min(np.linalg.norm(
                fc - res["anchor"], axis=1)) * 1000), 2)   # DIAGNOSTIC ONLY

        rec["point_3d_base_m"] = (None if res["anchor"] is None
                                  else np.round(res["anchor"], 6).tolist())
        rec["anchor_offset_from_centroid_mm"] = (
            None if res["anchor"] is None or scene.centroid is None
            else round(float(np.linalg.norm(res["anchor"] - scene.centroid) * 1000), 2))

        # ---- gates. ONLY thin and off-object abstain. Abstaining on the hard
        # half and winning on the easy half is not winning (Scout 4, T4).
        gate_thin = res["n_pts"] < MIN_PATCH_POINTS
        gate_off = (res["anchor"] is not None and scene.r95 is not None
                    and np.linalg.norm(res["anchor"] - scene.centroid)
                    > OFFOBJECT_R95_MULT * scene.r95)
        abstain_reason = "thin" if gate_thin else ("off_object" if gate_off else None)
        if res["ranked"] is None and abstain_reason is None:
            abstain_reason = "thin"
        flags = []
        if res["valid_frac"] < MIN_DEPTH_VALID_FRAC:
            flags.append("low_valid")
        if res["rms"] is not None and res["rms"] > MAX_PLANE_RMS_M:
            flags.append("high_rms")
        if res["fan"] > MAX_FAN_WIDTH_DEG:
            flags.append("unstable_fan")
        rec["flags"] = flags
        rec["abstained"] = abstain_reason is not None
        rec["abstain_reason"] = abstain_reason
        if abstain_reason:
            if abstain_reason == "thin":
                detail = (f"{res['n_pts']} patch points (min {MIN_PATCH_POINTS}); "
                          f"mask admits {rec['mask_box_overlap_frac']} of the box, "
                          f"{rec['n_patch_points_no_mask']} points without it")
            else:
                detail = (f"anchor {rec['anchor_offset_from_centroid_mm']} mm from "
                          f"the view's object centroid, r95 "
                          f"{rec['view_object_r95_m']} m, threshold "
                          f"{round(OFFOBJECT_R95_MULT * (scene.r95 or 0) * 1000, 1)} mm; "
                          f"reference cloud came from {scene.object_source}")
            rejections.append({"case_id": cid, "reason": abstain_reason,
                               "detail": detail})

        rec["normal_base"] = (None if res["normal"] is None
                              else np.round(res["normal"], 6).tolist())
        rec["normal_by_radius"] = {
            str(int(rho * 1000)): (None if v is None else {
                "normal": np.round(v["n"], 6).tolist(), "k": v["k"],
                "rms_mm": round(v["rms"] * 1000, 3),
                "planarity": round(v["planarity"], 4),
                "cell": list(v.get("cell", ())) or None, "accepted": True})
            for rho, v in res["per"].items()}
        rec["fan_width_deg"] = round(res["fan"], 3)
        rec["plane_rms_mm_headline"] = (None if res["rms"] is None
                                        else round(res["rms"] * 1000, 3))

        # patch azimuth span about the sphere centre — how much of the ring the
        # crop itself covers, in cells. A span > 1 cell means the "best cell"
        # question is under-determined by the crop, whatever the normal says.
        rec["patch_azimuth_span_deg"] = None
        rec["patch_span_cells"] = None
        if res["n_pts"] >= MIN_PCA_POINTS:
            az = np.degrees(np.arctan2(res["pts"][:, 1] - centre[1],
                                       res["pts"][:, 0] - centre[0]))
            a0 = math.degrees(math.atan2(res["anchor"][1] - centre[1],
                                         res["anchor"][0] - centre[0]))
            rel = (az - a0 + 180) % 360 - 180
            span = float(np.percentile(rel, 97.5) - np.percentile(rel, 2.5))
            rec["patch_azimuth_span_deg"] = round(span, 2)
            rec["patch_span_cells"] = round(span / (360 / store["grid"]["h_bins"]), 3)

        # ---- prediction
        pred = res["ranked"][0][0] if res["ranked"] else None
        rec["predicted_cell"] = list(pred) if pred else None
        rec["predicted_top3"] = ([list(c) for c, _ in res["ranked"][:3]]
                                 if res["ranked"] else [])
        rec["predicted_scores"] = ([round(s, 6) for _, s in res["ranked"][:3]]
                                   if res["ranked"] else [])
        rec["predicted_cell_centre_ray"] = None
        rec["predicted_cell_source_centre"] = None
        if res["normal"] is not None:
            vs = res["vs"]
            rec["predicted_cell_centre_ray"] = list(max(
                vs.cells(), key=lambda c: vs.cell_dir(*c) @ res["normal"]))
            # T3 sensitivity: the centre averaged over the run's poses is the
            # only quantity that touches other captures. Re-derive it from THIS
            # view alone and re-predict.
            c_src, _ = sphere_centre(store, views=[v for v in cellviews
                                                   if v["cap_dir"] == cap])
            rec["predicted_cell_source_centre"] = list(rank_cells(
                ViewSphere(c_src, r=r, elevations=elevs), c_src, r,
                res["anchor"], res["normal"])[0][0])
        # Diagnostic only, never scored: what the SAME estimator says with the
        # object-only mask gate switched off. Separates "the depth in this box
        # is unusable" from "the SAM 3 mask in this capture is unusable".
        rec["predicted_cell_no_mask"] = None
        if scene.mask is not None:
            nm = probe_case(scene, box_up, centre, r, elevs, use_mask=False)
            if nm["ranked"]:
                rec["predicted_cell_no_mask"] = list(nm["ranked"][0][0])
        rec["predicted_captured"] = bool(pred in captured) if pred else False
        rec["predicted_reachable"] = (None if reach[run] is None or pred is None
                                      else bool(pred in reach[run]))

        # ---- label
        rec["label_cells"] = [list(c) for c in labels]
        rec["label_alt_cells"] = [list(c) for c in LABEL_ALT.get(cid, [])]
        rec["label_primary"] = list(labels[0]) if labels else None
        li = label_info.get((run, labels[0])) if labels else None
        rec["label_source"] = ("human-audit" if cid == "A10" else
                               ("prose-FULL" if li else None))
        rec["label_confidence"] = ("weak" if strength == "weak" else
                                   ("strong" if strength == "STRONG" else None))
        rec["label_prose"] = li["prose"] if li else None
        rec["label_crop_box_upright"] = li["box"] if li else None
        rec["true_gap_cells"] = int(gap)

        # ---- scoring
        err, matched = (cells_err(pred, labels) if (pred and labels)
                        else (None, None))
        rec["label_matched"] = list(matched) if matched and err == 0 else None
        rec["err_cells"] = err
        rec["err_deg"] = (None if (pred is None or matched is None) else
                          round(deg_between(res["vs"], centre, r, res["anchor"],
                                            pred, matched), 2))
        rec["moved_cells"] = (None if pred is None else
                              abs(step_delta(src_cell[0], pred[0]))
                              + abs(pred[1] - src_cell[1]))

        if abstain_reason:
            bucket = "abstain"
        elif pred is None:
            bucket = "abstain"
        elif labels and err == 0:
            bucket = "correct"
        elif rec["predicted_reachable"] is False:
            bucket = "unreachable"
        elif not rec["predicted_captured"]:
            bucket = "not_captured"
        else:
            bucket = "wrong_but_captured"
        rec["bucket"] = bucket

        # motion saved. `views_probe_would_need` is the pre-registered
        # definition verbatim (section 3): 1 on a hit, otherwise the step
        # distance from the prediction to the nearest label. It UNDER-counts a
        # 1-cell miss (which also costs 1) — kept as written rather than
        # re-defined after seeing the data.
        src_idx = caps.index(cap) if cap in caps else 0
        lab_idx = None
        for lab in labels:
            for i, c in enumerate(captured):
                if c == lab and (lab_idx is None or i < lab_idx):
                    lab_idx = i
        took = (lab_idx - src_idx + 1) if (lab_idx is not None
                                           and lab_idx > src_idx) else 1
        need = (took if (bucket == "abstain" or err is None) else
                (1 if err == 0 else max(1, err)))
        rec["views_run_actually_took"] = int(took)
        rec["views_probe_would_need"] = int(need)
        rec["views_saved"] = int(took - need)

        # ---- baselines, identical shape for every one
        def score_cell(cell, hint=None):
            # B1 is null-valued whenever the transcript has no `framing` block
            # — only 9 findings in the whole corpus have one. Report the null
            # count, never fabricate a hint.
            if cell is None:
                return {"cell": None, "err_cells": None, "err_deg": None,
                        "bucket": None, **({"hint": hint} if hint is not None else {})}
            e, m = (cells_err(cell, labels) if labels else (None, None))
            b = ("correct" if (labels and e == 0) else
                 ("wrong_but_captured" if cell in captured else "not_captured"))
            d = (None if (m is None or res["anchor"] is None) else
                 round(deg_between(res["vs"], centre, r, res["anchor"], cell, m), 2))
            o = {"cell": list(cell), "err_cells": e, "err_deg": d, "bucket": b}
            if hint is not None:
                o["hint"] = hint
            return o

        bl = {"B0_stay": score_cell(tuple(src_cell))}
        fr = t.get("framing") or {}
        better = fr.get("better")
        if better in B1_MAPPING:
            dh, dv = B1_MAPPING[better]
            b1 = ((src_cell[0] + dh) % store["grid"]["h_bins"],
                  int(np.clip(src_cell[1] + dv, 0, len(elevs) - 1)))
            bl["B1_hint"] = score_cell(b1, hint=better)
        else:
            bl["B1_hint"] = score_cell(None, hint=better)
        cand = sorted(set(captured))
        pick = tuple(cand[int(rng.integers(len(cand)))]) if cand else None
        b2 = score_cell(pick)
        b2["p_exact_captured"] = round(1.0 / max(len(cand), 1), 4)
        b2["p_exact_reachable"] = round(
            1.0 / (len(reach[run]) if reach[run] else 26), 4)
        bl["B2_random"] = b2
        bl["B3_antipode"] = score_cell(((src_cell[0] + 6)
                                        % store["grid"]["h_bins"], src_cell[1]))
        if res["anchor"] is not None:
            radial = unit(res["anchor"] - centre)
            vs = res["vs"]
            bl["B4_radial"] = score_cell(max(vs.cells(),
                                             key=lambda c: vs.cell_dir(*c) @ radial))
        else:
            bl["B4_radial"] = score_cell(None)
        others = [c for c in cand if tuple(c) != tuple(src_cell)]
        mv = moves_from(tuple(src_cell), others,
                        h_bins=store["grid"]["h_bins"], elevations=elevs)
        bl["B5_nearest"] = score_cell(tuple(mv[0].cell) if mv else None)
        rec["baselines"] = bl

        # ---- T2d mirror injection
        mres = probe_case(scene, box_up, centre, r, elevs, mirror=True)
        mpred = mres["ranked"][0][0] if mres["ranked"] else None
        me, mm_ = (cells_err(mpred, labels) if (mpred and labels) else (None, None))
        rec["mirror_control"] = {
            "predicted_cell": list(mpred) if mpred else None,
            "err_cells": me,
            "bucket": (None if mpred is None else
                       ("correct" if (labels and me == 0) else
                        ("wrong_but_captured" if mpred in captured
                         else "not_captured")))}

        # ---- pictures
        rec["source_frame_rel"] = f"{run}/eyes/frames/{cap}.png"
        wanted_frames.add(rec["source_frame_rel"])
        rec["source_crop_rel"] = save_crop(out, f"{cid}_source", run, cap, box_up)
        rec["source_validity_rel"] = save_validity(
            out, f"{cid}_source", run, cap, box_up, scene,
            res["valid_map"], res["valid_frac"], res["n_pts"])

        lab_cap = None
        if labels:
            for v in cellviews:
                if tuple(v["cell"]) == tuple(labels[0]):
                    lab_cap = v["cap_dir"]
        rec["label_frame_rel"] = (f"{run}/eyes/frames/{lab_cap}.png"
                                  if lab_cap else None)
        if rec["label_frame_rel"]:
            wanted_frames.add(rec["label_frame_rel"])
        rec["label_crop_rel"] = (save_crop(out, f"{cid}_label", run, lab_cap,
                                           li["box"]) if (lab_cap and li
                                                          and li["box"]) else None)
        pred_cap = None
        if pred:
            for v in cellviews:
                if tuple(v["cell"]) == tuple(pred):
                    pred_cap = v["cap_dir"]
        rec["predicted_frame_rel"] = (f"{run}/eyes/frames/{pred_cap}.png"
                                      if pred_cap else None)
        if rec["predicted_frame_rel"]:
            wanted_frames.add(rec["predicted_frame_rel"])
        rec["predicted_crop_rel"] = (
            save_crop(out, f"{cid}_predicted", run, pred_cap, li["box"])
            if (pred_cap and li and li["box"]) else None)

        strip = []
        for v in cellviews:
            c = tuple(v["cell"])
            rc, prose, _ = strip_prose.get((run, c), ("NONE", None, None))
            rel = f"{run}/eyes/frames/{v['cap_dir']}.png"
            wanted_frames.add(rel)
            strip.append({"cell": list(c), "cap": v["cap_dir"], "frame_rel": rel,
                          "is_source": c == tuple(src_cell),
                          "is_predicted": pred is not None and c == tuple(pred),
                          "is_label": any(c == tuple(l) for l in labels),
                          "read_class": rc, "prose": prose})
        rec["strip"] = strip
        cases_out.append(rec)

        top3 = " ".join(f"{h}/{v}" for h, v in rec["predicted_top3"])
        rms_s = f"{res['rms'] * 1000:.2f}" if res["rms"] else "-"
        say(f"{cid:>4} {run:>13} {str(list(src_cell)):>7} "
            f"{str([list(l) for l in labels]):>12} {str(rec['predicted_cell']):>7} "
            f"{top3:>24} {str(err):>4} {str(rec['err_deg']):>6} "
            f"{res['valid_frac']:>6.3f} {res['n_pts']:>6} "
            f"{rms_s:>6} "
            f"{res['fan']:>6.1f} {bucket:>19} "
            f"{str(bl['B0_stay']['err_cells']):>6} "
            f"{str(bl['B1_hint']['err_cells']):>6} "
            f"{str(bl['B4_radial']['err_cells']):>6} "
            f"{str(bl['B5_nearest']['err_cells']):>6} "
            f"{str(rec['mirror_control']['err_cells']):>7}")

    mirror_frames(out, wanted_frames)

    # ------------------------------------------------------------- summary
    def tally(rows, key):
        ex = w1 = 0
        for rr in rows:
            e = (rr["err_cells"] if key == "probe"
                 else rr["baselines"][key]["err_cells"])
            if e is None:
                continue
            ex += int(e == 0)
            w1 += int(e <= 1)
        return ex, w1

    A = [c for c in cases_out if c["tier"] == "A" and c.get("error") is None]
    B = [c for c in cases_out if c["tier"] == "B" and c.get("error") is None]
    C = [c for c in cases_out if c["tier"] == "C"]

    def block(rows):
        ex, w1 = tally(rows, "probe")
        out_ = {"probe": {"exact": ex, "within1": w1,
                          "abstain": sum(1 for r in rows if r["abstained"]),
                          "not_captured": sum(1 for r in rows
                                              if r["bucket"] == "not_captured")}}
        for k in ("B0_stay", "B1_hint", "B2_random", "B3_antipode",
                  "B4_radial", "B5_nearest"):
            e, w = tally(rows, k)
            out_[k] = {"exact": e, "within1": w}
        out_["B1_hint"]["n_defined"] = sum(
            1 for r in rows if r["baselines"]["B1_hint"]["cell"] is not None)
        out_["B2_random"]["expected_exact"] = round(sum(
            r["baselines"]["B2_random"]["p_exact_captured"] for r in rows), 3)
        return out_

    tier_a, tier_b = block(A), block(B)

    delta_hist = {str(k): 0 for k in range(-6, 7)}
    for r in A:
        if r.get("predicted_cell"):
            d = step_delta(r["source_cell"][0], r["predicted_cell"][0])
            delta_hist[str(int(d))] += 1

    # Cross-view consistency: the base-frame azimuth of the SAME rigid feature
    # as estimated from every source view of a run. Null shape = the arrows
    # follow the camera round the circle (the normal is a copy of where you
    # stand). Runs with a single case get `null`, not 0.0 — one arrow has no
    # spread and must not be allowed to pass the test for free.
    spread = {}
    for run in runs_used:
        azs = [math.degrees(math.atan2(r["normal_base"][1], r["normal_base"][0]))
               for r in cases_out
               if r["run"] == run and r["tier"] in ("A", "B")
               and r.get("normal_base") and not r["abstained"]]
        spread[run] = round(circular_spread(azs), 2) if len(azs) >= 2 else None

    probe_ex = tier_a["probe"]["exact"]
    mirror_ex = sum(1 for r in A if r["mirror_control"]["err_cells"] == 0)
    ge2_cases = [r for r in A if r["true_gap_cells"] >= 2]
    strong = [r for r in A if r["strength"] == "STRONG"]
    weak = [r for r in A if r["strength"] == "weak"]
    s_rate = (sum(1 for r in strong if r["err_cells"] == 0) / len(strong)
              if strong else 0.0)
    w_rate = (sum(1 for r in weak if r["err_cells"] == 0) / len(weak)
              if weak else 0.0)

    conditions = {
        "1_does_not_beat_free_baselines": not (
            probe_ex > tier_a["B0_stay"]["exact"]
            and probe_ex > tier_a["B1_hint"]["exact"]),
        "2_never_jumps": not any(r["moved_cells"] and r["moved_cells"] >= 2
                                 for r in ge2_cases),
        "3_normal_contributes_nothing": tier_a["B4_radial"]["exact"] >= probe_ex,
        "4_estimate_is_a_property_of_the_view": any(
            v is not None and v > 30.0 for v in spread.values()),
        "5_probe_reads_the_pose_not_the_image": not (mirror_ex < probe_ex),
        "6_gates_are_decoration": (len(rejections) == 0 or not any(
            r["case_id"] == "C02" for r in rejections)),
        "7_wins_only_where_easy": (
            sum(1 for r in A if r["abstained"]) > 3 or s_rate < w_rate),
    }
    core_fail = (conditions["1_does_not_beat_free_baselines"]
                 or conditions["3_normal_contributes_nothing"])
    overall = ("NULL" if core_fail else
               ("SIGNAL" if not any(conditions.values()) else "PARTIAL"))

    summary_same_run = max(
        [sum(1 for r in ge2_cases if r["run"] == run) for run in runs_used],
        default=0)
    summary = {
        "n_tier_a": len(A), "n_tier_b": len(B), "n_tier_c": len(C),
        "n_runs": len({r["run"] for r in A}),
        "n_physical_features": 2,
        "n_distinct_source_cells": len({tuple(r["source_cell"]) for r in A}),
        "n_gap_ge_2": len(ge2_cases),
        "n_gap_ge_2_same_run": summary_same_run,
        "tier_a": tier_a, "tier_b": tier_b,
        "delta_hist": delta_hist,
        "cross_view_azimuth_spread_deg": spread,
        "rejections": rejections,
        "failures": [{"case_id": c, "reason": w} for c, w in failures],
        "wilson_95ci_probe_exact": wilson(probe_ex, len(A)),
        # Case-sums, not run-sums: several Tier-A cases share a run, so these
        # count the same physical views more than once. The per-run ceiling is
        # runs[<run>].max_possible_saving_views — that is the number a caption
        # should quote for "size of the prize".
        "total_views_saved": sum(r["views_saved"] for r in A),
        "total_views_available_to_save": sum(
            r["views_run_actually_took"] - 1 for r in A),
        "per_run_max_saving_views": {r: run_ctx[r]["max_possible_saving_views"]
                                     for r in sorted({c["run"] for c in A})},
        "n_strong": len(strong), "n_weak": len(weak),
        "strong_exact_rate": round(s_rate, 3),
        "weak_exact_rate": round(w_rate, 3),
        "mirror_control_exact": mirror_ex,
        "label_alt_hits_not_scored": [
            r["case_id"] for r in A if r["label_alt_cells"]
            and r.get("predicted_cell") in r["label_alt_cells"]],
        "c02_gate_note":
            "C02 (the tape-measure belt clip) was NOT rejected, and cannot be "
            "by this gate. Cell [1,1] of 2508-aiduck2 looks down on the tape "
            "measure with the duck not visible at all, so the SAM 3 mask for "
            "that capture IS the tape measure: 11 239 of 11 239 mask pixels "
            "fall inside the crop box, the anchor sits 3.3 mm from the view's "
            "own object centroid, and 1.4 mm from the run's fused cloud. The "
            "off-object gate measures self-consistency WITHIN one view, not "
            "identity across a run, and the run's fused identity already "
            "contains the tape measure. Reported, not repaired: null "
            "condition 6 therefore triggers.",
        "null_conditions": conditions,
        "null_verdict": {
            "beats_B0": probe_ex > tier_a["B0_stay"]["exact"],
            "beats_B1": probe_ex > tier_a["B1_hint"]["exact"],
            "beats_B4": probe_ex > tier_a["B4_radial"]["exact"],
            "produced_a_ge2_jump_on_a_ge2_case": not conditions["2_never_jumps"],
            "cross_view_spread_under_30deg": not conditions[
                "4_estimate_is_a_property_of_the_view"],
            "mirror_control_collapsed": mirror_ex < probe_ex,
            "overall": overall,
        },
        # Where the COMPUTED corpus facts differ from the build spec's prose.
        # Every one of these is the measurement, not the spec's number: a
        # figure that quotes the spec's version will be quoting something this
        # run could not reproduce.
        "spec_discrepancies": [
            f"spec prose says Tier A spans 4 runs; the spec's own case table "
            f"spans {len({c['run'] for c in A})} "
            f"({', '.join(sorted({c['run'] for c in A}))}).",
            f"spec prose says '5 STRONG, 5 weak'; the table has "
            f"{len(strong)} STRONG and {len(weak)} weak.",
            f"spec example JSON says n_distinct_source_cells 9; computed "
            f"{len({tuple(r['source_cell']) for r in A})} distinct cell "
            f"addresses over Tier A.",
            f"spec prose says gap>=2 in 4 of 10 with 3 from one run; computed "
            f"{len(ge2_cases)} cases, at most "
            f"{summary_same_run} of them in any one run.",
            "T2a: spec predicted ~26% off-object correct vs ~64% inverted and "
            f"~379 mm median displacement. MEASURED "
            f"{ab['correct']['frac']:.3f} vs {ab['inverted']['frac']:.3f} and "
            f"{ab['inverted']['median_disp_mm']:.1f} mm — the corpus-level "
            "orientation A/B does not discriminate. T2b (survey captures) and "
            "T2c (mask containment) do.",
            "T5: spec required C02 to trip the off-object gate. It cannot — "
            "see c02_gate_note.",
        ],
        # COMPUTED, so a caption that prints it verbatim prints the truth. The
        # spec's own sentence is kept beside it because Figure 2 was told to
        # quote it verbatim — and it names a run count and a hit rate this run
        # does not reproduce.
        "statistical_note":
            f"n={len(A)} over 2 physical feature instances and "
            f"{len({c['run'] for c in A})} runs, with {len(ge2_cases)} cases at "
            f"a true gap of >=2 cells. The measured exact rate is "
            f"{probe_ex}/{len(A)}, Wilson 95% CI "
            f"{wilson(probe_ex, len(A))} — wide enough to cover anywhere from "
            f"6 to 10 hits. B1, the hint step the system uses today, is DEFINED "
            f"ON ONLY {tier_a['B1_hint']['n_defined']} of {len(A)} cases "
            f"because only 9 findings in the whole corpus carry a `framing` "
            f"block, so the comparison the hypothesis most needs cannot be made "
            f"on this corpus at all. This corpus cannot establish significance.",
        "statistical_note_spec_verbatim":
            "n=10 over 2 physical feature instances and 4 runs. A 60% hit rate "
            "has a 95% CI of roughly 30-85% and cannot be distinguished from "
            "B1's 50%. This corpus cannot establish significance.",
        "ceiling_note":
            "Counted from the first feature crop onward the recorded runs had "
            "1-4 views of homing available to save. The '8 views for an answer "
            "that needed 2' framing is a SEARCH problem; this probe only "
            "touches HOMING. In 2508-aiduck2, 5 of 7 views were blind search "
            "before any feature crop existed — maximum possible saving there "
            "is ONE view.",
    }

    say("\n" + "=" * 100)
    say("SUMMARY — Tier A (homing, headline)")
    say("=" * 100)
    say(f"{'estimator':>12} {'exact':>6} {'within1':>8}  n={len(A)}")
    say(f"{'probe':>12} {tier_a['probe']['exact']:>6} "
        f"{tier_a['probe']['within1']:>8}   abstain="
        f"{tier_a['probe']['abstain']} not_captured={tier_a['probe']['not_captured']}")
    for k in ("B0_stay", "B1_hint", "B2_random", "B3_antipode", "B4_radial",
              "B5_nearest"):
        extra = (f"   n_defined={tier_a[k]['n_defined']}" if k == "B1_hint" else
                 (f"   E[exact]={tier_a[k]['expected_exact']}"
                  if k == "B2_random" else ""))
        say(f"{k:>12} {tier_a[k]['exact']:>6} {tier_a[k]['within1']:>8}{extra}")
    say(f"{'mirror(T2d)':>12} {mirror_ex:>6}")
    say(f"\nWilson 95% CI on the probe's exact rate: {summary['wilson_95ci_probe_exact']}")
    say(f"views saved {summary['total_views_saved']} of "
        f"{summary['total_views_available_to_save']} available "
        f"(case-sums; runs are shared, so this double counts. Per-run ceiling: "
        f"{summary['per_run_max_saving_views']})")
    say(f"STRONG exact rate {s_rate:.2f} (n={len(strong)}) vs weak "
        f"{w_rate:.2f} (n={len(weak)})")
    say("\nSUMMARY — Tier B (already there, correct answer = STAY)")
    say(f"{'probe':>12} {tier_b['probe']['exact']:>6} / {len(B)}")
    for k in ("B0_stay", "B4_radial", "B5_nearest"):
        say(f"{k:>12} {tier_b[k]['exact']:>6} / {len(B)}")
    say("\nbuckets: " + ", ".join(
        f"{b}={sum(1 for r in A if r['bucket'] == b)}"
        for b in ("correct", "wrong_but_captured", "not_captured",
                  "unreachable", "abstain")))
    say("abstained Tier A ids: " + (", ".join(
        r["case_id"] for r in A if r["abstained"]) or "none"))
    say("\ncross-view azimuth spread of the SAME feature, per run "
        "(null verdict if >30 deg; n/a = fewer than 2 cases in that run):")
    for run, v in spread.items():
        say(f"     {run:>13} {'n/a' if v is None else f'{v:.1f} deg':>9}")
    say("\nrejections (T5 — a probe with zero rejections is broken):")
    for r in rejections:
        say(f"     {r['case_id']}: {r['reason']} — {r['detail']}")
    if not rejections:
        say("     NONE — the gates are decoration (null condition 6)")
    if not any(r["case_id"] == "C02" for r in rejections):
        say("     ! C02 NOT REJECTED — " + summary["c02_gate_note"])
    say("\nfailures (cases that could not be computed):")
    for c, w in failures:
        say(f"     {c}: {w}")
    if not failures:
        say("     none")
    say("\ndelta histogram (signed azimuth steps, Tier A): " + ", ".join(
        f"{k}:{v}" for k, v in delta_hist.items() if v))
    say("\nnull conditions (section 7 — declared BEFORE the run):")
    for k, v in conditions.items():
        say(f"     {k:>42} = {v}")
    say(f"\n     OVERALL VERDICT: {overall}")
    say("\nwhere the COMPUTED facts differ from the build spec's prose:")
    for s in summary["spec_discrepancies"]:
        say(f"     - {s}")
    say(f"\nNOT A PROOF: {len(A)} cases, {len({c['run'] for c in A})} runs, "
        "2 physical feature instances, one genuinely multi-step example family.")
    say(summary["statistical_note"])
    say(summary["ceiling_note"])

    results = {
        "schema_version": "normal-probe-1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_root": str(RUNS),
        "derived_root": str(out),
        "image_note":
            "All *_rel PNGs are BGR-on-disk (cv2 convention). Read with "
            "cv2.imread and cv2.cvtColor(...,COLOR_BGR2RGB) before display. "
            "eyes/frames/<cap>.png is the UPRIGHT frame the VLM saw, so crop "
            "boxes overlay it directly with no transform. Paths without the "
            "crops/ or validity/ prefix are relative to runs_root; a copy of "
            "every one of them also exists under derived_root/frames/<same "
            "relative path>.",
        "config": {
            "rho_sweep_m": list(RHO_SWEEP),
            "headline_normal": "mean-of-accepted-unit-normals",
            "min_patch_points": MIN_PATCH_POINTS,
            "offobject_r95_mult": OFFOBJECT_R95_MULT,
            "min_depth_valid_frac": MIN_DEPTH_VALID_FRAC,
            "max_plane_rms_m": MAX_PLANE_RMS_M,
            "max_fan_width_deg": MAX_FAN_WIDTH_DEG,
            "deproject_stride": 1,
            "intrinsics_used": "color",
            "depth_array_used": "depth_aligned",
            "b1_mapping": B1_MAPPING,
            "b1_mapping_calibrated_n": B1_CALIBRATED_N,
            "random_seed": RANDOM_SEED,
        },
        "corpus_audit": {
            "n_crops_total": len(crops),
            "box_convention_mismatches": len(bad),
            "box_convention_mismatch_ids": bad,
            "orientation_ab": {
                "correct_map_offobject_frac": round(ab["correct"]["frac"], 4),
                "inverted_map_offobject_frac": round(ab["inverted"]["frac"], 4),
                "inverted_median_displacement_mm": round(
                    ab["inverted"]["median_disp_mm"], 1),
                "discriminates": bool(discriminates),
                "n_scored": ab["correct"]["tot"],
                "note": "MEASURED, not the spec's predicted 0.26/0.64/379 mm. "
                        "The corpus-level A/B does NOT separate the two "
                        "orientations: the camera is aimed at the object, so a "
                        "half-turned box near the frame centre still lands on "
                        "the object (Scout 4's F5). T2b and T2c carry the "
                        "orientation evidence instead.",
            },
            "survey_anchor_check": sab,
            "mask_orientation_proof": (
                {"run": head["run"], "cap": head["cap"],
                 "raw_mapped_containment": head["raw_mapped_containment"],
                 "unrotated_containment": head["unrotated_containment"]}
                if head else None),
            "mask_orientation_proof_all": mo,
            "stub_backend": tool_counts,
            "self_test_passed": self_ok,
            "self_test_detail": st,
            "captures_unusable": skipped,
        },
        "excluded": EXCLUDED,
        "runs": run_ctx,
        "cases": cases_out,
        "summary": summary,
    }

    def jsonable(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"{type(o)} is not JSON serialisable")

    (out / "results.json").write_text(
        json.dumps(results, indent=1, default=jsonable) + "\n")
    (out / "console.txt").write_text("\n".join(log) + "\n")
    print(f"\nwrote {out/'results.json'}  ({len(cases_out)} cases, "
          f"{len(failures)} failed)  in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
