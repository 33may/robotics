"""The view-plan contract: `PlannedView`, `StepPlan`, and their JSON.

One rule dominates this file: **a planned pose is stored exactly like a
recorded one**. `T_base_cam` is 4x4 row-major, OpenCV camera convention
(+Z boresight, +X right, +Y down), robot base frame, metres — bit-for-bit
the same shape as `runs/<run>/NNN/meta.json:T_base_cam`. The viewsphere
+X-forward convention (`view/viewsphere.py`) never appears here; anything
that wants it converts on the way out.

Serialisation is total and lossless: `plan_from_json(plan_to_json(p)) == p`.
Floats are plain Python floats and indices plain ints, so `json.dump` needs
no custom encoder — the same promise `fit/contract.py` makes.

Abstaining is a first-class answer. A 900-point stub of a cup has no honest
mesh in it, and a plan built on a hallucinated mesh is worse than no plan:
it looks like an answer. When `abstain=True` there are no views and `reason`
says why in one human sentence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

#: Package semver, bumped when the *meaning* of a field changes.
SEMVER = "0.1"

#: Rotation sanity for a stored pose. Poses built by `look_at_cv` are
#: orthonormal to ~1e-15 and survive the JSON round trip exactly (repr of a
#: float round-trips in Python 3), so this is not a tolerance we need — it is
#: a tripwire for poses that came from somewhere else and are not rigid at
#: all. 1e-5 is loose enough never to reject our own output and tight enough
#: that a 0.01 % scale error still fails.
ROT_TOL = 1e-5


def _jsonify(obj):
    """numpy -> plain python, recursively, so `json.dump` needs no encoder."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _pose_list(pose) -> List[List[float]]:
    """4x4 -> nested plain floats, with the rotation actually checked.

    A pose that is not a rigid transform is not a camera; catching it here
    means the renderer, the rerun workspace and any future executor can all
    trust the file without re-deriving this check.
    """
    T = np.asarray(pose, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_cam must be 4x4, got {T.shape}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=ROT_TOL):
        raise ValueError("T_base_cam rotation is not orthonormal")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > ROT_TOL:
        raise ValueError(f"T_base_cam rotation is not right-handed (det={det:.6f})")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"T_base_cam bottom row must be [0,0,0,1], got {T[3]}")
    return [[float(v) for v in row] for row in T]


@dataclass
class PlannedView:
    """One camera pose the planner committed to, plus what it buys.

    `covered` is the *whole* measurability row of this view — every surface
    sample it measures within the gates, not only the ones it was the first
    to reach. That is what makes a render checkable: the claim "this view
    frames this surface" is the list, and the render either shows it or does
    not (`render.py` splats them).
    """

    view_id: int
    T_base_cam: List[List[float]]
    standoff_m: float = 0.0
    gain: float = 0.0
    covered: List[int] = field(default_factory=list)
    n_covered: int = 0
    area_m2: float = 0.0
    mean_incidence_deg: float = 0.0
    p95_incidence_deg: float = 0.0
    cum_area_frac: float = 0.0
    render: str = ""
    frame_fill: float = 0.0

    def __post_init__(self):
        self.view_id = int(self.view_id)
        self.T_base_cam = _pose_list(self.T_base_cam)
        self.standoff_m = float(self.standoff_m)
        self.gain = float(self.gain)
        self.covered = [int(i) for i in self.covered]
        n = int(self.n_covered)
        if n and n != len(self.covered):
            raise ValueError(f"n_covered={n} but covered has {len(self.covered)}")
        self.n_covered = len(self.covered)
        self.area_m2 = float(self.area_m2)
        self.mean_incidence_deg = float(self.mean_incidence_deg)
        self.p95_incidence_deg = float(self.p95_incidence_deg)
        self.cum_area_frac = float(self.cum_area_frac)
        self.render = str(self.render)
        self.frame_fill = float(self.frame_fill)

    # -- convenience for consumers that want matrices back ----------------
    @property
    def T(self) -> np.ndarray:
        return np.array(self.T_base_cam, dtype=float)

    @property
    def position(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def boresight(self) -> np.ndarray:
        """Unit +Z column: where the camera looks (OpenCV convention)."""
        return self.T[:3, 2]

    def to_json(self) -> Dict[str, Any]:
        return {"view_id": self.view_id, "T_base_cam": self.T_base_cam,
                "standoff_m": self.standoff_m, "gain": self.gain,
                "covered": self.covered, "n_covered": self.n_covered,
                "area_m2": self.area_m2,
                "mean_incidence_deg": self.mean_incidence_deg,
                "p95_incidence_deg": self.p95_incidence_deg,
                "cum_area_frac": self.cum_area_frac,
                "render": self.render, "frame_fill": self.frame_fill}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "PlannedView":
        return PlannedView(
            view_id=d["view_id"], T_base_cam=d["T_base_cam"],
            standoff_m=d.get("standoff_m", 0.0), gain=d.get("gain", 0.0),
            covered=d.get("covered", []), n_covered=d.get("n_covered", 0),
            area_m2=d.get("area_m2", 0.0),
            mean_incidence_deg=d.get("mean_incidence_deg", 0.0),
            p95_incidence_deg=d.get("p95_incidence_deg", 0.0),
            cum_area_frac=d.get("cum_area_frac", 0.0),
            render=d.get("render", ""), frame_fill=d.get("frame_fill", 0.0))


@dataclass
class StepPlan:
    """What one variant says about one step of one run.

    `params` is the FULL parameter dict, verbatim — the file is
    self-describing, so a plan read a month from now needs no lookup table
    to say what standoff or cutoff produced it. `version` fingerprints the
    same dict, so a parameter tweak is visible on disk without a diff.
    """

    run: str
    step: int
    variant: str
    version: str
    seed: int = 0
    abstain: bool = False
    reason: str = ""
    table_plane: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0])
    mesh: Dict[str, Any] = field(default_factory=dict)
    samples: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    views: List[PlannedView] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.run = str(self.run)
        self.step = int(self.step)
        self.variant = str(self.variant)
        self.version = str(self.version)
        self.seed = int(self.seed)
        self.abstain = bool(self.abstain)
        self.reason = str(self.reason)
        plane = [float(v) for v in self.table_plane]
        if len(plane) != 4:
            raise ValueError(f"table_plane must be [a,b,c,d], got {plane}")
        self.table_plane = plane
        self.mesh = _jsonify(self.mesh)
        self.samples = _jsonify(self.samples)
        self.params = _jsonify(self.params)
        self.metrics = _jsonify(self.metrics)
        self.views = [v if isinstance(v, PlannedView) else PlannedView.from_json(v)
                      for v in self.views]
        if self.abstain and self.views:
            raise ValueError("an abstaining plan carries no views")

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def poses(self) -> np.ndarray:
        """(n, 4, 4) stack — the thing an executor would actually consume."""
        if not self.views:
            return np.empty((0, 4, 4))
        return np.stack([v.T for v in self.views])

    def to_json(self) -> Dict[str, Any]:
        return {"run": self.run, "step": self.step, "variant": self.variant,
                "version": self.version, "seed": self.seed,
                "abstain": self.abstain, "reason": self.reason,
                "table_plane": self.table_plane, "mesh": self.mesh,
                "samples": self.samples, "params": self.params,
                "views": [v.to_json() for v in self.views],
                "metrics": self.metrics}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "StepPlan":
        return StepPlan(
            run=d["run"], step=d["step"], variant=d["variant"],
            version=d["version"], seed=d.get("seed", 0),
            abstain=d.get("abstain", False), reason=d.get("reason", ""),
            table_plane=d.get("table_plane", [0.0, 0.0, 1.0, 0.0]),
            mesh=d.get("mesh", {}), samples=d.get("samples", {}),
            params=d.get("params", {}),
            views=[PlannedView.from_json(v) for v in d.get("views", [])],
            metrics=d.get("metrics", {}))


def plan_to_json(p: StepPlan) -> Dict[str, Any]:
    return p.to_json()


def plan_from_json(d: Dict[str, Any]) -> StepPlan:
    return StepPlan.from_json(d)


#: Metric keys that record how long something took rather than what it is.
#: They belong in the file (roadmap M9 — the whole point of asking "is this
#: cheap enough to run in the live loop") but they are not part of the plan,
#: so any comparison of two plans must step over them.
WALLCLOCK_KEYS = ("seconds", "seconds_split")


def stable_json(p: StepPlan) -> str:
    """Canonical text of a plan with the wall-clock out — the identity of it.

    Two runs of the planner on the same cloud with the same seed must produce
    the same string. `json.dumps(to_json())` would not: it carries timings,
    which differ by microseconds and would turn a determinism check into a
    stopwatch check.
    """
    d = p.to_json()
    d["metrics"] = {k: v for k, v in d["metrics"].items()
                    if k not in WALLCLOCK_KEYS}
    return json.dumps(d, sort_keys=True)


def fingerprint(variant: str, params: Dict[str, Any]) -> str:
    """`<variant>/<semver>+p:<sha1[:8]>` over every answer-changing param."""
    blob = json.dumps(_jsonify(params), sort_keys=True, default=str).encode()
    return f"{variant}/{SEMVER}+p:{hashlib.sha1(blob).hexdigest()[:8]}"


def abstained(run: str, step: int, variant: str, params: Dict[str, Any],
              seed: int, reason: str) -> StepPlan:
    """The canonical refusal. Every planner call returns one of these or a plan."""
    return StepPlan(run=run, step=step, variant=variant,
                    version=fingerprint(variant, params), seed=seed,
                    abstain=True, reason=reason, params=params)


def look_at_cv(position: Sequence[float], boresight: Sequence[float],
               down: Sequence[float] = (0.0, 0.0, -1.0)) -> np.ndarray:
    """4x4 `T_base_cam` from a camera position and where it looks.

    THE only pose constructor in this package. OpenCV convention: the
    rotation's columns are the camera axes in base frame — X right, Y image
    -down, Z boresight.

    Roll is pinned by the world-down rule (image-down ~ world-down, i.e. an
    upright picture), which is not an aesthetic choice: rebuilding all 30
    recorded `T_base_cam` of 2408-cup3 from position + recorded boresight
    with this rule reproduces the recorded rotation to a **median 0.0 deg**.
    The only exceptions are exactly the poses where `cell/geometry.py
    is_half_turn()` is true — the executor's own 180 deg roll for
    reachability, a physically identical view and an execution-time concern,
    not a planning one.

    Degenerate case (looking straight down or straight up) falls back to a
    fixed +Y reference so the frame stays deterministic instead of NaN.
    """
    pos = np.asarray(position, dtype=float).reshape(3)
    z = np.asarray(boresight, dtype=float).reshape(3)
    n = float(np.linalg.norm(z))
    if n < 1e-12:
        raise ValueError("degenerate boresight")
    z = z / n
    d = np.asarray(down, dtype=float).reshape(3)
    y = d - float(d @ z) * z
    if float(np.linalg.norm(y)) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0])
        y = ref - float(ref @ z) * z
    y = y / float(np.linalg.norm(y))
    x = np.cross(y, z)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)                     # re-orthogonalise, kill float drift
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T
