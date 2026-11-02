"""The primitive-fit contract: `Primitive`, `Decomposition`, and their JSON.

Kinds are CLOSED. A fitter that cannot express its answer as one of

    box · cylinder · plane_patch · superquadric · mesh

must degrade to `mesh` rather than invent a sixth kind — that is the promise
that lets the Rerun workspace and the later best-view derivation switch on
`kind` alone and never learn about methods.

Every kind poses the same way: a 4x4 row-major `pose` in the robot base
frame, Z-up, whose rotation columns are the primitive's own axes and whose
translation is its centre. `params` is the only per-kind part:

    box           extents      [3]   full edge lengths, along the pose axes
    cylinder      radius, length     axis is the pose's +Z column
    plane_patch   extents      [2]   in-plane, along the pose's X and Y
    superquadric  scale [3], eps [2] scale is SEMI-axes (the EMS convention)
    mesh          vertices, faces    inline, base frame, pose applied already

Serialisation is total and lossless: `decomposition_from_json(
decomposition_to_json(d)) == d`. Floats are plain Python floats and indices
plain ints so `json.dump` needs no custom encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

KINDS = ("box", "cylinder", "plane_patch", "superquadric", "mesh")

#: required `params` keys per kind — enforced at construction, because a
#: renderer that trusts the vocabulary must never meet a half-filled one.
PARAM_KEYS: Dict[str, tuple] = {
    "box": ("extents",),
    "cylinder": ("radius", "length"),
    "plane_patch": ("extents",),
    "superquadric": ("scale", "eps"),
    "mesh": ("vertices", "faces"),
}


def _pose_list(pose) -> List[List[float]]:
    T = np.asarray(pose, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {T.shape}")
    return [[float(v) for v in row] for row in T]


@dataclass
class Primitive:
    """One shape in a decomposition, addressed by its geometric content."""

    kind: str
    pose: List[List[float]]
    params: Dict[str, Any]
    label: str
    inliers: List[int] = field(default_factory=list)
    residual_mm: float = 0.0
    support: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not in closed set {KINDS}")
        missing = [k for k in PARAM_KEYS[self.kind] if k not in self.params]
        if missing:
            raise ValueError(f"{self.kind} params missing {missing}")
        self.pose = _pose_list(self.pose)
        self.params = _jsonify(self.params)
        self.inliers = [int(i) for i in self.inliers]
        self.residual_mm = float(self.residual_mm)
        self.support = int(self.support) or len(self.inliers)

    # -- convenience for consumers that want matrices back ----------------
    @property
    def T(self) -> np.ndarray:
        return np.array(self.pose, dtype=float)

    @property
    def axes(self) -> np.ndarray:
        """3x3 rotation; columns are the primitive's own X, Y, Z axes."""
        return self.T[:3, :3]

    @property
    def center(self) -> np.ndarray:
        return self.T[:3, 3]

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "pose": self.pose, "params": self.params,
                "label": self.label, "inliers": self.inliers,
                "residual_mm": self.residual_mm, "support": self.support}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Primitive":
        return Primitive(kind=d["kind"], pose=d["pose"], params=d["params"],
                         label=d["label"], inliers=d.get("inliers", []),
                         residual_mm=d.get("residual_mm", 0.0),
                         support=d.get("support", 0))


@dataclass
class Decomposition:
    """What one method says about one cloud.

    `abstain=True` is a first-class answer, not a failure: a 200-point stub
    of a cup has no honest cylinder in it, and emitting one anyway is the
    exact garbage this layer exists to avoid. When abstaining, `primitives`
    is empty and `reason` says why in one human sentence.
    """

    method: str
    version: str
    abstain: bool = False
    reason: str = ""
    primitives: List[Primitive] = field(default_factory=list)
    coverage: float = 0.0
    residual_mm: float = 0.0
    n_primitives: int = 0

    def __post_init__(self):
        self.abstain = bool(self.abstain)
        self.coverage = float(self.coverage)
        self.residual_mm = float(self.residual_mm)
        self.n_primitives = len(self.primitives)
        if self.abstain and self.primitives:
            raise ValueError("an abstaining decomposition carries no primitives")

    def to_json(self) -> Dict[str, Any]:
        return {"method": self.method, "version": self.version,
                "abstain": self.abstain, "reason": self.reason,
                "primitives": [p.to_json() for p in self.primitives],
                "coverage": self.coverage, "residual_mm": self.residual_mm,
                "n_primitives": self.n_primitives}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Decomposition":
        return Decomposition(
            method=d["method"], version=d["version"],
            abstain=d.get("abstain", False), reason=d.get("reason", ""),
            primitives=[Primitive.from_json(p) for p in d.get("primitives", [])],
            coverage=d.get("coverage", 0.0),
            residual_mm=d.get("residual_mm", 0.0),
            n_primitives=d.get("n_primitives", 0))


def decomposition_to_json(d: Decomposition) -> Dict[str, Any]:
    return d.to_json()


def decomposition_from_json(d: Dict[str, Any]) -> Decomposition:
    return Decomposition.from_json(d)


def abstained(method: str, version: str, reason: str) -> Decomposition:
    """The canonical refusal. Every fitter returns one of these or a fit."""
    return Decomposition(method=method, version=version, abstain=True,
                         reason=reason)


def _jsonify(obj):
    """numpy -> plain python, recursively, so `json.dump` needs no encoder."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def pose_from(axes: Sequence[Sequence[float]], center: Sequence[float]) -> np.ndarray:
    """4x4 from a 3x3 axis matrix (columns = axes) and a centre.

    The axis matrix is re-orthonormalised and made right-handed, because a
    least-squares axis plus two arbitrary tangents is rarely either.
    """
    R = np.asarray(axes, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(center, dtype=float)
    return T


def frame_from_axis(axis: Sequence[float], center: Sequence[float]) -> np.ndarray:
    """4x4 whose +Z is `axis` — the pose convention for cylinders.

    The X/Y tangents are chosen deterministically (from the world axis least
    parallel to `axis`) so the same cylinder always gets the same frame; a
    random tangent would make two identical fits look different on disk.
    """
    z = np.asarray(axis, dtype=float)
    n = np.linalg.norm(z)
    if n < 1e-12:
        raise ValueError("degenerate axis")
    z = z / n
    ref = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return pose_from(np.column_stack([x, y, z]), center)
