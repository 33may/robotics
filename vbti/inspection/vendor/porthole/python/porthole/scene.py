"""Payload builders for the 3D scene protocol.

Implements `docs/scene-protocol.md`. Like the rest of Porthole this is
domain-agnostic: it knows about boxes, meshes and 4x4 transforms, and nothing
about robots, links or joints.

A backend uses it in two places::

    from porthole import PortholeBus
    from porthole.scene import box_node, mesh_node, pose_frame, scene_description

    bus.declare("scene/description", qos="stream", kind="scene")
    bus.declare("scene/poses", qos="stream", kind="scene")

    # once, and again whenever the world is re-modelled
    bus.publish("scene/description", scene_description([
        box_node("cell/table", size=(0.53, 1.0, 0.03), transform=T_table,
                 color="#8a8a8a"),
        mesh_node("robot/wrist_3", url="/meshes/ur5e/wrist3.dae",
                  transform=T_wrist),
    ]))

    # per animation frame
    bus.publish("scene/poses", pose_frame({name: T for name, T in placements}))

`pose_frame` is the hot path and is written to be cheap: one float32 array of
shape [N, 4, 4] and a list of names, which for a UR5e is under a kilobyte.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import codec

Transform = Any  # a 4x4 array-like, or a flat 16-element row-major sequence


def _flat16(transform: Transform | None) -> list[float]:
    """Normalise a transform to 16 row-major floats.

    Accepts a 4x4 nested sequence, a numpy 4x4, a `pinocchio.SE3`, or a flat
    16-element sequence. Anything else is a mistake worth failing on loudly:
    silently accepting a 3x4 would place geometry somewhere plausible and wrong.
    """
    if transform is None:
        return [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]

    rotation = getattr(transform, "rotation", None)
    if rotation is not None and hasattr(transform, "translation"):
        import numpy as np

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = transform.translation
        return [float(v) for v in matrix.reshape(16)]

    values = list(transform)
    if len(values) == 16 and not hasattr(values[0], "__len__"):
        return [float(v) for v in values]
    if len(values) == 4:
        flat = [float(v) for row in values for v in row]
        if len(flat) == 16:
            return flat
    raise ValueError(
        f"transform must be 4x4 or 16 row-major values, got {len(values)} rows"
    )


def _material(
    color: str | None,
    opacity: float | None,
    wireframe: bool,
    side: str | None,
) -> dict | None:
    material: dict[str, Any] = {}
    if color is not None:
        material["color"] = color
    if opacity is not None:
        material["opacity"] = float(opacity)
    if wireframe:
        material["wireframe"] = True
    if side is not None:
        material["side"] = side
    return material or None


def _node(path: str, geometry: dict, transform, material, visible: bool) -> dict:
    node: dict[str, Any] = {
        "path": path,
        "geometry": geometry,
        "transform": _flat16(transform),
    }
    if material is not None:
        node["material"] = material
    if not visible:
        node["visible"] = False
    return node


def box_node(
    path: str,
    size: Sequence[float],
    transform: Transform | None = None,
    *,
    color: str | None = None,
    opacity: float | None = None,
    wireframe: bool = False,
    side: str | None = None,
    visible: bool = True,
) -> dict:
    """A box centred on the node origin.

    ``size`` is FULL extents — what a tape measure reads and what a URDF or a
    cell file records. `pinocchio`'s ``Box.halfSide`` is the odd one out, so a
    backend reading geometry from there must double it.
    """
    if len(size) != 3:
        raise ValueError(f"box size must be 3 values, got {len(size)}")
    return _node(
        path,
        {"kind": "box", "size": [float(v) for v in size]},
        transform,
        _material(color, opacity, wireframe, side),
        visible,
    )


def sphere_node(
    path: str,
    radius: float,
    transform: Transform | None = None,
    *,
    color: str | None = None,
    opacity: float | None = None,
    visible: bool = True,
) -> dict:
    return _node(
        path,
        {"kind": "sphere", "radius": float(radius)},
        transform,
        _material(color, opacity, False, None),
        visible,
    )


def cylinder_node(
    path: str,
    radius: float,
    length: float,
    transform: Transform | None = None,
    *,
    color: str | None = None,
    opacity: float | None = None,
    visible: bool = True,
) -> dict:
    """A cylinder whose axis is +Z, matching every robotics geometry library."""
    return _node(
        path,
        {"kind": "cylinder", "radius": float(radius), "length": float(length)},
        transform,
        _material(color, opacity, False, None),
        visible,
    )


def axes_node(
    path: str,
    size: float = 0.15,
    transform: Transform | None = None,
    *,
    visible: bool = True,
) -> dict:
    """An origin triad: R=X, G=Y, B=Z. The cheapest orientation sanity check."""
    return _node(path, {"kind": "axes", "size": float(size)}, transform, None, visible)


def mesh_node(
    path: str,
    url: str,
    transform: Transform | None = None,
    *,
    scale: Sequence[float] | None = None,
    color: str | None = None,
    opacity: float | None = None,
    visible: bool = True,
) -> dict:
    """A mesh fetched over HTTP from the server that serves the UI.

    ``url`` is a path on that server, not a filesystem path — mount the
    directory with ``serve_ui(dist, assets={"/meshes/ur5e": ...})`` and publish
    ``/meshes/ur5e/wrist3.dae``.

    Passing ``color`` replaces the materials baked into the file, which is how a
    link gets tinted without shipping a second mesh. Omit it to keep them.
    """
    geometry: dict[str, Any] = {"kind": "mesh", "url": url}
    if scale is not None:
        geometry["scale"] = [float(v) for v in scale]
    return _node(
        path,
        geometry,
        transform,
        _material(color, opacity, False, None),
        visible,
    )


def points_node(
    path: str,
    topic: str,
    *,
    field: str = "points",
    size: float = 0.003,
    max_points: int = 200_000,
    color: str | None = None,
    transform: Transform | None = None,
    visible: bool = True,
) -> dict:
    """A point cloud whose data streams on ``topic``, not in the description.

    The node says a cloud lives here; the cloud arrives as
    ``{field: array_payload(points)}`` on ``topic``.

    **Why the indirection:** a description is republished in full whenever
    anything in the world changes, and a fused cloud is megabytes. Inlining it
    would resend the entire cloud every time a marker changes colour. This also
    lets a dedicated cloud panel read the same topic without it being published
    twice.
    """
    geometry: dict[str, Any] = {
        "kind": "points",
        "topic": topic,
        "field": field,
        "size": float(size),
        "maxPoints": int(max_points),
    }
    return _node(
        path,
        geometry,
        transform,
        _material(color, None, False, None),
        visible,
    )


def scene_description(nodes: Sequence[dict], frame: str = "base") -> dict:
    """The complete node set. Publish all of it, every time.

    Retention keeps the last message only, so an incremental "add this node"
    message would leave a reconnecting UI with a scene missing everything
    published before it. Nodes absent from a new description are removed.
    """
    paths = [node["path"] for node in nodes]
    duplicates = {path for path in paths if paths.count(path) > 1}
    if duplicates:
        raise ValueError(f"duplicate node paths in scene description: {sorted(duplicates)}")
    return {"frame": frame, "nodes": list(nodes)}


def pose_frame(
    poses: Mapping[str, Transform],
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    background: Sequence[str] | None = None,
) -> dict:
    """One animation frame: where every moving node is right now.

    ``overrides`` are per-frame material changes — a red tint on a colliding
    link, say — and they are NOT sticky: a node not mentioned reverts to its
    described material. That means a backend cannot forget to clear a tint, and
    it is why the tint travels in the same message as the pose it belongs to.
    A separate topic would let a dropped message leave a green robot standing
    inside a table, and `stream` QoS drops messages by design.

    ``background`` is ``[top, bottom]`` — an unmissable, glanceable verdict.
    """
    import numpy as np

    names = list(poses.keys())
    if not names:
        matrices = np.zeros((0, 4, 4), dtype=np.float32)
    else:
        matrices = np.asarray(
            [_flat16(poses[name]) for name in names], dtype=np.float32
        ).reshape(len(names), 4, 4)

    frame: dict[str, Any] = {
        "names": names,
        "transforms": codec.array_payload(matrices),
    }
    if overrides:
        frame["overrides"] = {path: dict(spec) for path, spec in overrides.items()}
    if background is not None:
        frame["background"] = list(background)
    return frame
