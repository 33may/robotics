"""Inspection loop -> porthole bus. The UI seam.

This is a library, not a participant. It takes plain data — arrays, dicts,
tuples — and turns it into bus messages. It imports nothing from
``inspection.run`` (the existing "nothing imports from run/" contract holds in
both directions), it never decides anything, and it never raises into a caller
that might have a robot in the middle of a move.

Contract it implements: ``inspection/ui/AGENTS.md``.
Wire protocol: ``~/projects/porthole/docs/scene-protocol.md``.

Typical use, from the dispatcher (`run/machine.py`'s `Supervisor`)::

    pub = InspectionPublisher(bus, run_dir=outdir)
    pub.publish_world(world)                        # once, after RobotCell()
    pub.publish_survey_only(survey_state)            # before a sphere exists
    pub.publish_pose(world, q, colliding=...)        # preview replay / live mirror, 30 Hz
    pub.publish_capture(cap)                         # after rig.capture()
    pub.publish_object(acc.points, dims, mid)        # after _recenter()
    pub.publish_views(sphere, reach, visited=visited, blocked=blocked,
                      current=current, captures=captures,
                      pending=pending, previewing=previewing, survey=survey_state)

`replay_path` is the same pose-per-config port of `RobotCell.replay()`, kept
for standalone debug CLIs (design: "meshcat for debug CLIs") — the
dispatcher's own preview loop calls `publish_pose` directly instead, one
config at a time, so it can be cancelled mid-loop.

Every method is best-effort: a UI that is not connected, a closed socket or a
killed browser must never abort a run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from porthole.scene import (
    axes_node,
    box_node,
    mesh_node,
    points_node,
    pose_frame,
    scene_description,
    sphere_node,
)

log = logging.getLogger("inspection.ui")

# Topics. Declared here so there is one list to read, and so a typo is a
# NameError rather than a message nobody receives.
TOPIC_DESCRIPTION = "scene/description"
TOPIC_POSES = "scene/poses"
TOPIC_CLOUD = "cloud/fused"
TOPIC_CAMERA = "camera/wrist"
TOPIC_VIEWS = "views/state"
TOPIC_STATUS = "run/status"
TOPIC_META = "run/meta"
TOPIC_LOG = "log/events"
TOPIC_CHAIN = "chain/latest"
TOPIC_TRACE = "trace/state"

MESH_MOUNT = "/meshes"
CAPTURE_MOUNT = "/captures"

#: Marker colours by cell state. `blocked` and `unreachable` differ on purpose:
#: blocked is "planning refused it from where the arm is standing and it may
#: work after the next move" (loop clears plan_failed on every real move);
#: unreachable is "no IK branch at any roll for this object position".
STATE_COLOR = {
    "visited": "#7ee2a8",
    "current": "#8ab4f8",
    "available": "#6d737d",
    "blocked": "#f2c76b",
    "unreachable": "#5a3a3a",
    "pending": "#b0a0f0",
    "previewing": "#f08bd4",
}

#: Cosmetic only. Mirrors the table in `inspection/cell/viewer.py`, duplicated
#: rather than imported because that module pulls meshcat at import time and
#: this path must not depend on it. If they drift, colours differ; nothing breaks.
CELL_STYLE = [
    ("shelf", "#ff8800", 0.30),
    ("floor", "#556655", 0.25),
    ("bar", "#4a4a4a", 1.0),
    ("pedestal", "#333333", 1.0),
    ("slab", "#8a8a8a", 1.0),
    ("post", "#3f444d", 1.0),
]
CELL_DEFAULT = ("#2288cc", 0.8)

TOOL_COLOR = "#c8a24a"
COLLIDING_COLOR = "#dd2222"
BACKGROUND_BAD = ["#5a2b2b", "#2a1414"]


def _cell_style(name: str) -> tuple[str, float]:
    for key, color, opacity in CELL_STYLE:
        if key in name:
            return color, opacity
    return CELL_DEFAULT


def cell_key(h: int, v: int) -> str:
    """Node path suffix and marker id. Zero-padded so paths sort sensibly."""
    return f"h{int(h):02d}v{int(v)}"


def mesh_dir_for(world) -> Path | None:
    """Directory holding the robot's visual meshes, for `serve_ui(assets=...)`.

    Derived from the model rather than hardcoded: `robot_descriptions` caches
    under a path that depends on its own version and on the machine.
    """
    try:
        parents = {Path(str(g.meshPath)).parent for g in world.visual_model.geometryObjects}
    except Exception:
        log.exception("mesh_dir_for failed")
        return None
    if len(parents) != 1:
        # Every UR5e visual mesh lives in one directory today. If a model ever
        # spreads them, mounting one parent would 404 the rest — say so loudly
        # rather than serving a scene with holes in it.
        log.warning("visual meshes span %d directories: %s", len(parents), parents)
        return None
    return parents.pop()


class InspectionPublisher:
    """Publishes loop state onto a `PortholeBus`. Safe to call from any thread."""

    def __init__(
        self,
        bus,
        run_dir: str | Path | None = None,
        mesh_mount: str = MESH_MOUNT,
        capture_mount: str = CAPTURE_MOUNT,
        max_cloud_points: int = 400_000,
        live_stride: int = 2,
    ) -> None:
        self.bus = bus
        self.run_dir = Path(run_dir).resolve() if run_dir else None
        self.mesh_mount = mesh_mount.rstrip("/")
        self.capture_mount = capture_mount.rstrip("/")
        self.max_cloud_points = max_cloud_points
        #: Downsample factor for the LIVE camera stream only — see
        #: `publish_frame`. 1 disables it (and freezes the 3D on a machine
        #: without hardware compositing).
        self.live_stride = live_stride

        # The description is republished whole on every change (protocol §3), so
        # it is assembled here from independently-updated sections.
        self._sections: dict[str, list[dict]] = {
            "world": [axes_node("world/axes", 0.2)],
            "cell": [],
            "robot": [],
            "tool": [],
            "object": [],
            "views": [],
        }
        self._pose_names: list[str] = []       # robot visual + tool, in publish order
        self._data = None                      # our own pinocchio data: never touch the loop's
        self._visual_data = None
        self._geom_data = None
        self._geom_ngeoms = -1
        self._declared = False

    # ── declaration ────────────────────────────────────────────────────────

    def declare(self) -> None:
        """Declare every topic. Optional — publishing works without it — but the
        declarations are what a developer sees in `$/hello`."""
        try:
            self.bus.declare(TOPIC_DESCRIPTION, qos="stream", kind="scene")
            self.bus.declare(TOPIC_POSES, qos="stream", kind="scene")
            self.bus.declare(TOPIC_CLOUD, qos="stream", kind="ndarray")
            self.bus.declare(TOPIC_CAMERA, qos="stream", kind="image/jpeg")
            self.bus.declare(TOPIC_VIEWS, qos="stream", kind="json")
            self.bus.declare(TOPIC_STATUS, qos="stream", kind="json")
            self.bus.declare(TOPIC_META, qos="stream", kind="json")
            self.bus.declare(TOPIC_LOG, qos="event")
            self.bus.declare(TOPIC_CHAIN, qos="stream", kind="json")
            self.bus.declare(TOPIC_TRACE, qos="stream", kind="json")
            self._declared = True
        except Exception:
            log.exception("declare failed")

    # ── world ──────────────────────────────────────────────────────────────

    def publish_world(self, world) -> None:
        """Static geometry: cell boxes, robot visual meshes, tool envelope.

        Reads `world.geom_model` and `world.visual_model`, both documented as
        public in `cell/world.py`. Robot links use the VISUAL meshes; cell and
        tool boxes come from the collision model, because that is the geometry
        the planner actually respects and showing anything else would be a lie.
        """
        try:
            import pinocchio as pin

            self._data = world.model.createData()
            self._visual_data = world.visual_model.createData()

            cell_nodes, tool_nodes, robot_nodes = [], [], []
            pose_names: list[str] = []

            # Cell boxes are welded to joint 0, so their placement is fixed and
            # belongs in the description rather than the pose stream.
            geom_data = self._fresh_geom_data(world)
            pin.updateGeometryPlacements(
                world.model, self._data, world.geom_model, geom_data,
                np.zeros(world.model.nq),
            )
            for gid, gobj in enumerate(world.geom_model.geometryObjects):
                name = gobj.name
                if name.startswith("cell_"):
                    color, opacity = _cell_style(name)
                    cell_nodes.append(box_node(
                        f"cell/{name[5:]}",
                        2 * np.asarray(gobj.geometry.halfSide),   # halfSide -> full extents
                        geom_data.oMg[gid],
                        color=color, opacity=opacity,
                    ))
                elif name.startswith("tool_"):
                    tool_nodes.append(box_node(
                        f"tool/{name[5:]}",
                        2 * np.asarray(gobj.geometry.halfSide),
                        color=TOOL_COLOR, opacity=0.55,
                    ))
                    pose_names.append(f"tool/{name[5:]}")

            for gobj in world.visual_model.geometryObjects:
                url = f"{self.mesh_mount}/{Path(str(gobj.meshPath)).name}"
                scale = [float(s) for s in gobj.meshScale]
                robot_nodes.append(mesh_node(
                    f"robot/{gobj.name}", url,
                    scale=None if np.allclose(scale, 1.0) else scale,
                ))
                pose_names.append(f"robot/{gobj.name}")

            self._sections["cell"] = cell_nodes
            self._sections["tool"] = tool_nodes
            self._sections["robot"] = robot_nodes
            self._pose_names = pose_names
            self._publish_description()
        except Exception:
            log.exception("publish_world failed")

    def _fresh_geom_data(self, world):
        """Our own geom data, rebuilt when the world gains or loses geometry.

        `world.set_object` recreates the world's geom_data; a cached copy of our
        own would silently index past the end after that.
        """
        import pinocchio as pin  # noqa: F401 - imported for the caller's benefit

        if self._geom_data is None or self._geom_ngeoms != world.geom_model.ngeoms:
            self._geom_data = world.geom_model.createData()
            self._geom_ngeoms = world.geom_model.ngeoms
        return self._geom_data

    # ── object ─────────────────────────────────────────────────────────────

    def publish_object(
        self,
        points: np.ndarray | None = None,
        dims: Sequence[float] | None = None,
        center: Sequence[float] | None = None,
    ) -> None:
        """The reconstructed object: its collision box and its fused cloud.

        `dims`/`center` are the same values handed to `world.set_object`. Pass
        them explicitly rather than reading them back — the world's object
        registry is private, and this way the UI shows exactly what the planner
        was told.
        """
        try:
            nodes: list[dict] = []
            if dims is not None and center is not None:
                T = np.eye(4)
                T[:3, 3] = np.asarray(center, dtype=float)
                nodes.append(box_node(
                    "object/obb", dims, T,
                    color="#ddcc22", opacity=0.22, side="double",
                ))
            # The cloud is referenced, not carried: the description is
            # republished whenever any marker changes colour.
            nodes.append(points_node(
                "object/cloud", TOPIC_CLOUD,
                size=0.0025, color="#c8d4e4", max_points=self.max_cloud_points,
            ))
            self._sections["object"] = nodes
            self._publish_description()

            if points is not None:
                self.publish_cloud(points)
        except Exception:
            log.exception("publish_object failed")

    def publish_cloud(self, points: np.ndarray) -> None:
        """Fused object cloud, base frame, float32 [N,3]."""
        try:
            array = np.ascontiguousarray(points, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"cloud must be [N,3], got {array.shape}")
            if len(array) > self.max_cloud_points:
                array = array[: self.max_cloud_points]
            self.bus.publish(TOPIC_CLOUD, {"points": self.bus.array_payload(array)})
        except Exception:
            log.exception("publish_cloud failed")

    # ── viewsphere ─────────────────────────────────────────────────────────

    def publish_views(
        self,
        sphere,
        reach: Mapping[tuple, Any],
        visited: Iterable[tuple] = (),
        blocked: Iterable[tuple] = (),
        current: tuple | None = None,
        captures: Mapping[tuple, Mapping[str, Any]] | None = None,
        glosses: Mapping[tuple, str] | None = None,
        pending: tuple | None = None,
        previewing: tuple | None = None,
        survey: str = "visited",
    ) -> None:
        """Publish the viewsphere twice: as 3D markers and as clickable data.

        Both come out of this one call so they cannot disagree. `glosses` is
        passed in rather than computed — the egocentric wording lives in
        `run/decider.py`, and nothing here imports from `run/`.

        `captures` maps a cell to `{"step": int, "dir": str, "comment": str}`;
        `dir` is a capture directory on disk, which is turned into a URL under
        the capture mount.

        `pending`/`previewing` mark at most one cell each — the target of an
        in-flight plan, or a looping preview replay awaiting `view/confirm` —
        and take priority over every other state for that cell. `survey`
        carries the survey button's own state (`available|pending|previewing|
        visited`) alongside the sphere cells, in the same top-level payload,
        so the actions panel can render it from one topic.
        """
        try:
            visited_set = {tuple(c) for c in visited}
            blocked_set = {tuple(c) for c in blocked}
            current_cell = tuple(current) if current is not None else None
            pending_cell = tuple(pending) if pending is not None else None
            previewing_cell = tuple(previewing) if previewing is not None else None
            captures = captures or {}
            glosses = glosses or {}

            center = np.asarray(sphere.center, dtype=float)
            markers, cells = [], []

            for h, v in sphere.cells():
                cell = (h, v)
                position = center + sphere.cell_dir(h, v) * sphere.r
                state = self._cell_state(
                    cell, reach, visited_set, blocked_set, current_cell,
                    pending_cell, previewing_cell)

                T = np.eye(4)
                T[:3, 3] = position
                markers.append(sphere_node(
                    f"views/{cell_key(h, v)}",
                    0.012 if state not in ("current", "previewing") else 0.02,
                    T,
                    color=STATE_COLOR[state],
                    opacity=0.35 if state == "unreachable" else 0.95,
                ))

                entry: dict[str, Any] = {
                    "h": h, "v": v, "state": state,
                    "pos": [float(x) for x in position],
                }
                if cell in glosses:
                    entry["gloss"] = glosses[cell]
                capture = captures.get(cell)
                if capture:
                    if capture.get("step") is not None:
                        entry["step"] = int(capture["step"])
                    if capture.get("comment"):
                        entry["comment"] = capture["comment"]
                    url = self._capture_url(capture.get("dir"))
                    if url:
                        entry["image"] = url
                cells.append(entry)

            self._sections["views"] = markers
            self._publish_description()

            self.bus.publish(TOPIC_VIEWS, {
                "center": [float(x) for x in center],
                "radius": float(sphere.r),
                "current": list(current_cell) if current_cell else None,
                "cells": cells,
                "survey": {"state": survey},
            })
        except Exception:
            log.exception("publish_views failed")

    def publish_survey_only(self, state: str) -> None:
        """Publish survey state without sphere data. Used before boot capture."""
        try:
            self.bus.publish(TOPIC_VIEWS, {
                "survey": {"state": state},
                "center": None,
                "radius": None,
                "current": None,
                "cells": [],
            })
        except Exception:
            log.exception("publish_survey_only failed")

    @staticmethod
    def _cell_state(cell, reach, visited, blocked, current, pending=None, previewing=None) -> str:
        if cell == previewing:
            return "previewing"
        if cell == pending:
            return "pending"
        if cell == current:
            return "current"
        if cell in visited:
            return "visited"
        if reach.get(cell) is None:
            return "unreachable"
        if cell in blocked:
            return "blocked"
        return "available"

    def _capture_url(self, directory: str | Path | None) -> str | None:
        """Capture directory on disk -> URL under the capture mount."""
        if directory is None or self.run_dir is None:
            return None
        try:
            relative = Path(directory).resolve().relative_to(self.run_dir)
        except ValueError:
            log.warning("capture %s is outside run_dir %s", directory, self.run_dir)
            return None
        return f"{self.capture_mount}/{relative.as_posix()}/rgb.png"

    def _asset_url(self, relative: str) -> str | None:
        """A run-relative path from the trace -> URL under the capture mount.

        The brain writes paths, not URLs: it has no idea a UI exists. The
        mapping belongs here, and so does the refusal — anything absolute or
        climbing out of the run dir is dropped rather than served.
        """
        rel = str(relative).strip()
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            return None
        return f"{self.capture_mount}/{rel}"

    def publish_trace(self, events: list[dict]) -> None:
        """The whole run trace, RETAINED, image paths rewritten as URLs.

        Whole-state rather than per-event: the bus rule is "publish state, not
        deltas" (AGENTS.md), and it buys the thing that matters here — a panel
        opened halfway through a run, or three days later, sees the entire
        thought stream immediately instead of only what arrives next. Runs are
        5-30 steps, so republishing all of it on every append is cheap.
        """
        try:
            out = []
            for ev in events:
                ev = dict(ev)
                if ev.get("images"):
                    ev["images"] = [u for u in (self._asset_url(p)
                                                for p in ev["images"]) if u]
                # `sub_step` carries ONE image — a crop announced while the
                # subagent is still running. Same mapping, or it streams a
                # path the browser cannot fetch.
                if isinstance(ev.get("image"), str):
                    ev["image"] = self._asset_url(ev["image"])
                sub = ev.get("sub")
                if isinstance(sub, dict):
                    sub = dict(sub)
                    if sub.get("image"):
                        sub["image"] = self._asset_url(sub["image"])
                    sub["turns"] = [
                        {**t, "image": self._asset_url(t["image"])}
                        if isinstance(t, dict) and t.get("image") else t
                        for t in sub.get("turns", [])]
                    ev["sub"] = sub
                out.append(ev)
            self.bus.publish(TOPIC_TRACE, {"events": out})
        except Exception:
            log.exception("publish_trace failed")

    def publish_chain(self, directory, files: Mapping[str, str],
                      **stats: Any) -> None:
        """The identity chain for the newest capture: four images + the numbers.

        RETAINED, and it carries URLs rather than pixels — the images are
        static files the browser fetches over the same mount as every other
        capture image and caches off the main thread (AGENTS.md §7). A panel
        opened mid-run therefore shows the last chain immediately.

        `files` maps stage -> filename inside the capture dir; `rgb` is added
        here because it is the capture's own image and no one rewrites it.
        Stages that did not happen (a view that fell back to depth growth has
        no box and no mask) are simply absent — the panel renders what exists
        rather than inventing a placeholder.
        """
        try:
            base = self._capture_url(directory)
            if base is None:
                return
            base = base.rsplit("/", 1)[0]
            images = {"rgb": f"{base}/rgb.png"}
            images.update({stage: f"{base}/{name}"
                           for stage, name in dict(files).items()})
            self.bus.publish(TOPIC_CHAIN, {"images": images, **stats})
        except Exception:
            log.exception("publish_chain failed")

    # ── motion ─────────────────────────────────────────────────────────────

    def publish_pose(self, world, q, colliding: bool = False) -> None:
        """One frame: every moving node's absolute placement at configuration q."""
        try:
            self.bus.publish(TOPIC_POSES, self._pose_payload(world, q, colliding))
        except Exception:
            log.exception("publish_pose failed")

    def _pose_payload(self, world, q, colliding: bool) -> dict:
        import pinocchio as pin

        q = np.asarray(q, dtype=float)
        if self._data is None:
            self._data = world.model.createData()
            self._visual_data = world.visual_model.createData()

        pin.forwardKinematics(world.model, self._data, q)
        pin.updateGeometryPlacements(
            world.model, self._data, world.visual_model, self._visual_data)
        geom_data = self._fresh_geom_data(world)
        pin.updateGeometryPlacements(
            world.model, self._data, world.geom_model, geom_data)

        poses: dict[str, Any] = {}
        for gid, gobj in enumerate(world.visual_model.geometryObjects):
            poses[f"robot/{gobj.name}"] = self._visual_data.oMg[gid]
        for gid, gobj in enumerate(world.geom_model.geometryObjects):
            if gobj.name.startswith("tool_"):
                poses[f"tool/{gobj.name[5:]}"] = geom_data.oMg[gid]

        overrides = None
        background = None
        if colliding:
            # The tint travels with the pose it describes: on a separate topic a
            # dropped message would leave the tool green inside the table.
            overrides = {
                name: {"color": COLLIDING_COLOR, "opacity": 0.75}
                for name in poses if name.startswith("tool/")
            }
            background = BACKGROUND_BAD
        return pose_frame(poses, overrides=overrides, background=background)

    def replay_path(self, world, path, hz: float = 30.0, step: float | None = None) -> bool:
        """Animate a planned path. The port of `RobotCell.replay()`.

        Freezes at the first colliding configuration exactly as meshcat does
        today — it stops publishing and leaves the retained frame showing the
        collision, so a UI that connects afterwards still sees why the path was
        refused. Returns True if the path replayed clean.

        Blocking, like `world.replay()`, and not used by the run path — the
        dispatcher's cancellable preview loop calls `publish_pose` per config
        instead. This is for the standalone debug CLIs.
        """
        try:
            from inspection.cell.world import DEFAULT_STEP

            dense = world.discretize(path, DEFAULT_STEP if step is None else step)
            period = 1.0 / max(hz, 1.0)
            for q in dense:
                bad = world.is_colliding(q)
                self.publish_pose(world, q, colliding=bad)
                if bad:
                    self.log("error", "preview: path INVALID — frozen at impact")
                    return False
                time.sleep(period)
            return True
        except Exception:
            log.exception("replay_path failed")
            return False

    # ── camera, status, log ────────────────────────────────────────────────

    def publish_capture(self, capture: Mapping[str, Any], quality: int = 80) -> None:
        """Push a capture bundle's RGB frame to the camera topic."""
        try:
            rgb = capture.get("rgb")
            if rgb is None:
                return
            self.bus.publish(TOPIC_CAMERA, self.bus.jpeg_payload(
                np.ascontiguousarray(rgb), quality=quality))
        except Exception:
            log.exception("publish_capture failed")

    def publish_frame(self, rgb, quality: int = 70) -> None:
        """Push one live camera frame, downscaled. Same topic as `publish_capture`.

        The live view is downscaled and the SAVED capture is not: this is a
        monitor, and the frames that matter for the record go to disk at full
        resolution through `publish_capture`/`save_bundle`.

        Measured on this box: streaming 848x480 at 10 Hz froze the 3D panel
        (0.17% of its pixels changing per second, versus 4.2% with the camera
        off) while poses kept arriving at a healthy 30 Hz. The window is
        QtWebEngine on a software Vulkan fallback ("GBM is not supported",
        printed at every startup), so each full-size frame decoded into a 2D
        canvas competes with the WebGL canvas for the same path and the robot
        stops moving on screen. A quarter of the pixels is the difference
        between a live 3D view and a frozen one.
        """
        try:
            frame = np.ascontiguousarray(rgb)
            if self.live_stride > 1 and frame.ndim >= 2:
                # Stride sampling rather than an interpolating resize: it costs
                # nothing, needs no image library on this path, and the live
                # view is for a human to glance at, not to measure from.
                frame = np.ascontiguousarray(
                    frame[::self.live_stride, ::self.live_stride])
            self.bus.publish(TOPIC_CAMERA, self.bus.jpeg_payload(
                frame, quality=quality))
        except Exception:
            log.exception("publish_frame failed")

    def publish_run_meta(
        self,
        source: str,
        name: str,
        object: str | None = None,
        question: str | None = None,
    ) -> None:
        """Retained run identity — RETAINED, so a UI that connects late still
        knows what it is looking at without waiting for the next event.

        Published once at boot, before anything else that depends on it, and
        again whenever the question is set (`brain/ask`). `source` is what the
        frontend gates on: `"data-engine"` runs have no brain in the loop, so
        the trace panel and the Ask composer render nothing for them — see
        `InspectionApp.tsx`. `"live"` is every run driven by `run/machine.py`'s
        `Supervisor` with a brain attached.
        """
        try:
            if source not in ("live", "data-engine"):
                log.warning("publish_run_meta: unexpected source %r", source)
            self.bus.publish(TOPIC_META, {
                "source": source,
                "name": name,
                "object": object,
                "question": question,
            })
        except Exception:
            log.exception("publish_run_meta failed")

    def status(self, **fields: Any) -> None:
        """Merge fields into the retained status strip."""
        try:
            self._status = {**getattr(self, "_status", {}), **fields}
            self.bus.publish(TOPIC_STATUS, self._status)
        except Exception:
            log.exception("status failed")

    def log(self, level: str, msg: str) -> None:
        try:
            self.bus.publish(TOPIC_LOG, {"level": level, "msg": msg})
        except Exception:
            log.exception("log failed")

    # ── internals ──────────────────────────────────────────────────────────

    def _publish_description(self) -> None:
        """Republish the WHOLE description. Required by the protocol: retention
        keeps one message, so a partial description is what a reconnecting UI
        would believe the world to be."""
        nodes = [node for section in self._sections.values() for node in section]
        if not nodes:
            return
        self.bus.publish(TOPIC_DESCRIPTION, scene_description(nodes, frame="base"))
