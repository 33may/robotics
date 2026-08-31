"""RobotCell — the boolean collision world for the UR5e inspection cell (MAY-183).

One object, one truth: every motion tier (direct move, radial retract, RRT)
asks the same two questions — `is_colliding(q)` and `path_valid(path)` —
against the same geometry: UR5e collision meshes + measured tool envelope
welded to the flange + calibrated cell boxes from cell.yaml (bounds are
modeled as boxes too — e.g. the floor slab; there is no keep-in mechanism).

Usage (from repo root, robo env active):
    p inspection/cell/world.py            # headless smoke: truth table + timings
    p inspection/cell/world.py --demo     # meshcat: show truth-table poses colored
    p inspection/cell/world.py --replay   # meshcat: replay a path, freeze at impact

Design contract (do not break — MAY-184 builds on it):
- pinocchio internals are PUBLIC: .model .data .geom_model .geom_data — used by
  the viewer, bench and diagnostics. The PLANNER does not touch them: OMPL asks
  `is_colliding` through a validity callback, which is what lets one world with
  split margins serve both planning and validation.
- padding/step are query-time knobs. Resolution contract: step_L1 * LEVER_M
  must stay < padding — enforced here, not assumed.
- The inspected object IS a hard obstacle (Anton 2026-08-18, reverses the
  MAY-183 soft-object rule): `set_object` swaps the reconstructed primitive
  into the collision world every perception step. We look, we never touch —
  endpoint validity at env padding doubles as the standoff.
- World answers, never decides: no planning logic in this file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import pinocchio as pin
import coal

CELL_YAML = Path(__file__).parent / "cell.yaml"

# --- resolution-margin contract (MAY-181): step_L1 [rad] * lever [m] < padding [m]
LEVER_M = 1.10          # base-axis -> closed-tip worst-case sweep radius
DEFAULT_STEP = 0.015    # rad, L1  -> 16.5 mm sweep, under the 20 mm padding
DEFAULT_PADDING = 0.020  # m, coal security_margin, environment pairs
# Self/tool-vs-robot pairs use a smaller margin: the UR5e's own design
# clearances at park are 17-19 mm (base/upper-arm, wrist1/wrist3 measured),
# so a 20 mm blanket margin false-positives the HOME pose. Links are thick —
# 5 mm margin + the step contract is safe against tunneling in practice.
SELF_PADDING = 0.005

UR5E_HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# --- taught safety keep-outs (freedrive taps, 2026-08-31) -------------------
# Walls (`planes:`) become thick slabs with one face exactly ON the plane —
# not coal Halfspaces, whose narrow phase vs the UR5e BVH link meshes is not
# a path we have validated; slabs reuse the box pipeline end to end.
# Corners (`corners:`) become ONE convex prism each: the keep-out is the
# INTERSECTION of the half-spaces behind the two walls (Anton 2026-08-31) —
# two separate geoms would OR them and eat the free space beside each wall.
# Sizing: a keep-out only has to cover (true obstacle ∩ reachable set), and
# nothing the arm+tool can touch lies outside a ~1.35 m ball around the base
# (0.85 reach + wrist stack + 0.24 tool, stretched straight). Shapes are cut
# just past that — big enough to be safe, small enough to look like the room.
_WORKSPACE_MID = np.array([0.0, 0.0, 0.65])  # slab faces center on this
PLANE_SLAB = (3.0, 1.5, 0.3)   # wall panel: 3 m along, z -0.1..1.4, 0.3 thick
PRISM_L = 2.4                  # wedge clip length: > max apex-to-base distance
                               # (0.90 m, corner4) + the 1.35 m reach ball
PRISM_Z = (-0.10, 1.40)        # corner prism vertical extent, base frame [m]
_PRISM_TRIS = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5),
               (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
               (3, 7, 4), (3, 4, 0)]


REACH_BALL = 1.35  # m — nothing on the arm+tool can leave this ball (0.85
                   # reach + wrist stack + 0.24 tool, stretched straight)


def _wall_basis(n):
    """Orthonormal (x along the wall ~horizontal, y up the wall, z = normal)."""
    n = n / np.linalg.norm(n)
    seed = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(seed, n)
    x /= np.linalg.norm(x)
    return x, np.cross(n, x), n


def build_keepouts(cfg, frame_T):
    """cell.yaml keep-outs -> concrete shapes, shared by the collision world
    and the viewer (what you see IS what the solver checks). frame_T maps a
    frame name to its 4x4 base transform.

    Returns (panels, prisms): panels = (name, R, center, dims) boxes,
    prisms = (name, 8 wedge vertices).

    Wall panels are mutually CLIPPED at each other's plane so walls meet in a
    clean corner instead of crossing, and EVERYTHING is cut to the top-view
    bounding square: wall outer faces on the walled sides, the reach limit on
    the open sides (Anton 2026-08-31 — the model should look like the room).
    Beyond the square nothing is reachable without first crossing a kept
    shape, which the dense path check catches; single-config checks lose
    coverage only outside the square."""
    walls = []
    for name, pl in cfg.get("planes", {}).items():
        T = frame_T(pl.get("parent", "base"))
        n = T[:3, :3] @ np.asarray(pl["normal"], dtype=float)
        walls.append((name, T[:3, :3] @ np.asarray(pl["point"], float) + T[:3, 3],
                      n / np.linalg.norm(n)))

    # top-view bounding square: wall outer faces where walls exist, the reach
    # limit on the open sides — nothing may stick out of it
    bounds = np.array([-REACH_BALL, REACH_BALL, -REACH_BALL, REACH_BALL])
    for _, p, n in walls:
        outer = p - n * PLANE_SLAB[2]
        if abs(n[0]) > 0.9:
            bounds[0 if n[0] > 0 else 1] = outer[0]
        elif abs(n[1]) > 0.9:
            bounds[2 if n[1] > 0 else 3] = outer[1]

    def clipped_prism(name, verts8):
        """Cut a wedge prism to the square; re-hull the survivors."""
        bot = _clip_poly_xy(verts8[:4], bounds)
        top = _clip_poly_xy(verts8[4:], bounds)
        if len(bot) < 3 or len(top) < 3:
            return None
        from scipy.spatial import ConvexHull
        pts = np.array(bot + top)
        return name, pts, ConvexHull(pts, qhull_options="QJ").simplices

    panels, prisms = [], []
    thick = PLANE_SLAB[2]
    for name, p, n in walls:
        x, y, n = _wall_basis(n)
        foot = _WORKSPACE_MID - n * ((_WORKSPACE_MID - p) @ n)
        d0 = min(abs(p @ n), REACH_BALL)          # base-to-plane distance
        half = max(np.sqrt(REACH_BALL ** 2 - d0 ** 2), 0.3)
        s_min, s_max = -half, half                # extent along x, from foot
        for name2, p2, n2 in walls:
            if name2 == name:
                continue
            slope = x @ n2
            if abs(slope) < 1e-3:                 # ~parallel, no corner
                continue
            s_star = ((p2 - foot) @ n2) / slope   # crossing of the other plane
            if slope > 0:
                s_min = max(s_min, s_star)
            else:
                s_max = min(s_max, s_star)
        if s_max - s_min < 0.05:                  # fully behind another wall
            continue
        center = foot + x * ((s_min + s_max) / 2) - n * (thick / 2)
        panels.append((f"plane_{name}", np.column_stack([x, y, n]), center,
                       (s_max - s_min, PLANE_SLAB[1], thick)))

    # auto room corners: the region behind BOTH walls of a reachable pair
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            (name_a, pa, na), (name_b, pb, nb) = walls[i], walls[j]
            d = np.cross(na, nb)
            if np.linalg.norm(d) < 0.1:
                continue
            d /= np.linalg.norm(d)
            # a point on the corner line (component along d pinned to mid)
            A = np.vstack([na, nb, d])
            q0 = np.linalg.solve(A, [na @ pa, nb @ pb, d @ _WORKSPACE_MID])
            if np.linalg.norm(q0 - d * (d @ q0)) > REACH_BALL + 0.1:
                continue                          # corner line out of reach
            prisms.append(clipped_prism(
                f"corner_{name_a}_{name_b}", wedge_vertices(
                    {"point": q0, "normal": na}, {"point": q0, "normal": nb})))

    # declared obstacle corners
    for name, corner in cfg.get("corners", {}).items():
        T = frame_T(corner.get("parent", "base"))
        pl_a, pl_b = (
            {"point": T[:3, :3] @ np.asarray(p["point"], float) + T[:3, 3],
             "normal": T[:3, :3] @ np.asarray(p["normal"], float)}
            for p in corner["planes"].values())
        prisms.append(clipped_prism(f"corner_{name}",
                                    wedge_vertices(pl_a, pl_b)))
    return panels, [pr for pr in prisms if pr is not None]


def wedge_vertices(plane_a, plane_b, L=PRISM_L, z_range=PRISM_Z):
    """Corner keep-out (intersection of the half-spaces BEHIND two planes)
    -> 8 prism vertices, same frame as the planes. Cross-section is the
    parallelogram spanned by the two wall rays from the apex; extrusion runs
    along the walls' intersection line, cut at the z_range heights."""
    pa, na = (np.asarray(plane_a[k], dtype=float) for k in ("point", "normal"))
    pb, nb = (np.asarray(plane_b[k], dtype=float) for k in ("point", "normal"))
    na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
    if abs((pa - pb) @ nb) > 0.02:
        raise ValueError("corner planes do not share their point (apex)")
    d = np.cross(na, nb)
    d /= np.linalg.norm(d)
    if d[2] < 0:
        d = -d                       # extrusion direction, upward
    if d[2] < 0.5:
        raise ValueError("corner line is far from vertical — check the taps")
    walls = []
    for n_own, n_other in ((na, nb), (nb, na)):
        w = np.cross(n_own, d)
        w /= np.linalg.norm(w)
        walls.append(w if w @ n_other < 0 else -w)  # ray runs BEHIND the other
    quad = [pa, pa + L * walls[0], pa + L * (walls[0] + walls[1]),
            pa + L * walls[1]]
    return np.array([q + d * ((z - q[2]) / d[2])
                     for z in z_range for q in quad])


def _convex_prism(vertices, triangles=_PRISM_TRIS):
    verts = coal.StdVec_Vec3s()
    tris = coal.StdVec_Triangle()
    for v in vertices:
        verts.append(np.asarray(v, dtype=float))
    for i, j, k in triangles:
        tris.append(coal.Triangle(int(i), int(j), int(k)))
    return coal.Convex(verts, tris)


def _clip_poly_xy(poly, bounds):
    """Sutherland–Hodgman: convex polygon (list of 3D points) clipped to the
    top-view rectangle bounds = (x0, x1, y0, y1); z interpolates on edges."""
    pts = [np.asarray(p, dtype=float) for p in poly]
    x0, x1, y0, y1 = bounds
    for axis, lim, keep_less in ((0, x1, True), (0, x0, False),
                                 (1, y1, True), (1, y0, False)):
        nxt = []
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            ina = a[axis] <= lim if keep_less else a[axis] >= lim
            inb = b[axis] <= lim if keep_less else b[axis] >= lim
            if ina:
                nxt.append(a)
            if ina != inb:
                nxt.append(a + (lim - a[axis]) / (b[axis] - a[axis]) * (b - a))
        pts = nxt
        if not pts:
            return []
    return pts


def _pose_to_se3(pose):
    x, y, z, roll, pitch, yaw = pose
    R = (pin.utils.rotate("z", yaw) @ pin.utils.rotate("y", pitch)
         @ pin.utils.rotate("x", roll))
    return pin.SE3(R, np.array([x, y, z]))


def _frame_to_base(frames, name):
    if name in (None, "base"):
        return pin.SE3.Identity()
    f = frames[name]
    return _frame_to_base(frames, f.get("parent", "base")) * _pose_to_se3(f["pose"])


#: `base` (UR controller) -> `base_link` (URDF, REP-103) and back: pi about Z.
#: Its own inverse, which is why one constant serves both directions.
_RZ_PI = pin.SE3(np.diag([-1.0, -1.0, 1.0]), np.zeros(3))


def _mount_in_controller_base(model, *geom_models):
    """Re-express the robot in the UR controller's `base` frame.

    ur5e_description is rooted at `base_link`, which REP-103 aligns X+ forward;
    the controller's `base` — the frame the teach pendant reports, and the frame
    every number we measure is written in (`cell.yaml`'s probed table,
    `calib/`'s artifacts) — is that rotated pi about Z. The URDF ships both and
    says so on `base_link-base_fixed_joint`.

    We put the WORLD in `base` rather than converting at every boundary, so
    there is exactly one frame in the system: what the pendant shows, what the
    planner solves in, what the UI draws. Rotating the mount (not the cell) is
    what makes `cell.yaml` correct as measured.

    Load-bearing: this must run BEFORE any cell box is added, because those are
    attached to joint 0 too and are already in `base` — rotating them as well
    would just move the bug. Postcondition: the `base` frame coincides with the
    world, which is asserted below and is the cheapest proof this worked.
    """
    model.jointPlacements[1] = _RZ_PI * model.jointPlacements[1]
    for frame in model.frames:
        # Frame 0 IS the world; rotating it would rotate the thing we are
        # expressing everything else in.
        if frame.parentJoint == 0 and frame.name != "universe":
            frame.placement = _RZ_PI * frame.placement
    for geom_model in geom_models:
        for gobj in geom_model.geometryObjects:
            if gobj.parentJoint == 0:
                gobj.placement = _RZ_PI * gobj.placement

    base = model.frames[model.getFrameId("base")]
    if not np.allclose(base.placement.homogeneous, np.eye(4), atol=1e-9):
        raise RuntimeError(
            "mount failed: `base` should coincide with the world frame, got\n"
            f"{base.placement}")


def merged_keepout_mesh(panels, prisms):
    """Union of all keep-out footprints, extruded once -> [(vertices, faces)].

    DISPLAY ONLY: every keep-out spans the same PRISM_Z band (panels are
    sized/centered to match), so their union is a 2D polygon union — one
    watertight mesh per connected blob, no overlapping translucent faces to
    z-fight (Anton 2026-08-31). Collision keeps the separate convex shapes;
    the covered REGION is identical either way."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import trimesh

    polys = []
    for _, R, center, dims in panels:
        x, n = R[:, 0], R[:, 2]
        polys.append(Polygon([
            (center + sx * x * dims[0] / 2 + sn * n * dims[2] / 2)[:2]
            for sx, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    z_mid = sum(PRISM_Z) / 2
    for _, verts, _ in prisms:
        bot = verts[verts[:, 2] < z_mid]       # ordered ring from the clipper
        polys.append(Polygon(bot[:, :2]))
    union = unary_union([p.buffer(1e-3) for p in polys])  # weld shared edges
    meshes = []
    for poly in getattr(union, "geoms", [union]):
        m = trimesh.creation.extrude_polygon(poly, height=PRISM_Z[1] - PRISM_Z[0])
        v = np.array(m.vertices)
        v[:, 2] += PRISM_Z[0]
        meshes.append((v, np.array(m.faces)))
    return meshes


class RobotCell:
    def __init__(self, yaml_path=CELL_YAML, padding=DEFAULT_PADDING):
        """The ONE world. Split margins live here (env vs self), and the
        planner queries them through a validity callback — there is no second,
        inflated "planning world" any more (removed 2026-08-19 with the move
        from pyroboplan to OMPL). See motion/AGENTS.md for why that mattered."""
        from robot_descriptions.loaders.pinocchio import load_robot_description

        robot = load_robot_description("ur5e_description")
        self.model = robot.model
        self.geom_model = robot.collision_model
        self.visual_model = robot.visual_model
        _mount_in_controller_base(self.model, self.geom_model, self.visual_model)
        self.data = self.model.createData()
        self.padding = padding

        cfg = yaml.safe_load(Path(yaml_path).read_text())
        self._frames = cfg.get("frames", {})

        n_robot = self.geom_model.ngeoms  # UR5e links come first
        self._robot_geoms = list(range(n_robot))

        # --- tool envelope boxes welded to the flange (frame: +X out of tool face)
        flange = self.model.frames[self.model.getFrameId("flange")]
        self._tool_geoms = []
        for name, box in cfg.get("tool", {}).items():
            gid = self._add_box_geom(
                f"tool_{name}", flange.parentJoint,
                flange.placement * _pose_to_se3(box["pose"]), box["dims"])
            self._tool_geoms.append(gid)

        # --- static cell boxes (world-fixed, joint 0)
        self._env_geoms = []
        for name, box in cfg.get("boxes", {}).items():
            T = _frame_to_base(self._frames, box.get("parent", "base")) \
                * _pose_to_se3(box["pose"])
            gid = self._add_box_geom(f"cell_{name}", 0, T, box["dims"])
            self._env_geoms.append(gid)

        # --- taught safety keep-outs: clipped wall panels + corner prisms,
        # from the same builder the viewer draws (one truth)
        panels, prisms = build_keepouts(
            cfg, lambda f: _frame_to_base(self._frames, f).homogeneous)
        for name, R, center, dims in panels:
            gid = self._add_box_geom(f"cell_{name}", 0, pin.SE3(R, center), dims)
            self._env_geoms.append(gid)
        for name, verts, tris in prisms:
            gobj = pin.GeometryObject(f"cell_{name}", 0, pin.SE3.Identity(),
                                      _convex_prism(verts, tris))
            self._env_geoms.append(self.geom_model.addGeometryObject(gobj))

        self._build_pairs()
        self.geom_data = self.geom_model.createData()
        self.set_padding(padding)

        self._viz = None
        self._objects = {}  # soft geometry: display/standoff only, never pairs

    # ------------------------------------------------------------------ setup
    def _add_box_geom(self, name, joint_id, placement, dims):
        gobj = pin.GeometryObject(name, joint_id, placement,
                                  coal.Box(*[float(d) for d in dims]))
        return self.geom_model.addGeometryObject(gobj)

    def _build_pairs(self):
        gm = self.geom_model
        jid = lambda g: gm.geometryObjects[g].parentJoint
        # self-collision: robot link pairs, adjacent joints filtered (no SRDF ships)
        for i in self._robot_geoms:
            for j in self._robot_geoms:
                if i < j and abs(jid(i) - jid(j)) > 1:
                    gm.addCollisionPair(pin.CollisionPair(i, j))
        # tool vs base..forearm only: wrist_1..3 are kinematically welded to the
        # tool (the guessed cable-loop box overlaps wrist_1 by design)
        for t in self._tool_geoms:
            for i in self._robot_geoms:
                if jid(i) <= 3:
                    gm.addCollisionPair(pin.CollisionPair(i, t))
        # environment vs everything that moves (skip base link: bolted to the cell)
        for e in self._env_geoms:
            for i in self._robot_geoms + self._tool_geoms:
                if jid(i) >= 1:
                    gm.addCollisionPair(pin.CollisionPair(i, e))

    def set_padding(self, padding):
        """Margins classified per pair by its GEOMETRY (robust to runtime
        object add/remove): robot/tool internal pairs get SELF_PADDING;
        anything involving the environment or an object gets `padding`."""
        self.padding = padding
        moving = set(self._robot_geoms) | set(self._tool_geoms)
        for k, pair in enumerate(self.geom_model.collisionPairs):
            is_self = pair.first in moving and pair.second in moving
            self.geom_data.collisionRequests[k].security_margin = \
                SELF_PADDING if is_self else padding

    # ---------------------------------------------------------------- queries
    def is_colliding(self, q, padding=None):
        """True = config is INVALID (some collision pair hits)."""
        if padding is not None and padding != self.padding:
            self.set_padding(padding)
        return pin.computeCollisions(self.model, self.data, self.geom_model,
                                     self.geom_data, q, True)  # stop at first

    def first_collision(self, q):
        """Diagnose q: ('pair', names, contact point) or None."""
        pin.computeCollisions(self.model, self.data, self.geom_model,
                              self.geom_data, q, False)  # evaluate ALL pairs
        for k, res in enumerate(self.geom_data.collisionResults):
            if res.isCollision():
                pair = self.geom_model.collisionPairs[k]
                names = (self.geom_model.geometryObjects[pair.first].name,
                         self.geom_model.geometryObjects[pair.second].name)
                return ("pair", names, np.array(res.getContact(0).pos))
        return None

    def min_distance(self, q):
        """Smallest clearance over all pairs [m] (no padding applied)."""
        pin.computeDistances(self.model, self.data, self.geom_model,
                             self.geom_data, q)
        return min(dr.min_distance for dr in self.geom_data.distanceResults)

    @staticmethod
    def discretize(path, step=DEFAULT_STEP):
        """Waypoint list -> dense config list, consecutive L1 distance <= step."""
        path = [np.asarray(q, dtype=float) for q in path]
        out = [path[0]]
        for a, b in zip(path, path[1:]):
            n = max(1, int(np.ceil(np.abs(b - a).sum() / step)))
            for i in range(1, n + 1):
                out.append(a + (b - a) * (i / n))
        return out

    def path_valid(self, path, step=DEFAULT_STEP, padding=None):
        """Validate a waypoint path. Returns (ok, info): info is None when ok,
        else (index into the discretized path, colliding config, diagnosis)."""
        pad = self.padding if padding is None else padding
        assert step * LEVER_M < pad, (
            f"resolution contract violated: step {step} * lever {LEVER_M} = "
            f"{step * LEVER_M:.4f} m >= padding {pad} m")
        dense = self.discretize(path, step)
        for i, q in enumerate(dense):
            if self.is_colliding(q, padding=pad):
                return False, (i, q, self.first_collision(q))
        return True, None

    # --------------------------------------------- objects (the cup) — HARD
    def set_object(self, name, dims, pose, parent="base"):
        """Swap the reconstructed object primitive into the collision world
        (Anton 2026-08-18: the object IS a hard obstacle — we look, never
        touch; endpoint validity at env padding doubles as the standoff).
        Call again with the same name to update dims/pose every perception
        step — pairs are only built once, updates are cheap."""
        T = _frame_to_base(self._frames, parent) * _pose_to_se3(pose)
        self._objects[name] = (np.asarray(dims, dtype=float), T)
        gname = f"obj_{name}"
        if self.geom_model.existGeometryName(gname):
            gobj = self.geom_model.geometryObjects[
                self.geom_model.getGeometryId(gname)]
            gobj.placement = T
            gobj.geometry = coal.Box(*[float(d) for d in dims])
        else:
            gid = self._add_box_geom(gname, 0, T, dims)
            for i in self._robot_geoms + self._tool_geoms:
                if self.geom_model.geometryObjects[i].parentJoint >= 1:
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, gid))
        self.geom_data = self.geom_model.createData()
        self.set_padding(self.padding)
        if self._viz is not None:
            self._draw_object(name)

    def add_object(self, name, dims, pose, parent="base"):
        """Alias of set_object (older demos)."""
        self.set_object(name, dims, pose, parent)

    def remove_object(self, name):
        self._objects.pop(name, None)
        gname = f"obj_{name}"
        if self.geom_model.existGeometryName(gname):
            self.geom_model.removeGeometryObject(gname)
            self.geom_data = self.geom_model.createData()
            self.set_padding(self.padding)
        if self._viz is not None:
            self._viz.viewer["objects"][name].delete()

    # ---------------------------------------------------------------- visuals
    def init_viewer(self):
        import meshcat
        import meshcat.geometry as g
        from pinocchio.visualize import MeshcatVisualizer

        vis = meshcat.Visualizer()
        print(f"meshcat URL: {vis.url()}", flush=True)
        self._viz = MeshcatVisualizer(self.model, self.geom_model, self.visual_model)
        self._viz.initViewer(viewer=vis)
        self._viz.loadViewerModel(rootNodeName="ur5e")
        # cell boxes + keep-in ghost, reusing the yaml viewer's draw
        from inspection.cell.viewer import draw_cell
        draw_cell(vis, yaml.safe_load(CELL_YAML.read_text()))
        for name in self._objects:
            self._draw_object(name)
        return self._viz

    def _draw_object(self, name):
        import meshcat.geometry as g
        dims, T = self._objects[name]
        self._viz.viewer["objects"][name].set_object(
            g.Box(dims), g.MeshLambertMaterial(color=0xDDCC22, opacity=0.7,
                                               transparent=True))
        M = np.eye(4); M[:3, :3] = T.rotation; M[:3, 3] = T.translation
        self._viz.viewer["objects"][name].set_transform(M)

    def _draw_tool(self, color):
        """Tool boxes tinted by verdict — green free, red colliding."""
        import meshcat.geometry as g
        pin.updateGeometryPlacements(self.model, self.data,
                                     self.geom_model, self.geom_data)
        for gid in self._tool_geoms:
            gobj = self.geom_model.geometryObjects[gid]
            M4 = np.eye(4)
            M = self.geom_data.oMg[gid]
            M4[:3, :3], M4[:3, 3] = M.rotation, M.translation
            node = self._viz.viewer["tool"][gobj.name]
            node.set_object(g.Box(2 * gobj.geometry.halfSide),
                            g.MeshLambertMaterial(color=color, opacity=0.5,
                                                  transparent=True))
            node.set_transform(M4)

    def _set_status(self, bad):
        """Unmissable 3D verdict: background + tool tint. Green free, red bad."""
        self._draw_tool(0xDD2222 if bad else 0x22CC44)
        bg = self._viz.viewer["/Background"]
        bg.set_property("top_color", [0.95, 0.55, 0.55] if bad else [0.60, 0.85, 0.65])
        bg.set_property("bottom_color", [0.55, 0.25, 0.25] if bad else [0.30, 0.50, 0.35])

    def show(self, q, label=""):
        """Display q, flash verdict in 3D, mark the contact point. -> valid?"""
        import meshcat.geometry as g
        if self._viz is None:
            self.init_viewer()
        self._viz.display(np.asarray(q, dtype=float))
        diag = self.first_collision(q)
        self._set_status(diag is not None)
        marker = self._viz.viewer["contact"]
        if diag is not None and diag[2] is not None:
            marker.set_object(g.Sphere(0.03),
                              g.MeshLambertMaterial(color=0xFF0000))
            M = np.eye(4); M[:3, 3] = diag[2]
            marker.set_transform(M)
        else:
            marker.delete()
        verdict = "FREE" if diag is None else f"COLLIDING {diag[0]}: {diag[1]}"
        print(f"  {label or 'q'}: {verdict}", flush=True)
        return diag is None

    def replay(self, path, step=DEFAULT_STEP, dt=0.03):
        """Animate the discretized path; freeze at the first invalid config."""
        dense = self.discretize(path, step)
        print(f"replaying {len(dense)} configs "
              f"({len(path)} waypoints, step {step} rad L1) ...", flush=True)
        for i, q in enumerate(dense):
            if self._viz is None:
                self.init_viewer()
            self._viz.display(q)
            bad = self.is_colliding(q)
            self._set_status(bad)
            if bad:
                diag = self.first_collision(q)
                self.show(q, label=f"step {i}/{len(dense)}")
                print(f"  >> path INVALID at step {i}: frozen at impact", flush=True)
                return False
            time.sleep(dt)
        print("  path VALID end to end", flush=True)
        return True


# ---------------------------------------------------------------- smoke test
# Truth-table configs (found by random search against the calibrated cell).
# NOTE: the GENERIC UR home pose is NOT free in this cell — the wrist grazes
# the top-right post (measured -0.3 mm vs its envelope). Q_PARK is a searched
# collision-free pose over the table, tool facing down, max clearance.
Q_PARK = np.array([2.353, -1.325, -2.037, -1.289, 1.484, -2.872])
Q_TABLE = np.array([-0.80, -1.47, 2.10, -2.21, -1.84, 1.35])  # gripper into slab
Q_SELF = np.array([-2.60, -1.65, 1.89, 0.52, -2.55, -0.42])   # forearm vs wrist_3
Q_FLOOR = np.array([-1.96, -2.80, -1.41, 0.99, 0.39, -2.20])  # dives below table level
# one reach into each taught keep-out (found by seeded random search; each
# config's ONLY environment contact is the named shape)
Q_KEEPOUT = {
    "cell_plane_back": [-2.275, -0.689, -0.014, -1.346, 0.665, 0.644],
    "cell_plane_side_left": [-3.043, -1.894, -0.552, 1.179, -0.387, 0.084],
    "cell_corner_corner1": [-0.874, -2.511, -0.914, 1.465, -0.998, 1.145],
    "cell_corner_corner2": [1.707, 0.076, -1.196, -2.432, -1.735, 1.824],
    "cell_corner_corner3": [-1.509, -2.961, 1.148, -2.822, 0.703, 0.799],
    "cell_corner_corner4": [-1.439, -3.035, -0.267, -2.262, -0.421, 1.861],
}


def smoke(cell):
    print("\n== truth table ==")
    cases = [("park (expect FREE)", Q_PARK, False),
             ("through-table (expect COLLIDING)", Q_TABLE, True),
             ("self-fold (expect COLLIDING)", Q_SELF, True),
             ("below-table (expect COLLIDING floor)", Q_FLOOR, True)]
    cases += [(f"into {name} (expect COLLIDING)", np.array(q), True)
              for name, q in Q_KEEPOUT.items()]
    ok = True
    for label, q, expect_bad in cases:
        diag = cell.first_collision(q)
        bad = diag is not None
        status = "PASS" if bad == expect_bad else "FAIL"
        detail = "free" if diag is None else f"{diag[0]} {diag[1]}"
        ok &= bad == expect_bad
        # a keep-out case must be caught by ITS shape, not merely any contact
        if expect_bad and label.startswith("into ") and diag is not None:
            want = label.split()[1]
            if want not in diag[1]:
                status, ok = "FAIL", False
                detail += f"  (expected contact with {want})"
        print(f"  [{status}] {label}: {detail}")

    print("\n== timing (MAY-183 targets: 21us/check, 0.43ms/40-waypoint, 34ms/1600) ==")
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        cell.is_colliding(Q_PARK)
    per = (time.perf_counter() - t0) / n
    print(f"  is_colliding (free config): {per * 1e6:.1f} us")

    path40 = [Q_PARK + (i / 39) * 0.3 * np.sin(np.arange(6) + i) for i in range(40)]
    t0 = time.perf_counter()
    for q in path40:
        cell.is_colliding(q)
    print(f"  40-config path check: {(time.perf_counter() - t0) * 1e3:.2f} ms")

    rng = np.random.default_rng(7)
    qs = rng.uniform(-np.pi, np.pi, size=(1600, 6))
    t0 = time.perf_counter()
    for q in qs:
        cell.is_colliding(q)
    print(f"  1600 random configs: {(time.perf_counter() - t0) * 1e3:.1f} ms")

    ok_path, info = cell.path_valid([Q_PARK, Q_PARK + np.array([0.4, 0, 0, 0, 0, 0])])
    print(f"\n  small joint move path_valid: {ok_path}")
    bad_path, info = cell.path_valid([Q_PARK, Q_TABLE])
    print(f"  home->through-table path_valid: {bad_path}"
          + (f" (blocked at dense step {info[0]}, {info[2][1]})" if info else ""))
    print("\nsmoke:", "ALL PASS" if ok and ok_path and not bad_path else "FAILURES ABOVE")
    return ok and ok_path and not bad_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="meshcat truth-table poses")
    ap.add_argument("--replay", action="store_true", help="meshcat path replay demo")
    args = ap.parse_args()

    cell = RobotCell()
    print(f"world: {cell.geom_model.ngeoms} geoms, "
          f"{len(cell.geom_model.collisionPairs)} pairs, "
          f"padding {cell.padding * 1e3:.0f} mm")

    if args.demo:
        cell.init_viewer()
        print("cycling truth-table poses in the browser — ctrl-C to quit")
        try:
            while True:
                for label, q in [("park", Q_PARK), ("through-table", Q_TABLE),
                                 ("self-fold", Q_SELF), ("below-table", Q_FLOOR)]:
                    cell.show(q, label=label)
                    time.sleep(3.0)
        except KeyboardInterrupt:
            return 0
    if args.replay:
        cell.init_viewer()
        print("replaying in the browser: valid move, then a move into the table")
        try:
            while True:
                cell.replay([Q_PARK, Q_PARK + np.array([0.6, -0.2, 0.3, 0, 0, 0])])
                time.sleep(2.0)
                cell.replay([Q_PARK, Q_TABLE])   # freezes red at impact
                time.sleep(4.0)
        except KeyboardInterrupt:
            return 0
    return 0 if smoke(cell) else 1


if __name__ == "__main__":
    sys.exit(main())
