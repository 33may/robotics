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


class RobotCell:
    def __init__(self, yaml_path=CELL_YAML, padding=DEFAULT_PADDING):
        """The ONE world. Split margins live here (env vs self), and the
        planner queries them through a validity callback — there is no second,
        inflated "planning world" any more (removed 2026-08-19 with the move
        from pyroboplan to OMPL). See motion/AGENTS.md for why that mattered."""
        from robot_descriptions.loaders.pinocchio import load_robot_description

        robot = load_robot_description("ur5e_description")
        self.model = robot.model
        self.data = self.model.createData()
        self.geom_model = robot.collision_model
        self.visual_model = robot.visual_model
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


def smoke(cell):
    print("\n== truth table ==")
    cases = [("park (expect FREE)", Q_PARK, False),
             ("through-table (expect COLLIDING)", Q_TABLE, True),
             ("self-fold (expect COLLIDING)", Q_SELF, True),
             ("below-table (expect COLLIDING floor)", Q_FLOOR, True)]
    ok = True
    for label, q, expect_bad in cases:
        diag = cell.first_collision(q)
        bad = diag is not None
        status = "PASS" if bad == expect_bad else "FAIL"
        detail = "free" if diag is None else f"{diag[0]} {diag[1]}"
        ok &= bad == expect_bad
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
