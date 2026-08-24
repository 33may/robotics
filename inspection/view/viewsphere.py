"""Viewsphere around the inspected object — cells, reachability, planning (v2, UR5e).

Keeps the SO-101 addressing scheme (Anton, 2026-08-18): cells are {h, v} on
ONE shell of configurable radius r — 12 azimuth bins of 30 deg with h=0
facing the robot base, elevations [20, 45, 70] deg. Roll about the camera
boresight is a FREE parameter: each cell tries the upright roll first, then
rolled variants — image rotation is acceptable, unreachability is not.

The camera pose is derived from the flange via a NOMINAL hand-eye guess
(top camera, boresight along the tool axis) — replace T_FLANGE_CAM after
the real hand-eye calibration.

The object itself is a HARD obstacle in the world (set_object) — cells whose
every branch/roll collides are honestly blocked; there is no separate
standoff (endpoint validity at env padding IS the standoff). Radius must
respect the tool: the closed tip rides ~19 cm AHEAD of the camera.

Usage (from repo root, robo env active):
    p inspection/view/viewsphere.py             # reachability map for the demo cup
    p inspection/view/viewsphere.py --r 0.40    # different shell radius
    p inspection/view/viewsphere.py --demo      # meshcat: cells colored, tour them
"""

import argparse
import sys
import time

import numpy as np

from inspection.cell.geometry import load_T_flange_cam
from inspection.cell.world import RobotCell, DEFAULT_STEP
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint, CUP_POS, CUP_DIMS, DEMO_PARK
# Cell addressing lives in the light half (grid.py) so the AI tiers can speak
# about cells without importing IK/motion. Re-exported here: callers that
# already say `from ...viewsphere import H_BINS` keep working.
from inspection.view.grid import H_BINS, V_ELEVATIONS, DEFAULT_R

# CALIBRATED hand-eye (2026-08-19, calib/T_flange_cam_*.npy): left eye at
# flange [+142.4, -8.3, -47.5] mm, boresight 1.07 deg off the tool axis.
# Loaded in this module's camera convention (+X boresight, +Z image-up).
T_FLANGE_CAM = load_T_flange_cam("xfwd")

# Roll preference: upright first, then a half turn — and NOTHING else
# (Anton 2026-08-24). Both are LOSSLESS to undo in image space and both keep
# the frame landscape, so every capture can be presented to the VLM in one
# orientation. Measured on the demo cup at r=0.244: {0, 180} reaches 26/36
# cells — exactly what the old twelve-roll ladder reached. The nine
# intermediate rolls bought no reachability at all, they only produced
# tilted images that no de-rotation can fix without interpolating.
ROLLS = np.deg2rad([0, 180])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class ViewSphere:
    """One shell of {h, v} camera cells around a center point (base frame)."""

    def __init__(self, center, r=DEFAULT_R, elevations=V_ELEVATIONS,
                 t_flange_cam=T_FLANGE_CAM):
        self.center = np.asarray(center, dtype=float)
        self.r = float(r)
        self.elevations = tuple(elevations)   # deg ABOVE THE TABLE PLANE
        self._t_fc_inv = np.linalg.inv(t_flange_cam)
        # h=0 faces the base: azimuth of the center->base direction
        d = -self.center[:2]
        self._az0 = np.arctan2(d[1], d[0])

    def cells(self):
        return [(h, v) for v in range(len(self.elevations)) for h in range(H_BINS)]

    def cell_dir(self, h, v):
        """Unit vector center -> camera position of cell {h, v}."""
        az = self._az0 + h * (2 * np.pi / H_BINS)
        el = np.deg2rad(self.elevations[v])
        return np.array([np.cos(el) * np.cos(az),
                         np.cos(el) * np.sin(az),
                         np.sin(el)])

    def cam_pose(self, h, v, roll=0.0):
        """Camera pose 4x4: +X boresight at the center, +Z image-up (world-up
        preferred, radial fallback near the pole), rolled about the boresight."""
        out = self.cell_dir(h, v)
        pos = self.center + self.r * out
        x = -out                                        # boresight
        ref = np.array([0.0, 0.0, 1.0])
        if abs(x @ ref) > 0.95:                         # top-down: up degenerate
            ref = np.array([np.cos(self._az0), np.sin(self._az0), 0.0])
        y = np.cross(ref, x); y /= np.linalg.norm(y)    # image-right
        z = np.cross(x, y)                              # image-up
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
        T[:3, :3] = T[:3, :3] @ _rot_x(roll)
        return T

    def flange_pose(self, h, v, roll=0.0):
        """Where the FLANGE must be so the camera sits at cell {h, v}."""
        return self.cam_pose(h, v, roll) @ self._t_fc_inv

    # ------------------------------------------------------------ queries
    def cell_reachable(self, world, ik, h, v):
        """First roll whose flange pose has a collision-free IK branch.
        Returns (roll, configs) or (None, [])."""
        for roll in ROLLS:
            T = self.flange_pose(h, v, roll)
            free = [q for q in ik.branches(T) if not world.is_colliding(q)]
            if free:
                return roll, free
        return None, []

    def reachability(self, world, ik):
        """{(h, v): roll | None} over all cells."""
        return {(h, v): self.cell_reachable(world, ik, h, v)[0]
                for h, v in self.cells()}

    def plan_to_cell(self, world, ik, q_now, h, v, seed=0, step=DEFAULT_STEP):
        """Ladder-plan to cell {h, v}, walking the roll preferences.
        Returns (path, report, roll) — path None if every roll fails."""
        rep = None
        for roll in ROLLS:
            T = self.flange_pose(h, v, roll)
            free = [q for q in ik.branches(T) if not world.is_colliding(q)]
            if not free:
                continue
            path, rep = plan_viewpoint(world, ik, q_now, T,
                                       seed=seed, step=step)
            if path is not None:
                return path, rep, roll
        return None, rep, None


def map_str(reach, elevations=V_ELEVATIONS):
    """ASCII map: rows = v (top row highest), cols = h. #=reachable .=blocked"""
    lines = []
    n_v = len(elevations)
    for v in reversed(range(n_v)):
        row = "".join("#" if reach[(h, v)] is not None else "."
                      for h in range(H_BINS))
        lines.append(f"v{v} ({elevations[v]:.0f}deg)  {row}")
    lines.append(f"            {''.join(str(h % 10) for h in range(H_BINS))}"
                 "  (h0 faces base)")
    return "\n".join(lines)


# ------------------------------------------------------------------ demo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=DEFAULT_R, help="shell radius [m]")
    ap.add_argument("--elev", type=float, nargs="+", default=list(V_ELEVATIONS),
                    help="ring elevations, deg ABOVE the table plane (default 20 45 70)")
    ap.add_argument("--demo", action="store_true", help="meshcat cell tour")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world, ik = RobotCell(), UR5eIK()
    world.set_object("cup", CUP_DIMS, [*CUP_POS, 0, 0, 0], parent="base")
    center = CUP_POS + np.array([0.0, 0.0, CUP_DIMS[2] / 2])
    sphere = ViewSphere(center, r=args.r, elevations=args.elev)

    t0 = time.perf_counter()
    reach = sphere.reachability(world, ik)
    n_ok = sum(1 for r in reach.values() if r is not None)
    print(f"shell r={args.r:.2f} m around the cup — reachability "
          f"({(time.perf_counter() - t0) * 1e3:.0f} ms):")
    print(map_str(reach, sphere.elevations))
    print(f"{n_ok}/{len(reach)} cells reachable")

    if not args.demo:
        return 0

    import meshcat.geometry as g
    world.init_viewer()
    world.show(DEMO_PARK, label="park")
    # paint every cell on the sphere
    for (h, v), roll in reach.items():
        node = world._viz.viewer["sphere"][f"h{h}v{v}"]
        node.set_object(g.Sphere(0.012), g.MeshLambertMaterial(
            color=0x22CC44 if roll is not None else 0xDD2222))
        M = np.eye(4)
        M[:3, 3] = sphere.center + sphere.r * sphere.cell_dir(h, v)
        node.set_transform(M)

    # tour: RANDOM cell order — consecutive hops show real reorientation
    q_now = DEMO_PARK.copy()
    visited = failed = 0
    tally = {1: 0, 2: 0, 3: 0}
    order = [c for c in sphere.cells() if reach[c] is not None]
    np.random.default_rng(args.seed).shuffle(order)
    try:
        for h, v in order:
            if True:
                path, rep, roll = sphere.plan_to_cell(
                    world, ik, q_now, h, v, seed=args.seed)
                if path is None:
                    failed += 1
                    print(f"cell h{h} v{v}: REFUSED (no tier found a path)")
                    continue
                visited += 1
                tally[rep["tier"]] += 1
                print(f"cell h{h} v{v}: tier {rep['tier']}, "
                      f"roll {np.rad2deg(roll):+.0f} deg, "
                      f"{len(path)} waypoints ({rep['ms']:.0f} ms)")
                world._viz.viewer["sphere"][f"h{h}v{v}"].set_object(
                    g.Sphere(0.014), g.MeshLambertMaterial(color=0x2288DD))
                world.replay(path)
                q_now = path[-1]
                time.sleep(0.5)
        print(f"\ntour: {visited} cells visited "
              f"(t1/t2/t3 {tally[1]}/{tally[2]}/{tally[3]}), {failed} refused")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
