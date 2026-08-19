#!/usr/bin/env python3
"""Inspection loop v1 (POC) — the D3 orchestrator with a human decider.

Spec: inspection/2026-08-19-loop-v1-design.md. Boot: plan to the taught
survey pose (approved, like every motion), capture 000, seed the object.
Turn: READ comment -> DECIDE look/answer -> plan -> meshcat preview ->
approve -> move (ENTER = software stop) -> capture -> fuse -> re-center.

The Decider is the AI seam; the rig bundles arm+camera so tests can run
the full loop headless (see tests/test_loop_fake.py).
"""
import json
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import CloudAccumulator, object_in_base
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.motion.plan import plan_viewpoint
from inspection.run.decider import Answer, Ctx, Look, Quit, build_menu
from inspection.view.viewsphere import ViewSphere, map_str

START_TOL_RAD = 0.02        # same gate as execute.py


class Loop:
    def __init__(self, rig, decider, console, outdir, q_survey, r=0.35,
                 max_turns=12, question="", seed=0, world=None, ik=None):
        self.rig, self.decider, self.console = rig, decider, console
        self.outdir = Path(outdir)
        self.q_survey = np.asarray(q_survey, dtype=float)
        self.r, self.max_turns = r, max_turns
        self.question, self.seed = question, seed
        self.world = world if world is not None else RobotCell()
        self.ik = ik if ik is not None else UR5eIK()
        self.acc = CloudAccumulator()
        self.sphere = None
        self.current = None                 # (h, v) after the first look
        self.visited = set()
        self.plan_failed = set()            # cleared after every real move
        self.turns = []
        self.latest_cap = None
        self.t0 = time.time()

    # ---------------------------------------------------------------- boot
    def boot(self):
        q_now = self.rig.q()
        if np.abs(q_now - self.q_survey).max() > START_TOL_RAD:
            print("boot: planning current -> survey pose")
            path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                       self.ik.fk(self.q_survey),
                                       seed=self.seed)
            if path is None:
                print(f"boot REFUSED: {rep.get('reason', rep)}")
                return False
            ok, _ = self._approve_and_move(path, "boot -> survey")
            if not ok:
                return False
        try:
            cap = self.rig.capture(0)
            view = object_in_base(cap["depth_raw"], self.rig.intr,
                                  self.rig.depth_scale, cap["T_base_cam"])
            if view["centroid"] is None:
                print("boot: NO OBJECT above the table")
                return False
            self.acc.add(view["points"])
            self._recenter()
        except RuntimeError as e:
            print(f"boot: capture failed: {e}")
            return False
        self.latest_cap = cap
        print(f"boot ok: object at {np.round(self.sphere.center, 3).tolist()}, "
              f"{len(view['points'])} pts")
        return True

    # ---------------------------------------------------------------- turn
    def turn(self):
        step = len(self.turns) + 1
        comment = self.decider.read(self.latest_cap)
        reach = self.sphere.reachability(self.world, self.ik)
        ctx = Ctx(question=self.question, step=step,
                  current_cell=self.current,
                  map_ascii=map_str(reach, self.sphere.elevations),
                  menu=build_menu(reach, self.visited | self.plan_failed,
                                  self.current,
                                  elevations=self.sphere.elevations),
                  comments=[t["comment"] for t in self.turns] + [comment])
        act = self.decider.decide(ctx)
        rec = {"step": step, "comment": comment, "t": time.time() - self.t0}
        self.turns.append(rec)

        if isinstance(act, (Answer, Quit)):
            rec["action"] = ("answer " + act.text) if isinstance(act, Answer) \
                else "quit"
            return False

        rec["action"] = f"look {act.h} {act.v}"
        if reach.get((act.h, act.v)) is None:
            rec["result"] = "unreachable cell — pick again"
            print(rec["result"])
            self._save()
            return True
        path, prep, roll = self.sphere.plan_to_cell(
            self.world, self.ik, self.rig.q(), act.h, act.v, seed=self.seed)
        if path is None:
            self.plan_failed.add((act.h, act.v))
            rec["result"] = "plan refused — cell dropped from menu this round"
            print(rec["result"])
            self._save()
            return True
        rec["tier"] = prep["tier"]
        ok, reason = self._approve_and_move(path, rec["action"])
        rec["approved"] = reason != "refused"
        rec["stopped"] = reason == "stopped"
        if not ok:
            rec["result"] = ("refused — nothing sent to the robot" if reason == "refused"
                             else "software stop — arm halted mid-move")
            self._save()
            return True
        self.visited.add((act.h, act.v))
        self.current = (act.h, act.v)
        self.plan_failed.clear()            # new q — refused cells may work now

        try:
            cap = self.rig.capture(step)
            view = object_in_base(cap["depth_raw"], self.rig.intr,
                                  self.rig.depth_scale, cap["T_base_cam"])
            if len(view["points"]):
                self.acc.add(view["points"])
            self._recenter()
        except RuntimeError as e:
            print(f"capture failed: {e}")
            rec["result"] = f"capture failed: {e} — back to menu"
            self._save()
            return True
        self.latest_cap = cap
        rec["result"] = (f"+{len(view['points'])} pts, "
                         f"fused {len(self.acc.points)}")
        self._save()                        # every turn is recorded, not just successes
        return True

    # ------------------------------------------------------------- helpers
    def _approve_and_move(self, path, label):
        """Returns (ok, reason) — reason in {"refused", "stopped", None}."""
        self.rig.preview(path)
        ans = self.console.readline(
            f"{label}: {len(path)} waypoints previewed — approve? [y/n] ")
        if ans.strip().lower() != "y":
            print("refused — nothing sent to the robot")
            return False, "refused"
        print("executing (ENTER = software stop)")
        self.console.arm_stop()
        try:
            rep = self.rig.move(path)
        finally:
            self.console.disarm_stop()
        if rep.get("stopped"):
            print("SOFTWARE STOP — arm halted; back to the menu "
                  "(plans restart from wherever it stopped)")
            return False, "stopped"
        return True, None

    def _recenter(self):
        center = self.acc.centroid
        self.sphere = ViewSphere(center, r=self.r)
        mn, mx = self.acc.aabb()
        dims = np.maximum(mx - mn + 0.04, 0.05)         # 2 cm margin each side
        mid = (mn + mx) / 2
        self.world.set_object("object", dims.tolist(),
                              [*mid.tolist(), 0.0, 0.0, 0.0], parent="base")

    def _save(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "run.json").write_text(json.dumps(
            {"question": self.question, "q_survey": self.q_survey.tolist(),
             "r": self.r, "turns": self.turns}, indent=2) + "\n")

    # ----------------------------------------------------------------- run
    def run(self):
        ok = self.boot()
        if ok:
            while len(self.turns) < self.max_turns:
                if not self.turn():
                    break
        self._save()
        if len(self.acc.points):
            np.save(self.outdir / "fused_cloud.npy", self.acc.points)
        print(f"run saved: {self.outdir}  ({len(self.turns)} turns, "
              f"{len(self.acc.points)} fused pts)")


# ── real rig + CLI ──────────────────────────────────────────────────────────

ROBOT_IP = "192.168.2.50"
SURVEY_POSE_FILE = Path(__file__).resolve().parent / "survey_pose.json"


class RealRig:
    """UR5e + wrist D405 behind the rig contract. Owns session.json."""

    def __init__(self, world, console, outdir, ip=ROBOT_IP):
        from inspection.motion.execute import UR5eArm
        from inspection.perception.camera import (
            WRIST_SERIAL, capture_bundle, open_camera, session_metadata,
            t_flange_cam)
        self._capture_bundle = capture_bundle
        self.world, self.console = world, console
        self.outdir = Path(outdir)
        self.arm = UR5eArm(ip)
        self.pipe, profile, self.align, self.depth_scale = open_camera()
        meta = session_metadata(profile, self.depth_scale, WRIST_SERIAL)
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "session.json").write_text(
            json.dumps(meta, indent=2) + "\n")
        self.intr = meta["intrinsics"]["ir_left"]
        self._T_fc = t_flange_cam()
        self._ik = UR5eIK()

    def q(self):
        return self.arm.q()

    def preview(self, path):
        self.world.replay(path)

    def move(self, path):
        return self.arm.execute(path, world=self.world,
                                stop_event=self.console.stop_event)

    def capture(self, pose_id):
        from inspection.perception.capture import save_bundle
        bundle = self._capture_bundle(self.pipe, self.align)
        q = self.arm.q()
        T_bf = self._ik.fk(q)
        pose = {"joints_rad": q, "T_base_flange": T_bf,
                "T_base_cam": T_bf @ self._T_fc}
        d = save_bundle(self.outdir, pose_id, bundle, pose)
        return {"dir": str(d), "rgb": bundle["rgb"],
                "depth_raw": bundle["depth_raw"],
                "T_base_cam": pose["T_base_cam"], "q": q}

    def close(self):
        self.pipe.stop()
        self.arm.close()


def teach(ip: str = ROBOT_IP):
    """Freedrive the arm to the survey pose, then run this. Read-only."""
    from rtde_receive import RTDEReceiveInterface
    q = list(RTDEReceiveInterface(ip).getActualQ())
    SURVEY_POSE_FILE.write_text(json.dumps({"q_rad": q}, indent=2) + "\n")
    print(f"survey pose saved: {np.round(np.degrees(q), 1).tolist()} deg "
          f"-> {SURVEY_POSE_FILE}")


def run(outdir: str, ip: str = ROBOT_IP, r: float = 0.35,
        max_turns: int = 12, question: str = "", seed: int = 0):
    """The v1 loop. Requires: bringup done, Remote Control, taught survey."""
    from inspection.motion.execute import preflight
    from inspection.run.decider import Console, TerminalDecider

    pf = preflight(ip)
    if not pf["go"]:
        raise SystemExit(f"preflight NO-GO: {pf}")
    if not SURVEY_POSE_FILE.exists():
        raise SystemExit("no survey pose — freedrive there and run "
                         "`p inspection/run/loop.py teach` first")
    q_survey = np.array(json.loads(SURVEY_POSE_FILE.read_text())["q_rad"])

    world = RobotCell()
    world.init_viewer()                    # meshcat preview window
    console = Console()
    rig = RealRig(world, console, outdir, ip)
    try:
        Loop(rig, TerminalDecider(console), console, outdir,
             q_survey=q_survey, r=r, max_turns=max_turns,
             question=question, seed=seed, world=world).run()
    finally:
        rig.close()


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"teach": teach, "run": run})
