"""
eval_engine_logged.py — drop-in wrapper that records per-step (state, action)
trajectories during an eval session, WITHOUT modifying eval_engine.py.

Run it exactly like eval_engine.py — same flags:

    p inference/eval_engine_logged.py \
        --checkpoint=/.../SmolVLA-Duck-Co-Exp0-v1/pretrained_model/ \
        --protocol=ood_single_cup_30 \
        --port=/dev/ttyACM1 \
        --cameras=sim_4cam_opencv \
        --max_steps=10000 \
        --action_horizon=10 \
        --record=True \
        --task "pick up the blue duck and put it in the red bin"

For every trial it writes, alongside the existing session.json / videos/:

    <session_dir>/trajectories/trial_<id>_<result>.npz
        state  : (T, 6) float64 — robot joint positions in DEGREES, sampled each
                                   step right before the policy runs.
        action : (T, 6) float64 — the policy's commanded joint targets in DEGREES
                                   (de-normalized output of AsyncChunkRunner.step).
        meta   : ()    object   — JSON string: trial_id, result, steps, task, note,
                                   video, delta_actions.

    T == the trial's `steps` count, so state[k] is the observation the policy saw
    just before emitting action[k].

────────────────────────────────────────────────────────────────────────────────
HOW IT WORKS — three monkeypatched seams, no source edits

  1. eval_engine._init_robot     wrap the robot's get_observation() so every
                                  poll snapshots the raw joint state (degrees).
  2. AsyncChunkRunner.step        append (state, action) here. This seam fires
                                  ONLY during the inference phase, so the robot's
                                  rest moves (move_to_rest) and the guidance phase
                                  never leak spurious rows into the buffer.
  3. EvalSession.record           flush the buffer to one .npz per trial, at the
                                  exact point the outcome is recorded.

Note on --delta_actions: the logged `action` is the policy output. With
--delta_actions=True the value actually SENT is (state + action) with the gripper
held absolute; that sent target is still reconstructable from the logged pair.
Your eval command does not use delta actions, so logged action == sent target.
"""

import json
from pathlib import Path

import numpy as np

from vbti.logic.inference import eval_engine
from vbti.logic.inference.async_chunk_runner import AsyncChunkRunner

JOINT_NAMES = eval_engine.JOINT_NAMES

# ── module-level capture state ────────────────────────────────────────────────
_latest_obs: dict | None = None   # most recent full observation dict from robot
_buf_state:  list = []            # one (6,) joint-degree row per inference step
_buf_action: list = []            # one (6,) commanded-degree row per inference step
_patched_robot_classes: set = set()  # robot classes already wrapped (idempotency)


def _install_logging() -> None:
    """Patch the three seams in the imported eval_engine module."""

    # 1) snapshot raw joint state on every robot poll ─────────────────────────
    orig_init_robot = eval_engine._init_robot

    def _init_robot_logged(*args, **kwargs):
        robot = orig_init_robot(*args, **kwargs)
        cls = type(robot)
        # Patch at the class level (survives __slots__; only one robot anyway).
        if cls not in _patched_robot_classes:
            _patched_robot_classes.add(cls)
            orig_get = cls.get_observation

            def get_observation_logged(self, *a, **k):
                global _latest_obs
                obs = orig_get(self, *a, **k)
                _latest_obs = obs
                return obs

            cls.get_observation = get_observation_logged
        return robot

    eval_engine._init_robot = _init_robot_logged

    # 2) append (state, action) on every policy step ──────────────────────────
    orig_step = AsyncChunkRunner.step

    def step_logged(self, obs):
        action = orig_step(self, obs)
        if _latest_obs is not None:
            _buf_state.append(
                [float(_latest_obs[f"{n}.pos"]) for n in JOINT_NAMES]
            )
            _buf_action.append(np.asarray(action, dtype=float).ravel().copy())
        return action

    AsyncChunkRunner.step = step_logged

    # 3) flush one .npz per trial where the outcome is recorded ────────────────
    orig_record = eval_engine.EvalSession.record

    def record_logged(self, trial, result, steps, video=None, note=""):
        orig_record(self, trial, result, steps, video=video, note=note)
        if _buf_action:
            tdir = self.dir / "trajectories"
            tdir.mkdir(exist_ok=True)
            tid = trial.get("trial_id", len(self.data["trials"]) - 1)
            cfg = self.data.get("config", {})
            meta = {
                "trial_id":      tid,
                "result":        result,
                "steps":         steps,
                "task":          trial.get("task") or cfg.get("task"),
                "note":          note,
                "video":         Path(video).name if video else None,
                "delta_actions": cfg.get("delta_actions", False),
            }
            out = tdir / f"trial_{tid:02d}_{result}.npz"
            np.savez(
                out,
                state=np.asarray(_buf_state, dtype=float),
                action=np.asarray(_buf_action, dtype=float),
                meta=json.dumps(meta),
            )
            print(f"  trajectory saved: trajectories/{out.name} "
                  f"({len(_buf_action)} steps)")
        _buf_state.clear()
        _buf_action.clear()

    eval_engine.EvalSession.record = record_logged


if __name__ == "__main__":
    import fire

    _install_logging()
    fire.Fire(eval_engine.run)
