"""runtime.py — the Orchestrator: conducts the brain loop.

`read → reason → act → write`, paced by the World stamp (D8): one policy step per
≥`policy_dt_ns` of advancing simulation time, NOT brain wall-clock, so the policy's
effective timestep stays correct at any real-time factor. Latest-wins (the Comm drains
to the newest Observation). The FIRST observation always steps — that is the
freeze-until-cmd handshake (D9): the brain bootstraps the World off its frozen pose,
and the World begins stepping on receipt of that first PolicyOut.

The Orchestrator is the only component that sees every contract, so it owns optional
recording. It depends only on the duck-typed Comm / reason / action interfaces — no
isaacsim, no limxsdk. Runs in the `brain` env.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np

from .contracts import PolicyOut

# 10 ms = the walk policy's trained decimation (decimation=10 @ 1 kHz).
_DEFAULT_POLICY_DT_NS = 10_000_000


class Orchestrator:
    """Conducts read→reason→act→write, stamp-paced and latest-wins."""

    def __init__(
        self,
        comm,
        reason,
        action,
        joystick=None,
        policy_dt_ns: int = _DEFAULT_POLICY_DT_NS,
        recorder: Optional[Callable] = None,
        idle_sleep: float = 0.0,
    ) -> None:
        self._comm = comm
        self._reason = reason
        self._action = action
        self._joystick = joystick
        self._policy_dt_ns = int(policy_dt_ns)
        self._recorder = recorder
        self._idle_sleep = idle_sleep
        self._last_step_stamp: Optional[int] = None
        # Env-gated obs/command trace for Isaac-vs-MuJoCo diffing (OLI_TRACE=/path.jsonl).
        # Same brain code → identical tap in both sims. Capped so the file stays small.
        self._trace_f = open(os.environ["OLI_TRACE"], "w") if os.environ.get("OLI_TRACE") else None
        self._trace_left = 400

    def step_once(self) -> Optional[PolicyOut]:
        """One iteration: read latest obs; step iff the stamp has advanced enough."""
        obs = self._comm.read_observation()
        if obs is None:
            return None
        # Stamp-pacing: always step the first obs (freeze handshake), then gate on Δ.
        if (
            self._last_step_stamp is not None
            and (obs.stamp_ns - self._last_step_stamp) < self._policy_dt_ns
        ):
            return None
        self._last_step_stamp = obs.stamp_ns

        joy = self._joystick.poll() if self._joystick is not None else None
        policy_in = self._reason.to_policy_in(obs, joy)
        action_out = self._action.step(policy_in)
        # send() dispatches by type: PolicyOut (walk) or GlideCmd (glide) — same loop.
        self._comm.send(action_out)
        if self._recorder is not None:
            self._recorder(obs, policy_in, action_out, joy)
        # The obs/command trace is walk-only (it logs q_des); glide's GlideCmd has none.
        if (self._trace_f is not None and self._trace_left > 0
                and isinstance(action_out, PolicyOut)):
            self._trace_left -= 1

            def _r(a, n):
                return np.round(np.asarray(a, dtype=float).reshape(-1)[:n], 3).tolist()

            self._trace_f.write(json.dumps({
                "t": int(obs.stamp_ns),
                "q": _r(obs.q, 12),          # leg joints (PR 0-11), absolute
                "dq": _r(obs.dq, 12),
                "gyro": _r(obs.gyro, 3),
                "quat": _r(obs.quat_wxyz, 4),
                "qdes": _r(action_out.q_des, 12),  # commanded leg targets = step size
            }) + "\n")
            self._trace_f.flush()
        return action_out

    def run(self, should_continue: Callable[[], bool] = lambda: True) -> None:
        """Connect, then loop `step_once` until `should_continue()` is False."""
        self._comm.connect()
        try:
            while should_continue():
                stepped = self.step_once()
                if stepped is None and self._idle_sleep > 0.0:
                    time.sleep(self._idle_sleep)
        finally:
            self._comm.close()
