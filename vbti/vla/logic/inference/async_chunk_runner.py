"""Async chunk runner with optional Real-Time Chunking (RTC).

Decouples policy inference from robot execution: a worker thread runs
``predict_action_chunk`` in the background while the main thread pops actions
at robot rate. When the worker delivers a fresh chunk, it is spliced in at the
time-aligned position so the new chunk's first actions match the trajectory
the robot already committed to.

When RTC is enabled, the leftover tail of the previous chunk is passed as
``prev_chunk_left_over`` to the denoiser, which inpaints the new chunk's early
actions to match it (PI's Real-Time Chunking).

Usage:
    runner = AsyncChunkRunner(policy, postprocessor, execution_horizon=10)
    runner.start()
    for trial in trials:
        runner.reset()
        while not done:
            obs = build_observation(...)
            action_deg = runner.step(obs)
            robot.send_action(action_deg)
    runner.stop()
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

import numpy as np
import torch


class AsyncChunkRunner:
    def __init__(
        self,
        policy,
        postprocessor,
        execution_horizon: int = 10,
        enable_rtc: bool = True,
        max_guidance_weight: float = 10.0,
    ):
        self.policy = policy
        self.postprocessor = postprocessor
        self.execution_horizon = int(execution_horizon)
        self.enable_rtc = bool(enable_rtc)

        if self.enable_rtc:
            self._enable_rtc(max_guidance_weight)

        # Main-thread chunk state.
        self._chunk_norm: torch.Tensor | None = None     # (T, A) normalized, on device
        self._chunk_deg:  np.ndarray  | None = None      # (T, A) degrees
        self._exec_idx:   int         = 0

        # Worker → main handoff (latest chunk wins).
        self._next_chunk_norm: torch.Tensor | None = None
        self._next_chunk_deg:  np.ndarray  | None = None
        self._handoff_lock = threading.Lock()
        self._policy_lock = threading.Lock()

        # Main → worker request queue (depth 1 — only the latest matters).
        # Includes a generation id so reset() can invalidate in-flight predictions.
        self._generation = 0
        self._pending_request: tuple[int, Any, torch.Tensor] | None = None
        self._request_lock  = threading.Lock()
        self._request_event = threading.Event()
        self._request_in_flight = False
        self._stop_event    = threading.Event()
        self._worker: threading.Thread | None = None

    def _enable_rtc(self, max_guidance_weight: float):
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=self.execution_horizon,
            max_guidance_weight=max_guidance_weight,
        )
        self.policy.init_rtc_processor()

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._inference_loop, name="AsyncChunkRunner", daemon=True
        )
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self._request_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._worker = None

    def reset(self):
        """Clear all chunk state. Call between trials or after prompt changes."""
        with self._request_lock:
            self._generation += 1
            self._pending_request = None
        with self._handoff_lock:
            self._next_chunk_norm = None
            self._next_chunk_deg  = None
        self._chunk_norm = None
        self._chunk_deg  = None
        self._exec_idx   = 0
        self._request_in_flight = False
        with self._policy_lock:
            self.policy.reset()  # clears the (now-unused) internal action queue

    def step(self, obs: dict) -> np.ndarray:
        """Return the next action (degrees) to send to the robot.

        Bootstraps synchronously on the very first call. Subsequent calls pop
        from the cached chunk and trigger background re-planning at the
        ``execution_horizon`` boundary.
        """
        if self._chunk_deg is None:
            self._predict_sync(obs, prev_leftover=None)

        self._maybe_apply_handoff()

        assert self._chunk_deg is not None and self._chunk_norm is not None

        if self._exec_idx >= self._chunk_deg.shape[0]:
            # Worker missed its window — fall back to a sync re-plan.
            self._predict_sync(obs, prev_leftover=None)
            self._maybe_apply_handoff()
            assert self._chunk_deg is not None and self._chunk_norm is not None

        action = self._chunk_deg[self._exec_idx]
        self._exec_idx += 1

        if (self._exec_idx >= self.execution_horizon
                and not self._request_in_flight
                and self.execution_horizon < self._chunk_norm.shape[0]):
            leftover = self._chunk_norm[self.execution_horizon:].clone()
            self._request_in_flight = True
            self._enqueue_request(obs, leftover)

        return action

    def _maybe_apply_handoff(self):
        with self._handoff_lock:
            new_norm = self._next_chunk_norm
            new_deg  = self._next_chunk_deg
            self._next_chunk_norm = None
            self._next_chunk_deg  = None

        if new_deg is None:
            return

        # Time-align: the new chunk's first ``execution_horizon`` actions are
        # inpainted to match old_chunk[K:K+T]. We've already executed
        # (exec_idx - K) of those, so jump that far into the new chunk.
        adjust = max(0, self._exec_idx - self.execution_horizon)
        adjust = min(adjust, new_deg.shape[0] - 1)

        self._chunk_norm = new_norm
        self._chunk_deg  = new_deg
        self._exec_idx   = adjust
        self._request_in_flight = False

    def _enqueue_request(self, obs: dict, prev_leftover: torch.Tensor):
        with self._request_lock:
            self._pending_request = (self._generation, obs, prev_leftover)
        self._request_event.set()

    def _predict_sync(self, obs: dict, prev_leftover: torch.Tensor | None):
        actions_norm, actions_deg = self._predict(obs, prev_leftover)
        self._chunk_norm = actions_norm
        self._chunk_deg  = actions_deg
        self._exec_idx   = 0

    def _predict(
        self, obs: dict, prev_leftover: torch.Tensor | None
    ) -> tuple[torch.Tensor, np.ndarray]:
        kwargs: dict[str, Any] = {}
        if self.enable_rtc and prev_leftover is not None:
            kwargs["prev_chunk_left_over"] = prev_leftover
            kwargs["inference_delay"] = 0
            kwargs["execution_horizon"] = self.execution_horizon

        with self._policy_lock:
            actions_norm = self.policy.predict_action_chunk(obs, **kwargs)
            actions_deg  = self.postprocessor({"action": actions_norm})["action"]
        return (
            actions_norm[0].detach(),
            actions_deg[0].detach().cpu().numpy(),
        )

    def _inference_loop(self):
        while not self._stop_event.is_set():
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._request_lock:
                req = self._pending_request
                self._pending_request = None
            if req is None:
                continue

            generation, obs, prev_leftover = req
            try:
                actions_norm, actions_deg = self._predict(obs, prev_leftover)
            except Exception:
                print("[AsyncChunkRunner] inference error:")
                traceback.print_exc()
                continue

            with self._request_lock:
                if generation != self._generation:
                    continue

            with self._handoff_lock:
                self._next_chunk_norm = actions_norm
                self._next_chunk_deg  = actions_deg
