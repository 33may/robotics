"""base.py — the brain-side Communication interface the Orchestrator depends on.

The Orchestrator talks to the World only through this ABC; it never sees the world
realization (D4). `BrainComm` (client.py) implements it over a UDS socket to the
World server (`SimComm`/`RealComm`). A future in-process or fake Comm can implement
it too — that is what keeps Reason/Action deployment-invariant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..contracts import CameraFrame, Observation, PolicyOut
from ..glide import GlideCmd


class Comm(ABC):
    """Brain-side edge: connect to the World, read the latest Observation, write PolicyOut."""

    @abstractmethod
    def connect(self, timeout: float = 10.0) -> None:
        """Establish the connection to the World and complete the handshake."""

    @abstractmethod
    def read_observation(self) -> Optional[Observation]:
        """Return the newest Observation (latest-wins drain), or None if none pending."""

    def read_camera_frame(self) -> Optional[CameraFrame]:
        """Return the newest CameraFrame (latest-wins), or None if none pending or this
        Comm has no camera channel. Concrete default: no cameras (override to provide).

        Cameras travel a SEPARATE channel from the control socket (design.md D6), so a
        walk-only Comm simply never wires one and this stays None.
        """
        return None

    @abstractmethod
    def write_policy_out(self, policy_out: PolicyOut) -> None:
        """Send a PolicyOut to the World (non-blocking; drop if the buffer is full)."""

    def write_glide_cmd(self, glide_cmd: GlideCmd) -> None:
        """Send a glide motion command to the World (glide mode — MAY-172).

        Concrete (not abstract) so walk-only Comms need not implement it; glide-capable
        Comms (e.g. `BrainComm`) override it. See `send` for how the mode is dispatched.
        """
        raise NotImplementedError("this Comm does not support glide mode")

    def send(self, msg) -> None:
        """Send the Action's output, dispatching by message type — the single seam that
        lets ONE Orchestrator loop carry either mode: a `PolicyOut` (walk) goes to
        `write_policy_out`, a `GlideCmd` (glide) to `write_glide_cmd`. The walk path is
        byte-identical to calling `write_policy_out` directly.
        """
        if isinstance(msg, GlideCmd):
            self.write_glide_cmd(msg)
        else:
            self.write_policy_out(msg)

    @abstractmethod
    def close(self) -> None:
        """Idempotently tear down the connection."""
