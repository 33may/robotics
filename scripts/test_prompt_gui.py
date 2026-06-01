#!/usr/bin/env python3
"""Standalone prompt GUI test, no robot/cameras/model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vbti.logic.inference.prompt_gui import FullInferenceGui, PromptGuiStatus

RED_PROMPT = "Pick up the duck and place it in the red cup"
BLACK_PROMPT = "Pick up the duck and place it in the black cup"

state = {"prompt": RED_PROMPT, "version": 0, "source": "initial", "mode": "smooth", "step": 0}


def get_status():
    return PromptGuiStatus(
        prompt=state["prompt"],
        version=state["version"],
        source=state["source"],
        mode=state["mode"],
        step=state["step"],
        running=True,
    )


def on_prompt(prompt: str, source: str):
    state["prompt"] = prompt
    state["source"] = source
    state["version"] += 1
    print(f"PROMPT v{state['version']} from {source}: {prompt}")


def on_mode(mode: str):
    state["mode"] = mode
    print(f"MODE: {mode}")


def main():
    gui = FullInferenceGui(get_status, on_prompt, on_mode, RED_PROMPT, BLACK_PROMPT)
    gui.start()
    try:
        while not gui.closed:
            state["step"] += 1
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:] = (25, 18, 8)
            cv2.putText(frame, "SIMULATED CAMERA FEED", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, f"step {state['step']}", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 200, 80), 3)
            cv2.circle(frame, (420, 430), 90, (40, 40, 220), -1)
            cv2.putText(frame, "RED", (360, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.circle(frame, (830, 430), 90, (5, 5, 5), -1)
            cv2.putText(frame, "BLACK", (760, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            gui.update_frame(frame)
            time.sleep(0.2)
    except KeyboardInterrupt:
        gui.close()


if __name__ == "__main__":
    main()
