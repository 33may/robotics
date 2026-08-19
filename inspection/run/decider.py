#!/usr/bin/env python3
"""Decider seam for the v1 loop — the AI slot from the project definition.

D3/D4 (inspection/2026-08-12-ai-inspection-project-definition.md): the
decider READs the newest capture (a comment) and DECIDEs look(h, v) or
answer(text). v1 is Anton at the terminal; BusDecider (UI toolkit) and
AgentDecider swap in later without touching loop.py.
"""
import queue
import sys
import threading
from dataclasses import dataclass, field

from inspection.view.viewsphere import H_BINS, V_ELEVATIONS

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


@dataclass(frozen=True)
class Look:
    h: int
    v: int


@dataclass(frozen=True)
class Answer:
    text: str


@dataclass(frozen=True)
class Quit:
    pass


@dataclass
class MenuItem:
    h: int
    v: int
    gloss: str
    visited: bool = False


@dataclass
class Ctx:
    """Everything the decider sees — D4: 'injected every turn, not tools'."""
    question: str
    step: int
    current_cell: tuple | None      # None before the first look (survey)
    map_ascii: str
    menu: list                      # [MenuItem], nearest first
    comments: list = field(default_factory=list)   # prior READs, oldest first


def _dh(cur_h, h, n_h):
    """Signed shortest azimuth steps cur -> cell; +1 = one step right."""
    return (h - cur_h + n_h // 2) % n_h - n_h // 2


def gloss(cur, cell, n_h=H_BINS, elevations=V_ELEVATIONS):
    """Egocentric label for cell relative to cur (D4 menu gloss)."""
    if cur is None:
        return f"elevation {elevations[cell[1]]:.0f} deg"
    dh = _dh(cur[0], cell[0], n_h)
    dv = cell[1] - cur[1]
    if abs(dh) == n_h // 2:
        side = "opposite side"
    elif dh == 0:
        side = "same side"
    else:
        n = abs(dh)
        side = f"{_WORDS[n]} step{'s' if n > 1 else ''} " \
               f"{'right' if dh > 0 else 'left'}"
    height = "same height" if dv == 0 else ("higher" if dv > 0 else "lower")
    return f"{side}, {height}"


def build_menu(reach, visited, cur, elevations=V_ELEVATIONS):
    """Feasible unvisited cells, nearest to cur first."""
    items = [MenuItem(h, v, gloss(cur, (h, v), elevations=elevations))
             for (h, v), roll in sorted(reach.items())
             if roll is not None and (h, v) not in visited]

    def key(it):
        if cur is None:
            return (it.v, it.h, 0)
        dh = abs(_dh(cur[0], it.h, H_BINS))
        return (dh + abs(it.v - cur[1]), dh, it.h)

    return sorted(items, key=key)


def parse_command(line):
    """'look 6 1'/'l 6 1' -> Look; 'answer <text>'/'a <text>' -> Answer;
    'quit'/'q' -> Quit; None on anything unparseable."""
    toks = line.strip().split()
    if not toks:
        return None
    cmd = toks[0].lower()
    if cmd in ("look", "l") and len(toks) == 3:
        try:
            return Look(int(toks[1]), int(toks[2]))
        except ValueError:
            return None
    if cmd in ("answer", "a") and len(toks) >= 2:
        return Answer(line.strip().split(None, 1)[1])
    if cmd in ("quit", "q") and len(toks) == 1:
        return Quit()
    return None


class Console:
    """Single owner of terminal input. A daemon thread reads lines; while
    the stop is ARMED (robot in motion) any line fires stop_event instead
    of queueing — ENTER is the software stop button. The hardware e-stop
    is unaffected and always available."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._q = queue.Queue()
        self.stop_event = threading.Event()
        self._armed = False
        self._lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self._stream:
            with self._lock:
                if self._armed:
                    self.stop_event.set()
                else:
                    self._q.put(line.rstrip("\n"))

    def arm_stop(self):
        with self._lock:
            self.stop_event.clear()
            self._armed = True

    def disarm_stop(self):
        with self._lock:
            self._armed = False

    def readline(self, prompt=""):
        if prompt:
            print(prompt, end="", flush=True)
        return self._q.get()


def show_capture(rgb):
    """Stopgap viewer until the UI toolkit lands: one reused cv2 window."""
    import cv2
    cv2.imshow("newest capture", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(200)        # render once; window persists between turns


class TerminalDecider:
    """Anton in the model slot. The printed ctx IS the future model prompt."""

    def __init__(self, console):
        self.console = console

    def read(self, cap):
        if cap.get("rgb") is not None:
            show_capture(cap["rgb"])
        print(f"\n[READ] capture: {cap.get('dir', '(fake)')}")
        return self.console.readline("comment> ")

    def decide(self, ctx):
        print(f"\n[DECIDE] step {ctx.step}   question: {ctx.question!r}")
        print(ctx.map_ascii)
        cur = (f"h={ctx.current_cell[0]} v={ctx.current_cell[1]}"
               if ctx.current_cell else "survey pose")
        print(f"you are at: {cur}")
        for it in ctx.menu[:12]:
            print(f"  look {it.h} {it.v}    {it.gloss}")
        if len(ctx.menu) > 12:
            print(f"  ... {len(ctx.menu) - 12} more cells on the map above")
        while True:
            act = parse_command(
                self.console.readline("look H V | answer TEXT | quit > "))
            if act is not None:
                return act
            print("could not parse — try 'look 6 1', 'answer no logo', 'quit'")
