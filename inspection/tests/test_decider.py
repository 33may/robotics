#!/usr/bin/env python3
"""Tests for the decider data model. Run: p inspection/tests/test_decider.py"""
import queue
import time

from inspection.run.decider import (Look, Answer, Quit, MenuItem, gloss,
                                    build_menu, parse_command, Console,
                                    TerminalDecider, Ctx)


def test_gloss():
    # +1 azimuth step = "one step right" (D4 example: h=5 -> h=6)
    assert gloss((5, 1), (6, 1)) == "one step right, same height"
    assert gloss((5, 1), (3, 1)) == "two steps left, same height"
    assert gloss((11, 0), (0, 0)) == "one step right, same height"  # wrap
    assert gloss((5, 1), (11, 2)) == "opposite side, higher"        # |dh|=6
    assert gloss((5, 1), (5, 0)) == "same side, lower"
    assert gloss(None, (3, 2)) == "elevation 70 deg"                # survey


def test_build_menu():
    reach = {(0, 0): 0.0, (1, 0): 0.5, (6, 0): 0.0, (2, 0): None}
    menu = build_menu(reach, visited={(0, 0)}, cur=(0, 0))
    cells = [(it.h, it.v) for it in menu]
    assert (2, 0) not in cells          # unreachable filtered
    assert (0, 0) not in cells          # visited filtered
    assert cells[0] == (1, 0)           # nearest first
    assert cells[-1] == (6, 0)          # opposite side last
    assert menu[0].gloss == "one step right, same height"


def test_parse_command():
    assert parse_command("look 6 1") == Look(6, 1)
    assert parse_command("l 6 1") == Look(6, 1)
    assert parse_command("answer no logo visible") == Answer("no logo visible")
    assert parse_command("a yes") == Answer("yes")
    assert parse_command("quit") == Quit()
    assert parse_command("look six 1") is None
    assert parse_command("look 6") is None
    assert parse_command("") is None
    assert parse_command("garbage") is None


class FeedStream:
    """Line source the test controls in real time (stdin stand-in)."""
    def __init__(self):
        self.q = queue.Queue()

    def feed(self, line):
        self.q.put(line + "\n")

    def __iter__(self):
        while True:
            line = self.q.get()
            if line is None:
                return
            yield line


def test_console_stop_arming():
    fs = FeedStream()
    con = Console(stream=fs)
    fs.feed("first")
    assert con.readline() == "first"
    con.arm_stop()                      # robot "moving" now
    fs.feed("anything")                 # any line while armed = STOP
    assert con.stop_event.wait(timeout=1.0)
    con.disarm_stop()
    fs.feed("after")
    assert con.readline() == "after"    # queue not polluted by the stop line


def test_terminal_decider_parses_until_valid():
    fs = FeedStream()
    dec = TerminalDecider(Console(stream=fs))
    ctx = Ctx(question="logo?", step=1, current_cell=(0, 0),
              map_ascii="(map)", menu=[MenuItem(1, 0, "one step right")])
    fs.feed("nonsense")                 # rejected, re-prompts
    fs.feed("look 1 0")
    assert dec.decide(ctx) == Look(1, 0)


def main():
    test_gloss(); test_build_menu(); test_parse_command()
    test_console_stop_arming(); test_terminal_decider_parses_until_valid()
    print("OK test_decider")


if __name__ == "__main__":
    main()
