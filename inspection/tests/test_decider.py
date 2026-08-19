#!/usr/bin/env python3
"""Tests for the decider data model. Run: p inspection/tests/test_decider.py"""
from inspection.run.decider import (Look, Answer, Quit, MenuItem, gloss,
                                    build_menu, parse_command)


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


def main():
    test_gloss(); test_build_menu(); test_parse_command()
    print("OK test_decider")


if __name__ == "__main__":
    main()
