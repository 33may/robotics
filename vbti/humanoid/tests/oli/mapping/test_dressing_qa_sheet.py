"""Tests for the dressing QA contact sheet composer (MAY-173 locdev dressing).

Before/after rows x pose columns, labeled, one PNG artifact for the QA gate.
Pure python (PIL) — runs in the `hum` env.
"""

import numpy as np
import pytest
from PIL import Image

from humanoid.logic.simulation.mapping.dressing.qa_sheet import compose_sheet

pytestmark = pytest.mark.brain

_W, _H = 64, 36


def _frame(color, path):
    Image.new("RGB", (_W, _H), color).save(path)
    return path


def test_sheet_is_two_rows_by_n_pose_columns(tmp_path):
    befores = [_frame((200, 0, 0), tmp_path / f"b{i}.png") for i in range(3)]
    afters = [_frame((0, 200, 0), tmp_path / f"a{i}.png") for i in range(3)]
    out = tmp_path / "sheet.png"
    compose_sheet(befores, afters, ["t=1s", "t=2s", "t=3s"], out)
    sheet = Image.open(out)
    assert sheet.width == 3 * _W
    assert sheet.height > 2 * _H  # two image rows + label band


def test_sheet_places_before_row_above_after_row(tmp_path):
    befores = [_frame((200, 0, 0), tmp_path / "b.png")]
    afters = [_frame((0, 200, 0), tmp_path / "a.png")]
    out = tmp_path / "sheet.png"
    compose_sheet(befores, afters, ["t=1s"], out)
    arr = np.asarray(Image.open(out))
    # sample a pixel well inside each row (rows sit at the bottom, band on top)
    before_px = arr[arr.shape[0] - 2 * _H - 1 + _H // 2 + 1, _W // 2]
    after_px = arr[arr.shape[0] - _H // 2, _W // 2]
    assert before_px[0] > before_px[1]  # red row on top
    assert after_px[1] > after_px[0]  # green row below


def test_mismatched_lengths_raise(tmp_path):
    b = [_frame((1, 1, 1), tmp_path / "b.png")]
    with pytest.raises(ValueError):
        compose_sheet(b, [], ["t=1s"], tmp_path / "s.png")
