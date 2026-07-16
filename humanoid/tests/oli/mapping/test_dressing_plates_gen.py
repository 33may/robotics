"""Tests for dressing plate/poster texture generation (MAY-173 locdev dressing).

Number plates are anti-aliasing landmarks for cuVSLAM: each must be *globally
unique after grayscale conversion* (the tracker doesn't care about hue), so
style variation is shape/luminance-driven: alternating polarity (dark-on-light
vs light-on-dark), cycling border styles, and the unique number itself.

Pure python (PIL + stdlib) — runs in the `hum` env.
"""

import numpy as np
import pytest
from PIL import Image

from humanoid.logic.simulation.mapping.dressing.plates_gen import (
    plate_style,
    render_plate,
    render_poster,
)

pytestmark = pytest.mark.brain


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def test_render_plate_returns_square_rgb_of_requested_size():
    img = render_plate(7, px=256)
    assert img.mode == "RGB"
    assert img.size == (256, 256)


def test_render_plate_is_deterministic():
    a = render_plate(42, px=128)
    b = render_plate(42, px=128)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_plate_is_not_blank():
    # A drawn number + border must produce real structure, not a flat fill.
    g = _gray(render_plate(3, px=256))
    assert g.std() > 20


def test_consecutive_plates_differ_in_grayscale():
    # The cuVSLAM-facing requirement: uniqueness survives grayscale.
    imgs = [_gray(render_plate(n, px=256)) for n in range(12)]
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            diff = np.abs(imgs[i] - imgs[j]).mean()
            assert diff > 2.0, f"plates {i} and {j} too similar in grayscale"


def test_polarity_alternates_between_consecutive_numbers():
    # Even index: dark-on-light (bright background). Odd: light-on-dark.
    g_even = _gray(render_plate(0, px=256))
    g_odd = _gray(render_plate(1, px=256))
    assert np.median(g_even) > 128  # light background
    assert np.median(g_odd) < 128  # dark background


def test_style_cycles_polarity_color_and_border():
    s0, s1 = plate_style(0), plate_style(1)
    assert s0.dark_on_light != s1.dark_on_light
    # Border styles cycle with a period coprime to 2 so (polarity, border)
    # combinations don't repeat every other plate.
    borders = {plate_style(i).border for i in range(8)}
    assert len(borders) >= 3
    colors = {plate_style(i).color for i in range(12)}
    assert len(colors) >= 4


def test_render_poster_returns_rgb_with_requested_aspect(tmp_path):
    # Poster generator takes the VBTI logo file; use a synthetic logo here.
    logo = tmp_path / "logo.png"
    Image.new("RGBA", (100, 40), (200, 30, 30, 255)).save(logo)
    img = render_poster(logo, px_w=400, px_h=300)
    assert img.mode == "RGB"
    assert img.size == (400, 300)


def test_poster_contains_logo_pixels(tmp_path):
    logo = tmp_path / "logo.png"
    Image.new("RGBA", (100, 40), (200, 30, 30, 255)).save(logo)
    arr = np.asarray(render_poster(logo, px_w=400, px_h=300))
    # The red synthetic logo must land somewhere on the canvas.
    reddish = (arr[..., 0] > 150) & (arr[..., 1] < 100)
    assert reddish.any()


def test_module_is_pure():
    import sys

    import humanoid.logic.simulation.mapping.dressing.plates_gen  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules


def test_write_textures_renders_every_layout_plate_and_poster(tmp_path):
    from humanoid.logic.simulation.mapping.dressing.plates_gen import write_textures

    logo = tmp_path / "logo.png"
    Image.new("RGBA", (100, 40), (200, 30, 30, 255)).save(logo)
    layout = {
        "plates": [{"number": 4, "texture": "plate_004.png"},
                   {"number": 7, "texture": "plate_007.png"}],
        "posters": [{"texture": "poster_vbti.png"}],
    }
    out = tmp_path / "textures"
    n = write_textures(layout, logo, out)
    assert n == 3
    assert (out / "plate_004.png").exists()
    assert (out / "plate_007.png").exists()
    assert (out / "poster_vbti.png").exists()
