"""plates_gen.py — number-plate + poster texture generator (MAY-173 dressing).

Anti-aliasing landmarks for cuVSLAM: every plate must stay unique after
grayscale conversion, so identity is carried by the number glyphs, alternating
luminance polarity, and cycling border *shapes* — hue is a bonus, never the
only discriminator. Fully deterministic (style = f(number)): re-running the
bake reproduces byte-identical textures.

Pure python (PIL + stdlib). Runs in the `hum` env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Saturated base hues; light/dark variants are derived per polarity.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (200, 40, 40),    # red
    (40, 90, 200),    # blue
    (30, 140, 60),    # green
    (210, 130, 20),   # orange
    (120, 50, 170),   # purple
    (0, 140, 150),    # teal
    (150, 110, 20),   # ochre
)
_BORDERS: tuple[str, ...] = ("solid", "double", "dashed", "thick", "corners")


@dataclass(frozen=True)
class PlateStyle:
    dark_on_light: bool
    color: tuple[int, int, int]
    border: str


def plate_style(number: int) -> PlateStyle:
    """Deterministic style for a plate number. Cycle lengths 2/5/7 are pairwise
    coprime → the (polarity, border, color) triple repeats only every 70."""
    return PlateStyle(
        dark_on_light=(number % 2 == 0),
        color=_PALETTE[number % len(_PALETTE)],
        border=_BORDERS[number % len(_BORDERS)],
    )


def _find_font() -> str:
    candidates = []
    try:
        import matplotlib

        candidates.append(
            Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    except ImportError:
        pass
    candidates += [
        Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError("no DejaVuSans-Bold.ttf found (install matplotlib or dejavu fonts)")


def _shades(style: PlateStyle) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(background, foreground) honoring luminance polarity."""
    r, g, b = style.color
    light = (min(255, r + 190), min(255, g + 190), min(255, b + 190))
    dark = (r // 4, g // 4, b // 4)
    return (light, dark) if style.dark_on_light else (dark, light)


def _draw_border(d: ImageDraw.ImageDraw, px: int, style: str, fg) -> None:
    m = max(2, px // 32)          # margin
    w = max(2, px // 42)          # line width
    box = (m, m, px - 1 - m, px - 1 - m)
    if style == "solid":
        d.rectangle(box, outline=fg, width=w)
    elif style == "thick":
        d.rectangle(box, outline=fg, width=3 * w)
    elif style == "double":
        d.rectangle(box, outline=fg, width=w)
        g = 3 * w
        d.rectangle((m + g, m + g, px - 1 - m - g, px - 1 - m - g), outline=fg, width=w)
    elif style == "dashed":
        step = px // 8
        for i in range(0, px - 2 * m, step):
            a = m + i
            b = min(a + step // 2, px - 1 - m)
            d.rectangle((a, m, b, m + w), fill=fg)                       # top
            d.rectangle((a, px - 1 - m - w, b, px - 1 - m), fill=fg)     # bottom
            d.rectangle((m, a, m + w, b), fill=fg)                       # left
            d.rectangle((px - 1 - m - w, a, px - 1 - m, b), fill=fg)     # right
    elif style == "corners":
        arm = px // 5
        for cx, cy, sx, sy in ((m, m, 1, 1), (px - 1 - m, m, -1, 1),
                               (m, px - 1 - m, 1, -1), (px - 1 - m, px - 1 - m, -1, -1)):
            d.rectangle(sorted_box(cx, cy, cx + sx * arm, cy + sy * 3 * w), fill=fg)
            d.rectangle(sorted_box(cx, cy, cx + sx * 3 * w, cy + sy * arm), fill=fg)
    else:  # pragma: no cover - style enum is closed
        raise ValueError(f"unknown border style {style!r}")


def sorted_box(x0: float, y0: float, x1: float, y1: float) -> tuple:
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def render_plate(number: int, px: int = 512) -> Image.Image:
    """Square RGB plate texture: big centered number, styled border, polarity."""
    style = plate_style(number)
    bg, fg = _shades(style)
    img = Image.new("RGB", (px, px), bg)
    d = ImageDraw.Draw(img)
    _draw_border(d, px, style.border, fg)
    font = ImageFont.truetype(_find_font(), int(px * 0.45))
    d.text((px / 2, px / 2), str(number), fill=fg, font=font, anchor="mm")
    return img


def render_poster(logo_path: str | Path, px_w: int, px_h: int) -> Image.Image:
    """Poster texture: logo centered on white with a thin dark frame."""
    img = Image.new("RGB", (px_w, px_h), (245, 245, 245))
    logo = Image.open(logo_path).convert("RGBA")
    box_w, box_h = int(px_w * 0.8), int(px_h * 0.8)
    scale = min(box_w / logo.width, box_h / logo.height)
    logo = logo.resize((max(1, int(logo.width * scale)), max(1, int(logo.height * scale))))
    img.paste(logo, ((px_w - logo.width) // 2, (px_h - logo.height) // 2), logo)
    d = ImageDraw.Draw(img)
    w = max(2, min(px_w, px_h) // 60)
    d.rectangle((0, 0, px_w - 1, px_h - 1), outline=(40, 40, 40), width=w)
    return img


def write_textures(layout: dict, logo_path: str | Path, out_dir: str | Path,
                   plate_px: int = 512, poster_px: tuple[int, int] = (1024, 768)) -> int:
    """Render every texture the layout references into out_dir. Returns count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in layout["plates"]:
        render_plate(p["number"], px=plate_px).save(out / p["texture"])
        n += 1
    poster_names = {po["texture"] for po in layout["posters"]}
    for name in sorted(poster_names):
        render_poster(logo_path, *poster_px).save(out / name)
        n += 1
    return n


def main() -> None:
    import argparse

    from humanoid.logic.simulation.mapping.dressing.layout_gen import load_layout

    ap = argparse.ArgumentParser(description="Render dressing textures for a layout.")
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--logo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="textures output dir")
    args = ap.parse_args()
    n = write_textures(load_layout(args.layout), args.logo, args.out)
    print(f"[plates_gen] {n} textures -> {args.out}")


if __name__ == "__main__":
    main()
