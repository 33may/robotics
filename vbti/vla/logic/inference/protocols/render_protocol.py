"""Unified renderer for any eval protocol — handles both schemas.

Schemas supported:
  - "entities" (dual_cup_30 etc.): trial.entities = [{name, kind, color, px, dir_deg?}, ...]
                                   trial.target = entity name
                                   trial.task   = per-trial prompt
  - "legacy"   (id_scale_60, checkpoint_sweep_*): trial = {duck_px, cup_px, duck_dir_deg, cup_group?}
                                                  proto.task = global prompt

Both schemas render the same way: a wide grid of per-trial thumbnails on the
top-camera base_frame.png, target highlighted, tags labelled.

Layout: auto-pick (rows, cols) targeting aspect ratio ≈1.33 (4:3 — wider than
tall but not extreme).
  - N=20 → 5×4   (or 4×5)
  - N=30 → 6×5
  - N=60 → 10×6

Marker sizes are tuned for 220×170 thumbnail size — visible at a glance.

Usage:
  python vbti/logic/inference/protocols/render_protocol.py <protocol-name>
  python vbti/logic/inference/protocols/render_protocol.py dual_cup_30
  python vbti/logic/inference/protocols/render_protocol.py id_scale_60
  python vbti/logic/inference/protocols/render_protocol.py checkpoint_sweep_no_ood
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROTO_DIR    = Path(__file__).resolve().parent
RENDERS_DIR  = PROTO_DIR / "renders"
BASE_FRAME   = PROTO_DIR / "base_frame.png"

# Entity colors — facecolor lookup
CUP_FILL_COLORS = {
    "red":     "#cc1f1f",
    "black":   "#1a1a1a",
    "blue":    "#1f55cc",
    "green":   "#1faa44",
    "yellow":  "#d8b020",
    "default": "#9966cc",   # fallback for legacy / uncolored cups
}
DUCK_FILL_COLOR = "#f4d000"

# Group palette for legacy cup_group rendering — distinct, accessible
GROUP_PALETTE = [
    "#cc1f1f", "#1faa44", "#1f55cc",
    "#cc8822", "#aa22cc", "#22aaaa",
    "#ee5577", "#77aa22",
]

# Marker sizing — tuned for thumbnails (in image-pixel coordinates).
# Workspace bbox is ~300 px wide, panels render at ~400 px → keep markers
# readable but not dominating the panel.
CUP_RADIUS       = 12
CUP_TARGET_RADIUS= 14
DUCK_RADIUS      = 11
ARROW_LEN        = 22
TARGET_OUTLINE_W = 1.5     # subtle white ring on target — small accent, not a halo
NORMAL_OUTLINE_W = 1.0


# ── Schema normalisation ─────────────────────────────────────────────────────

def _normalise_trial(proto: dict, trial: dict) -> dict:
    """Convert any-schema trial → entity-style trial.

    Returns a dict with: trial_id, zone, entities, target, task, tags
    """
    schema = proto.get("schema", "legacy")

    if schema == "entities":
        return {
            "trial_id": trial["trial_id"],
            "zone":     trial.get("zone", ""),
            "entities": trial["entities"],
            "target":   trial.get("target"),
            "task":     trial.get("task", proto.get("task", "")),
            "tags":     trial.get("tags", {}),
        }

    # Legacy schema → synthesise entities
    entities = [{
        "name":    "duck",
        "kind":    "duck",
        "color":   "yellow",
        "px":      list(trial["duck_px"]),
        "dir_deg": trial.get("duck_dir_deg"),
    }]
    cup_group = trial.get("cup_group")
    cup_color_hex = (
        GROUP_PALETTE[cup_group % len(GROUP_PALETTE)] if cup_group is not None
        else CUP_FILL_COLORS["default"]
    )
    entities.append({
        "name":      f"C{cup_group}" if cup_group is not None else "cup",
        "kind":      "cup",
        "color_hex": cup_color_hex,           # explicit hex (bypasses CUP_FILL_COLORS)
        "px":        list(trial["cup_px"]),
    })

    tags = {}
    if "zone" in trial:       tags["zone"]      = trial["zone"]
    if cup_group is not None: tags["cup_group"] = cup_group

    return {
        "trial_id": trial["trial_id"],
        "zone":     trial.get("zone", ""),
        "entities": entities,
        "target":   entities[1]["name"],     # only one cup → it is the target
        "task":     proto.get("task", ""),
        "tags":     tags,
    }


# ── Drawing primitives ───────────────────────────────────────────────────────

def _entity_facecolor(ent: dict) -> str:
    """Resolve facecolor: explicit hex first, then color-name lookup, else default."""
    if "color_hex" in ent:
        return ent["color_hex"]
    return CUP_FILL_COLORS.get(ent.get("color", "default"), CUP_FILL_COLORS["default"])


def _draw_trial(ax, trial: dict, base_img: Image.Image,
                bbox: tuple,
                nogo_bbox: tuple | None = None,
                duck_nogo_bbox: tuple | None = None):
    """Draw one trial layout on a matplotlib axes."""
    ax.imshow(base_img, extent=(0, base_img.width, base_img.height, 0))

    x0, y0, x1, y1 = bbox
    ax.add_patch(patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="#33dd33", linewidth=1.0, zorder=2,
    ))

    # Cup no-go zone — gray fill, orange-red dashed border
    if nogo_bbox is not None:
        rx0, ry0, rx1, ry1 = nogo_bbox
        ax.add_patch(patches.Rectangle(
            (rx0, ry0), rx1 - rx0, ry1 - ry0,
            facecolor="#888888", alpha=0.24,
            edgecolor="#cc4422", linewidth=1.0, linestyle="--", zorder=3,
        ))
    # Duck no-go zone — purple-ish, denser fill, drawn ON TOP of cup zone
    if duck_nogo_bbox is not None:
        dx0, dy0, dx1, dy1 = duck_nogo_bbox
        ax.add_patch(patches.Rectangle(
            (dx0, dy0), dx1 - dx0, dy1 - dy0,
            facecolor="#8855cc", alpha=0.32,
            edgecolor="#5522aa", linewidth=1.2, linestyle=":", zorder=3,
        ))

    target_name = trial.get("target")
    for ent in trial["entities"]:
        x, y = ent["px"]
        kind = ent["kind"]
        is_target = ent["name"] == target_name

        if kind == "cup":
            face = _entity_facecolor(ent)
            edge = "white" if is_target else "#444444"
            lw   = TARGET_OUTLINE_W if is_target else NORMAL_OUTLINE_W
            r    = CUP_TARGET_RADIUS if is_target else CUP_RADIUS
            ax.add_patch(patches.Circle(
                (x, y), r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=4,
            ))
            # Letters intentionally omitted — color + size + outline encode target

        elif kind == "duck":
            ax.add_patch(patches.Circle(
                (x, y), DUCK_RADIUS, facecolor=DUCK_FILL_COLOR,
                edgecolor="white", linewidth=NORMAL_OUTLINE_W, zorder=4,
            ))
            if ent.get("dir_deg") is not None:
                rad = math.radians(ent["dir_deg"])
                ex  = x + ARROW_LEN * math.cos(rad)
                ey  = y + ARROW_LEN * math.sin(rad)
                ax.annotate(
                    "", xy=(ex, ey), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#ff8800", lw=2.0),
                    zorder=6,
                )

    pad = 15   # tight zoom — workspace fills the panel
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)   # inverted y for image coords
    ax.set_xticks([])
    ax.set_yticks([])

    # Trial header — pull tag values via .get() to keep type narrowing happy
    tags = trial.get("tags", {})
    scene         = tags.get("scene")
    target_color  = tags.get("target_color")
    target_side   = tags.get("target_side")
    target_closer = tags.get("target_closer")
    cup_group     = tags.get("cup_group")
    zone          = tags.get("zone")

    tag_bits = []
    # scene is the primary descriptor — show first if present
    if scene: tag_bits.append(scene)
    # target_color: only add if not already encoded in scene name
    if target_color and (not scene or "_" + target_color not in scene):
        tag_bits.append(f"T:{target_color[0].upper()}")
    if target_side:                tag_bits.append(f"S:{target_side[0].upper()}")
    if target_closer is not None:  tag_bits.append(f"C:{'Y' if target_closer else 'N'}")
    if cup_group is not None:      tag_bits.append(f"G:{cup_group}")
    if zone and not tag_bits:      tag_bits.append(f"Z:{zone}")
    tag_str = "  ".join(tag_bits)

    # Short prompt — "→ red cup", "→ black cup" — derived from trial.task
    task = trial.get("task", "")
    if "place it in the " in task:
        short_prompt = "→ " + task.split("place it in the ", 1)[1].rstrip(".")
    elif task:
        short_prompt = "→ " + task[:32]
    else:
        short_prompt = ""

    title = f"#{trial['trial_id']:02d}  {tag_str}"
    if short_prompt:
        title = f"{title}\n{short_prompt}"
    ax.set_title(title, fontsize=8)


# ── Layout ───────────────────────────────────────────────────────────────────

def best_grid(n: int, target_ratio: float = 1.33) -> tuple[int, int]:
    """Pick (rows, cols) optimising for ultrawide (cols/rows ≈ target_ratio)
    while penalising empty cells."""
    best_cost  = math.inf
    best_rows  = 1
    best_cols  = n
    for rows in range(1, 8):
        cols  = math.ceil(n / rows)
        ratio = cols / rows
        empty = cols * rows - n
        cost  = abs(ratio - target_ratio) + 0.5 * empty
        if cost < best_cost:
            best_cost = cost
            best_rows = rows
            best_cols = cols
    return best_rows, best_cols


# ── Public render entrypoint ─────────────────────────────────────────────────

def render(protocol_name: str) -> dict[str, Path]:
    """Render both the per-trial overview grid AND the single-table positions
    plot for a protocol. Returns a dict {kind: path} with both outputs."""
    overview = render_overview(protocol_name)
    positions = render_positions(protocol_name)
    return {"overview": overview, "positions": positions}


def render_overview(protocol_name: str) -> Path:
    """Per-trial grid: one panel per trial showing the layout."""
    proto_path = PROTO_DIR / f"{protocol_name}.json"
    proto = json.loads(proto_path.read_text())
    raw_trials = proto["trials"]
    trials = [_normalise_trial(proto, t) for t in raw_trials]
    n = len(trials)

    rows, cols = best_grid(n)

    base = Image.open(BASE_FRAME).convert("RGB")
    bbox = tuple(proto.get(
        "workspace_bbox",
        proto.get("table_bbox", (160, 83, 535, 419))   # fall back to table mapping
    ))
    nogo_raw = proto.get("robot_nogo_bbox")
    nogo_bbox = tuple(nogo_raw) if nogo_raw else None
    duck_nogo_raw = proto.get("duck_nogo_bbox")
    duck_nogo_bbox = tuple(duck_nogo_raw) if duck_nogo_raw else None

    # Thumbnail target ~220×170 px → in matplotlib inches at dpi=180 ≈ 1.22×0.94
    panel_w_in = 2.2
    panel_h_in = 1.7
    fig, axes = plt.subplots(rows, cols, figsize=(cols * panel_w_in, rows * panel_h_in))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, trial in enumerate(trials):
        _draw_trial(axes[i], trial, base, bbox,
                    nogo_bbox=nogo_bbox, duck_nogo_bbox=duck_nogo_bbox)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    title = f"{protocol_name} — {n} trials   ({rows}×{cols} grid)"
    fig.suptitle(title, fontsize=12, y=0.995)

    # Legend — schema-aware
    schema = proto.get("schema", "legacy")
    legend_handles = [
        patches.Patch(facecolor=DUCK_FILL_COLOR, edgecolor="white", label="duck"),
    ]
    if schema == "entities":
        # Show all unique cup colors in this protocol
        seen_colors = []
        for t in trials:
            for e in t["entities"]:
                if e["kind"] == "cup" and e.get("color") and e["color"] not in seen_colors:
                    seen_colors.append(e["color"])
        for c in seen_colors:
            legend_handles.append(patches.Patch(
                facecolor=CUP_FILL_COLORS.get(c, CUP_FILL_COLORS["default"]),
                label=f"{c} cup",
            ))
        legend_handles.append(patches.Patch(
            facecolor="none", edgecolor="white", linewidth=TARGET_OUTLINE_W,
            label="target (thick white outline + *)",
        ))
    else:
        legend_handles.append(patches.Patch(
            facecolor="#9966cc", label="cup (color = cup_group)",
        ))

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_handles),
               fontsize=9, frameon=False)

    plt.tight_layout(rect=(0, 0.025, 1, 0.985))

    RENDERS_DIR.mkdir(exist_ok=True)
    out = RENDERS_DIR / f"{protocol_name}_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({rows}×{cols} grid, {n} trials)")
    return out


def render_positions(protocol_name: str) -> Path:
    """Single-frame summary: every cup + duck position overlaid on base_frame.

    Useful for verifying coverage at-a-glance without scanning per-trial panels.
    Cups are coloured by (color, target/distractor); ducks shown with a small
    arrow at their dir_deg.
    """
    proto_path = PROTO_DIR / f"{protocol_name}.json"
    proto      = json.loads(proto_path.read_text())
    raw_trials = proto["trials"]
    trials     = [_normalise_trial(proto, t) for t in raw_trials]

    base = Image.open(BASE_FRAME).convert("RGB")
    bbox = tuple(proto.get("workspace_bbox",
                  proto.get("table_bbox", (160, 80, 540, 480))))
    cup_bbox      = proto.get("cup_bbox")
    duck_bbox     = proto.get("duck_bbox")
    nogo_bbox     = proto.get("robot_nogo_bbox")
    duck_nogo     = proto.get("duck_nogo_bbox")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(base)

    # Workspace + duck-bbox + no-go zones for context
    x0, y0, x1, y1 = bbox
    ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                    fill=False, edgecolor="#33dd33", lw=1.5,
                                    label="workspace"))
    if cup_bbox:
        cx0, cy0, cx1, cy1 = cup_bbox
        ax.add_patch(patches.Rectangle((cx0, cy0), cx1-cx0, cy1-cy0,
                                        fill=False, edgecolor="#dd9933", lw=1.0,
                                        linestyle="--", label="cup bbox"))
    if duck_bbox:
        dx0, dy0, dx1, dy1 = duck_bbox
        ax.add_patch(patches.Rectangle((dx0, dy0), dx1-dx0, dy1-dy0,
                                        fill=False, edgecolor="#22aa22", lw=1.0,
                                        linestyle="--", label="duck bbox"))
    if nogo_bbox:
        rx0, ry0, rx1, ry1 = nogo_bbox
        ax.add_patch(patches.Rectangle((rx0, ry0), rx1-rx0, ry1-ry0,
                                        facecolor="#888888", alpha=0.20,
                                        edgecolor="#cc4422", lw=1.0,
                                        linestyle="--", label="cup no-go"))
    if duck_nogo:
        ux0, uy0, ux1, uy1 = duck_nogo
        ax.add_patch(patches.Rectangle((ux0, uy0), ux1-ux0, uy1-uy0,
                                        facecolor="#8855cc", alpha=0.30,
                                        edgecolor="#5522aa", lw=1.0,
                                        linestyle=":", label="duck no-go"))

    # Aggregate positions
    cup_red_xs,  cup_red_ys  = [], []
    cup_blk_xs,  cup_blk_ys  = [], []
    cup_oth_xs,  cup_oth_ys  = [], []
    duck_xs,     duck_ys     = [], []
    duck_dirs                = []
    target_red_xs, target_red_ys = [], []
    target_blk_xs, target_blk_ys = [], []

    for t in trials:
        target_name = t.get("target")
        for ent in t["entities"]:
            x, y = ent["px"]
            if ent["kind"] == "cup":
                col = ent.get("color", "default")
                is_target = ent.get("name") == target_name
                if col == "red":
                    if is_target: target_red_xs.append(x); target_red_ys.append(y)
                    else:         cup_red_xs.append(x);    cup_red_ys.append(y)
                elif col == "black":
                    if is_target: target_blk_xs.append(x); target_blk_ys.append(y)
                    else:         cup_blk_xs.append(x);    cup_blk_ys.append(y)
                else:
                    cup_oth_xs.append(x); cup_oth_ys.append(y)
            elif ent["kind"] == "duck":
                duck_xs.append(x); duck_ys.append(y)
                if ent.get("dir_deg") is not None:
                    duck_dirs.append(ent["dir_deg"])
                else:
                    duck_dirs.append(None)

    # Distractor cups (small)
    if cup_red_xs:
        ax.scatter(cup_red_xs, cup_red_ys, s=70, c=CUP_FILL_COLORS["red"],
                   edgecolor="#444444", lw=0.8, label=f"red cup distractor ({len(cup_red_xs)})", zorder=5)
    if cup_blk_xs:
        ax.scatter(cup_blk_xs, cup_blk_ys, s=70, c=CUP_FILL_COLORS["black"],
                   edgecolor="#444444", lw=0.8, label=f"black cup distractor ({len(cup_blk_xs)})", zorder=5)
    # Target cups (big with white outline)
    if target_red_xs:
        ax.scatter(target_red_xs, target_red_ys, s=140, c=CUP_FILL_COLORS["red"],
                   edgecolor="white", lw=1.6, label=f"red cup target ({len(target_red_xs)})", zorder=6)
    if target_blk_xs:
        ax.scatter(target_blk_xs, target_blk_ys, s=140, c=CUP_FILL_COLORS["black"],
                   edgecolor="white", lw=1.6, label=f"black cup target ({len(target_blk_xs)})", zorder=6)
    if cup_oth_xs:
        ax.scatter(cup_oth_xs, cup_oth_ys, s=80, c=CUP_FILL_COLORS["default"],
                   edgecolor="white", lw=1.0, label=f"cup ({len(cup_oth_xs)})", zorder=5)

    # Ducks with direction arrows
    if duck_xs:
        ax.scatter(duck_xs, duck_ys, s=70, c=DUCK_FILL_COLOR,
                   edgecolor="black", lw=0.8, label=f"ducks ({len(duck_xs)})", zorder=7)
        for x, y, deg in zip(duck_xs, duck_ys, duck_dirs):
            if deg is None: continue
            rad = math.radians(deg)
            ex, ey = x + 18 * math.cos(rad), y + 18 * math.sin(rad)
            ax.annotate("", xy=(ex, ey), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="#ff7000", lw=1.2),
                        zorder=8)

    ax.set_xlim(0, base.width)
    ax.set_ylim(base.height, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{protocol_name} — all entity positions ({len(trials)} trials)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    RENDERS_DIR.mkdir(exist_ok=True)
    out = RENDERS_DIR / f"{protocol_name}_positions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "dual_cup_30"
    render(name)   # produces both overview grid + positions plot
