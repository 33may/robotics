"""Protocol tooling — render, verify, and interactively edit eval protocols.

Subcommands:
    render <name>           Save an OpenCV overview PNG of <name>.json.
    verify <name>            Save a matplotlib distributional verification PNG.
    edit   <name>            Open the interactive editor on <name>.json.

Outputs go to ``protocols/renders/<name>_{overview,verify}.png`` by default.

Examples:
    python vbti/logic/inference/protocols/protocols.py render id_scale_60
    python vbti/logic/inference/protocols/protocols.py verify id_scale_60
    python vbti/logic/inference/protocols/protocols.py edit   id_scale_60
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

PROTO_DIR   = Path(__file__).resolve().parent
BASE_FRAME  = PROTO_DIR / "base_frame.png"
WORKSPACE   = (270, 191, 424, 313)  # x0, y0, x1, y1 of ID zone
RENDERS_DIR = PROTO_DIR / "renders"

CUP_COLORS = [
    (0, 200, 255), (60, 180, 255), (120, 160, 255),
    (180, 140, 255), (220, 120, 255), (255, 100, 200),
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_proto(protocol: str) -> Path:
    """Accept bare name (`id_scale_60`) or filename (`id_scale_60.json`)."""
    p = Path(protocol)
    if not p.suffix:
        p = PROTO_DIR / (p.name + ".json")
    elif not p.is_absolute():
        p = PROTO_DIR / p
    return p


def _load_proto(protocol: str) -> dict:
    return json.loads(_resolve_proto(protocol).read_text())


def _fold(angle: float) -> float:
    a = angle % 360.0
    return a - 360.0 if a > 180.0 else a


# ══ render ════════════════════════════════════════════════════════════════════

def render(protocol: str, save: str = "", show: bool = False):
    """Render an OpenCV overview of a protocol onto base_frame.png.

    Args:
        protocol: Name (e.g. "id_scale_60") or path to a protocol JSON.
        save:     Optional output path. Defaults to renders/<name>_overview.png.
        show:     If True, open an OpenCV window for inspection.
    """
    proto = _load_proto(protocol)

    base = cv2.imread(str(BASE_FRAME))
    if base is None:
        raise FileNotFoundError(f"Base frame not found: {BASE_FRAME}")
    img = base.copy()

    # Zone outlines
    cv2.rectangle(img, (160, 83),  (535, 419), (255, 180, 0), 1)   # table
    cv2.rectangle(img, (270, 191), (424, 313), (0, 180, 60),  1)   # ID zone

    for gi, (cx, cy) in enumerate(proto["cup_positions"]):
        col = CUP_COLORS[gi % len(CUP_COLORS)]
        cv2.circle(img, (cx, cy), 15, col, -1)
        cv2.circle(img, (cx, cy), 15, (255, 255, 255), 2)
        cv2.putText(img, f"C{gi}", (cx - 9, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2)

    for t in proto["trials"]:
        dx, dy = t["duck_px"]
        col = (50, 220, 50) if t["zone"] == "ID" else (50, 100, 255)
        cv2.circle(img, (dx, dy), 6, col, -1)
        cv2.circle(img, (dx, dy), 6, (255, 255, 255), 1)

    # Legend
    h = img.shape[0]
    cv2.rectangle(img, (5, h - 60), (200, h - 5), (20, 20, 20), -1)
    cv2.circle(img, (16, h - 46), 5, (50, 220, 50), -1)
    cv2.putText(img, f"ID duck ({proto['id_count']})",
                (26, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 220, 50), 1)
    cv2.circle(img, (16, h - 28), 5, (50, 100, 255), -1)
    cv2.putText(img, f"OOD duck ({proto['ood_count']})",
                (26, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 100, 255), 1)
    cv2.rectangle(img, (110, h - 46), (122, h - 36), (255, 255, 255), 1)
    cv2.putText(img, "Cup groups", (126, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(img, proto["name"], (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    RENDERS_DIR.mkdir(exist_ok=True)
    out_path = Path(save) if save else RENDERS_DIR / f"{proto['name']}_overview.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

    if show:
        cv2.namedWindow("Protocol Overview", cv2.WINDOW_NORMAL)
        cv2.imshow("Protocol Overview", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══ verify ════════════════════════════════════════════════════════════════════

def _enrich(trials):
    octants, pterrs = [], []
    for t in trials:
        d2c = math.degrees(math.atan2(
            t["cup_px"][1] - t["duck_px"][1],
            t["cup_px"][0] - t["duck_px"][0],
        )) % 360.0
        pterrs.append(_fold(t["duck_dir_deg"] - d2c))
        octants.append(int(((t["duck_dir_deg"] + 22.5) % 360) // 45))
    return octants, pterrs


def verify(protocol: str, save: str = "", show: bool = False):
    """Render a multi-panel matplotlib verification figure.

    Panels: workspace overlay with heading arrows; octant histogram;
    pointing_error histogram; spatial-cell coverage; per-cup mini scatters.
    """
    proto = _load_proto(protocol)
    trials = proto["trials"]
    cups = proto["cup_positions"]
    octants, pterrs = _enrich(trials)

    by_cup: dict[int, list] = {i: [] for i in range(len(cups))}
    for t in trials:
        by_cup[t["cup_group"]].append(t)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[2.2, 1, 1.4])
    cmap = plt.get_cmap("viridis")(np.linspace(0.1, 0.95, len(cups)))

    # ---- Panel 1: full overlay with arrows
    ax1 = fig.add_subplot(gs[0, :4])
    base = cv2.imread(str(BASE_FRAME))
    if base is None:
        raise FileNotFoundError(BASE_FRAME)
    ax1.imshow(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = WORKSPACE
    ax1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, ec="lime", lw=1.5))
    for ci, (cx, cy) in enumerate(cups):
        ax1.add_patch(Circle((cx, cy), 14, color=cmap[ci], ec="white", lw=2))
        ax1.text(cx, cy + 2, f"C{ci}", color="black", ha="center",
                 va="center", fontsize=9, weight="bold")
    for t in trials:
        dx, dy = t["duck_px"]
        ang = math.radians(t["duck_dir_deg"])
        col = cmap[t["cup_group"]]
        ax1.add_patch(Circle((dx, dy), 4, color=col, ec="white", lw=0.7))
        ax1.arrow(dx, dy, 12 * math.cos(ang), 12 * math.sin(ang),
                  head_width=3.5, head_length=3.5, fc=col, ec=col, lw=1)
    ax1.set_xlim(120, 580)
    ax1.set_ylim(440, 60)
    ax1.set_title(f"{proto['name']} — {len(trials)} trials   color=cup_group, arrow=duck heading")
    ax1.set_xticks([]); ax1.set_yticks([])

    # ---- Panel 2: octant histogram
    ax2 = fig.add_subplot(gs[0, 4:])
    oct_names = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    counts = [Counter(octants).get(i, 0) for i in range(8)]
    bars = ax2.bar(oct_names, counts, color="steelblue", ec="black")
    ax2.axhline(len(trials) / 8, color="red", ls="--", lw=1.2,
                label=f"uniform={len(trials)/8:.1f}")
    ax2.set_ylabel("# trials")
    ax2.set_title("Duck-direction octants")
    ax2.legend(loc="upper right", fontsize=8)
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2, c + 0.2, str(c),
                 ha="center", fontsize=8)

    # ---- Panel 3: pterr histogram
    ax3 = fig.add_subplot(gs[1, :3])
    bins = list(range(-180, 181, 30))
    ax3.hist(pterrs, bins=bins, color="orange", ec="black", alpha=0.85)
    ax3.axvspan(-45, 45, alpha=0.15, color="green",
                label="pointing_at_cup=yes")
    yes = sum(1 for p in pterrs if abs(p) <= 45)
    ax3.set_xlabel("pointing_error (deg, signed)")
    ax3.set_ylabel("# trials")
    ax3.set_title(f"pointing_error  (yes={yes}/{len(pterrs)}, no={len(pterrs)-yes}/{len(pterrs)})")
    ax3.legend(loc="upper right", fontsize=8)

    # ---- Panel 4: spatial-cell coverage
    ax4 = fig.add_subplot(gs[1, 3:])
    cols, rows = 4, 3
    cell_counts = np.zeros((rows, cols), dtype=int)
    for t in trials:
        if t.get("zone") == "OOD":
            continue
        dx, dy = t["duck_px"]
        cx_i = min(int((dx - x0) / (x1 - x0) * cols), cols - 1)
        ry_i = min(int((dy - y0) / (y1 - y0) * rows), rows - 1)
        cell_counts[ry_i, cx_i] += 1
    im = ax4.imshow(cell_counts, cmap="YlGnBu", aspect="auto")
    for r in range(rows):
        for c in range(cols):
            ax4.text(c, r, cell_counts[r, c], ha="center", va="center",
                     color="black" if cell_counts[r, c] < 6 else "white",
                     fontsize=10)
    ax4.set_title(f"Spatial-cell coverage  ({rows}×{cols} ID grid)")
    ax4.set_xticks(range(cols)); ax4.set_yticks(range(rows))
    plt.colorbar(im, ax=ax4, fraction=0.04, label="# trials")

    # ---- Panel 5: per-cup mini scatters
    n_cups = len(cups)
    for ci in range(min(n_cups, 6)):
        ax = fig.add_subplot(gs[2, ci])
        ts = by_cup[ci]
        cxp, cyp = cups[ci]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, ec="lime", lw=0.8))
        ax.scatter(cxp, cyp, s=120, c=[cmap[ci]], ec="white", zorder=3)
        for t in ts:
            d2c = math.degrees(math.atan2(
                t["cup_px"][1] - t["duck_px"][1],
                t["cup_px"][0] - t["duck_px"][0],
            )) % 360.0
            pterr = _fold(t["duck_dir_deg"] - d2c)
            mk = "o" if abs(pterr) <= 45 else "x"
            ax.scatter(t["duck_px"][0], t["duck_px"][1], marker=mk,
                       s=35, c=[cmap[ci]], ec="black", lw=0.6)
        ax.set_xlim(x0 - 20, x1 + 20)
        ax.set_ylim(y1 + 20, y0 - 20)
        ax.set_title(f"C{ci}  n={len(ts)} (o=yes, x=no)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    RENDERS_DIR.mkdir(exist_ok=True)
    out_path = Path(save) if save else RENDERS_DIR / f"{proto['name']}_verify.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ══ edit ══════════════════════════════════════════════════════════════════════

ROTATION_STEP = 15
WINDOW = "Protocol Editor"


def _draw_arrow(img, cx, cy, deg, length=20, color=(255, 255, 255), thickness=2):
    rad = np.radians(deg)
    ex, ey = int(cx + length * np.cos(rad)), int(cy + length * np.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, thickness, tipLength=0.38)


def _is_entities_trial(t: dict) -> bool:
    return "entities" in t


# Entity colors in BGR for the editor canvas
ENTITY_BGR = {
    "red":   ( 32,  32, 200),
    "black": ( 32,  32,  32),
    "blue":  (200,  32,  32),
    "green": ( 32, 180,  32),
    "yellow":(  0, 210, 255),
}


def _entity_helpers(trial):
    """Return (duck_ent, cups_list, target_ent_or_None) views into the trial.

    `cups_list` is in `entities` order (so [0] is the first non-duck cup).
    """
    duck = next((e for e in trial["entities"] if e["kind"] == "duck"), None)
    cups = [e for e in trial["entities"] if e["kind"] == "cup"]
    target_name = trial.get("target")
    target = next((c for c in cups if c["name"] == target_name), None)
    return duck, cups, target


def _editor_render_entity(base, trial, total, proto, dirty):
    """Render a single entity-schema trial onto a copy of the base frame."""
    img = base.copy()

    # Workspace + duck/cup bbox + no-go zones for visual reference
    if proto.get("workspace_bbox"):
        x0, y0, x1, y1 = (int(v) for v in proto["workspace_bbox"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 180, 0), 1)
    if proto.get("cup_bbox"):
        x0, y0, x1, y1 = (int(v) for v in proto["cup_bbox"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (50, 150, 220), 1)
    if proto.get("duck_bbox"):
        x0, y0, x1, y1 = (int(v) for v in proto["duck_bbox"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 200, 80), 1)
    if proto.get("robot_nogo_bbox"):
        rx0, ry0, rx1, ry1 = (int(v) for v in proto["robot_nogo_bbox"])
        nogo = img.copy()
        cv2.rectangle(nogo, (rx0, ry0), (rx1, ry1), (128, 128, 128), -1)
        cv2.addWeighted(nogo, 0.30, img, 0.70, 0, img)
        cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (40, 80, 220), 1)
    if proto.get("duck_nogo_bbox"):
        ux0, uy0, ux1, uy1 = (int(v) for v in proto["duck_nogo_bbox"])
        nogo = img.copy()
        cv2.rectangle(nogo, (ux0, uy0), (ux1, uy1), (180, 100, 200), -1)
        cv2.addWeighted(nogo, 0.30, img, 0.70, 0, img)
        cv2.rectangle(img, (ux0, uy0), (ux1, uy1), (140, 60, 180), 1)

    duck, cups, target = _entity_helpers(trial)

    # Cups: target gets thick white ring + bigger
    for ent in cups:
        x, y  = int(ent["px"][0]), int(ent["px"][1])
        bgr   = ENTITY_BGR.get(ent.get("color", "default"), (180, 180, 180))
        is_t  = (target is not None) and (ent["name"] == target["name"])
        r     = 18 if is_t else 14
        cv2.circle(img, (x, y), r, bgr, -1)
        cv2.circle(img, (x, y), r, (255, 255, 255), 3 if is_t else 1)

    # Duck
    if duck is not None:
        x, y = int(duck["px"][0]), int(duck["px"][1])
        cv2.circle(img, (x, y), 15, (0, 210, 255), -1)
        cv2.circle(img, (x, y), 15, (255, 255, 255), 2)
        if duck.get("dir_deg") is not None:
            _draw_arrow(img, x, y, duck["dir_deg"], length=22)

    # ── HUD ───────────────────────────────────────────────────────────────
    tid     = trial["trial_id"]
    tags    = trial.get("tags", {})
    scene   = tags.get("scene", "")
    tcolor  = tags.get("target_color", "?")
    task    = trial.get("task", "")
    dmark   = " *" if dirty else ""

    # scene-coloured top bar
    sc_col = ((60, 200, 60)  if scene.startswith("single") else
              (50, 120, 255) if scene == "both" else
              (200, 200, 200))

    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(img, f"Trial {tid+1}/{total}{dmark}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, sc_col, 2)
    info = f"{scene}  target={tcolor.upper()}  cups={len(cups)}"
    cv2.putText(img, info, (155, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    h = img.shape[0]
    cv2.rectangle(img, (0, h - 44), (img.shape[1], h), (15, 15, 15), -1)
    cv2.putText(img, f'"{task}"', (8, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 100), 1)
    cv2.putText(img, "L:duck  R:target  M:distractor  T:swap-target  "
                     "<-/->:trial  Up/Dn:rotate  R:rand  S:save  Q:quit",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)
    return img


def _editor_render(base, trial, total, cup_positions, dirty, proto=None):
    """Top-level render — dispatches on schema."""
    if _is_entities_trial(trial):
        return _editor_render_entity(base, trial, total, proto or {}, dirty)

    # ── Legacy schema render (unchanged) ─────────────────────────────────
    img = base.copy()
    cv2.rectangle(img, (160, 83),  (535, 419), (255, 180, 0), 1)
    cv2.rectangle(img, (270, 191), (424, 313), (0, 180, 60),  1)
    cv2.rectangle(img, (160, 83),  (535, 191), (180, 200, 255), 1)

    dx, dy = trial["duck_px"]
    cx, cy = trial["cup_px"]
    d      = trial["duck_dir_deg"]
    g      = trial["cup_group"]
    zone   = trial["zone"]
    tid    = trial["trial_id"]
    ccol   = CUP_COLORS[g % len(CUP_COLORS)]

    for gi, (cpx, cpy) in enumerate(cup_positions):
        if gi != g:
            cv2.circle(img, (cpx, cpy), 10, (60, 60, 60), -1)
            cv2.circle(img, (cpx, cpy), 10, (80, 80, 80),  1)

    cv2.circle(img, (cx, cy), 15, ccol, -1)
    cv2.circle(img, (cx, cy), 15, (255, 255, 255), 2)
    cv2.putText(img, f"C{g}", (cx - 9, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2)

    dcol = (0, 210, 255) if zone == "ID" else (0, 80, 255)
    cv2.circle(img, (dx, dy), 14, dcol, -1)
    cv2.circle(img, (dx, dy), 14, (255, 255, 255), 2)
    _draw_arrow(img, dx, dy, d)

    zone_col = (0, 220, 80) if zone == "ID" else (50, 120, 255)
    dirty_marker = " *" if dirty else ""
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(img, f"Trial {tid+1}/{total}{dirty_marker}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, zone_col, 2)
    cv2.putText(img, f"{zone}  grp:{g}  dir:{d:.0f}deg  duck:({dx},{dy})  cup:({cx},{cy})",
                (135, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    h = img.shape[0]
    cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (15, 15, 15), -1)
    cv2.putText(img, "LClick:duck  RClick:cup  Arrows:nav/rotate  R:rand-dir  S:save  Q:quit",
                (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)
    return img


def _make_mouse_cb(state):
    """Schema-aware mouse callback.

    Legacy:  L=duck, R=cup
    Entity:  L=duck, R=target cup, M=distractor cup (if 2 cups present)
    """
    def cb(event, x, y, _flags, _param):
        trial = state["trial"]
        if _is_entities_trial(trial):
            duck, cups, target = _entity_helpers(trial)
            if event == cv2.EVENT_LBUTTONDOWN and duck is not None:
                duck["px"] = [x, y]
                state["dirty"]  = True
                state["redraw"] = True
            elif event == cv2.EVENT_RBUTTONDOWN and target is not None:
                target["px"] = [x, y]
                state["dirty"]  = True
                state["redraw"] = True
            elif event == cv2.EVENT_MBUTTONDOWN:
                # distractor = the cup that is NOT the target
                distractor = next((c for c in cups
                                   if c.get("name") != trial.get("target")), None)
                if distractor is not None:
                    distractor["px"] = [x, y]
                    state["dirty"]  = True
                    state["redraw"] = True
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                trial["duck_px"] = [x, y]
                state["dirty"]   = True
                state["redraw"]  = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                trial["cup_px"]  = [x, y]
                state["dirty"]   = True
                state["redraw"]  = True
    return cb


def _trial_duck_dir(trial: dict) -> float:
    """Read duck_dir_deg from either schema."""
    if _is_entities_trial(trial):
        for e in trial["entities"]:
            if e["kind"] == "duck":
                return float(e.get("dir_deg") or 0.0)
        return 0.0
    return float(trial.get("duck_dir_deg") or 0.0)


def _set_trial_duck_dir(trial: dict, deg: float) -> None:
    """Write duck_dir_deg to either schema."""
    if _is_entities_trial(trial):
        for e in trial["entities"]:
            if e["kind"] == "duck":
                e["dir_deg"] = round(deg, 1)
                return
    else:
        trial["duck_dir_deg"] = round(deg, 1)


def _swap_target(trial: dict) -> bool:
    """For entity-schema dual-cup trials: swap which named cup is the target.
    Updates trial.target AND trial.task to match the new target's color.
    Returns True if a swap happened."""
    if not _is_entities_trial(trial):
        return False
    cups = [e for e in trial["entities"] if e["kind"] == "cup"]
    if len(cups) < 2:
        return False
    cur     = trial.get("target")
    nxt     = next((c["name"] for c in cups if c["name"] != cur), None)
    if nxt is None:
        return False
    trial["target"] = nxt
    new_color = next((c.get("color") for c in cups if c["name"] == nxt), None)
    if new_color and trial.get("task"):
        trial["task"] = f"Pick up the duck and place it in the {new_color} cup"
        if "tags" in trial:
            trial["tags"]["target_color"] = new_color
    return True


def edit(protocol: str, base_image: str | None = None):
    """Open the interactive OpenCV editor for a protocol JSON.

    Controls (entity schema):
      L-click=duck, R-click=target cup, M-click=distractor cup,
      T=swap target↔distractor, ←/→=nav, ↑/↓=rotate duck, R=randomize, S=save, Q=quit
    Controls (legacy schema):
      L-click=duck, R-click=cup, ←/→=nav, ↑/↓=rotate, R=rand, S=save, Q=quit
    """
    proto_path = _resolve_proto(protocol)
    proto = json.loads(proto_path.read_text())

    trials        = proto["trials"]
    cup_positions = proto.get("cup_positions", [])
    total         = len(trials)
    is_entity     = bool(trials) and _is_entities_trial(trials[0])

    img_path = Path(base_image) if base_image is not None else BASE_FRAME
    if not img_path.exists():
        print(f"Base image not found: {img_path}")
        print("Pass --base_image=<path> to your top camera frame.")
        return
    base = cv2.imread(str(img_path))

    state = {
        "idx":    0,
        "trial":  dict(trials[0]),
        "dirty":  False,
        "redraw": True,
    }
    # Deep copy entities so editing the working trial doesn't mutate the source
    if is_entity:
        state["trial"]["entities"] = [dict(e) for e in state["trial"]["entities"]]
        state["trial"]["tags"]     = dict(state["trial"].get("tags", {}))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, _make_mouse_cb(state))

    print(f"Editing: {proto_path}  ({total} trials, schema={'entity' if is_entity else 'legacy'})")
    if is_entity:
        print("Controls: L=duck  R=target  M=distractor  T=swap-target  "
              "<-/->:nav  Up/Dn:rotate  R:rand  S:save  Q:quit")
    else:
        print("Controls: LClick=duck  RClick=cup  <-/->:nav  Up/Dn:rotate  R:rand  S:save  Q:quit")

    def _commit_to_trials():
        """Push the working trial back into the trials list (deep copy entities)."""
        clone = dict(state["trial"])
        if is_entity:
            clone["entities"] = [dict(e) for e in clone["entities"]]
            clone["tags"]     = dict(clone.get("tags", {}))
        trials[state["idx"]] = clone

    def _load_trial(new_idx: int):
        state["idx"]   = new_idx % total
        src            = trials[state["idx"]]
        state["trial"] = dict(src)
        if is_entity:
            state["trial"]["entities"] = [dict(e) for e in src["entities"]]
            state["trial"]["tags"]     = dict(src.get("tags", {}))
        state["dirty"]  = False
        state["redraw"] = True

    while True:
        if state["redraw"]:
            frame = _editor_render(base, state["trial"], total,
                                   cup_positions, state["dirty"], proto=proto)
            cv2.imshow(WINDOW, frame)
            state["redraw"] = False

        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue

        if key in (ord('q'), 27):
            if state["dirty"]:
                print("Unsaved changes — press S to save or Q again to discard.")
                state["dirty"] = False
            else:
                break
        elif key == ord('s'):
            _commit_to_trials()
            with open(proto_path, "w") as f:
                json.dump(proto, f, indent=2)
            state["dirty"] = False
            state["redraw"] = True
            print(f"Saved {proto_path}")
        elif key == 81 or key == ord('a'):
            if state["dirty"]: _commit_to_trials()
            _load_trial(state["idx"] - 1)
        elif key == 83 or key == ord('d'):
            if state["dirty"]: _commit_to_trials()
            _load_trial(state["idx"] + 1)
        elif key == 82:
            _set_trial_duck_dir(state["trial"],
                                _trial_duck_dir(state["trial"]) + ROTATION_STEP)
            state["dirty"]  = True
            state["redraw"] = True
        elif key == 84:
            _set_trial_duck_dir(state["trial"],
                                _trial_duck_dir(state["trial"]) - ROTATION_STEP)
            state["dirty"]  = True
            state["redraw"] = True
        elif key == ord('t') and is_entity:
            if _swap_target(state["trial"]):
                state["dirty"]  = True
                state["redraw"] = True
        elif key == ord('r'):
            _set_trial_duck_dir(state["trial"], random.uniform(0, 360))
            state["dirty"]  = True
            state["redraw"] = True

    cv2.destroyAllWindows()
    print("Editor closed.")


# ══ Fire CLI ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import fire
    fire.Fire({
        "render": render,
        "verify": verify,
        "edit":   edit,
    })
