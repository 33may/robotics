"""Eval-session visualization helpers — reusable plot functions.

Right now: ``render_session_heatmap`` renders a 3-panel duck-position
success-rate heatmap (ALL + per target_color) for a single eval session.

Usage as a CLI:
    python -m vbti.logic.inference.eval_render heatmap \\
        /path/to/session_dir [--output=/path/to/plot.png]

Programmatic:
    from vbti.logic.inference.eval_render import render_session_heatmap
    render_session_heatmap(session_dir, output_png)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


PROTO_DIR = Path(__file__).resolve().parent / "protocols"

# Diverging colormap: red (0, all-fail) → light yellow (0.5, mixed) → green (1, all-succ)
_DIVERGING = LinearSegmentedColormap.from_list("rg", [
    (0.0, "#c0392b"), (0.5, "#f9e79f"), (1.0, "#27ae60"),
])

# Cup color → display facecolor
_CUP_RGB = {"red": "#c0392b", "black": "#1c1c1c"}


def _resolve_session_path(session) -> Path:
    """Accept either a session.json path or a session directory."""
    p = Path(session).expanduser()
    if p.is_file():
        return p
    if p.is_dir() and (p / "session.json").exists():
        return p / "session.json"
    raise FileNotFoundError(f"No session.json at {session}")


def _load_protocol(name: str) -> dict:
    path = PROTO_DIR / f"{name}.json"
    return json.loads(path.read_text())


def _flatten_trial(st: dict, proto_by_id: dict) -> dict:
    """Flatten one session-trial + its protocol-trial into the geometry we need."""
    pt = proto_by_id.get(st["trial_id"], {})
    # Prefer entities on the session itself; fall back to protocol.
    entities = st.get("entities") or pt.get("entities") or []
    target_name = st.get("target") or pt.get("target")
    ents = {e["name"]: e for e in entities}

    duck = ents.get("duck")
    target_ent = ents.get(target_name) if target_name else None
    distractor = next(
        (e for e in entities if e.get("kind") == "cup" and e.get("name") != target_name),
        None,
    )
    tags = st.get("tags") or pt.get("tags") or {}
    return {
        "result":              st["result"],
        "target_color":        tags.get("target_color"),
        "duck_px":             duck["px"]   if duck else None,
        "target_px":           target_ent["px"] if target_ent else None,
        "target_color_actual": target_ent.get("color") if target_ent else None,
        "distractor_px":       distractor["px"] if distractor else None,
        "distractor_color":    distractor.get("color") if distractor else None,
    }


def _kde_peak_norm(points, GX, GY, grid_pts, ws_w, bw=0.55):
    """Peak-normalized KDE so single-point fallback and scipy KDE share scale."""
    if not points:
        return np.zeros_like(GX)
    if len(points) < 2:
        Z = np.zeros_like(GX)
        sig = ws_w * 0.10
        for px, py in points:
            Z += np.exp(-((GX - px) ** 2 + (GY - py) ** 2) / (2 * sig ** 2))
    else:
        pts = np.array(points).T.astype(float)
        Z = gaussian_kde(pts, bw_method=bw)(grid_pts).reshape(GX.shape)
    return Z / Z.max() if Z.max() > 0 else Z


def _render_panel(ax, sub_trials, title, *, base_img, ws, GX, GY, grid_pts,
                  scope: str, target_color: str | None = None):
    """Draw one heatmap panel onto ``ax``."""
    x0, y0, x1, y1 = ws
    succ = [t["duck_px"] for t in sub_trials if t["result"] == "success" and t["duck_px"]]
    fail = [t["duck_px"] for t in sub_trials if t["result"] == "failure" and t["duck_px"]]

    Zs = _kde_peak_norm(succ, GX, GY, grid_pts, x1 - x0)
    Zf = _kde_peak_norm(fail, GX, GY, grid_pts, x1 - x0)
    Zt = Zs + Zf
    eps = max(Zt.max() * 1e-3, 1e-9)
    score = Zs / (Zt + eps)
    alpha = np.clip(np.maximum(Zs, Zf), 0, 1) ** 0.7

    rgba = _DIVERGING(score)
    rgba[..., -1] = alpha * 0.82

    ax.imshow(base_img, alpha=0.55)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, ec="lime", lw=1.5, alpha=0.7))
    ax.imshow(rgba, extent=(x0, x1, y1, y0), origin="upper",
              interpolation="bilinear", zorder=2)

    score_masked = np.where(alpha > 0.05, score, np.nan)
    cs = ax.contour(GX, GY, score_masked, levels=[0.1, 0.25, 0.5, 0.75, 0.9],
                    colors=["#922b21", "#a93226", "#7d6608", "#229954", "#1e8449"],
                    linewidths=1.0, alpha=0.65, zorder=3)
    ax.clabel(cs, fmt={0.1: "10%", 0.25: "25%", 0.5: "50%", 0.75: "75%", 0.9: "90%"},
              fontsize=7, inline=True)

    # Connection lines
    for t in sub_trials:
        if t["duck_px"] is None or t["target_px"] is None:
            continue
        line_color = "#196f3d" if t["result"] == "success" else "#922b21"
        dx_, dy_ = t["duck_px"]; tx_, ty_ = t["target_px"]
        ax.plot([dx_, tx_], [dy_, ty_], color=line_color, lw=1.3, alpha=0.40, zorder=3.5)
        if t["distractor_px"] is not None:
            dxx_, dyy_ = t["distractor_px"]
            ax.plot([dx_, dxx_], [dy_, dyy_], color="#555555", ls=":", lw=1.0,
                    alpha=0.55, zorder=3.5)

    # Duck markers
    if succ:
        sx, sy = zip(*succ)
        ax.scatter(sx, sy, s=60, marker="o", facecolor="#27ae60",
                   edgecolor="white", lw=1.2, zorder=5, label="duck — success")
    if fail:
        fx, fy = zip(*fail)
        ax.scatter(fx, fy, s=85, marker="X", color="#c0392b",
                   edgecolor="white", lw=1.4, zorder=5, label="duck — failure")

    # Cup markers — coloring depends on whether the panel is per-color or pooled
    if scope == "all":
        for color, face in _CUP_RGB.items():
            txs = [t["target_px"][0] for t in sub_trials
                   if t["target_color_actual"] == color and t["target_px"]]
            tys = [t["target_px"][1] for t in sub_trials
                   if t["target_color_actual"] == color and t["target_px"]]
            if txs:
                ax.scatter(txs, tys, s=100, marker="s", facecolor=face,
                           edgecolor="white", lw=1.4, zorder=6,
                           label=f"target cup ({color})")
    else:
        face = _CUP_RGB.get(target_color, "#666")
        txs = [t["target_px"][0] for t in sub_trials if t["target_px"]]
        tys = [t["target_px"][1] for t in sub_trials if t["target_px"]]
        if txs:
            ax.scatter(txs, tys, s=110, marker="s", facecolor=face,
                       edgecolor="white", lw=1.4, zorder=6,
                       label=f"target cup ({target_color})")
    dxs = [t["distractor_px"][0] for t in sub_trials if t["distractor_px"]]
    dys = [t["distractor_px"][1] for t in sub_trials if t["distractor_px"]]
    if dxs:
        ax.scatter(dxs, dys, s=65, marker="s", facecolor="#888",
                   edgecolor="white", lw=1.0, alpha=0.55, zorder=6,
                   label="distractor cup")

    ax.set_xlim(120, 580); ax.set_ylim(440, 60)
    ax.set_xticks([]); ax.set_yticks([])
    nS, nF = len(succ), len(fail)
    n = nS + nF
    sr_str = f"{nS}/{n} = {nS/n:.0%}" if n else "n=0"
    ax.set_title(f"{title}  —  {nS} succ / {nF} fail   (SR {sr_str})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)


def render_session_heatmap(session, output, *,
                           protocol_dir: Path | str | None = None,
                           grid: int = 180) -> Path:
    """Render a 3-panel success-rate heatmap for a single eval session.

    Panels: ALL trials | per target_color (one per unique color in the session).

    Args:
        session:       path to session.json or session directory
        output:        output PNG path
        protocol_dir:  override for protocols/ dir (default: vbti/logic/inference/protocols)
        grid:          KDE grid resolution per axis

    Returns:
        Path to the written PNG.
    """
    proto_dir = Path(protocol_dir) if protocol_dir else PROTO_DIR
    sess_path = _resolve_session_path(session)
    sess = json.loads(sess_path.read_text())
    proto_name: str | None = (sess.get("config") or {}).get("protocol") or sess.get("protocol")
    if not proto_name:
        raise ValueError(f"No protocol name in session config: {sess_path}")
    proto = json.loads((proto_dir / f"{proto_name}.json").read_text())
    proto_by_id = {t["trial_id"]: t for t in proto["trials"]}

    trials = [_flatten_trial(st, proto_by_id) for st in sess.get("trials", [])]
    if not trials:
        raise ValueError(f"No trials in session {sess_path}")

    base_path = proto_dir / "base_frame.png"
    raw = cv2.imread(str(base_path))
    if raw is None:
        raise FileNotFoundError(f"Could not read base frame: {base_path}")
    base_img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    ws = proto["workspace_bbox"]
    x0, y0, x1, y1 = ws
    gx = np.linspace(x0, x1, grid)
    gy = np.linspace(y0, y1, grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.vstack([GX.ravel(), GY.ravel()])

    # Discover unique target_colors present in this session (preserves order seen).
    seen_colors: list[str] = []
    for t in trials:
        c = t["target_color"]
        if c and c not in seen_colors:
            seen_colors.append(c)

    panels = [("all", trials, "ALL trials")]
    for c in seen_colors:
        sub = [t for t in trials if t["target_color"] == c]
        panels.append((c, sub, f"Target = {c.upper()}"))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels) + 4, 8))
    if len(panels) == 1:
        axes = [axes]

    for ax, (scope, sub, title) in zip(axes, panels):
        target_color = scope if scope != "all" else None
        _render_panel(ax, sub, title,
                      base_img=base_img, ws=ws, GX=GX, GY=GY, grid_pts=grid_pts,
                      scope=scope, target_color=target_color)

    label = (sess.get("config") or {}).get("checkpoint_label", "?")
    fig.suptitle(
        f"{sess.get('experiment','?')}/{sess.get('version','?')} — "
        f"{label} on {proto_name}   "
        "(lines: duck→target solid, duck→distractor dotted)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout()

    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_session_for_render(session, protocol_dir: Path):
    """Load a session and its protocol; return (sess_dict, trials_list, ws, base_img, proto_name)."""
    sess_path = _resolve_session_path(session)
    sess = json.loads(sess_path.read_text())
    proto_name: str | None = (sess.get("config") or {}).get("protocol") or sess.get("protocol")
    if not proto_name:
        raise ValueError(f"No protocol name in session config: {sess_path}")
    proto = json.loads((protocol_dir / f"{proto_name}.json").read_text())
    proto_by_id = {t["trial_id"]: t for t in proto["trials"]}
    trials = [_flatten_trial(st, proto_by_id) for st in sess.get("trials", [])]
    if not trials:
        raise ValueError(f"No trials in session {sess_path}")
    base_path = protocol_dir / "base_frame.png"
    raw = cv2.imread(str(base_path))
    if raw is None:
        raise FileNotFoundError(f"Could not read base frame: {base_path}")
    base_img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return sess, trials, proto["workspace_bbox"], base_img, proto_name


def render_sessions_grid(sessions: list, output, *,
                         protocol_dir: Path | str | None = None,
                         grid: int = 180) -> Path:
    """Render a multi-row grid: each row is one session, columns are ALL/per-color panels.

    All sessions share the same workspace bbox + base frame (assumes the protocols
    use the same canonical table). The number of columns = max #target_colors found
    + 1 (for the ALL panel).
    """
    proto_dir = Path(protocol_dir) if protocol_dir else PROTO_DIR
    rows = [_load_session_for_render(s, proto_dir) for s in sessions]

    # Discover global set of target colors across all sessions, in stable order.
    all_colors: list[str] = []
    for _, trials, _, _, _ in rows:
        for t in trials:
            c = t["target_color"]
            if c and c not in all_colors:
                all_colors.append(c)
    n_cols = 1 + len(all_colors)

    # Use the first session's workspace + base frame for the grid (assume shared).
    ws = rows[0][2]
    base_img = rows[0][3]
    x0, y0, x1, y1 = ws
    gx = np.linspace(x0, x1, grid)
    gy = np.linspace(y0, y1, grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.vstack([GX.ravel(), GY.ravel()])

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ri, (sess, trials, _, _, proto_name) in enumerate(rows):
        cfg = sess.get("config") or {}
        label = cfg.get("checkpoint_label", "?")
        version = sess.get("version") or cfg.get("version", "?")
        row_label = f"{version} · {label} · {proto_name} · n={len(trials)}"

        # Panel: ALL
        _render_panel(axes[ri, 0], trials, f"{row_label}\n— ALL trials",
                      base_img=base_img, ws=ws, GX=GX, GY=GY, grid_pts=grid_pts,
                      scope="all", target_color=None)
        # Per-color panels
        for ci, color in enumerate(all_colors, start=1):
            sub = [t for t in trials if t["target_color"] == color]
            _render_panel(axes[ri, ci], sub, f"Target = {color.upper()}",
                          base_img=base_img, ws=ws, GX=GX, GY=GY, grid_pts=grid_pts,
                          scope=color, target_color=color)

    fig.suptitle(
        "Eval-session heatmap grid   "
        "(rows = checkpoint, columns = ALL · per target color)",
        fontsize=15, fontweight="bold", y=0.997,
    )
    plt.tight_layout()

    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def grid(*sessions, output: str = "eval_grid.png",
         protocol_dir: str | None = None) -> None:
    """CLI wrapper — render multi-session grid."""
    if not sessions:
        raise SystemExit("usage: grid <session1> [session2 …] [--output=PATH]")
    out = render_sessions_grid(list(sessions), output, protocol_dir=protocol_dir)
    print(f"saved: {out}")


def render_pooled_heatmap(sessions: list, output, *,
                          protocol_dir: Path | str | None = None,
                          grid: int = 180) -> Path:
    """Pool trials across multiple sessions; render ONE 3-panel heatmap.

    Use to visualize *aggregate* coverage and success-rate across the whole
    eval history — useful for planning where to collect more training data.

    All sessions must share the same canonical workspace + base frame.
    """
    proto_dir = Path(protocol_dir) if protocol_dir else PROTO_DIR
    rows = [_load_session_for_render(s, proto_dir) for s in sessions]

    # Pool every trial from every session into one big list.
    all_trials: list = []
    proto_names: set = set()
    for _, trials, _, _, proto_name in rows:
        all_trials.extend(trials)
        proto_names.add(proto_name)

    # Discover unique target_colors in stable order.
    seen_colors: list[str] = []
    for t in all_trials:
        c = t["target_color"]
        if c and c not in seen_colors:
            seen_colors.append(c)

    ws = rows[0][2]
    base_img = rows[0][3]
    x0, y0, x1, y1 = ws
    gx = np.linspace(x0, x1, grid)
    gy = np.linspace(y0, y1, grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.vstack([GX.ravel(), GY.ravel()])

    panels = [("all", all_trials, "ALL trials (pooled)")]
    for c in seen_colors:
        sub = [t for t in all_trials if t["target_color"] == c]
        panels.append((c, sub, f"Target = {c.upper()} (pooled)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels) + 4, 8))
    if len(panels) == 1:
        axes = [axes]

    for ax, (scope, sub, title) in zip(axes, panels):
        target_color = scope if scope != "all" else None
        _render_panel(ax, sub, title,
                      base_img=base_img, ws=ws, GX=GX, GY=GY, grid_pts=grid_pts,
                      scope=scope, target_color=target_color)

    n_total = len(all_trials)
    n_succ = sum(1 for t in all_trials if t["result"] == "success")
    fig.suptitle(
        f"Pooled coverage across {len(rows)} eval sessions  ·  "
        f"{n_total} trials  ·  pooled SR {n_succ}/{n_total} = {n_succ/n_total:.0%}  ·  "
        f"protocols: {sorted(proto_names)}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout()

    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pooled(*sessions, output: str = "eval_pooled.png",
           protocol_dir: str | None = None) -> None:
    """CLI wrapper — pool trials across sessions into one 3-panel heatmap."""
    if not sessions:
        raise SystemExit("usage: pooled <session1> [session2 …] [--output=PATH]")
    out = render_pooled_heatmap(list(sessions), output, protocol_dir=protocol_dir)
    print(f"saved: {out}")


def heatmap(session: str, output: str | None = None,
            protocol_dir: str | None = None) -> None:
    """CLI wrapper — render and save the 3-panel heatmap.

    If ``output`` is omitted, writes ``eval_<sessionname>_heatmap.png`` next to
    the session directory's parent (i.e., the version directory).
    """
    sess_path = _resolve_session_path(session)
    sess_dir = sess_path.parent
    if output is None:
        # version dir is two parents up: <version>/eval_sessions/<session>/session.json
        version_dir = sess_dir.parent.parent
        output = str(version_dir / f"eval_{sess_dir.name}_heatmap.png")
    out = render_session_heatmap(sess_dir, output, protocol_dir=protocol_dir)
    print(f"saved: {out}")


if __name__ == "__main__":
    import fire
    fire.Fire({"heatmap": heatmap, "grid": grid, "pooled": pooled})
