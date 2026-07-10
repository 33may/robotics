"""harness.py — headless closed-loop SE(2) kinematic sim for the nav-stack localization budget.

WHAT THIS MEASURES
------------------
The Oli nav stack is DONE and world-invariant: an offline-baked 2D `OccupancyGrid` costmap →
8-connected A* (`plan_path`) → holonomic pure-pursuit (`PurePursuit.command`) → body-frame
`(vx, vy, wz)`. What is NOT decided is which *localizer* to drop into the `Localizer` seam
(`reason/nav/localizer.py`). This harness answers the prerequisite question empirically:

    how much localization ERROR, and how low a localization RATE, does the EXISTING nav stack
    tolerate before it starts failing (colliding / missing goals)?

It never touches Isaac or limxsdk — it is a pure-kinematic closed loop over the real nav modules.
The robot PLANS and STEERS on a NOISY estimated pose, but the TRUE pose integrates the resulting
body twist. Success/collision are judged on the TRUE pose against the RAW (un-inflated) grid.

    per tick:
      1. est   = true_pose + injected localization noise (per E1/E2 policy)
      2. path  = plan_path(inflated_grid, (est.x, est.y), goal)
      3. twist = pursuit.command(est_as_RobotPose, path)      # body frame (vx, vy, wz)
      4. integrate TRUE pose by that twist over dt, rotating body vel into world by TRUE yaw:
            true_x  += (vx*cos(yaw) - vy*sin(yaw)) * dt
            true_y  += (vx*sin(yaw) + vy*cos(yaw)) * dt
            true_yaw+= wz * dt
    terminate:
      reached   dist(true, goal) <= goal_tol                  -> success
      collision raw_grid.is_occupied(true_x, true_y)          -> fail
      timeout   tick >= max_ticks                             -> fail

The "gap" between est and true is exactly what a real localizer's error would inject. Two sweeps:

  E1 accuracy tolerance   — Gaussian position noise, Gaussian yaw noise, constant position bias.
  E2 reloc-rate/staleness — refresh the estimate only every k ticks; between fixes either HOLD the
                            last fix or DEAD-RECKON by integrating the commanded twist (perfect-
                            odometry proxy). Directly tests whether a slow relocalizer NEEDS a fast
                            odometry layer.

Deterministic: fixed RNG seed, no wall-clock randomness. Re-runnable via CLI (`--help`).

Pure numpy/stdlib + the invariant nav modules; matplotlib/scipy only for optional plots/inflate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Import the real, invariant nav stack (the thing under test) --------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../humanoid
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from logic.oli.reason.nav.controller import PurePursuit  # noqa: E402
from logic.oli.reason.nav.costmap import OccupancyGrid  # noqa: E402
from logic.oli.reason.nav.planner import plan_path  # noqa: E402
from logic.oli.reason.nav.types import RobotPose  # noqa: E402

Point = Tuple[float, float]

# --- Fixed simulation parameters (documented, deterministic) ------------------------------------
DT = 0.1                # s, integration step (10 Hz control tick)
ROBOT_RADIUS = 0.3      # m, circle footprint for inflation
MAX_TICKS = 600         # 60 s wall of sim time before timeout
GOAL_TOL = 0.15         # m, PurePursuit arrival radius (its stop condition)
RESOLUTION = 0.10       # m/cell for all test maps
SEED = 20260710         # master seed — deterministic across the whole run

# Start/goal endpoints are sampled free in a grid inflated by ROBOT_RADIUS + SAMPLE_MARGIN (not
# just ROBOT_RADIUS). Reason: PurePursuit has no path-tracking recovery — its lookahead can corner-
# cut into the inflated band, landing the robot in a cell the planner then can't escape (A* returns
# None, robot freezes). A margin gives the baseline breathing room so the zero-noise cell is a clean
# ~near-100% reference and the noise sweeps stay attributable to the injected error, not to hard
# geometry. Any residual baseline failure (an inherent property of this hand-rolled nav stack, no
# recovery behavior) shows up honestly in the sigma=0 cell. PLANNING still inflates by ROBOT_RADIUS.
SAMPLE_MARGIN = 0.10    # m, extra clearance for endpoint sampling only

# Success radius for the TRUE pose. The planner snaps the goal to a CELL CENTER, and PurePursuit
# stops within GOAL_TOL of that snapped waypoint (path[-1]) — NOT of the true continuous goal. The
# snap adds up to resolution*sqrt(2)/2 (~0.07 m at 0.1 m/cell). So a robot that has legitimately
# "arrived" (controller commands zero) can sit up to GOAL_TOL + snap from the true goal. Judging
# TRUE-pose success at GOAL_TOL alone would spuriously fail perfect-pose runs (verified: zero-noise
# runs park at ~0.154 m). We use GOAL_TOL + RESOLUTION to absorb the cell-snap; this is a
# measurement alignment, not a loosening of the nav stack's own arrival criterion.
SUCCESS_RADIUS = GOAL_TOL + RESOLUTION  # 0.25 m

# Sweep grids (from the task spec).
E1_POS_SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]        # m
E1_YAW_SIGMAS_DEG = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]    # deg
E1_POS_BIASES = [0.0, 0.05, 0.1, 0.2]                   # m
E2_KS = [1, 2, 5, 10, 20, 50]                            # reloc period (ticks)
E2_MODES = ["hold", "dead_reckon"]
E2_POS_SIGMA = 0.05                                      # m, modest fixed noise so rate effect shows
TRIALS_PER_CELL = 24                                    # >= 20 per spec, across maps


# ================================================================================================
# Test maps
# ================================================================================================
@dataclass
class TestMap:
    name: str
    grid: OccupancyGrid            # RAW occupancy (collision test)
    inflated: OccupancyGrid        # inflated by ROBOT_RADIUS (planner input)
    sample_inflated: OccupancyGrid  # inflated by ROBOT_RADIUS + SAMPLE_MARGIN (endpoint sampling)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """World (xmin, ymin, xmax, ymax) of the map extent."""
        ox, oy = self.grid.origin
        xmax = ox + self.grid.ncols * self.grid.resolution
        ymax = oy + self.grid.nrows * self.grid.resolution
        return (ox, oy, xmax, ymax)


def _make_empty(size_m: float = 6.0) -> np.ndarray:
    n = int(round(size_m / RESOLUTION))
    grid = np.zeros((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True  # walls
    return grid


def _make_corridor(size_m: float = 6.0) -> np.ndarray:
    """A wall spanning the map with a single doorway gap — forces a narrow passage."""
    n = int(round(size_m / RESOLUTION))
    grid = np.zeros((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    wall_row = n // 2
    grid[wall_row, :] = True
    grid[wall_row - 1, :] = True  # 2-cell thick wall
    door_c = n // 2
    door_half = max(1, int(round(0.9 / RESOLUTION)))  # ~1.8 m doorway (> 2*robot_radius)
    grid[wall_row - 1:wall_row + 1, door_c - door_half:door_c + door_half] = False
    return grid


def _make_obstacle_field(size_m: float = 6.0, n_obs: int = 14, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random scattered square obstacles (deterministic given rng)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = int(round(size_m / RESOLUTION))
    grid = np.zeros((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    obs_half = max(1, int(round(0.3 / RESOLUTION)))  # ~0.6 m blocks
    margin = 3
    for _ in range(n_obs):
        r = int(rng.integers(margin + obs_half, n - margin - obs_half))
        c = int(rng.integers(margin + obs_half, n - margin - obs_half))
        grid[r - obs_half:r + obs_half + 1, c - obs_half:c + obs_half + 1] = True
    return grid


def build_maps() -> List[TestMap]:
    rng = np.random.default_rng(SEED + 999)
    specs = [
        ("empty", _make_empty()),
        ("corridor", _make_corridor()),
        ("obstacle_field", _make_obstacle_field(rng=rng)),
    ]
    maps: List[TestMap] = []
    for name, raw in specs:
        g = OccupancyGrid(raw, RESOLUTION, (0.0, 0.0))
        maps.append(TestMap(
            name=name,
            grid=g,
            inflated=g.inflate(ROBOT_RADIUS),
            sample_inflated=g.inflate(ROBOT_RADIUS + SAMPLE_MARGIN),
        ))
    return maps


# ================================================================================================
# Start/goal sampling — collision-free, planner-reachable pairs
# ================================================================================================
def _free_world_point(m: TestMap, rng: np.random.Generator) -> Point:
    """Sample a world point free in the sampling-inflated grid (robot_radius + margin clearance)."""
    xmin, ymin, xmax, ymax = m.bounds
    for _ in range(4000):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if not m.sample_inflated.is_occupied(x, y):
            return (x, y)
    raise RuntimeError(f"could not sample a free point in map {m.name}")


def sample_start_goal(m: TestMap, rng: np.random.Generator, min_sep: float = 2.0) -> Tuple[Point, Point]:
    """A start/goal pair that is far enough apart AND provably reachable by the planner."""
    for _ in range(500):
        start = _free_world_point(m, rng)
        goal = _free_world_point(m, rng)
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < min_sep:
            continue
        if plan_path(m.inflated, start, goal) is not None:
            return start, goal
    raise RuntimeError(f"could not sample a reachable start/goal in map {m.name}")


# ================================================================================================
# Noise model
# ================================================================================================
@dataclass
class NoiseSpec:
    """Localization-error injection policy for one trial."""
    pos_sigma: float = 0.0     # m, zero-mean Gaussian on x and y
    yaw_sigma: float = 0.0     # rad, zero-mean Gaussian on yaw
    pos_bias: float = 0.0      # m, constant offset magnitude, fixed random direction per trial
    reloc_k: int = 1           # refresh estimate every k ticks
    mode: str = "fresh"        # "fresh" (k==1), "hold", or "dead_reckon"


def _apply_noise(true: RobotPose, spec: NoiseSpec, bias_vec: Tuple[float, float], rng: np.random.Generator) -> RobotPose:
    ex = true.x + spec.pos_bias * bias_vec[0]
    ey = true.y + spec.pos_bias * bias_vec[1]
    eyaw = true.yaw
    if spec.pos_sigma > 0.0:
        ex += float(rng.normal(0.0, spec.pos_sigma))
        ey += float(rng.normal(0.0, spec.pos_sigma))
    if spec.yaw_sigma > 0.0:
        eyaw += float(rng.normal(0.0, spec.yaw_sigma))
    return RobotPose(true.stamp_ns, ex, ey, eyaw)


# ================================================================================================
# Single closed-loop trial
# ================================================================================================
@dataclass
class TrialResult:
    success: bool
    collided: bool
    timed_out: bool
    final_dist: float          # m, TRUE pose distance to goal at termination
    ticks: int
    map_name: str


def run_trial(
    m: TestMap,
    start: Point,
    goal: Point,
    spec: NoiseSpec,
    rng: np.random.Generator,
    pursuit: PurePursuit,
) -> TrialResult:
    """One closed-loop episode. Plans/steers on the noisy estimate; integrates the TRUE pose."""
    # TRUE pose starts at `start`, heading roughly toward the goal.
    true_yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    true = RobotPose(0, start[0], start[1], true_yaw)

    # Fixed per-trial bias direction (unit vector) so bias is a constant offset, not noise.
    theta = float(rng.uniform(-math.pi, math.pi))
    bias_vec = (math.cos(theta), math.sin(theta))

    # State for reloc-rate modes: the estimate the robot currently believes.
    est = _apply_noise(true, spec, bias_vec, rng)

    for tick in range(MAX_TICKS):
        stamp = tick * int(DT * 1e9)

        # --- (1) localization estimate, per reloc-rate policy ---
        if spec.mode in ("fresh", "hold", "dead_reckon"):
            if tick % spec.reloc_k == 0:
                # Fresh fix from the (noisy) localizer.
                est = _apply_noise(RobotPose(stamp, true.x, true.y, true.yaw), spec, bias_vec, rng)
            # else: between fixes — est is either held (unchanged) or dead-reckoned (updated below).
        else:
            raise ValueError(f"unknown mode {spec.mode}")

        # --- (2) plan on the estimate ---
        path = plan_path(m.inflated, (est.x, est.y), goal)
        if path is None:
            # Planner gave up from the (noisy) estimate — steer nowhere this tick.
            vx = vy = wz = 0.0
        else:
            # --- (3) pursuit command on the estimate (body frame) ---
            vx, vy, wz = pursuit.command(est, path)

        # --- (4) integrate the TRUE pose by the commanded body twist ---
        cy, sy = math.cos(true.yaw), math.sin(true.yaw)
        nx = true.x + (vx * cy - vy * sy) * DT
        ny = true.y + (vx * sy + vy * cy) * DT
        nyaw = true.yaw + wz * DT
        true = RobotPose(stamp, nx, ny, nyaw)

        # Dead-reckon mode: propagate the ESTIMATE by the same commanded twist between fixes.
        # (Perfect-odometry proxy: odometry integrates the exact command, so the estimate keeps
        # up with motion but never re-anchors to truth until the next fix.)
        if spec.mode == "dead_reckon":
            ecy, esy = math.cos(est.yaw), math.sin(est.yaw)
            enx = est.x + (vx * ecy - vy * esy) * DT
            eny = est.y + (vx * esy + vy * ecy) * DT
            enyaw = est.yaw + wz * DT
            est = RobotPose(stamp, enx, eny, enyaw)

        # --- termination checks on the TRUE pose ---
        dist_goal = math.hypot(goal[0] - true.x, goal[1] - true.y)
        if m.grid.is_occupied(true.x, true.y):
            return TrialResult(False, True, False, dist_goal, tick + 1, m.name)
        if dist_goal <= SUCCESS_RADIUS:
            return TrialResult(True, False, False, dist_goal, tick + 1, m.name)

    final_dist = math.hypot(goal[0] - true.x, goal[1] - true.y)
    return TrialResult(False, False, True, final_dist, MAX_TICKS, m.name)


# ================================================================================================
# Cell aggregation (a "cell" = one sweep point evaluated over N trials across maps)
# ================================================================================================
@dataclass
class CellResult:
    label: str
    params: Dict[str, Any]
    n_trials: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_final_dist_m: float
    per_map: Dict[str, float] = field(default_factory=dict)  # per-map success rate


def run_cell(
    label: str,
    params: Dict[str, Any],
    spec: NoiseSpec,
    maps: List[TestMap],
    seed: int,
) -> CellResult:
    """Run TRIALS_PER_CELL trials, spread round-robin across maps, all deterministic from `seed`."""
    pursuit = PurePursuit(max_lin=1.0, max_wz=1.5, lookahead=0.5, goal_tol=GOAL_TOL)
    results: List[TrialResult] = []
    per_map_hits: Dict[str, List[int]] = {m.name: [] for m in maps}
    for i in range(TRIALS_PER_CELL):
        m = maps[i % len(maps)]
        # Per-trial RNG derived from (cell seed, trial index) — reproducible, independent streams.
        rng = np.random.default_rng((seed * 1_000_003 + i) & 0xFFFFFFFFFFFFFFFF)
        start, goal = sample_start_goal(m, rng)
        res = run_trial(m, start, goal, spec, rng, pursuit)
        results.append(res)
        per_map_hits[m.name].append(1 if res.success else 0)

    n = len(results)
    succ = sum(r.success for r in results)
    coll = sum(r.collided for r in results)
    tout = sum(r.timed_out for r in results)
    mean_fd = float(np.mean([r.final_dist for r in results]))
    per_map = {name: (float(np.mean(hits)) if hits else float("nan")) for name, hits in per_map_hits.items()}
    return CellResult(
        label=label,
        params=params,
        n_trials=n,
        success_rate=succ / n,
        collision_rate=coll / n,
        timeout_rate=tout / n,
        mean_final_dist_m=mean_fd,
        per_map=per_map,
    )


# ================================================================================================
# Sweeps
# ================================================================================================
def sweep_e1(maps: List[TestMap]) -> List[CellResult]:
    cells: List[CellResult] = []
    sidx = 0
    # (a) Gaussian position noise
    for sig in E1_POS_SIGMAS:
        spec = NoiseSpec(pos_sigma=sig)
        cells.append(run_cell(
            label=f"pos_gauss/sigma={sig:g}m",
            params={"noise_type": "pos_gauss", "magnitude": f"{sig:g} m", "sigma_m": sig},
            spec=spec, maps=maps, seed=SEED + sidx))
        sidx += 1
    # (b) Gaussian yaw noise
    for sig_deg in E1_YAW_SIGMAS_DEG:
        spec = NoiseSpec(yaw_sigma=math.radians(sig_deg))
        cells.append(run_cell(
            label=f"yaw_gauss/sigma={sig_deg:g}deg",
            params={"noise_type": "yaw_gauss", "magnitude": f"{sig_deg:g} deg", "sigma_deg": sig_deg},
            spec=spec, maps=maps, seed=SEED + 100 + sidx))
        sidx += 1
    # (c) constant position bias
    for bias in E1_POS_BIASES:
        spec = NoiseSpec(pos_bias=bias)
        cells.append(run_cell(
            label=f"pos_bias/{bias:g}m",
            params={"noise_type": "pos_bias", "magnitude": f"{bias:g} m", "bias_m": bias},
            spec=spec, maps=maps, seed=SEED + 200 + sidx))
        sidx += 1
    return cells


def sweep_e2(maps: List[TestMap]) -> List[CellResult]:
    cells: List[CellResult] = []
    sidx = 0
    for mode in E2_MODES:
        for k in E2_KS:
            spec = NoiseSpec(pos_sigma=E2_POS_SIGMA, reloc_k=k, mode=mode)
            cells.append(run_cell(
                label=f"{mode}/k={k}",
                params={"reloc_every_ticks": k, "between_fix_mode": mode, "pos_sigma_m": E2_POS_SIGMA},
                spec=spec, maps=maps, seed=SEED + 500 + sidx))
            sidx += 1
    return cells


# ================================================================================================
# Output: JSON, CSV, plots / markdown
# ================================================================================================
def _cell_row(c: CellResult) -> Dict[str, object]:
    row = {
        "label": c.label,
        "n_trials": c.n_trials,
        "success_rate": round(c.success_rate, 4),
        "collision_rate": round(c.collision_rate, 4),
        "timeout_rate": round(c.timeout_rate, 4),
        "mean_final_dist_m": round(c.mean_final_dist_m, 4),
    }
    row.update(c.params)
    for mname, sr in c.per_map.items():
        row[f"sr_{mname}"] = round(sr, 4) if sr == sr else None
    return row


def write_json(path: Path, e1: List[CellResult], e2: List[CellResult]) -> None:
    payload = {
        "config": {
            "dt_s": DT, "robot_radius_m": ROBOT_RADIUS, "max_ticks": MAX_TICKS,
            "goal_tol_m": GOAL_TOL, "resolution_m": RESOLUTION, "seed": SEED,
            "trials_per_cell": TRIALS_PER_CELL,
            "maps": ["empty", "corridor", "obstacle_field"],
        },
        "e1_accuracy": [_cell_row(c) for c in e1],
        "e2_reloc_rate": [_cell_row(c) for c in e2],
    }
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, e1: List[CellResult], e2: List[CellResult]) -> None:
    rows = [{"sweep": "E1", **_cell_row(c)} for c in e1] + [{"sweep": "E2", **_cell_row(c)} for c in e2]
    # Union of keys, stable order.
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_plots(out_dir: Path, e1: List[CellResult], e2: List[CellResult]) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: List[Path] = []

    # --- E1 accuracy: success vs noise magnitude (3 panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    pos = [c for c in e1 if c.params["noise_type"] == "pos_gauss"]
    axes[0].plot([c.params["sigma_m"] for c in pos], [c.success_rate for c in pos], "o-", color="#2266cc")
    axes[0].set_title("Position noise (Gaussian)")
    axes[0].set_xlabel("pos sigma [m]"); axes[0].set_ylabel("success rate")

    yaw = [c for c in e1 if c.params["noise_type"] == "yaw_gauss"]
    axes[1].plot([c.params["sigma_deg"] for c in yaw], [c.success_rate for c in yaw], "o-", color="#cc6622")
    axes[1].set_title("Yaw noise (Gaussian)")
    axes[1].set_xlabel("yaw sigma [deg]"); axes[1].set_ylabel("success rate")

    bias = [c for c in e1 if c.params["noise_type"] == "pos_bias"]
    axes[2].plot([c.params["bias_m"] for c in bias], [c.success_rate for c in bias], "o-", color="#229955")
    axes[2].set_title("Constant position bias")
    axes[2].set_xlabel("bias [m]"); axes[2].set_ylabel("success rate")
    for ax in axes:
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    fig.suptitle("E1 — nav-stack accuracy tolerance (success vs localization error)")
    fig.tight_layout()
    p1 = out_dir / "e1_accuracy.png"
    fig.savefig(p1, dpi=110); plt.close(fig); paths.append(p1)

    # --- E2 reloc-rate: success vs reloc period, hold vs dead-reckon ---
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.6))
    for mode, color in (("hold", "#cc3333"), ("dead_reckon", "#2266cc")):
        cells = [c for c in e2 if c.params["between_fix_mode"] == mode]
        cells.sort(key=lambda c: c.params["reloc_every_ticks"])
        ks = [c.params["reloc_every_ticks"] for c in cells]
        axa.plot(ks, [c.success_rate for c in cells], "o-", color=color, label=mode)
        axb.plot(ks, [c.mean_final_dist_m for c in cells], "o-", color=color, label=mode)
    axa.set_xscale("log"); axa.set_title("Success vs reloc period")
    axa.set_xlabel("reloc every k ticks (0.1 s/tick)"); axa.set_ylabel("success rate")
    axa.set_ylim(-0.05, 1.05); axa.grid(alpha=0.3, which="both"); axa.legend()
    axb.set_xscale("log"); axb.set_title("Mean final distance vs reloc period")
    axb.set_xlabel("reloc every k ticks (0.1 s/tick)"); axb.set_ylabel("mean final dist [m]")
    axb.grid(alpha=0.3, which="both"); axb.legend()
    fig.suptitle(f"E2 — reloc-rate / staleness tolerance (fixed pos sigma {E2_POS_SIGMA} m)")
    fig.tight_layout()
    p2 = out_dir / "e2_reloc_rate.png"
    fig.savefig(p2, dpi=110); plt.close(fig); paths.append(p2)

    return paths


def write_markdown(path: Path, e1: List[CellResult], e2: List[CellResult]) -> None:
    lines = ["# Localization budget — results\n"]
    lines.append("## E1 accuracy tolerance\n")
    lines.append("| noise | magnitude | success | collision | mean final dist (m) |")
    lines.append("|---|---|---|---|---|")
    for c in e1:
        lines.append(f"| {c.params['noise_type']} | {c.params['magnitude']} | "
                     f"{c.success_rate:.2f} | {c.collision_rate:.2f} | {c.mean_final_dist_m:.3f} |")
    lines.append("\n## E2 reloc-rate / staleness tolerance\n")
    lines.append("| mode | k (ticks) | success | collision | mean final dist (m) |")
    lines.append("|---|---|---|---|---|")
    for c in e2:
        lines.append(f"| {c.params['between_fix_mode']} | {c.params['reloc_every_ticks']} | "
                     f"{c.success_rate:.2f} | {c.collision_rate:.2f} | {c.mean_final_dist_m:.3f} |")
    path.write_text("\n".join(lines) + "\n")


# ================================================================================================
# Budget derivation (headline numbers)
# ================================================================================================
def derive_accuracy_budget(e1: List[CellResult], threshold: float = 0.9) -> str:
    """Largest position sigma keeping success >= threshold."""
    pos = sorted([c for c in e1 if c.params["noise_type"] == "pos_gauss"],
                 key=lambda c: c.params["sigma_m"])
    ok = [c for c in pos if c.success_rate >= threshold]
    if not ok:
        return f"no pos-noise level meets success >= {threshold:.0%}"
    best = max(ok, key=lambda c: c.params["sigma_m"])
    return f"pos sigma <= {best.params['sigma_m']:g} m keeps success >= {threshold:.0%} (={best.success_rate:.0%})"


def derive_rate_budget(e2: List[CellResult], threshold: float = 0.9) -> str:
    """Largest reloc period keeping success >= threshold, per mode."""
    parts = []
    for mode in E2_MODES:
        cells = sorted([c for c in e2 if c.params["between_fix_mode"] == mode],
                       key=lambda c: c.params["reloc_every_ticks"])
        ok = [c for c in cells if c.success_rate >= threshold]
        if ok:
            best = max(ok, key=lambda c: c.params["reloc_every_ticks"])
            k = best.params["reloc_every_ticks"]
            parts.append(f"{mode}: up to k={k} ticks ({k * DT:.1f}s, {1.0/(k*DT):.1f} Hz) @ {best.success_rate:.0%}")
        else:
            parts.append(f"{mode}: fails even at k=1")
    return "; ".join(parts)


# ================================================================================================
# CLI
# ================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results",
                    help="output directory for JSON/CSV/plots")
    ap.add_argument("--sweep", choices=["all", "e1", "e2"], default="all")
    ap.add_argument("--trials", type=int, default=None, help="override trials per cell (>=20 for real runs)")
    args = ap.parse_args()

    global TRIALS_PER_CELL
    if args.trials is not None:
        TRIALS_PER_CELL = args.trials

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[harness] seed={SEED} dt={DT}s robot_radius={ROBOT_RADIUS}m max_ticks={MAX_TICKS} "
          f"trials/cell={TRIALS_PER_CELL}")
    maps = build_maps()
    for m in maps:
        occ = m.grid.grid.mean()
        print(f"[harness] map '{m.name}': {m.grid.nrows}x{m.grid.ncols} cells, {occ:.1%} occupied (raw)")

    e1: List[CellResult] = []
    e2: List[CellResult] = []
    if args.sweep in ("all", "e1"):
        print("[harness] running E1 (accuracy tolerance)...")
        e1 = sweep_e1(maps)
        for c in e1:
            print(f"    E1 {c.label:28s} success={c.success_rate:.2f} coll={c.collision_rate:.2f} "
                  f"mean_final={c.mean_final_dist_m:.3f}m")
    if args.sweep in ("all", "e2"):
        print("[harness] running E2 (reloc-rate / staleness)...")
        e2 = sweep_e2(maps)
        for c in e2:
            print(f"    E2 {c.label:20s} success={c.success_rate:.2f} coll={c.collision_rate:.2f} "
                  f"mean_final={c.mean_final_dist_m:.3f}m")

    # --- persist ---
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    write_json(json_path, e1, e2)
    write_csv(csv_path, e1, e2)
    print(f"[harness] wrote {json_path}")
    print(f"[harness] wrote {csv_path}")

    plot_paths = try_plots(out_dir, e1, e2) if (e1 and e2) else []
    if plot_paths:
        for p in plot_paths:
            print(f"[harness] wrote plot {p}")
    else:
        md_path = out_dir / "results.md"
        write_markdown(md_path, e1, e2)
        print(f"[harness] matplotlib unavailable or partial sweep — wrote {md_path}")

    # --- headline budgets ---
    if e1:
        acc = derive_accuracy_budget(e1)
        print(f"[budget] ACCURACY: {acc}")
    if e2:
        rate = derive_rate_budget(e2)
        print(f"[budget] RATE: {rate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
