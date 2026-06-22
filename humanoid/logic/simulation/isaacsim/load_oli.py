"""
load_oli.py — Isaac Sim driver for HU_D04_01, the reference host app.

Two modes:
  --bridge (default): make Isaac a sim peer on the LimX MROS bus. Spawns the
      Py 3.8 sidecar, constructs Oli(bridge=...), and runs the physics loop so
      any humanoid-rl-deploy-python controller can drive Oli at `--ip`.
  --no-bridge: kinematics-only smoke loader (loads Oli, holds rest pose, no SDK).

This is intentionally thin — all Oli/bridge logic lives in `oli.py` / `bridge/`.
Copy this loop into any recon / nav / SLAM / RL host app.

Run:
  conda activate isaac
  python humanoid/logic/simulation/isaacsim/load_oli.py            # bridge mode
  python humanoid/logic/simulation/isaacsim/load_oli.py --no-bridge
  python humanoid/logic/simulation/isaacsim/load_oli.py --headless --ip 127.0.0.1

Then (bridge mode), in the limx env:
  ROBOT_TYPE=HU_D04_01 python <humanoid-rl-deploy-python>/main.py 127.0.0.1
"""

# ── 1. CLI parse BEFORE booting Kit (cheap, and picks headless) ───────────────
import argparse


def _parse():
    ap = argparse.ArgumentParser(description="Isaac HU_D04_01 driver")
    ap.add_argument("--no-bridge", action="store_true",
                    help="kinematics-only: load + hold rest pose, no SDK bridge")
    ap.add_argument("--ip", default="127.0.0.1", help="MROS peer IP")
    ap.add_argument("--socket", default="/tmp/limx-isaac-bridge.sock",
                    help="bridge UDS socket path")
    ap.add_argument("--headless", action="store_true", help="no viewport")
    ap.add_argument("--render-decimation", type=int, default=20,
                    help="render every N physics ticks (default 20)")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="stop after N ticks (0 = run until window closed; "
                         "use for headless smoke tests)")
    ap.add_argument("--debug", action="store_true", help="verbose sidecar log")
    return ap.parse_args()


ARGS = _parse()

# ── 2. Bootstrap Kit BEFORE any other isaac/pxr imports ───────────────────────
from pathlib import Path as _Path  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

_KIT_NAME = "isaacsim.exp.base.kit" if ARGS.headless else "isaacsim.exp.full.kit"
KIT = _Path(
    f"/home/may33/miniconda3/envs/isaac/lib/python3.11/site-packages/"
    f"isaacsim/apps/{_KIT_NAME}"
)
SIM_APP = SimulationApp({"headless": ARGS.headless, "experience": str(KIT)})

# ── 3. Now safe to import the rest ────────────────────────────────────────────
import sys  # noqa: E402

from isaacsim.core.api import World  # noqa: E402

sys.path.insert(0, "/home/may33/projects/ml_portfolio/robotics")
from humanoid.logic.simulation.isaacsim.oli import Oli  # noqa: E402

PHYSICS_DT = 1.0 / 1000.0


def log(msg: str) -> None:
    # print (not logging) — omni's carb logger swallows the root logger.
    print(f"[load_oli] {msg}", flush=True)


def run(bridge) -> None:
    """Build the world + Oli, then run the physics/render loop."""
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)
    world.scene.add_default_ground_plane()
    oli = Oli(world, bridge=bridge, spawn_pose=(0.0, 0.0, 1.05), pin_root=True)
    log(f"Oli ready: {oli.num_dof} DOFs, base={oli.base_link_path}")
    if bridge is not None:
        log(f"bridge attached — Oli is a sim peer on MROS at {ARGS.ip}")
    else:
        log("no bridge — kinematics-only; Oli sags to zero-effort pose")

    tick = 0
    last_stats = 0
    while SIM_APP.is_running():
        oli.tick()
        world.step(render=(tick % ARGS.render_decimation == 0))
        tick += 1
        if bridge is not None and tick - last_stats >= 5000:
            s = oli.tick_latency_stats()
            log(f"tick {tick} | latency p50={s['p50_us']:.0f}us "
                f"p99={s['p99_us']:.0f}us")
            last_stats = tick
        if ARGS.max_ticks and tick >= ARGS.max_ticks:
            log(f"reached --max-ticks={ARGS.max_ticks}, stopping")
            break


def main() -> int:
    if ARGS.no_bridge:
        run(bridge=None)
        SIM_APP.close()
        return 0

    # Bridge mode — spawn the Py 3.8 sidecar and run as a sim peer.
    from humanoid.logic.simulation.isaacsim.bridge import OliBridge, BridgeClosedError
    try:
        with OliBridge.spawn_sidecar(
            ip=ARGS.ip, socket=ARGS.socket, debug=ARGS.debug
        ) as bridge:
            run(bridge)
    except BridgeClosedError as e:
        log(f"bridge closed: {e} — shutting down")
    finally:
        SIM_APP.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
