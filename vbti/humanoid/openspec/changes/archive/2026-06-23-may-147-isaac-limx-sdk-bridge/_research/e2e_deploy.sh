#!/usr/bin/env bash
# e2e_deploy.sh — Phase 7 end-to-end: deploy-python (stand) drives Isaac via bridge.
#
# Launches load_oli.py (isaac env → spawns sidecar in limx env, Isaac becomes a
# MROS sim peer), then humanoid-rl-deploy-python main.py (limx env, autostarts
# the stand controller → publishes RobotCmd). Verifies the sidecar's cmd_recv
# climbs (deploy → Isaac cmd flow) and Isaac stays alive.
#
# Self-cleaning: kills the whole process group on exit.
set -u

ROOT=/home/may33/projects/ml_portfolio/robotics
ISAAC_PY=/home/may33/miniconda3/envs/isaac/bin/python
LIMX_PY=/home/may33/miniconda3/envs/limx/bin/python
DEPLOY=$ROOT/humanoid/vendor/humanoid-rl-deploy-python
SOCK=/tmp/limx-isaac-bridge.sock
RUN_SECS=${1:-40}

ISAAC_LOG=/tmp/e2e_isaac.log
DEPLOY_LOG=/tmp/e2e_deploy.log
PIDS=()

cleanup() {
  echo "[e2e] cleanup — killing children"
  for pid in "${PIDS[@]}"; do
    kill -INT "$pid" 2>/dev/null
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
  done
  # belt-and-suspenders: any stragglers
  pkill -f "load_oli.py" 2>/dev/null
  pkill -f "bridge/sidecar.py" 2>/dev/null
  pkill -f "humanoid-rl-deploy-python/main.py" 2>/dev/null
  pkill -f "limxsdk.ability.cli" 2>/dev/null
  rm -f "$SOCK"
}
trap cleanup EXIT INT TERM

rm -f "$SOCK" "$ISAAC_LOG" "$DEPLOY_LOG"

echo "[e2e] starting Isaac sim peer (load_oli.py --headless) ..."
# Big max-ticks so it stays alive for the whole test even at GPU speed.
$ISAAC_PY -u "$ROOT/humanoid/logic/simulation/isaacsim/load_oli.py" \
  --headless --ip 127.0.0.1 --socket "$SOCK" --max-ticks 200000 --debug \
  >"$ISAAC_LOG" 2>&1 &
PIDS+=($!)

echo "[e2e] waiting for sidecar handshake (socket + 'subscribed to RobotCmdForSim') ..."
for i in $(seq 1 60); do
  if grep -q "subscribed to RobotCmdForSim" "$ISAAC_LOG" 2>/dev/null; then
    echo "[e2e] sidecar ready after ${i}s"
    break
  fi
  sleep 1
done
if ! grep -q "subscribed to RobotCmdForSim" "$ISAAC_LOG" 2>/dev/null; then
  echo "[e2e] FAIL — sidecar never became ready. Isaac log tail:"
  tail -20 "$ISAAC_LOG"
  exit 1
fi

echo "[e2e] starting kinematic_projection (MROS relay: converts+republishes state/cmd) ..."
SIM_DIR=$ROOT/humanoid/vendor/humanoid-mujoco-sim
( MROS_IP_LIST=127.0.0.x ROBOT_TYPE=HU_D04_01 \
  MROS_ETC_PATH="$SIM_DIR/prebuild/etc" MROS_LOG_LEVEL=0 \
  "$SIM_DIR/prebuild/kinematic_projection" >/tmp/e2e_kinproj.log 2>&1 ) &
PIDS+=($!)
sleep 4  # let it finish joint calibration

echo "[e2e] starting deploy-python (autostarts stand controller) ..."
# IMPORTANT: the deploy stack shells out with bare `python3 -m limxsdk.ability.cli`,
# so the limx conda env must be ACTIVATED (PATH python3 = limx python), not just
# invoked by interpreter path. Source conda activate before running main.py.
( source /home/may33/miniconda3/etc/profile.d/conda.sh && conda activate limx && \
  cd "$DEPLOY" && ROBOT_TYPE=HU_D04_01 MROS_IP_LIST=127.0.0.x \
  python main.py 127.0.0.1 \
  >"$DEPLOY_LOG" 2>&1 ) &
PIDS+=($!)

echo "[e2e] running for ${RUN_SECS}s ..."
sleep "$RUN_SECS"

echo ""
echo "=============== RESULTS ==============="
echo "--- sidecar cmd/state stats (climbing cmd_recv = deploy→Isaac cmd flow) ---"
grep -E "stats: cmd_recv" "$ISAAC_LOG" | tail -5
echo ""
echo "--- Isaac tick latency ---"
grep -E "tick .* latency" "$ISAAC_LOG" | tail -3
echo ""
echo "--- deploy-python output tail ---"
tail -15 "$DEPLOY_LOG"
echo "======================================="
