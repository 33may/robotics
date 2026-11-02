#!/usr/bin/env bash
# cell_audit.sh — MAY-173 micro-cell experiment audit (16-07).
#
# One experiment arm on one recorded cell: dump → ROS2 bag → container cuVSLAM
# (edex + compute_poses only, no cuSFM/cuVGL) → GT audit + impossible-jump count.
# Cells are minutes long, so no staged deletion — bag and map are kept.
#
# Usage:  bash logic/simulation/mapping/cell_audit.sh <dump_dir> <out_name> [extra -o opts…]
# e.g. :  bash …/cell_audit.sh data/coverage_drives/cellrun_ep0_5hz_base cell_ep0_5hz_base
#         bash …/cell_audit.sh data/coverage_drives/cellrun_ep0_30hz cell_ep0_30hz \
#              -o cuvslam.cfg_enable_imu_fusion=true

set -u
H=/home/may33/projects/ml_portfolio/robotics/humanoid
DUMP=${1:?dump dir}
OUT=${2:?output map name}
shift 2
EXTRA_OPTS=("$@")
PY=~/miniconda3/envs/hum/bin/python
LEROBOT_PY=~/miniconda3/envs/lerobot/bin/python
PODRUN="podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v $H/data:/data -v $H/logic/simulation/mapping/container:/cfg \
  -e ISAAC_ROS_WS=/data/ws oli-isaac-mapping bash -lc"

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }
cd "$H" || die "no repo dir"

# Variant support: one master dump → many bag variants. Override the bag path
# (so variants don't collide) and pass synth flags, e.g.:
#   SYNTH_ARGS="--every-n 6 --no-imu" BAG=data/coverage_drives/foo_bagA bash cell_audit.sh …
BAG="${BAG:-${DUMP%/}_bag}"

say "=== A: synthesize bag${SYNTH_ARGS:+ ($SYNTH_ARGS)} ==="
$PY logic/simulation/mapping/rosbag_synth.py --dump "$DUMP" --out "$BAG" ${SYNTH_ARGS:-} \
  || die "bag synth failed"

say "=== B: cuVSLAM (planar, unlimited graph${EXTRA_OPTS[*]:+, ${EXTRA_OPTS[*]}}) ==="
$PODRUN "ros2 run isaac_mapping_ros create_map_offline.py \
  --sensor_data_bag /${BAG} \
  --base_output_folder /data/maps/$OUT \
  --camera_topic_config /cfg/camera_topic_config_oli.yaml \
  --steps_to_run edex compute_poses \
  -o cuvslam.cfg_planar=true -o cuvslam.cfg_slam_max_map_size=0 ${EXTRA_OPTS[*]} \
  --print_mode tail" || die "edex/compute_poses failed"

M=$(ls -dt "$H"/data/maps/$OUT/2026-* 2>/dev/null | head -1)
[ -n "$M" ] || die "no map output dir found"

say "=== C: GT audit + jump count ==="
REPORT="$M/cell_report.txt"
: > "$REPORT"
find "$M/poses" -name '*.tum' | sort | while read -r tum; do
  rel=${tum#"$M"/}
  stats=$($PY logic/simulation/mapping/map_audit.py --tum "$tum" --dump "$DUMP" 2>/dev/null) \
    || stats="AUDIT-FAILED"
  jumps=$($LEROBOT_PY - "$tum" <<'EOF'
import sys
import numpy as np
tum = np.loadtxt(sys.argv[1], ndmin=2)
d = np.linalg.norm(np.diff(tum[:, 1:4], axis=0), axis=1)
print(f"jumps>0.5m={int((d > 0.5).sum())} jumps>2m={int((d > 2.0).sum())} maxstep={d.max():.2f}m" if len(d) else "empty")
EOF
  )
  echo "$rel  $stats  $jumps" >> "$REPORT"
done
say "cell report ($REPORT):"
cat "$REPORT"
