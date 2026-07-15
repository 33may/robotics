#!/usr/bin/env bash
# full_bake_overnight.sh — MAY-173 T3: full-drive cuVGL-via-cuSFM bake (overnight run).
#
# Encodes everything the 15-07 smoke day established:
#   * staged deletion: the 68 GB bag exists only until edex succeeds (disk fits)
#   * batch PGO config (cusfm_configs_isaac/) — incremental mode OOMs this box
#   * real keyframe spacing for cusfm_cli (workflow hardcodes 0.0 → OOM)
#   * planar + unlimited pose graph for the cuVSLAM pass
#   * GT audit of every produced .tum → audit_report.txt for the morning
#
# Launch:  nohup bash logic/simulation/mapping/full_bake_overnight.sh [delay_s] &
# Watch :  tail -f /tmp/overnight_full_bake.log

set -u
H=/home/may33/projects/ml_portfolio/robotics/humanoid
DUMP=data/coverage_drives/warehouse_coverage_v1
BAG=data/coverage_drives/warehouse_coverage_v1_bag_full
OUT=wc_v1_full
DELAY=${1:-3600}
PY=~/miniconda3/envs/hum/bin/python
PODRUN="podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v $H/data:/data -v $H/logic/simulation/mapping/container:/cfg \
  -e ISAAC_ROS_WS=/data/ws oli-isaac-mapping bash -lc"

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }

cd "$H" || die "no repo dir"

say "sleeping ${DELAY}s (PC cleanup window) …"
sleep "$DELAY"

say "waiting for any running cuSFM mapper to finish (max 3 h) …"
for _ in $(seq 1 180); do
  pgrep -f keypoints_mapper_main >/dev/null || break
  sleep 60
done
pgrep -f keypoints_mapper_main >/dev/null && die "mapper still running after 3 h"

FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
say "free space: ${FREE_GB} G"
[ "$FREE_GB" -ge 110 ] || die "need >=110 G free for the staged bake, have ${FREE_GB} G"

say "=== A: synthesize full bag (skip 10 s head) ==="
$PY logic/simulation/mapping/rosbag_synth.py \
  --dump "$DUMP" --out "$BAG" --skip-seconds 10 || die "bag synth failed"
say "bag size: $(du -sh "$BAG" | cut -f1)"

say "=== B: edex + compute_poses (cuVSLAM, planar, unlimited graph) ==="
$PODRUN "ros2 run isaac_mapping_ros create_map_offline.py \
  --sensor_data_bag /data/coverage_drives/warehouse_coverage_v1_bag_full \
  --base_output_folder /data/maps/$OUT \
  --camera_topic_config /cfg/camera_topic_config_oli.yaml \
  --steps_to_run edex compute_poses \
  -o cuvslam.cfg_planar=true -o cuvslam.cfg_slam_max_map_size=0 \
  --print_mode tail" || die "edex/compute_poses failed"

M=$(ls -d "$H"/data/maps/$OUT/2026-* 2>/dev/null | head -1)
[ -n "$M" ] || die "no map output dir found"
MB=$(basename "$M")
say "map dir: $MB"

say "=== C: staged deletion — removing OUR bag (regenerable from dump) ==="
[ -f "$M/poses/keyframe_pose_optimized.tum" ] || die "no optimized poses; keeping bag"
rm -rf "$BAG"
say "bag removed; free now: $(df -h /home | awk 'END{print $4}')"

say "=== D: cuSFM chain (spacing 0.3 m/5°, batch PGO, downsampled matches) ==="
$PODRUN "ros2 run isaac_ros_visual_mapping cusfm_cli \
  --input_dir=/data/maps/$OUT/$MB/map_frames/rectified \
  --cusfm_base_dir=/data/maps/$OUT/$MB/cusfm \
  --model_dir=/opt/ros/jazzy/share/isaac_ros_visual_mapping/models \
  --config_dir=/cfg/cusfm_configs_isaac \
  --binary_dir=/opt/ros/jazzy/lib/isaac_ros_visual_mapping \
  --skip_cuvslam \
  --min_inter_frame_distance=0.3 --min_inter_frame_rotation_degrees=5 \
  --downsampling_matches" || say "WARN: cusfm chain failed — audits will show how far it got"

say "=== E: GT audit of every trajectory produced ==="
REPORT="$M/audit_report.txt"
: > "$REPORT"
find "$M" -name '*.tum' | sort | while read -r tum; do
  rel=${tum#"$M"/}
  stats=$($PY logic/simulation/mapping/map_audit.py --tum "$tum" --dump "$DUMP" 2>/dev/null) \
    && echo "$rel  $stats" >> "$REPORT" \
    || echo "$rel  AUDIT-FAILED" >> "$REPORT"
done
say "audit report:"
cat "$REPORT"

say "=== F: cuVGL map build (BoW over baked keyframes) ==="
$PODRUN "ros2 run isaac_mapping_ros create_map_offline.py \
  --map_dir /data/maps/$OUT/$MB \
  --camera_topic_config /cfg/camera_topic_config_oli.yaml \
  --steps_to_run cuvgl --print_mode tail" || say "WARN: cuvgl step failed"

say "DONE — morning entry points: $REPORT ; $M ; this log"
