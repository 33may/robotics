#!/usr/bin/env bash
# bake_map.sh — MAY-173: ONE command teleop dump → complete map dir.
#
# PRODUCTION recipe (Anton 17-07 evening, measured on teleop_demo_v2):
# every artifact the runtime consumes is BORN in the cuVSLAM map frame **M** —
# the mesh is TSDF-fused directly on cuVSLAM slam poses (0.074 m mean vs GT on
# the long drive; cuSFM PGO degraded to 0.553 m), so there is NO registration
# and NO transform stage anymore. cuVSLAM z-drifts ~1 m (no planar constraint)
# → occupancy uses per-tile local floors (--local-floor-tile 1.0). Quality bar
# hit on v2 (occupancy_slam_lf): boundary p95 0.05 m, precision 94.2 %.
# cuSFM ate 2445 s of the 3227 s bake and only feeds the cuVGL recovery branch
# → cuSFM/cuVGL/REGISTER are OPT-IN via RUN_CUSFM=1 (default OFF).
# GT appears ONLY in the eval-side compare stage.
#
# Usage:
#   bash logic/simulation/mapping/bake_map.sh <dump_dir> <out_map_dir> [--dry-run]
#     <dump_dir>     teleop coverage dump (frames/ + poses.jsonl + rig.json),
#                    e.g. data/coverage_drives/teleop_demo_v2 — must live under data/
#     <out_map_dir>  base output folder, e.g. data/maps/teleop_demo_v2 — the bake
#                    lands in a timestamped subdir (container tool's naming)
#
# Default stages, ≈13 min total (each skippable via SKIP_<STAGE>=1; resumable —
# a stage is skipped when its output already exists and is non-empty; per-stage
# wall timings logged):
#   1 BAG        rosbag_synth.py                      → <dump>_bag  (override: BAG=…, SYNTH_ARGS=…)  ~343s
#   2 EDEX       container edex + compute_poses       → <map>/edex, poses, map_frames                ~296s
#   3 PYCUVSLAM  pycuvslam map rebuild (bench env)    → <map>/pycuvslam_map (+ slam_poses.tum)       ~70s
#   4 SLAMTUM    slam vehicle poses → head_left camera TUM (optical convention, still M)
#                                                     → <map>/slam_fused/slam_head_left.tum         ~5s
#   5 FUSE       TSDF fuse on slam poses, NO --mirror-fix (that was a cusfm-frame artifact)
#                                                     → <map>/slam_fused/dense_mesh_slam.ply        ~31s
#   6 OCCUPANCY  mesh_to_occupancy --traj + --local-floor-tile 1.0 (z-drift-proof)
#                                                     → <map>/$OCC_DIR  (occupancy.npy/.json)       ~39s
#   7 COMPARE    map_compare vs GT grid (EVAL only)   → <map>/$OCC_DIR/compare_report.json
#   8 STARTPOSE  first slam pose (already in M)       → <map>/start_pose.json
#
# Opt-in branch — RUN_CUSFM=1 (only needed for the cuVGL recovery branch):
#   9 CUSFM      container cusfm_cli batch-PGO chain  → <map>/cusfm  (PGO-ONLY consumption:
#                poses from pose_graph/, NEVER output_poses/ TUMs — mapper BA corrupts)
#  10 CUVGL      container cuvgl BoW map              → <map>/cuvgl_map
#  11 REGISTER   GT-free SE(2) fit slam↔cusfm         → <map>/registration_mesh.json — audit
#                ONLY, never gates (this fit is what caught the 17-07 cuSFM degradation)
#
# Hard-won facts encoded here (do not deviate): 30 Hz recording recipe; mesh is
# fused on SLAM poses (born in M — no registration/transform, no --mirror-fix);
# occupancy needs --traj AND --local-floor-tile 1.0; batch-PGO config dir
# cusfm_configs_isaac (incremental OOMs this box); real keyframe spacing for
# cusfm_cli (workflow's hardcoded 0.0 OOMs); PGO-ONLY poses.

set -u

H="$(cd "$(dirname "$0")/../../.." && pwd)"
PY_HUM=~/miniconda3/envs/hum/bin/python
PY_ISAAC=~/miniconda3/envs/isaac/bin/python
PODRUN="podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v $H/data:/data -v $H/logic/simulation/mapping/container:/cfg \
  -e ISAAC_ROS_WS=/data/ws oli-isaac-mapping bash -lc"
export PYTHONPATH="$(dirname "$H")${PYTHONPATH:+:$PYTHONPATH}"  # humanoid.* imports

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }

# ── args ──────────────────────────────────────────────────────────────────────
[ $# -ge 2 ] || die "usage: bake_map.sh <dump_dir> <out_map_dir> [--dry-run]"
DUMP="$(cd "$1" 2>/dev/null && pwd)" || die "no dump dir: $1"
mkdir -p "$2" || die "cannot create out dir: $2"
OUTBASE="$(cd "$2" && pwd)"
DRY=0; [ "${3:-}" = "--dry-run" ] && DRY=1

[ -f "$DUMP/rig.json" ] && [ -f "$DUMP/poses.jsonl" ] || die "not a dump (rig.json/poses.jsonl missing): $DUMP"
case "$DUMP"    in "$H"/data/*) ;; *) die "dump must live under $H/data (container mount): $DUMP";; esac
case "$OUTBASE" in "$H"/data/*) ;; *) die "out dir must live under $H/data (container mount): $OUTBASE";; esac

BAG="${BAG:-${DUMP%/}_bag}"
OCC_DIR="${OCC_DIR:-occupancy_mesh}"
GT_OCC="${GT_OCC:-$H/assets/envs/warehouse_nvidia/nav_maps/v1}"
RUN_CUSFM="${RUN_CUSFM:-0}"
REL_BAG="${BAG#"$H"/}"; REL_OUT="${OUTBASE#"$H"/}"

# the container tool names the bake dir itself (<timestamp>_<bag>); resolve latest
MAPDIR=""
resolve_mapdir() { MAPDIR="$(ls -dt "$OUTBASE"/2*-*/ 2>/dev/null | head -1)"; MAPDIR="${MAPDIR%/}"; }
resolve_mapdir

cd "$H" || die "no repo dir"
say "bake_map: dump=$DUMP"
say "bake_map: out=$OUTBASE  map_dir=${MAPDIR:-<created by stage 2>}  dry_run=$DRY  run_cusfm=$RUN_CUSFM"

# ── stage driver ──────────────────────────────────────────────────────────────
TIMINGS=""
stage() {  # stage <num> <NAME> <done_predicate_fn> <run_fn>
  local num="$1" name="$2" done_fn="$3" run_fn="$4" skip
  eval "skip=\${SKIP_$name:-0}"
  if [ "$skip" = "1" ]; then
    say "── [$num $name] SKIP (SKIP_$name=1)"; return 0
  fi
  if "$done_fn"; then
    say "── [$num $name] SKIP (output exists — resume)"; return 0
  fi
  if [ "$DRY" = "1" ]; then
    say "── [$num $name] WOULD RUN"; return 0
  fi
  say "── [$num $name] RUN"
  local t0; t0=$(date +%s)
  "$run_fn" || die "stage $num $name failed"
  "$done_fn" || die "stage $num $name finished but its output is missing"
  local dt=$(( $(date +%s) - t0 ))
  TIMINGS="$TIMINGS\n  $num $name  ${dt}s"
  say "── [$num $name] done in ${dt}s"
}

# ── 1 BAG ─────────────────────────────────────────────────────────────────────
# bag exists, or edex already baked (bag is regenerable — staged-deletion safe)
done_bag() { { [ -d "$BAG" ] && [ -n "$(ls -A "$BAG" 2>/dev/null)" ]; } \
             || { [ -n "$MAPDIR" ] && [ -n "$(ls -A "$MAPDIR/edex" 2>/dev/null)" ]; }; }
run_bag() { $PY_HUM logic/simulation/mapping/rosbag_synth.py \
              --dump "$DUMP" --out "$BAG" ${SYNTH_ARGS:-}; }
stage 1 BAG done_bag run_bag

# ── 2 EDEX (container: edex + compute_poses, planar, unlimited graph) ─────────
done_edex() { [ -n "$MAPDIR" ] && [ -s "$MAPDIR/poses/keyframe_pose_optimized.tum" ]; }
run_edex() {
  $PODRUN "ros2 run isaac_mapping_ros create_map_offline.py \
    --sensor_data_bag /$REL_BAG \
    --base_output_folder /$REL_OUT \
    --camera_topic_config /cfg/camera_topic_config_oli.yaml \
    --steps_to_run edex compute_poses \
    -o cuvslam.cfg_planar=true -o cuvslam.cfg_slam_max_map_size=0 \
    --print_mode tail" || return 1
  resolve_mapdir
  [ -n "$MAPDIR" ]
}
stage 2 EDEX done_edex run_edex
[ "$DRY" = "1" ] || [ -n "$MAPDIR" ] || die "no map dir under $OUTBASE after stage 2"
M_SET=$([ -n "$MAPDIR" ] && echo 1 || echo 0)
REL_MAP="${MAPDIR#"$H"/}"

# Fused-on-slam artifacts live in <map>/slam_fused/ — adopting the paths of the
# 17-07 manual run so existing bakes resume onto the validated files. NOT
# <map>/mesh_M.ply: that name is already taken in pre-17-07 bakes by the deleted
# TRANSFORM stage's cusfm-derived mesh and would false-SKIP the fuse.
SLAM_TUM="$MAPDIR/pycuvslam_map/slam_poses.tum"
CAM_TUM="$MAPDIR/slam_fused/slam_head_left.tum"
SLAM_MESH="$MAPDIR/slam_fused/dense_mesh_slam.ply"

# ── 3 PYCUVSLAM (bench env map rebuild — container LMDB not pycuvslam-loadable)
done_pycuvslam() { [ "$M_SET" = 1 ] && [ -s "$SLAM_TUM" ]; }
run_pycuvslam() { conda run -n bench-cuvslam python \
    logic/oli/reason/localization/realizations/cuvslam/build_map.py \
    --bake "$MAPDIR" --out "$MAPDIR/pycuvslam_map"; }
stage 3 PYCUVSLAM done_pycuvslam run_pycuvslam

# ── 4 SLAMTUM (slam vehicle poses → head_left camera TUM, optical, still M) ───
done_slamtum() { [ "$M_SET" = 1 ] && [ -s "$CAM_TUM" ]; }
run_slamtum() { $PY_ISAAC logic/simulation/mapping/slam_to_camera_tum.py \
    --slam-tum "$SLAM_TUM" --out "$CAM_TUM"; }
stage 4 SLAMTUM done_slamtum run_slamtum

# ── 5 FUSE (TSDF fuse on slam poses — born in M; NO --mirror-fix here) ────────
done_fuse() { [ "$M_SET" = 1 ] && [ -s "$SLAM_MESH" ]; }
run_fuse() { $PY_ISAAC logic/simulation/mapping/fuse_reconstruction.py \
    --dump "$DUMP" --tum "$CAM_TUM" --out "$SLAM_MESH"; }
stage 5 FUSE done_fuse run_fuse

# ── 6 OCCUPANCY (M-frame grid; --traj clears the robot's blind-spot footprint;
#                 --local-floor-tile 1.0 absorbs cuVSLAM's ~1 m z-drift) ───────
done_occupancy() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/$OCC_DIR/occupancy.npy" ]; }
run_occupancy() { $PY_ISAAC logic/simulation/mapping/mesh_to_occupancy.py \
    --mesh "$SLAM_MESH" --out "$MAPDIR/$OCC_DIR" \
    --traj "$CAM_TUM" --local-floor-tile 1.0; }
stage 6 OCCUPANCY done_occupancy run_occupancy

# ── 7 COMPARE (EVAL-ONLY: GT allowed here, never in the runtime path) ─────────
done_compare() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/$OCC_DIR/compare_report.json" ]; }
run_compare() { $PY_ISAAC logic/simulation/mapping/map_compare.py \
    --cand "$MAPDIR/$OCC_DIR" --gt "$GT_OCC" \
    --tum "$MAPDIR/pycuvslam_map/slam_poses.tum" --dump "$DUMP" \
    --observed "$MAPDIR/$OCC_DIR/observed.npy" \
    --out "$MAPDIR/$OCC_DIR/compare_report.json" \
    --overlay "$MAPDIR/$OCC_DIR/compare_overlay.png"; }
if [ -d "$GT_OCC" ]; then
  stage 7 COMPARE done_compare run_compare
else
  say "── [7 COMPARE] SKIP (no GT grid at $GT_OCC — eval stage only)"
fi

# ── 8 STARTPOSE (first slam pose — already in M, no transform) ────────────────
done_startpose() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/start_pose.json" ]; }
run_startpose() { $PY_ISAAC logic/simulation/mapping/register_slam_to_mesh.py start-pose \
    --slam-tum "$MAPDIR/pycuvslam_map/slam_poses.tum" \
    --out "$MAPDIR/start_pose.json"; }
stage 8 STARTPOSE done_startpose run_startpose

# ══ Opt-in cuSFM branch (RUN_CUSFM=1) — only the cuVGL recovery branch needs it

# ── 9 CUSFM (batch-PGO config; spacing 0.3 m/5° — workflow's 0.0 OOMs) ────────
done_cusfm() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/cusfm/pose_graph/vehicle_pose.tum" ]; }
run_cusfm() {
  # chain may fail past pose_graph (mapper/extract) — PGO output is all we need
  $PODRUN "ros2 run isaac_ros_visual_mapping cusfm_cli \
    --input_dir=/$REL_MAP/map_frames/rectified \
    --cusfm_base_dir=/$REL_MAP/cusfm \
    --model_dir=/opt/ros/jazzy/share/isaac_ros_visual_mapping/models \
    --config_dir=/cfg/cusfm_configs_isaac \
    --binary_dir=/opt/ros/jazzy/lib/isaac_ros_visual_mapping \
    --skip_cuvslam \
    --min_inter_frame_distance=0.3 --min_inter_frame_rotation_degrees=5 \
    --downsampling_matches" \
    || say "WARN: cusfm chain reported failure — continuing iff PGO output exists"
  [ -s "$MAPDIR/cusfm/pose_graph/vehicle_pose.tum" ]
}

# ── 10 CUVGL (BoW over baked keyframes) ───────────────────────────────────────
done_cuvgl() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/cuvgl_map/bow_index.pb" ]; }
run_cuvgl() { $PODRUN "ros2 run isaac_mapping_ros create_map_offline.py \
    --map_dir /$REL_MAP \
    --camera_topic_config /cfg/camera_topic_config_oli.yaml \
    --steps_to_run cuvgl --print_mode tail"; }

# ── 11 REGISTER (GT-free slam↔cusfm fit; audit artifact ONLY, never gates) ────
done_register() { [ "$M_SET" = 1 ] && [ -s "$MAPDIR/registration_mesh.json" ]; }
run_register() { $PY_ISAAC logic/simulation/mapping/register_slam_to_mesh.py fit \
    --slam-tum "$MAPDIR/pycuvslam_map/slam_poses.tum" \
    --cusfm-tum "$MAPDIR/cusfm/pose_graph/vehicle_pose.tum" \
    --out "$MAPDIR/registration_mesh.json"; }

if [ "$RUN_CUSFM" = "1" ]; then
  stage 9  CUSFM    done_cusfm    run_cusfm
  stage 10 CUVGL    done_cuvgl    run_cuvgl
  stage 11 REGISTER done_register run_register
else
  say "── [9-11 CUSFM/CUVGL/REGISTER] SKIP (opt-in — set RUN_CUSFM=1 for the cuVGL recovery branch)"
fi

# ── summary ───────────────────────────────────────────────────────────────────
if [ "$DRY" = "1" ]; then
  say "dry run complete — nothing executed"
else
  say "DONE — map dir: $MAPDIR"
  [ -n "$TIMINGS" ] && say "stage timings:$(echo -e "$TIMINGS")"
  say "runtime artifacts (all born in M): pycuvslam_map/ $OCC_DIR/ start_pose.json"
  [ "$RUN_CUSFM" = "1" ] && say "recovery-branch artifacts: cuvgl_map/ registration_mesh.json (audit)"
  say "audit artifacts: slam_fused/ (camera TUM + fused mesh) $OCC_DIR/compare_report.json"
fi
