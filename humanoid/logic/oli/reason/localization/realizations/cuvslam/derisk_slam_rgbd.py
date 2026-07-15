"""De-risk: does cuVSLAM Slam (save_map / localize_in_map) work with a SINGLE mono-depth RGBD rig?

Synthetic, no Isaac: vendor test's multi-scale-noise plane (data_gen.ImageGenerator) gives RGB;
the plane is fronto-parallel at known distance z, so depth = constant uint16 z*1000 (mm).
Run inside bench-cuvslam:
    conda run -n bench-cuvslam python derisk_slam_rgbd.py [--map-cell-size 0.0] [--planar]

Phases:
  1. build map: 1-camera RGBD rig, Tracker(rig, odom_cfg RGBD, SlamConfig(sync_mode=True)),
     30 frames forward motion, assert slam_pose valid, save_map -> assert saved
  2. fresh Tracker (same process; process isolation is a later concern), track 1 frame,
     localize_in_map(guess close to true pose) -> print result pose + error message
Exit 0 iff localization returned a pose within 0.5 m of truth.
"""

import argparse
import sys
import threading
import numpy as np

sys.path.insert(0, "vendor/pycuvslam/python/test")  # data_gen
import cuvslam as vslam
import data_gen

W, H = 640, 480
BASELINE = 0.25   # only used by the generator's zoom math, NOT in the rig
STEPS = 30
DEPTH_UNITS_PER_M = 1000.0


def make_rig():
    cams = data_gen.generate_stereo_camera(W, H, BASELINE)
    return vslam.Rig([cams[0]]), cams  # rig has ONLY camera 0


def odom_config():
    return vslam.Tracker.OdometryConfig(
        odometry_mode=vslam.Tracker.OdometryMode.RGBD,
        async_sba=False,
        rgbd_settings=vslam.Tracker.OdometryRGBDSettings(
            depth_scale_factor=DEPTH_UNITS_PER_M,
            depth_camera_id=0,
            enable_depth_stereo_tracking=False,
        ),
    )


def slam_config(args):
    return vslam.Tracker.SlamConfig(
        sync_mode=True,
        planar_constraints=bool(args.planar),
        map_cell_size=float(args.map_cell_size),
        max_map_size=0,  # unlimited pose graph — offline map build
    )


def depth_image(z_m: float) -> np.ndarray:
    return np.full((H, W), int(round(z_m * DEPTH_UNITS_PER_M)), dtype=np.uint16)


def frame(img_gen, step):
    images, travelled = img_gen.generate_zoomed_images(step)
    z = img_gen.get_start_distance() - travelled  # current distance to the plane
    return images[0], depth_image(z), travelled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-cell-size", default=0.0, type=float)
    ap.add_argument("--planar", action="store_true")
    ap.add_argument("--map-dir", default="/tmp/cuvslam_derisk_map")
    args = ap.parse_args()

    rig, gen_cams = make_rig()
    img_gen = data_gen.ImageGenerator(gen_cams, STEPS)

    # ---- phase 1: map build -------------------------------------------------
    tracker = vslam.Tracker(rig, odom_config(), slam_config(args))
    print("primary cameras:", tracker.odom.get_primary_cameras())

    last_travel = None
    for i in range(STEPS):
        rgb, depth, travelled = frame(img_gen, i)
        pose_est, slam_pose = tracker.track(i * 100_000_000, [rgb], depths=[depth])
        assert pose_est.world_from_rig is not None, f"VO lost at {i}"
        if i % 10 == 0 or i == STEPS - 1:
            print(f"  {i}: gt_travel={travelled:.3f} "
                  f"odom_z={pose_est.world_from_rig.pose.translation[2]:.3f} "
                  f"slam_z={slam_pose.translation[2]:.3f}")
        last_travel = travelled

    saved = {}
    tracker.save_map(args.map_dir, lambda ok: saved.setdefault("ok", ok))
    assert saved.get("ok"), "save_map failed"
    print("map saved:", args.map_dir)
    m = tracker.get_slam_metrics()
    print("metrics after build:", m)
    del tracker

    # ---- phase 2: fresh tracker, localize_in_map ----------------------------
    tracker2 = vslam.Tracker(rig, odom_config(), slam_config(args))
    step = 12  # somewhere mid-map
    rgb, depth, travelled = frame(img_gen, step)
    ts = 1_000_000_000_000  # fresh epoch
    tracker2.track(ts, [rgb], depths=[depth])

    true_pose = vslam.Pose(translation=[0, 0, travelled], rotation=[0, 0, 0, 1])
    guess = vslam.Pose(translation=[0.1, 0.05, travelled - 0.1], rotation=[0, 0, 0, 1])

    loc = vslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=0.5, vertical_search_radius=0.25,
        horizontal_step=0.0625, vertical_step=0.0625, angular_step_rads=0.03125)

    done = threading.Event()
    result = {}

    def finish(pose, err):
        result["pose"], result["err"] = pose, err
        done.set()

    tracker2.localize_in_map(args.map_dir, ts, guess, [rgb], loc,
                             lambda: print("localization started"), finish)
    # sync_mode=True -> callback should already have fired
    ok = done.wait(timeout=30)
    print("finish_cb fired:", ok, "| error:", result.get("err"))
    pose = result.get("pose")
    if pose is None:
        print("LOCALIZE FAILED")
        return 1
    err_vec = np.array(pose.translation) - np.array(true_pose.translation)
    print(f"localized pose: {pose}")
    print(f"true pose z={travelled:.3f}; err vector={err_vec}, |err|={np.linalg.norm(err_vec):.3f} m")

    # ---- phase 3: does the SLAM pose keep tracking in the loaded map? -------
    for k in range(step + 1, step + 6):
        rgb, depth, travelled = frame(img_gen, k)
        ts += 100_000_000
        pose_est, slam_pose = tracker2.track(ts, [rgb], depths=[depth])
        print(f"  post-localize {k}: gt_z={travelled:.3f} slam_z={slam_pose.translation[2]:.3f}")

    return 0 if np.linalg.norm(err_vec) < 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
