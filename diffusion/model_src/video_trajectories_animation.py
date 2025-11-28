# ── CONFIG ───────────────────────────────────────────────────
N_DEMOS        = 200        # сколько траекторий показываем
STROKE_WIDTH   = 4         # толщина линий
TOTAL_RUNTIME  = 5         # секунд на все траектории
SCALE_K        = 3.5       # масштаб (увеличьте, если линии слишком мелкие)
# ------------------------------------------------------------------

from pathlib import Path
from manim import *
import h5py, numpy as np

config.media_dir  = str(Path(__file__).resolve().parent)   # сохранять рядом со скриптом
config.video_dir  = config.media_dir
config.preview    = False                                 # не открывать ролик после рендера
config.disable_caching_warning = True
config.background_color = WHITE

# ── data prep ─────────────────────────────────────────────────
def load_and_preprocess():
    h5 = h5py.File("../robomimic/datasets/tool_hang/ph/low_dim_v15.hdf5", "r")["data"]
    trajs = [d["obs"]["robot0_eef_pos"][:] for d in list(h5.values())[:N_DEMOS]]
    pts   = np.concatenate(trajs, axis=0)
    ctr   = pts.mean(0)
    scale = SCALE_K / np.abs(pts - ctr).max()   # подгоняем в куб
    return [(t - ctr) * scale for t in trajs]

# ── axes factory ──────────────────────────────────────────────
def make_axes():
    return ThreeDAxes(
        x_range=[-8, 8, 2], y_range=[-8, 8, 2], z_range=[-8, 8, 2],
        x_length=16, y_length=16, z_length=16,
        axis_config={"color": BLACK, "tick_size": 0.05},
    )

# ── drawing helper ────────────────────────────────────────────
def add_trajectories(scene, trajs, runtime_per):
    for traj in trajs:
        segments = VGroup(*[
            Line(
                traj[i], traj[i + 1],
                color=interpolate_color(PURPLE, YELLOW, i / (len(traj) - 1)),
                stroke_width=STROKE_WIDTH
            )
            for i in range(len(traj) - 1)
        ])
        scene.play(Create(segments), run_time=runtime_per)

# ── perspective view ─────────────────────────────────────────
class PerspectiveScene(ThreeDScene):
    def construct(self):
        trajs = load_and_preprocess()
        self.add(make_axes())
        self.set_camera_orientation(phi=75 * DEGREES, theta=-315 * DEGREES)
        add_trajectories(self, trajs, TOTAL_RUNTIME / N_DEMOS)
        self.wait()

# ── top-down view ────────────────────────────────────────────
class TopDownScene(ThreeDScene):
    def construct(self):
        trajs = load_and_preprocess()
        self.add(make_axes())
        self.set_camera_orientation(phi=90 * DEGREES, theta=0 * DEGREES)
        add_trajectories(self, trajs, TOTAL_RUNTIME / N_DEMOS)
        self.wait()
