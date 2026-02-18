from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg

# joints: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

so101_cfg = ArticulationCfg(
    prim_path = "/World/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path = "/home/may33/projects/ml_portfolio/robotics/robots/SO-ARM100/convex_fripper_robot.usd",
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = False,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions = True,
            fix_root_link = True,
        )
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.0),
        joint_pos = {
            "shoulder_pan" : 0.0,
            "shoulder_lift" : 0.0,
            "elbow_flex" : 0.0,
            "wrist_flex" : 0.0,
            "wrist_roll" : 0.0,
            "gripper" : 0.0,
        }
    ),
    actuators = {
        "arm" : ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=10.0,
        )
    }
)