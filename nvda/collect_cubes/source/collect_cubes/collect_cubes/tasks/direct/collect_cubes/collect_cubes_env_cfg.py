# comments in English only
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg


from isaaclab import sim as sim_utils


@configclass
class CollectCubesEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    action_space = 9       # number of controllable joints (for Franka)
    observation_space = 24  # e.g. joint positions + velocities
    state_space = 0        # optional global state, keep 0 if unused

    action_scale = 7.5

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)

    # # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )

    # # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # ) 








    # robot
    # ==================================================================
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.fix_base = True
    # robot_cfg.ee_frame_name = "panda_hand_tcp"

    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


    # ===  SO101  ===============================================================


    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.fix_base = True
    # robot_cfg.ee_frame_name = "panda_hand_tcp"

    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # Bucket components (5 cuboids: floor + 4 walls)
    bucket_floor_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BucketFloor",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.5, 0.005),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    bucket_wall_left_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BucketWallLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.25, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5 - 0.125, 0.5, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    bucket_wall_right_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BucketWallRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.25, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5 + 0.125, 0.5, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    bucket_wall_front_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BucketWallFront",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.01, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.5 - 0.125, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    bucket_wall_back_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BucketWallBack",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.01, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.5, restitution=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.5 + 0.125, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # scene — ONLY the supported args
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.5,
        replicate_physics=True,
    )
