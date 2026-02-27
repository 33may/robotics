"""
IsaacLab configuration generation utilities.

Takes a composed scene USDA and generates leisaac task boilerplate:
scene assets, env configs (SceneCfg, ObsCfg, EnvCfg), and gym registration.

Usage:
    python vbti/utils/isaaclab_cfg_utils.py pipeline scene.usda vbti_mesh_table
    python vbti/utils/isaaclab_cfg_utils.py extract scene.usda
    python vbti/utils/isaaclab_cfg_utils.py gen_env_cfg scene.usda vbti_mesh_table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pxr import Usd, UsdPhysics, UsdLux


LEISAAC_ROOT = "/home/may33/projects/ml_portfolio/robotics/leisaac"


def create_no_robot_scene(scene_path: str, robot_prim_path: str = "/World/so101_simready_follower_leisaac",
                          save_path: str = None):
    """Strip robot, cameras, and GUI render config from a scene USD.

    Produces a clean scene that LeIsaac can load — it spawns
    the robot and cameras separately from the env config.

    Args:
        scene_path: Path to the input scene USD.
        robot_prim_path: Prim path of the robot to remove.
        save_path: Output path. Defaults to <scene>_no_robot.usda.
    """
    from pathlib import Path

    stage = Usd.Stage.Open(scene_path)

    # 1. Remove the robot prim
    stage.RemovePrim(robot_prim_path)
    print(f"  [OK] Removed robot: {robot_prim_path}")

    # 2. Remove world-level cameras (spawned from env config instead)
    for prim in list(stage.GetPrimAtPath("/World").GetChildren()):
        if prim.GetTypeName() == "Camera":
            stage.RemovePrim(prim.GetPath())
            print(f"  [OK] Removed camera: {prim.GetPath()}")

    # 3. Remove /Render (GUI viewport render products)
    if stage.GetPrimAtPath("/Render"):
        stage.RemovePrim("/Render")
        print(f"  [OK] Removed /Render")

    # 4. Remove lights (will be spawned by IsaacLab config instead)
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
            stage.RemovePrim(prim.GetPath())
            print(f"  [OK] Removed light: {prim.GetPath()}")

    # 5. Clean GUI camera metadata from customLayerData
    layer = stage.GetRootLayer()
    metadata = dict(layer.customLayerData)
    if "cameraSettings" in metadata:
        del metadata["cameraSettings"]
        layer.customLayerData = metadata
        print(f"  [OK] Removed cameraSettings metadata")

    # 6. Export
    if save_path is None:
        p = Path(scene_path)
        save_path = str(p.with_name(p.stem + "_no_robot" + p.suffix))

    layer.Export(save_path)
    print(f"\n  Saved to {save_path}")


def extract_scene_config(scene_path: str, robot_prim_path=None,
                          robot_usd_path: str = None,
                          save_path: str = None) -> dict:
    """Extract robot, camera, light, and joint config from a composed scene USDA.

    When robot_usd_path is provided, also extracts drive parameters
    (stiffness, damping, limits) from the robot USD via robot_utils.

    Args:
        scene_path: Path to the scene USD file.
        robot_prim_path: Prim path of the robot. If None, auto-detects via ArticulationRootAPI.
        robot_usd_path: Path to the robot USD file for drive/limit extraction.
        save_path: If set, writes the config dict to this JSON file.
    """
    import json
    from pathlib import Path

    stage = Usd.Stage.Open(scene_path)

    # Auto-detect robot if not specified
    if robot_prim_path is None:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                robot_prim_path = str(prim.GetParent().GetPath())
                break
        if robot_prim_path is None:
            print("  [ERR] No robot found (no ArticulationRootAPI)")
            return {}
    print(f"  Robot: {robot_prim_path}")

    robot = stage.GetPrimAtPath(robot_prim_path)

    # get robot position and orientation
    robot_pos = list(robot.GetAttribute("xformOp:translate").Get())
    robot_orient_q = robot.GetAttribute("xformOp:orient").Get()
    robot_orient = [robot_orient_q.GetReal(), *robot_orient_q.GetImaginary()]

    # get joint target positions
    joint_prim = robot.GetPrimAtPath("joints")
    joint_targets = {}
    for joint in joint_prim.GetChildren():
        target = joint.GetAttribute("drive:angular:physics:targetPosition").Get()
        if target is not None:
            joint_targets[joint.GetName()] = float(target)

    # get gripper camera override (inside robot prim)
    gripper_cam = {}
    gripper_prim = robot.GetPrimAtPath("gripper/gripper_cam")
    if gripper_prim:
        orient_attr = gripper_prim.GetAttribute("xformOp:orient")
        if orient_attr and orient_attr.Get():
            q = orient_attr.Get()
            gripper_cam["orientation"] = [q.GetReal(), *q.GetImaginary()]

    # get world-level cameras
    cameras = {}
    world = stage.GetPrimAtPath("/World")
    for prim in world.GetChildren():
        if prim.GetTypeName() == "Camera":
            cam = {}
            pos = prim.GetAttribute("xformOp:translate").Get()
            if pos:
                cam["position"] = list(pos)
            orient = prim.GetAttribute("xformOp:orient").Get()
            if orient:
                cam["orientation"] = [orient.GetReal(), *orient.GetImaginary()]
            focal = prim.GetAttribute("focalLength").Get()
            if focal:
                cam["focal_length"] = float(focal)
            aperture = prim.GetAttribute("horizontalAperture").Get()
            if aperture:
                cam["horizontal_aperture"] = float(aperture)
            clip = prim.GetAttribute("clippingRange").Get()
            if clip:
                cam["clipping_range"] = list(clip)
            focus = prim.GetAttribute("focusDistance").Get()
            if focus:
                cam["focus_distance"] = float(focus)
            cameras[prim.GetName()] = cam

    # get lights (traverse entire stage — lights may be anywhere)
    lights = {}
    scene_dir = str(Path(scene_path).parent)
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(prim)
            tex_attr = dome.GetTextureFileAttr().Get()
            # resolve relative asset path to absolute
            tex_path = None
            if tex_attr and str(tex_attr) not in ("", "@@@"):
                resolved = tex_attr.resolvedPath
                if resolved:
                    tex_path = str(resolved)
                else:
                    # manual resolve relative to scene file
                    tex_path = str(Path(scene_dir) / str(tex_attr.path))
            light_info = {
                "type": "DomeLight",
                "intensity": float(dome.GetIntensityAttr().Get() or 1.0),
                "texture_file": tex_path,
                "texture_format": str(dome.GetTextureFormatAttr().Get() or "automatic"),
            }
            # get orientation if present
            orient_attr = prim.GetAttribute("xformOp:orient")
            if orient_attr and orient_attr.Get():
                q = orient_attr.Get()
                light_info["orientation"] = [q.GetReal(), *q.GetImaginary()]
            lights[prim.GetName()] = light_info

        elif prim.IsA(UsdLux.DistantLight):
            dist = UsdLux.DistantLight(prim)
            light_info = {
                "type": "DistantLight",
                "intensity": float(dist.GetIntensityAttr().Get() or 1.0),
                "angle": float(dist.GetAngleAttr().Get() or 0.53),
            }
            # get orientation if present
            orient_attr = prim.GetAttribute("xformOp:orient")
            if orient_attr and orient_attr.Get():
                q = orient_attr.Get()
                light_info["orientation"] = [q.GetReal(), *q.GetImaginary()]
            lights[prim.GetName()] = light_info

    config = {
        "scene_path": str(Path(scene_path).resolve()),
        "robot": {
            "prim_path": robot_prim_path,
            "position": robot_pos,
            "orientation": robot_orient,
            "joint_targets": joint_targets,
        },
        "cameras": cameras,
        "gripper_cam": gripper_cam,
        "lights": lights,
    }

    # Merge robot drive/limit config from robot USD
    if robot_usd_path:
        from utils.robot_utils import extract_robot_config
        robot_cfg = extract_robot_config(str(Path(robot_usd_path).resolve()))
        config["robot"]["usd_path"] = str(Path(robot_usd_path).resolve())
        config["robot"]["joints"] = robot_cfg["joints"]

    # Persist to disk
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  [OK] Saved scene config → {save_path}")

    print(json.dumps(config, indent=2))
    return config


def generate_scene_asset(task_name: str, scene_usd_path: str,
                         leisaac_root: str = LEISAAC_ROOT):
    """Generate a leisaac scene asset config that points to a clean scene USD.

    Args:
        task_name: Task name in snake_case (e.g., "vbti_table").
        scene_usd_path: Path to the no-robot scene USD.
        leisaac_root: Root of the leisaac package.
    """
    from pathlib import Path
    from textwrap import dedent

    usd_abs = str(Path(scene_usd_path).resolve())
    name_upper = task_name.upper()  # vbti_table → VBTI_TABLE

    scenes_dir = Path(leisaac_root) / "source" / "leisaac" / "leisaac" / "assets" / "scenes"

    # 1. Write scene asset file
    asset_file = scenes_dir / f"{task_name}.py"
    content = dedent(f"""\
        from pathlib import Path

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg

        {name_upper}_USD_PATH = "{usd_abs}"

        {name_upper}_CFG = AssetBaseCfg(
            spawn=sim_utils.UsdFileCfg(usd_path={name_upper}_USD_PATH)
        )
    """)
    asset_file.write_text(content)
    print(f"  [OK] Wrote {asset_file}")

    # 2. Append import to __init__.py (if not already there)
    init_file = scenes_dir / "__init__.py"
    import_line = f"from .{task_name} import {name_upper}_CFG, {name_upper}_USD_PATH\n"
    existing = init_file.read_text()
    if import_line.strip() not in existing:
        with open(init_file, "a") as f:
            f.write(import_line)
        print(f"  [OK] Updated {init_file}")
    else:
        print(f"  [SKIP] Import already in {init_file}")


def create_task_boilerplate(task_name: str, tasks_root: str = LEISAAC_ROOT + "/source/leisaac/leisaac/tasks",
                            gym_id: str = None, env_cfg_class: str = None):
    """Generate a minimal leisaac task folder with gym registration.

    Creates the folder structure:
        tasks/{task_name}/__init__.py          — gym.register()
        tasks/{task_name}/mdp/__init__.py      — MDP base re-exports
        tasks/{task_name}/{task_name}_env_cfg.py — placeholder

    Auto-discovered by leisaac's import_packages() — no parent __init__.py change needed.

    Args:
        task_name: Snake_case task name, e.g. 'vbti_table'.
        tasks_root: Path to leisaac/tasks/ directory.
        gym_id: Gym env ID. Default: LeIsaac-SO101-{PascalName}-v0.
        env_cfg_class: Config class name. Default: {PascalName}EnvCfg.
    """
    from pathlib import Path

    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    gym_id = gym_id or f"LeIsaac-SO101-{pascal}-v0"
    env_cfg_class = env_cfg_class or f"{pascal}EnvCfg"

    root = Path(tasks_root).resolve()
    task_dir = root / task_name
    mdp_dir = task_dir / "mdp"

    if not root.is_dir():
        print(f"  [ERR] Tasks root not found: {root}")
        return
    if task_dir.exists():
        print(f"  [SKIP] Task directory already exists: {task_dir}")
        return

    mdp_dir.mkdir(parents=True)
    print(f"  [OK] Created {task_dir}/")
    print(f"  [OK] Created {task_dir}/mdp/")

    # tasks/{task_name}/__init__.py
    init_content = (
        "import gymnasium as gym\n\n"
        "gym.register(\n"
        f'    id="{gym_id}",\n'
        '    entry_point="isaaclab.envs:ManagerBasedRLEnv",\n'
        "    disable_env_checker=True,\n"
        "    kwargs={\n"
        f'        "env_cfg_entry_point": f"{{__name__}}.{task_name}_env_cfg:{env_cfg_class}",\n'
        "    },\n"
        ")\n"
    )
    (pkg_dir / "__init__.py").write_text(init_content)
    print(f"  [OK] __init__.py  →  gym.register(id='{gym_id}')")

    # tasks/{task_name}/mdp/__init__.py
    (mdp_dir / "__init__.py").write_text(
        "from isaaclab.envs.mdp import *\nfrom leisaac.enhance.envs.mdp import *\n"
    )
    print(f"  [OK] mdp/__init__.py")

    # tasks/{task_name}/{task_name}_env_cfg.py (placeholder)
    (task_dir / f"{task_name}_env_cfg.py").write_text(
        f'"""Environment configuration for {task_name} task."""\n\n'
        f"# TODO: Define {env_cfg_class} here.\n"
        f"# See lift_cube/lift_cube_env_cfg.py for the full pattern.\n"
    )
    print(f"  [OK] {task_name}_env_cfg.py  (placeholder)")

    print(f"\n  Task '{task_name}' ready — next: implement {task_name}_env_cfg.py")


def _cam_field(cam_name: str, cam_cfg: dict, width: int = 640, height: int = 480,
               cosmos_sensors: bool = False) -> str:
    """Generate a TiledCameraCfg field with full spawn config and offset.

    Camera is spawned from code (not from USD), enabling depth/seg capture
    and camera position randomization for domain randomization.
    """
    pos = tuple(cam_cfg.get("position", (0, 0, 0)))
    rot = tuple(cam_cfg.get("orientation", (1, 0, 0, 0)))
    focal = cam_cfg.get("focal_length", 24.0)
    aperture = cam_cfg.get("horizontal_aperture", 36.0)
    clip = tuple(cam_cfg.get("clipping_range", [0.01, 50.0]))
    focus = cam_cfg.get("focus_distance", 400.0)

    if cosmos_sensors:
        data_types = '["rgb", "distance_to_camera", "instance_segmentation_fast"]'
    else:
        data_types = '["rgb"]'

    return (
        f'    {cam_name}: TiledCameraCfg = TiledCameraCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Scene/{cam_name}",\n'
        f'        offset=TiledCameraCfg.OffsetCfg(\n'
        f'            pos={pos}, rot={rot}, convention="opengl"\n'
        f'        ),\n'
        f'        spawn=sim_utils.PinholeCameraCfg(\n'
        f'            focal_length={focal},\n'
        f'            focus_distance={focus},\n'
        f'            horizontal_aperture={aperture},\n'
        f'            clipping_range={clip},\n'
        f'            lock_camera=True,\n'
        f'        ),\n'
        f'        data_types={data_types},\n'
        f'        width={width},\n'
        f'        height={height},\n'
        f'        update_period=1 / 30.0,\n'
        f'    )\n'
    )


def _light_field(name: str, light_cfg: dict) -> str:
    """Generate an AssetBaseCfg field for a light parsed from the scene USDA."""
    if light_cfg["type"] == "DomeLight":
        spawn_args = [f'intensity={light_cfg["intensity"]}']
        if light_cfg.get("texture_file"):
            spawn_args.append(f'texture_file="{light_cfg["texture_file"]}"')
        if light_cfg.get("texture_format", "automatic") != "automatic":
            spawn_args.append(f'texture_format="{light_cfg["texture_format"]}"')
        spawn = f'sim_utils.DomeLightCfg({", ".join(spawn_args)})'
    elif light_cfg["type"] == "DistantLight":
        spawn_args = [f'intensity={light_cfg["intensity"]}']
        if light_cfg.get("angle"):
            spawn_args.append(f'angle={light_cfg["angle"]}')
        spawn = f'sim_utils.DistantLightCfg({", ".join(spawn_args)})'
    else:
        return ""

    # snake_case field name
    field_name = name[0].lower() + name[1:]
    lines = (
        f'    {field_name} = AssetBaseCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/{field_name}",\n'
        f'        spawn={spawn},\n'
    )
    # add init_state orientation if present
    if light_cfg.get("orientation"):
        o = tuple(light_cfg["orientation"])
        lines += f'        init_state=AssetBaseCfg.InitialStateCfg(rot={o}),\n'
    lines += f'    )\n'
    return lines


def _discover_subassets(scene_usda_path: str) -> list:
    """Find RigidBody/Articulation prims that parse_usd_and_create_subassets will discover.

    Returns paths relative to the default prim (root), since USD maps
    the root prim to the spawn prim_path when loaded via UsdFileCfg.
    """
    stage = Usd.Stage.Open(scene_usda_path)
    root = stage.GetDefaultPrim()
    root_path = root.GetPath().pathString if root else ""
    assets = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            name = prim.GetPath().pathString.split("/")[-1]
            kind = "Articulation" if prim.HasAPI(UsdPhysics.ArticulationRootAPI) else "RigidBody"
            # Path relative to root prim (e.g., /World/duck → /duck)
            full_path = str(prim.GetPath())
            rel_path = full_path[len(root_path):] if root_path and full_path.startswith(root_path) else full_path
            assets.append({"name": name, "path": rel_path, "kind": kind})
    return assets


def _obs_cam_field(cam_name: str, cosmos_sensors: bool = False) -> str:
    """Generate ObsTerms for a camera — name must match SceneCfg field."""
    lines = (
        f'        {cam_name} = ObsTerm(\n'
        f'            func=mdp.image,\n'
        f'            params={{\n'
        f'                "sensor_cfg": SceneEntityCfg("{cam_name}"),\n'
        f'                "data_type": "rgb",\n'
        f'                "normalize": False\n'
        f'            }}\n'
        f'        )\n'
    )
    if cosmos_sensors:
        lines += (
            f'        {cam_name}_depth = ObsTerm(\n'
            f'            func=mdp.image,\n'
            f'            params={{\n'
            f'                "sensor_cfg": SceneEntityCfg("{cam_name}"),\n'
            f'                "data_type": "distance_to_camera",\n'
            f'                "normalize": False\n'
            f'            }}\n'
            f'        )\n'
            f'        {cam_name}_seg = ObsTerm(\n'
            f'            func=mdp.image,\n'
            f'            params={{\n'
            f'                "sensor_cfg": SceneEntityCfg("{cam_name}"),\n'
            f'                "data_type": "instance_segmentation_fast",\n'
            f'                "normalize": False\n'
            f'            }}\n'
            f'        )\n'
        )
    return lines


def generate_leisaac_env(scene_usda_path: str, task_name: str,
                     no_robot_usda_path: str = None,
                     robot_usd_path: str = None,
                     cosmos_sensors: bool = False,
                     tasks_root: str = LEISAAC_ROOT + "/source/leisaac/leisaac/tasks"):
    """Generate a leisaac env config file from a scene USDA.

    Generates SceneCfg (scene + cameras), ObservationsCfg (joints + cameras),
    TerminationsCfg (pass with TODO), and EnvCfg (robot position + subasset
    parsing). Domain randomization is left as TODO for manual work.

    Args:
        scene_usda_path: Path to the original scene USDA (with robot, for extracting config).
        task_name: Snake_case task name, e.g. 'vbti_table'.
        no_robot_usda_path: Path to the no-robot scene USDA (for subasset discovery).
                            Defaults to {scene}_no_robot.usda.
        cosmos_sensors: If True, cameras also capture depth + segmentation for Cosmos Transfer.
        tasks_root: Path to leisaac/tasks/ directory.
    """
    from pathlib import Path

    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    task_upper = task_name.upper()

    # Default no-robot path
    if no_robot_usda_path is None:
        p = Path(scene_usda_path)
        no_robot_usda_path = str(p.with_name(p.stem + "_no_robot" + p.suffix))

    # --- Extract scene config from ORIGINAL scene (has robot + cameras) ---
    cfg = extract_scene_config(scene_usda_path)
    if not cfg:
        print("  [ERR] Could not extract scene config")
        return

    cam_names = list(cfg.get("cameras", {}).keys())
    robot_pos = tuple(cfg["robot"]["position"])
    robot_orient = tuple(cfg["robot"]["orientation"])
    all_cam_names = cam_names + ["wrist"]  # wrist kept from template
    lights = cfg.get("lights", {})

    # --- Discover subassets from NO-ROBOT scene (what parse_usd will find) ---
    subassets = _discover_subassets(no_robot_usda_path)

    print(f"  Scene cameras: {cam_names}")
    print(f"  Robot position: {robot_pos}")
    print(f"  Lights: {list(lights.keys())}")
    print(f"  Subassets: {[a['name'] for a in subassets]}")

    # --- Imports ---
    imports = (
        f'import math\n'
        f'\n'
        f'import isaaclab.sim as sim_utils\n'
        f'from isaaclab.assets import AssetBaseCfg\n'
        f'from isaaclab.managers import ObservationGroupCfg as ObsGroup\n'
        f'from isaaclab.managers import ObservationTermCfg as ObsTerm\n'
        f'from isaaclab.managers import SceneEntityCfg\n'
        f'from isaaclab.managers import TerminationTermCfg as DoneTerm\n'
        f'from isaaclab.sensors import TiledCameraCfg\n'
        f'from isaaclab.utils import configclass\n'
        f'from leisaac.assets.scenes.{task_name} import {task_upper}_CFG, {task_upper}_USD_PATH\n'
        f'from leisaac.utils.domain_randomization import (\n'
        f'    domain_randomization,\n'
        f'    randomize_light_rotation_uniform,\n'
        f'    randomize_object_uniform,\n'
        f')\n'
        f'from leisaac.utils.env_utils import delete_attribute\n'
        f'from leisaac.utils.general_assets import parse_usd_and_create_subassets\n'
        f'\n'
        f'from ..template import (\n'
        f'    SingleArmObservationsCfg,\n'
        f'    SingleArmTaskEnvCfg,\n'
        f'    SingleArmTaskSceneCfg,\n'
        f'    SingleArmTerminationsCfg,\n'
        f')\n'
        f'from . import mdp\n'
    )

    # --- SceneCfg ---
    cameras_cfg = cfg.get("cameras", {})
    scene_cameras = "\n".join(_cam_field(name, cameras_cfg[name], cosmos_sensors=cosmos_sensors) for name in cam_names)
    scene_lights = "\n".join(_light_field(name, lcfg) for name, lcfg in lights.items())
    scene_cfg = (
        f'\n\n@configclass\n'
        f'class {pascal}SceneCfg(SingleArmTaskSceneCfg):\n'
        f'    """Scene configuration for the {task_name} task."""\n'
        f'\n'
        f'    scene: AssetBaseCfg = {task_upper}_CFG.replace(prim_path="{{ENV_REGEX_NS}}/Scene")\n'
        f'\n'
        f'{scene_cameras}\n'
        f'{scene_lights}\n'
        f'    def __post_init__(self):\n'
        f'        super().__post_init__()\n'
        f'        delete_attribute(self, "front")\n'
        f'        delete_attribute(self, "light")  # remove template default light\n'
    )
    if cosmos_sensors:
        scene_cfg += (
            f'        # Patch wrist camera (inherited from template) with cosmos sensors\n'
            f'        self.wrist.data_types = ["rgb", "distance_to_camera", "instance_segmentation_fast"]\n'
        )

    # --- ObservationsCfg ---
    # Joint states (always needed) + camera images for each camera.
    # Camera ObsTerm names must match SceneCfg field names.
    obs_cam_terms = "\n".join(_obs_cam_field(name, cosmos_sensors=cosmos_sensors) for name in all_cam_names)
    obs_cfg = (
        f'\n\n@configclass\n'
        f'class ObservationsCfg(SingleArmObservationsCfg):\n'
        f'\n'
        f'    @configclass\n'
        f'    class PolicyCfg(ObsGroup):\n'
        f'        """Observations for policy group."""\n'
        f'\n'
        f'        joint_pos = ObsTerm(func=mdp.joint_pos)\n'
        f'        joint_vel = ObsTerm(func=mdp.joint_vel)\n'
        f'        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)\n'
        f'        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)\n'
        f'        actions = ObsTerm(func=mdp.last_action)\n'
        f'\n'
        f'{obs_cam_terms}\n'
        f'        def __post_init__(self):\n'
        f'            self.enable_corruption = True\n'
        f'            self.concatenate_terms = False\n'
        f'\n'
        f'    policy: PolicyCfg = PolicyCfg()\n'
    )

    # --- TerminationsCfg ---
    term_cfg = (
        f'\n\n@configclass\n'
        f'class TerminationsCfg(SingleArmTerminationsCfg):\n'
        f'    # Inherits time_out from template.\n'
        f'    # TODO: add custom termination conditions if needed\n'
        f'    pass\n'
    )

    # --- EnvCfg ---
    # Subasset names as comments for reference when writing domain randomization
    subasset_lines = "\n".join(
        f'    #   "{a["name"]}" → {a["path"]}  [{a["kind"]}]'
        for a in subassets
    )
    env_cfg = (
        f'\n\n@configclass\n'
        f'class {pascal}EnvCfg(SingleArmTaskEnvCfg):\n'
        f'    """Configuration for the {task_name} environment."""\n'
        f'\n'
        f'    scene: {pascal}SceneCfg = {pascal}SceneCfg(env_spacing=8.0)\n'
        f'\n'
        f'    observations: ObservationsCfg = ObservationsCfg()\n'
        f'\n'
        f'    terminations: TerminationsCfg = TerminationsCfg()\n'
        f'\n'
        f'    def __post_init__(self) -> None:\n'
        f'        super().__post_init__()\n'
        f'\n'
        f'        self.scene.robot.init_state.pos = {robot_pos}\n'
        f'        self.scene.robot.init_state.rot = {robot_orient}\n'
        + (f'        self.scene.robot.spawn.usd_path = "{robot_usd_path}"\n' if robot_usd_path else '')
        + f'\n'
        f'        # Auto-discover objects in scene USD (creates SceneCfg attrs at runtime)\n'
        f'        # Subassets found:\n'
        f'{subasset_lines}\n'
        f'        parse_usd_and_create_subassets({task_upper}_USD_PATH, self)\n'
        f'\n'
        f'        # TODO: domain_randomization(self, random_options=[...])\n'
    )

    # --- Write ---
    full_content = imports + scene_cfg + obs_cfg + term_cfg + env_cfg

    out_path = Path(tasks_root) / task_name / f"{task_name}_env_cfg.py"
    out_path.write_text(full_content)
    print(f"\n  [OK] Wrote {out_path}")
    print(f"       {pascal}SceneCfg: scene + {len(cam_names)} USD cameras + wrist (template) + {len(lights)} lights")
    print(f"       ObservationsCfg: {len(all_cam_names)} cameras + joint states")
    print(f"       TerminationsCfg: time_out (inherited)")
    print(f"       {pascal}EnvCfg: robot pos + {len(subassets)} subassets")



def _robot_articulation_field(robot_cfg: dict) -> str:
    """Generate an inline ArticulationCfg from extracted robot config."""
    usd_path = robot_cfg.get("usd_path", "MISSING")
    joints = robot_cfg.get("joints", {})

    # Split joints into arm vs gripper by name
    gripper_joints = [j for j in joints if "gripper" in j.lower()]
    arm_joints = [j for j in joints if j not in gripper_joints]

    # Use first joint's drive params as representative (all same on SO101)
    sample = next(iter(joints.values()), {})
    stiffness = sample.get("stiffness", 17.8)
    damping = sample.get("damping", 0.6)
    max_force = sample.get("max_force", 10.0)
    max_velocity = sample.get("max_velocity", 10.0)

    # Joint init positions (all zero)
    joint_init = ", ".join(f'"{j}": 0.0' for j in joints)

    return (
        f'    robot: ArticulationCfg = ArticulationCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Robot",\n'
        f'        spawn=sim_utils.UsdFileCfg(\n'
        f'            usd_path=ROBOT_USD_PATH,\n'
        f'            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),\n'
        f'            articulation_props=sim_utils.ArticulationRootPropertiesCfg(\n'
        f'                enabled_self_collisions=True,\n'
        f'                solver_position_iteration_count=4,\n'
        f'                solver_velocity_iteration_count=4,\n'
        f'                fix_root_link=True,\n'
        f'            ),\n'
        f'        ),\n'
        f'        init_state=ArticulationCfg.InitialStateCfg(\n'
        f'            joint_pos={{{joint_init}}},\n'
        f'        ),\n'
        f'        actuators={{\n'
        f'            "arm": ImplicitActuatorCfg(\n'
        f'                joint_names_expr={arm_joints},\n'
        f'                effort_limit_sim={max_force},\n'
        f'                velocity_limit_sim={max_velocity},\n'
        f'                stiffness={stiffness},\n'
        f'                damping={damping},\n'
        f'            ),\n'
        f'            "gripper": ImplicitActuatorCfg(\n'
        f'                joint_names_expr={gripper_joints},\n'
        f'                effort_limit_sim={max_force},\n'
        f'                velocity_limit_sim={max_velocity},\n'
        f'                stiffness={stiffness},\n'
        f'                damping={damping},\n'
        f'            ),\n'
        f'        }},\n'
        f'        soft_joint_pos_limit_factor=1.0,\n'
        f'    )\n'
    )


def _subasset_field(asset: dict) -> str:
    """Generate a RigidObjectCfg or ArticulationCfg field for a discovered subasset."""
    name = asset["name"]
    path = asset["path"]
    if asset["kind"] == "Articulation":
        return (
            f'    {name}: ArticulationCfg = ArticulationCfg(\n'
            f'        prim_path="{{ENV_REGEX_NS}}/Scene{path}",\n'
            f'        spawn=None,  # Already in scene USD\n'
            f'    )\n'
        )
    else:
        return (
            f'    {name}: RigidObjectCfg = RigidObjectCfg(\n'
            f'        prim_path="{{ENV_REGEX_NS}}/Scene{path}",\n'
            f'        spawn=None,  # Already in scene USD\n'
            f'    )\n'
        )


def generate_isaaclab_env(scene_config: dict | str, task_name: str,
                           output_dir: str = None):
    """Generate a self-contained IsaacLab export package from scene config.

    Produces an export folder with:
        {task_name}/   — Self-contained task package (Python + data)
        setup_task.py  — Copies task into IsaacLab's tasks dir
        run_random.py  — Test script with random actions

    Args:
        scene_config: Scene config dict or path to scene_config.json.
        task_name: Snake_case task name.
        output_dir: Where to write the export folder.
    """
    import json
    import shutil
    from pathlib import Path

    # Load config
    if isinstance(scene_config, str):
        with open(scene_config) as f:
            cfg = json.load(f)
    else:
        cfg = scene_config

    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    out_dir = Path(output_dir or f"{task_name}_export")
    pkg_dir = out_dir / task_name  # task package directory
    data_dir = pkg_dir / "data"    # data lives inside the package
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "mdp").mkdir(exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    robot = cfg["robot"]
    cameras = cfg.get("cameras", {})
    lights = cfg.get("lights", {})
    cam_names = list(cameras.keys())
    all_cam_names = cam_names + ["wrist"]

    # Discover subassets
    scene_path = cfg.get("scene_usd_path") or cfg.get("scene_path", "")
    if scene_path:
        p = Path(scene_path)
        no_robot = p.with_name(p.stem + "_no_robot" + p.suffix)
        discover_path = str(no_robot) if no_robot.exists() else scene_path
        subassets = _discover_subassets(discover_path)
    else:
        subassets = []

    # --- Copy data assets (preserve dir structure for USD relative refs) ---
    scene_src = Path(scene_path)
    scene_root = scene_src.parent.parent  # e.g., so_v1/ (parent of scene/)

    # Scene USD
    scene_dest = data_dir / "scene" / "scene.usda"
    scene_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(no_robot if no_robot.exists() else scene_src), str(scene_dest))
    print(f"  [OK] Copied scene → data/scene/scene.usda")

    # Sub-asset dirs referenced by scene USD (../assets/, ../env/)
    for sub in ["assets", "env"]:
        src = scene_root / sub
        if src.exists():
            dst = data_dir / sub
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
            print(f"  [OK] Copied {sub}/ → data/{sub}/")

    # Robot USD
    robot_usd_src = robot.get("usd_path", "")
    if robot_usd_src and Path(robot_usd_src).exists():
        robot_dest = data_dir / "robot" / "robot.usd"
        robot_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(robot_usd_src, str(robot_dest))
        print(f"  [OK] Copied robot → data/robot/robot.usd")

    # HDRI file (from dome light config)
    for lname, lcfg in lights.items():
        tex = lcfg.get("texture_file")
        if tex and Path(tex).exists():
            hdri_dest = data_dir / "hdri" / Path(tex).name
            hdri_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tex, str(hdri_dest))
            # Rewrite the light config to use relative path
            lcfg["texture_file"] = str(Path("DATA_DIR") / "hdri" / Path(tex).name)
            print(f"  [OK] Copied HDRI → data/hdri/{Path(tex).name}")

    # --- __init__.py (gym registration) ---
    gym_id = f"Custom-SO101-{pascal}-v0"
    init_content = (
        "import gymnasium as gym\n\n"
        "gym.register(\n"
        f'    id="{gym_id}",\n'
        '    entry_point="isaaclab.envs:ManagerBasedRLEnv",\n'
        "    disable_env_checker=True,\n"
        "    kwargs={\n"
        f'        "env_cfg_entry_point": f"{{__name__}}.{task_name}_env_cfg:{pascal}EnvCfg",\n'
        "    },\n"
        ")\n"
    )
    (pkg_dir / "__init__.py").write_text(init_content)

    # --- mdp/__init__.py ---
    (pkg_dir / "mdp" / "__init__.py").write_text(
        "from isaaclab.envs.mdp import *  # noqa: F403\n"
    )

    # --- env_cfg.py ---
    robot_pos = tuple(robot["position"])
    robot_orient = tuple(robot["orientation"])

    imports = (
        f'"""Self-contained IsaacLab env config for {task_name}.\n'
        f'\n'
        f'Generated by isaac_cfg_utils.generate_isaaclab_env().\n'
        f'No leisaac dependencies — runs with standard IsaacLab.\n'
        f'"""\n'
        f'from pathlib import Path\n'
        f'\n'
        f'import isaaclab.sim as sim_utils\n'
        f'from isaaclab.actuators import ImplicitActuatorCfg\n'
        f'from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg\n'
        f'from isaaclab.envs import ManagerBasedRLEnvCfg\n'
        f'from isaaclab.envs.mdp import *  # noqa: F403\n'
        f'from isaaclab.managers import EventTermCfg as EventTerm\n'
        f'from isaaclab.managers import ObservationGroupCfg as ObsGroup\n'
        f'from isaaclab.managers import ObservationTermCfg as ObsTerm\n'
        f'from isaaclab.managers import SceneEntityCfg\n'
        f'from isaaclab.managers import TerminationTermCfg as DoneTerm\n'
        f'from isaaclab.scene import InteractiveSceneCfg\n'
        f'from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg\n'
        f'from isaaclab.utils import configclass\n'
        f'\n'
        f'from . import mdp\n'
        f'\n'
        f'# --- Asset paths (set by setup_task.py) ---\n'
        f'DATA_DIR = Path(__file__).resolve().parent / "data"\n'
        f'SCENE_USD_PATH = str(DATA_DIR / "scene" / "scene.usda")\n'
        f'ROBOT_USD_PATH = str(DATA_DIR / "robot" / "robot.usd")\n'
    )

    # --- SceneCfg ---
    scene_cameras = "\n".join(
        _cam_field(name, cameras[name]) for name in cam_names
    )
    # Generate light fields — override HDRI paths to use DATA_DIR
    scene_light_parts = []
    for name, lcfg in lights.items():
        if lcfg["type"] == "DomeLight" and lcfg.get("texture_file"):
            # Use DATA_DIR code reference instead of hardcoded path
            hdri_filename = Path(lcfg["texture_file"]).name if not lcfg["texture_file"].startswith("DATA_DIR") else Path(lcfg["texture_file"]).name
            field_name = name[0].lower() + name[1:]
            spawn_args = [f'intensity={lcfg["intensity"]}']
            spawn_args.append(f'texture_file=str(DATA_DIR / "hdri" / "{hdri_filename}")')
            if lcfg.get("texture_format", "automatic") != "automatic":
                spawn_args.append(f'texture_format="{lcfg["texture_format"]}"')
            lines = (
                f'    {field_name} = AssetBaseCfg(\n'
                f'        prim_path="{{ENV_REGEX_NS}}/{field_name}",\n'
                f'        spawn=sim_utils.DomeLightCfg({", ".join(spawn_args)}),\n'
            )
            if lcfg.get("orientation"):
                o = tuple(lcfg["orientation"])
                lines += f'        init_state=AssetBaseCfg.InitialStateCfg(rot={o}),\n'
            lines += f'    )\n'
            scene_light_parts.append(lines)
        else:
            scene_light_parts.append(_light_field(name, lcfg))
    scene_lights = "\n".join(scene_light_parts)
    robot_field = _robot_articulation_field(robot)
    subasset_fields = "\n".join(_subasset_field(a) for a in subassets)

    # Wrist camera (from template, hardcoded)
    wrist_field = (
        f'    wrist: TiledCameraCfg = TiledCameraCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Robot/gripper/wrist_camera",\n'
        f'        offset=TiledCameraCfg.OffsetCfg(\n'
        f'            pos=(-0.001, 0.1, -0.04),\n'
        f'            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),\n'
        f'            convention="ros",\n'
        f'        ),\n'
        f'        data_types=["rgb"],\n'
        f'        spawn=sim_utils.PinholeCameraCfg(\n'
        f'            focal_length=36.5,\n'
        f'            focus_distance=400.0,\n'
        f'            horizontal_aperture=36.83,\n'
        f'            clipping_range=(0.01, 50.0),\n'
        f'            lock_camera=True,\n'
        f'        ),\n'
        f'        width=640, height=480,\n'
        f'        update_period=1 / 30.0,\n'
        f'    )\n'
    )

    # ee_frame (from template, hardcoded)
    ee_frame_field = (
        f'    ee_frame: FrameTransformerCfg = FrameTransformerCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Robot/base",\n'
        f'        debug_vis=False,\n'
        f'        target_frames=[\n'
        f'            FrameTransformerCfg.FrameCfg(\n'
        f'                prim_path="{{ENV_REGEX_NS}}/Robot/gripper", name="gripper",\n'
        f'            ),\n'
        f'            FrameTransformerCfg.FrameCfg(\n'
        f'                prim_path="{{ENV_REGEX_NS}}/Robot/jaw", name="jaw",\n'
        f'                offset=OffsetCfg(pos=(-0.021, -0.070, 0.02)),\n'
        f'            ),\n'
        f'        ],\n'
        f'    )\n'
    )

    scene_cfg = (
        f'\n\n@configclass\n'
        f'class {pascal}SceneCfg(InteractiveSceneCfg):\n'
        f'    """Scene: robot + cameras + lights + objects."""\n'
        f'\n'
        f'    scene: AssetBaseCfg = AssetBaseCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Scene",\n'
        f'        spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD_PATH),\n'
        f'    )\n'
        f'\n'
        f'{robot_field}\n'
        f'{ee_frame_field}\n'
        f'{wrist_field}\n'
        f'{scene_cameras}\n'
        f'{scene_lights}\n'
        f'{subasset_fields}\n'
    )

    # --- ObservationsCfg ---
    obs_cam_terms = "\n".join(
        _obs_cam_field(name) for name in all_cam_names
    )
    obs_cfg = (
        f'\n\n@configclass\n'
        f'class ObservationsCfg:\n'
        f'\n'
        f'    @configclass\n'
        f'    class PolicyCfg(ObsGroup):\n'
        f'        joint_pos = ObsTerm(func=mdp.joint_pos)\n'
        f'        joint_vel = ObsTerm(func=mdp.joint_vel)\n'
        f'        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)\n'
        f'        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)\n'
        f'        actions = ObsTerm(func=mdp.last_action)\n'
        f'\n'
        f'{obs_cam_terms}\n'
        f'        def __post_init__(self):\n'
        f'            self.enable_corruption = False\n'
        f'            self.concatenate_terms = False\n'
        f'\n'
        f'    policy: PolicyCfg = PolicyCfg()\n'
    )

    # --- ActionsCfg ---
    # Joint names from extracted robot config
    arm_joints = [j for j in cfg["robot"].get("joints", {}) if "gripper" not in j.lower()]
    gripper_joints = [j for j in cfg["robot"].get("joints", {}) if "gripper" in j.lower()]
    actions_cfg = (
        f'\n\n@configclass\n'
        f'class ActionsCfg:\n'
        f'    arm_action = mdp.JointPositionActionCfg(\n'
        f'        asset_name="robot",\n'
        f'        joint_names={arm_joints},\n'
        f'        scale=1.0,\n'
        f'        use_default_offset=True,\n'
        f'    )\n'
        f'    gripper_action = mdp.JointPositionActionCfg(\n'
        f'        asset_name="robot",\n'
        f'        joint_names={gripper_joints},\n'
        f'        scale=1.0,\n'
        f'        use_default_offset=True,\n'
        f'    )\n'
    )

    # --- RewardsCfg ---
    rewards_cfg = (
        f'\n\n@configclass\n'
        f'class RewardsCfg:\n'
        f'    """Reward terms — define task-specific rewards here."""\n'
        f'    pass\n'
    )

    # --- EventsCfg ---
    events_cfg = (
        f'\n\n@configclass\n'
        f'class EventsCfg:\n'
        f'    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")\n'
    )

    # --- TerminationsCfg ---
    term_cfg = (
        f'\n\n@configclass\n'
        f'class TerminationsCfg:\n'
        f'    time_out = DoneTerm(func=mdp.time_out, time_out=True)\n'
    )

    # --- EnvCfg ---
    env_cfg = (
        f'\n\n@configclass\n'
        f'class {pascal}EnvCfg(ManagerBasedRLEnvCfg):\n'
        f'    """Standalone IsaacLab environment for {task_name}."""\n'
        f'\n'
        f'    scene: {pascal}SceneCfg = {pascal}SceneCfg(env_spacing=8.0)\n'
        f'    observations: ObservationsCfg = ObservationsCfg()\n'
        f'    actions: ActionsCfg = ActionsCfg()\n'
        f'    rewards: RewardsCfg = RewardsCfg()\n'
        f'    events: EventsCfg = EventsCfg()\n'
        f'    terminations: TerminationsCfg = TerminationsCfg()\n'
        f'\n'
        f'    def __post_init__(self) -> None:\n'
        f'        super().__post_init__()\n'
        f'\n'
        f'        self.decimation = 1\n'
        f'        self.episode_length_s = 25.0\n'
        f'        self.viewer.eye = (1.4, -0.9, 1.2)\n'
        f'        self.viewer.lookat = (2.0, -0.5, 1.0)\n'
        f'\n'
        f'        self.sim.physx.bounce_threshold_velocity = 0.01\n'
        f'        self.sim.physx.friction_correlation_distance = 0.00625\n'
        f'        self.sim.render.enable_translucency = True\n'
        f'\n'
        f'        self.scene.robot.init_state.pos = {robot_pos}\n'
        f'        self.scene.robot.init_state.rot = {robot_orient}\n'
        f'\n'
        f'        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)\n'
    )

    # --- Write env_cfg to package ---
    full_content = imports + scene_cfg + obs_cfg + actions_cfg + rewards_cfg + events_cfg + term_cfg + env_cfg
    env_cfg_path = pkg_dir / f"{task_name}_env_cfg.py"
    env_cfg_path.write_text(full_content)

    # --- Generate setup_task.py ---
    setup_content = (
        '#!/usr/bin/env python3\n'
        '"""Install this task into an IsaacLab environment.\n'
        '\n'
        'Usage:\n'
        '    python setup_task.py /path/to/IsaacLab\n'
        '\n'
        'This copies the task code + data into IsaacLab\'s extension tasks\n'
        'so it can be discovered via `import_packages()` and used with\n'
        '`isaaclab train`, `gym.make()`, etc.\n'
        '"""\n'
        'import argparse\n'
        'import shutil\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        f'TASK_NAME = "{task_name}"\n'
        '\n'
        '\n'
        'def main():\n'
        '    parser = argparse.ArgumentParser(description="Install task into IsaacLab")\n'
        '    parser.add_argument("isaaclab_root", help="Path to IsaacLab installation")\n'
        '    args = parser.parse_args()\n'
        '\n'
        '    isaaclab_root = Path(args.isaaclab_root).resolve()\n'
        '    if not (isaaclab_root / "source").exists():\n'
        '        print(f"[ERR] {isaaclab_root} does not look like an IsaacLab installation")\n'
        '        print("      Expected to find a \'source/\' directory.")\n'
        '        sys.exit(1)\n'
        '\n'
        '    this_dir = Path(__file__).resolve().parent\n'
        '    pkg_src = this_dir / TASK_NAME\n'
        '\n'
        '    # Find tasks root\n'
        '    # IsaacLab uses source/isaaclab_tasks/isaaclab_tasks/ for custom tasks\n'
        '    tasks_root = isaaclab_root / "source" / "isaaclab_tasks" / "isaaclab_tasks"\n'
        '    if not tasks_root.exists():\n'
        '        tasks_root = isaaclab_root / "source" / "standalone" / "tasks"\n'
        '        if not tasks_root.exists():\n'
        '            print(f"[ERR] Cannot find tasks directory in {isaaclab_root}")\n'
        '            print("      Tried: source/isaaclab_tasks/isaaclab_tasks/")\n'
        '            sys.exit(1)\n'
        '\n'
        '    dest = tasks_root / TASK_NAME\n'
        '    if dest.exists():\n'
        '        print(f"[WARN] {dest} already exists, overwriting...")\n'
        '        shutil.rmtree(str(dest))\n'
        '\n'
        '    # Copy entire task package (code + data)\n'
        '    shutil.copytree(str(pkg_src), str(dest))\n'
        '    print(f"[OK] Installed task → {dest}")\n'
        '\n'
        '    print()\n'
        f'    print(f"Task installed! Test with:")\n'
        f'    print(f"  python {{isaaclab_root}}/source/standalone/environments/random_agent.py --task {gym_id} --num_envs 1")\n'
        '    print()\n'
        '    print("Or from Python:")\n'
        f'    print(f"  import {{TASK_NAME}}")\n'
        f'    print(f"  env = gym.make(\\"{gym_id}\\\", cfg=...)")\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    (out_dir / "setup_task.py").write_text(setup_content)

    # --- Generate run_random.py (test script) ---
    run_content = (
        '"""Test standalone IsaacLab env with random actions.\n'
        '\n'
        'Run after setup_task.py to verify the task works:\n'
        f'    python run_random.py --num_envs 1\n'
        '"""\n'
        'import argparse\n'
        'from isaaclab.app import AppLauncher\n'
        '\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--num_envs", type=int, default=1)\n'
        'AppLauncher.add_app_launcher_args(parser)\n'
        'args_cli = parser.parse_args()\n'
        '\n'
        'app_launcher = AppLauncher(args_cli)\n'
        'simulation_app = app_launcher.app\n'
        '\n'
        '"""Rest everything follows."""\n'
        '\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        '# Register our task (add export dir to path so vbti_so_v1 package is importable)\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        '\n'
        f'import {task_name}\n'
        f'from {task_name}.{task_name}_env_cfg import {pascal}EnvCfg\n'
        '\n'
        'import gymnasium as gym\n'
        'import torch\n'
        '\n'
        'def main():\n'
        f'    env_cfg = {pascal}EnvCfg()\n'
        '    env_cfg.scene.num_envs = args_cli.num_envs\n'
        '\n'
        f'    env = gym.make("{gym_id}", cfg=env_cfg)\n'
        '    print(f"[INFO] Observation space: {env.observation_space}")\n'
        '    print(f"[INFO] Action space: {env.action_space}")\n'
        '\n'
        '    env.reset()\n'
        '    count = 0\n'
        '    while simulation_app.is_running() and count < 200:\n'
        '        with torch.inference_mode():\n'
        '            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1\n'
        '            env.step(actions)\n'
        '            count += 1\n'
        '\n'
        '    env.close()\n'
        '    print(f"[OK] Ran {count} steps successfully")\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    (out_dir / "run_random.py").write_text(run_content)

    print(f"\n  [OK] Wrote standalone IsaacLab export → {out_dir}/")
    print(f"       {task_name}/  — self-contained task package (code + data)")
    print(f"       setup_task.py — copies {task_name}/ into IsaacLab")
    print(f"       run_random.py — test with random actions")
    print(f"       Gym ID: {gym_id}")

    return str(out_dir)


def pipeline(scene_usda_path: str, task_name: str,
             robot_prim_path: str = "/World/robot",
             robot_usd_path: str = None,
             cosmos_sensors: bool = False):
    """Run the full scene→task pipeline end-to-end.

    Takes a composed scene USDA (with robot, cameras, lights) and produces
    a complete leisaac task ready to run with teleop/training.

    Steps:
        1. create_no_robot_scene  → strips robot + lights + /Render
        2. generate_scene_asset   → creates leisaac asset bridge file
        3. create_task_boilerplate → creates task folder + gym registration
        4. generate_leisaac_env       → generates SceneCfg, ObsCfg, EnvCfg with lights

    Args:
        scene_usda_path: Path to the original scene USDA (with robot).
        task_name: Snake_case task name, e.g. 'vbti_mesh_table'.
        robot_prim_path: Prim path of the robot in the scene.
        robot_usd_path: Path to custom robot USD. If None, uses default leisaac asset.
        cosmos_sensors: If True, cameras capture depth + segmentation for Cosmos Transfer.

    Usage:
        python robot_utils.py pipeline vbti/data/vbti_table/scene/scene_v3.usda vbti_mesh_table
        python robot_utils.py pipeline scene.usda my_task --cosmos_sensors
    """
    from pathlib import Path

    scene_usda_path = str(Path(scene_usda_path).resolve())
    if robot_usd_path:
        robot_usd_path = str(Path(robot_usd_path).resolve())

    print(f"\n{'='*60}")
    print(f"  Pipeline: {scene_usda_path} → {task_name}")
    print(f"{'='*60}")

    # Step 1: Strip robot + lights → no_robot USDA
    p = Path(scene_usda_path)
    no_robot_path = str(p.with_name(p.stem + "_no_robot" + p.suffix))

    print(f"\n--- Step 1: create_no_robot_scene ---")
    create_no_robot_scene(scene_usda_path, robot_prim_path=robot_prim_path,
                          save_path=no_robot_path)

    # Step 2: Extract & persist scene config (source of truth)
    print(f"\n--- Step 2: extract_scene_config → scene_config.json ---")
    scene_config_path = str(p.with_name("scene_config.json"))
    scene_cfg = extract_scene_config(
        scene_usda_path,
        robot_prim_path=robot_prim_path,
        robot_usd_path=robot_usd_path,
        save_path=scene_config_path,
    )
    # Store the no-robot scene path for downstream generators
    scene_cfg["scene_usd_path"] = no_robot_path

    # Step 3: Generate scene asset bridge
    print(f"\n--- Step 3: generate_scene_asset ---")
    generate_scene_asset(task_name, no_robot_path)

    # Step 4: Create task boilerplate (skipped if folder exists)
    tasks_root = LEISAAC_ROOT + "/source/leisaac/leisaac/tasks"
    task_dir = Path(tasks_root) / task_name
    if task_dir.exists():
        print(f"\n--- Step 4: SKIPPED (task folder already exists) ---")
    else:
        print(f"\n--- Step 4: create_task_boilerplate ---")
        create_task_boilerplate(task_name)

    # Step 5: Generate leisaac env config
    print(f"\n--- Step 5: generate_leisaac_env ---")
    generate_leisaac_env(scene_usda_path, task_name, no_robot_usda_path=no_robot_path,
                     robot_usd_path=robot_usd_path, cosmos_sensors=cosmos_sensors)

    # Summary
    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    print(f"\n{'='*60}")
    print(f"  DONE — task '{task_name}' is ready")
    print(f"  Gym ID: LeIsaac-SO101-{pascal}-v0")
    if cosmos_sensors:
        print(f"  Cosmos sensors: depth + segmentation enabled on all cameras")
    print(f"")
    print(f"  Run with:")
    print(f"    python leisaac/scripts/environments/teleoperation/teleop_se3_agent.py \\")
    print(f"      --task LeIsaac-SO101-{pascal}-v0 \\")
    print(f"      --teleop_device so101leader --num_envs 1 --enable_cameras \\")
    print(f"      --dataset_file ./datasets/{task_name}.hdf5")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "no_robot_scene": create_no_robot_scene,
        "extract": extract_scene_config,
        "gen_scene": generate_scene_asset,
        "gen_task_folders": create_task_boilerplate,
        "gen_leisaac": generate_leisaac_env,
        "gen_isaaclab": generate_isaaclab_env,
        "pipeline": pipeline,
    })
