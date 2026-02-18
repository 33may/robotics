#!/usr/bin/env python3
"""USD robot asset utilities — inspect, fix base, and configure joint drives."""

from pxr import Usd, UsdPhysics, UsdLux


LEISAAC_ROOT = "/home/may33/projects/ml_portfolio/robotics/leisaac"


def fix_articulation_base(robot_path: str, save_path: str = None, unfix: bool = False):
    """Fix or unfix the articulation base of a robot USD.

    Sets kinematicEnabled on the prim with ArticulationRootAPI.
    When kinematic=True, physics cannot move the base (gravity, collisions ignored).
    When kinematic=False, the base is dynamic and will fall unless supported.

    Args:
        robot_path: Path to the input robot USD file.
        save_path: Path to save the modified USD. If None, overwrites in place.
        unfix: If True, sets kinematicEnabled=False (base becomes dynamic).
    """
    target_val = not unfix
    action = "Unfixing" if unfix else "Fixing"

    print(f"\n{'='*50}")
    print(f"  {action} articulation base")
    print(f"  kinematicEnabled → {target_val}")
    print(f"  File: {robot_path}")
    print(f"{'='*50}\n")

    stage = Usd.Stage.Open(robot_path)

    found = False
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            found = True
            kinematic_attr = prim.GetAttribute("physics:kinematicEnabled")
            old_val = kinematic_attr.Get()

            if old_val == target_val:
                print(f"  [SKIP] {prim.GetPath()} already kinematicEnabled={target_val}")
                return

            kinematic_attr.Set(target_val)
            print(f"  [OK] {prim.GetPath()} [{prim.GetTypeName()}]")
            print(f"        kinematicEnabled: {old_val} → {target_val}")

    if not found:
        print(f"  [WARN] No ArticulationRootAPI found in {robot_path}")
        return

    if save_path:
        stage.GetRootLayer().Export(save_path)
        print(f"\n  Saved to {save_path}")
    else:
        stage.GetRootLayer().Save()
        print(f"\n  Saved (overwritten) {robot_path}")


def set_drives(robot_path: str, save_path: str = None,
               stiffness: float = 17.8, damping: float = 0.60,
               max_force: float = 10.0, max_velocity: float = 10.0):
    """Set angular drive parameters on all revolute joints.

    Configures stiffness, damping, effort and velocity limits so the robot
    holds position against gravity. Matches leisaac ImplicitActuatorCfg defaults.

    Args:
        robot_path: Path to the input robot USD file.
        save_path: Path to save the modified USD. If None, overwrites in place.
        stiffness: Position gain — spring force toward target angle.
        damping: Velocity damping — resists motion, prevents oscillation.
        max_force: Effort limit (Nm).
        max_velocity: Velocity limit (rad/s in PhysX, stored as deg/s in USD).
    """
    print(f"\n{'='*50}")
    print(f"  Setting joint drives")
    print(f"  stiffness={stiffness}  damping={damping}")
    print(f"  max_force={max_force}  max_velocity={max_velocity}")
    print(f"  File: {robot_path}")
    print(f"{'='*50}\n")

    stage = Usd.Stage.Open(robot_path)

    count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            count += 1
            old_stiff = prim.GetAttribute("drive:angular:physics:stiffness").Get()
            old_damp = prim.GetAttribute("drive:angular:physics:damping").Get()

            prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            prim.GetAttribute("drive:angular:physics:damping").Set(damping)
            prim.GetAttribute("drive:angular:physics:maxForce").Set(max_force)
            prim.GetAttribute("physxJoint:maxJointVelocity").Set(max_velocity)

            print(f"  [OK] {prim.GetName()}")
            print(f"        stiffness: {old_stiff:.4f} → {stiffness}")
            print(f"        damping:   {old_damp:.4f} → {damping}")

    if count == 0:
        print(f"  [WARN] No revolute joints found in {robot_path}")
        return

    print(f"\n  Updated {count} joints")

    if save_path:
        stage.GetRootLayer().Export(save_path)
        print(f"  Saved to {save_path}")
    else:
        stage.GetRootLayer().Save()
        print(f"  Saved (overwritten) {robot_path}")


def make_ready(robot_path: str, save_path: str = None,
               stiffness: float = 17.8, damping: float = 0.60,
               max_force: float = 10.0, max_velocity: float = 10.0):
    """One-shot: fix base + set drives. Produces a GUI-ready robot USD.

    Args:
        robot_path: Path to the input robot USD file.
        save_path: Path to save. If None, overwrites in place.
        stiffness: Joint position gain.
        damping: Joint velocity damping.
        max_force: Joint effort limit.
        max_velocity: Joint velocity limit.
    """
    print(f"\n{'='*50}")
    print(f"  Making robot GUI-ready")
    print(f"  File: {robot_path}")
    print(f"{'='*50}")

    stage = Usd.Stage.Open(robot_path)

    # 1. Fix articulation base
    art_found = False
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_found = True
            prim.GetAttribute("physics:kinematicEnabled").Set(True)
            print(f"\n  [OK] Fixed base: {prim.GetPath()}")

    if not art_found:
        print(f"\n  [WARN] No ArticulationRootAPI found")

    # 2. Set drives on all joints
    count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            count += 1
            prim.GetAttribute("drive:angular:physics:stiffness").Set(stiffness)
            prim.GetAttribute("drive:angular:physics:damping").Set(damping)
            prim.GetAttribute("drive:angular:physics:maxForce").Set(max_force)
            prim.GetAttribute("physxJoint:maxJointVelocity").Set(max_velocity)
            print(f"  [OK] Drive: {prim.GetName()}  stiff={stiffness} damp={damping}")

    print(f"\n  Summary: base={'fixed' if art_found else 'NOT FOUND'}, {count} joints configured")

    if save_path:
        stage.GetRootLayer().Export(save_path)
        print(f"  Saved to {save_path}")
    else:
        stage.GetRootLayer().Save()
        print(f"  Saved (overwritten) {robot_path}")


def inspect_robot(robot_path: str, joint_info: bool = True):
    """Print the full prim hierarchy of a robot USD as a colored tree.

    Args:
        robot_path: Path to the robot USD file.
        joint_info: Show drive parameters on joints (default True).
    """
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    GREEN   = "\033[32m"
    MAGENTA = "\033[35m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    RED     = "\033[31m"

    TYPE_COLORS = {
        "Xform": CYAN,
        "Mesh": GREEN,
        "Scope": DIM,
        "Material": MAGENTA,
        "Shader": MAGENTA,
        "PhysicsRevoluteJoint": YELLOW,
        "PhysicsFixedJoint": YELLOW,
        "PhysicsScene": YELLOW,
    }

    stage = Usd.Stage.Open(robot_path)
    root = stage.GetDefaultPrim()
    print(f"{CYAN}{root.GetPath()}{RESET}\n")

    # Build parent → children map for tree connectors
    children_map = {}
    for prim in stage.Traverse():
        parent_path = str(prim.GetParent().GetPath())
        children_map.setdefault(parent_path, []).append(prim)

    def print_tree(prim, prefix=""):
        siblings = children_map.get(str(prim.GetParent().GetPath()), [])
        is_last = prim == siblings[-1]
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        type_name = prim.GetTypeName() or "Prim"
        color = TYPE_COLORS.get(type_name, RESET)

        schemas = prim.GetAppliedSchemas()
        schema_str = f" {DIM}{schemas}{RESET}" if schemas else ""

        print(f"{DIM}{prefix}{connector}{RESET}{color}{prim.GetName()}{RESET}  {DIM}[{type_name}]{RESET}{schema_str}")

        # Joint drive details
        if joint_info and type_name in ("PhysicsRevoluteJoint", "PhysicsFixedJoint"):
            detail_prefix = prefix + extension
            attrs = {
                "stiffness":    "drive:angular:physics:stiffness",
                "damping":      "drive:angular:physics:damping",
                "max force":    "drive:angular:physics:maxForce",
                "max velocity": "physxJoint:maxJointVelocity",
                "lower limit":  "physics:lowerLimit",
                "upper limit":  "physics:upperLimit",
                "axis":         "physics:axis",
            }
            for label, attr_name in attrs.items():
                attr = prim.GetAttribute(attr_name)
                if attr and attr.Get() is not None:
                    val = attr.Get()
                    warn = ""
                    if isinstance(val, float):
                        if label == "stiffness" and val < 1.0:
                            warn = f"  {RED}← low!{RESET}"
                        elif label == "damping" and val < 0.1:
                            warn = f"  {RED}← low!{RESET}"
                        val = f"{val:.4f}"
                    print(f"{DIM}{detail_prefix}  {label}: {RESET}{val}{warn}")

        # Articulation root details
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            detail_prefix = prefix + extension
            kinematic = prim.GetAttribute("physics:kinematicEnabled").Get()
            status = f"{GREEN}FIXED{RESET}" if kinematic else f"{RED}DYNAMIC (will fall!){RESET}"
            print(f"{DIM}{detail_prefix}  base: {RESET}{status}")

        for child in children_map.get(str(prim.GetPath()), []):
            print_tree(child, prefix + extension)

    for prim in children_map.get("/", []):
        print_tree(prim)


def create_no_robot_scene(scene_path: str, robot_prim_path: str = "/World/so101_simready_follower_leisaac",
                          save_path: str = None, remove_cameras: bool = False):
    """Strip robot and GUI render config from a scene USD.

    Produces a clean scene that LeIsaac can load — it spawns
    the robot separately, so it must not be in the scene USD.
    Cameras are kept by default so LeIsaac can reference them via spawn=None.

    Args:
        scene_path: Path to the input scene USD.
        robot_prim_path: Prim path of the robot to remove.
        save_path: Output path. Defaults to <scene>_no_robot.usda.
        remove_cameras: If True, also remove Camera prims under /World.
    """
    from pathlib import Path

    stage = Usd.Stage.Open(scene_path)

    # 1. Remove the robot prim
    stage.RemovePrim(robot_prim_path)
    print(f"  [OK] Removed robot: {robot_prim_path}")

    # 2. Optionally remove world-level cameras
    if remove_cameras:
        for prim in stage.GetPrimAtPath("/World").GetChildren():
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


def extract_scene_config(scene_path: str, robot_prim_path=None) -> dict:
    """Extract robot, camera, and joint config from a composed scene USDA.

    Args:
        scene_path: Path to the scene USD file.
        robot_prim_path: Prim path of the robot. If None, auto-detects via ArticulationRootAPI.
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
    (task_dir / "__init__.py").write_text(init_content)
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


def _cam_field(cam_name: str, width: int = 640, height: int = 480) -> str:
    """Generate a TiledCameraCfg field that references an existing camera in the scene USD."""
    return (
        f'    {cam_name}: TiledCameraCfg = TiledCameraCfg(\n'
        f'        prim_path="{{ENV_REGEX_NS}}/Scene/{cam_name}",\n'
        f'        spawn=None,\n'
        f'        data_types=["rgb"],\n'
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
    """Find RigidBody/Articulation prims that parse_usd_and_create_subassets will discover."""
    stage = Usd.Stage.Open(scene_usda_path)
    assets = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            name = prim.GetPath().pathString.split("/")[-1]
            kind = "Articulation" if prim.HasAPI(UsdPhysics.ArticulationRootAPI) else "RigidBody"
            assets.append({"name": name, "path": str(prim.GetPath()), "kind": kind})
    return assets


def _obs_cam_field(cam_name: str) -> str:
    """Generate an ObsTerm for a camera image — name must match SceneCfg field."""
    return (
        f'        {cam_name} = ObsTerm(\n'
        f'            func=mdp.image,\n'
        f'            params={{\n'
        f'                "sensor_cfg": SceneEntityCfg("{cam_name}"),\n'
        f'                "data_type": "rgb",\n'
        f'                "normalize": False\n'
        f'            }}\n'
        f'        )\n'
    )


def generate_env_cfg(scene_usda_path: str, task_name: str,
                     no_robot_usda_path: str = None,
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
        f'import isaaclab.sim as sim_utils\n'
        f'from isaaclab.assets import AssetBaseCfg\n'
        f'from isaaclab.managers import ObservationGroupCfg as ObsGroup\n'
        f'from isaaclab.managers import ObservationTermCfg as ObsTerm\n'
        f'from isaaclab.managers import SceneEntityCfg\n'
        f'from isaaclab.managers import TerminationTermCfg as DoneTerm\n'
        f'from isaaclab.sensors import TiledCameraCfg\n'
        f'from isaaclab.utils import configclass\n'
        f'from leisaac.assets.scenes.{task_name} import {task_upper}_CFG, {task_upper}_USD_PATH\n'
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
    scene_cameras = "\n".join(_cam_field(name) for name in cam_names)
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

    # --- ObservationsCfg ---
    # Joint states (always needed) + camera images for each camera.
    # Camera ObsTerm names must match SceneCfg field names.
    obs_cam_terms = "\n".join(_obs_cam_field(name) for name in all_cam_names)
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
        f'\n'
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



def pipeline(scene_usda_path: str, task_name: str,
             robot_prim_path: str = "/World/robot"):
    """Run the full scene→task pipeline end-to-end.

    Takes a composed scene USDA (with robot, cameras, lights) and produces
    a complete leisaac task ready to run with teleop/training.

    Steps:
        1. create_no_robot_scene  → strips robot + lights + /Render
        2. generate_scene_asset   → creates leisaac asset bridge file
        3. create_task_boilerplate → creates task folder + gym registration
        4. generate_env_cfg       → generates SceneCfg, ObsCfg, EnvCfg with lights

    Args:
        scene_usda_path: Path to the original scene USDA (with robot).
        task_name: Snake_case task name, e.g. 'vbti_mesh_table'.
        robot_prim_path: Prim path of the robot in the scene.

    Usage:
        python robot_utils.py pipeline vbti/data/vbti_table/scene/scene_v3.usda vbti_mesh_table
    """
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"  Pipeline: {scene_usda_path} → {task_name}")
    print(f"{'='*60}")

    # Step 1: Strip robot + lights → no_robot USDA
    p = Path(scene_usda_path)
    no_robot_path = str(p.with_name(p.stem + "_no_robot" + p.suffix))

    print(f"\n--- Step 1: create_no_robot_scene ---")
    create_no_robot_scene(scene_usda_path, robot_prim_path=robot_prim_path,
                          save_path=no_robot_path)

    # Step 2: Generate scene asset bridge
    print(f"\n--- Step 2: generate_scene_asset ---")
    generate_scene_asset(task_name, no_robot_path)

    # Step 3: Create task boilerplate (skipped if folder exists)
    tasks_root = LEISAAC_ROOT + "/source/leisaac/leisaac/tasks"
    task_dir = Path(tasks_root) / task_name
    if task_dir.exists():
        print(f"\n--- Step 3: SKIPPED (task folder already exists) ---")
    else:
        print(f"\n--- Step 3: create_task_boilerplate ---")
        create_task_boilerplate(task_name)

    # Step 4: Generate env config (reads original scene for config extraction)
    print(f"\n--- Step 4: generate_env_cfg ---")
    generate_env_cfg(scene_usda_path, task_name, no_robot_usda_path=no_robot_path)

    # Summary
    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    print(f"\n{'='*60}")
    print(f"  DONE — task '{task_name}' is ready")
    print(f"  Gym ID: LeIsaac-SO101-{pascal}-v0")
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
        "inspect": inspect_robot,
        "fix_base": fix_articulation_base,
        "set_drives": set_drives,
        "make_ready": make_ready,
        "no_robot_scene": create_no_robot_scene,
        "extract": extract_scene_config,
        "gen_scene": generate_scene_asset,
        "gen_task_folders": create_task_boilerplate,
        "gen_env_cfg": generate_env_cfg,
        "pipeline": pipeline,
    })
