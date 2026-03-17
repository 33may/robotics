#!/usr/bin/env python3
"""USD robot asset utilities — inspect, fix base, and configure joint drives."""

from pxr import Usd, UsdPhysics


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


def extract_robot_config(robot_path: str) -> dict:
    """Extract joint drive and limit config from a robot USD.

    Reads all PhysicsRevoluteJoint prims and returns per-joint:
    stiffness, damping, effort/velocity limits, angle limits, axis.

    Args:
        robot_path: Path to the robot USD file.

    Returns:
        Dict with "joints" mapping joint name to its config.
        Angle limits are in degrees (as stored in USD).
    """
    stage = Usd.Stage.Open(robot_path)

    joints = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue

        name = prim.GetName()
        joint = {}

        attr_map = {
            "axis":         "physics:axis",
            "lower_limit":  "physics:lowerLimit",
            "upper_limit":  "physics:upperLimit",
            "stiffness":    "drive:angular:physics:stiffness",
            "damping":      "drive:angular:physics:damping",
            "max_force":    "drive:angular:physics:maxForce",
            "max_velocity": "physxJoint:maxJointVelocity",
        }
        for key, attr_name in attr_map.items():
            attr = prim.GetAttribute(attr_name)
            if attr and attr.Get() is not None:
                val = attr.Get()
                joint[key] = float(val) if isinstance(val, float) else str(val)

        joints[name] = joint

    return {"joints": joints}


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


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "inspect": inspect_robot,
        "fix_base": fix_articulation_base,
        "set_drives": set_drives,
        "extract": extract_robot_config,
        "make_ready": make_ready,
    })
