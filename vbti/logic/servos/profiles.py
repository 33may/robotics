#!/usr/bin/env python3
"""Manage SO-ARM101 calibration profiles.

List, inspect, load, export, and register calibration profiles stored in
the LeRobot cache and tracked in calibration/registry.json.

Usage:
    python -m vbti.logic.servos.profiles list
    python -m vbti.logic.servos.profiles show frodeo-test
    python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
    python -m vbti.logic.servos.profiles export frodeo-test
    python -m vbti.logic.servos.profiles register my-new --description "New calib"
"""
import json
import shutil
from datetime import date
from pathlib import Path

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

LEROBOT_CALIB_DIR = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
)
LEROBOT_CALIB_DIR_ALT = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # vbti/logic/servos -> root
REGISTRY_PATH = PROJECT_ROOT / "calibration" / "registry.json"
PROFILES_BACKUP_DIR = PROJECT_ROOT / "calibration" / "profiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calib_path(name: str) -> Path:
    """Path to a calibration JSON in the LeRobot cache (checks both dirs)."""
    primary = LEROBOT_CALIB_DIR / f"{name}.json"
    if primary.exists():
        return primary
    alt = LEROBOT_CALIB_DIR_ALT / f"{name}.json"
    if alt.exists():
        return alt
    return primary  # default to primary for new files


def _load_registry() -> dict:
    """Load the registry, bootstrapping from cache scan if file is missing."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())

    # Bootstrap: scan cache dirs for existing calibration files
    profiles = {}
    for d in (LEROBOT_CALIB_DIR, LEROBOT_CALIB_DIR_ALT):
        if d.exists():
            for f in sorted(d.glob("*.json")):
                if f.stem not in profiles:
                    profiles[f.stem] = {
                        "description": "",
                        "created": date.today().isoformat(),
                        "status": "active",
                    }
    registry = {"profiles": profiles, "default": ""}
    _save_registry(registry)
    return registry


def _save_registry(registry: dict) -> None:
    """Write the registry JSON."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=4) + "\n")


def _load_calib_json(name: str) -> dict:
    """Load a calibration JSON from the LeRobot cache."""
    path = _calib_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def list_profiles() -> None:
    """Show all profiles with status, description, and cache presence."""
    registry = _load_registry()
    default = registry.get("default", "")
    profiles = registry.get("profiles", {})

    if not profiles:
        print("No profiles registered.")
        return

    print(f"{'Name':<20} {'Status':<10} {'Cached':<8} {'Description'}")
    print("-" * 65)
    for name, info in profiles.items():
        marker = " *" if name == default else ""
        path = _calib_path(name)
        cached = f"yes ({path.parent.name})" if path.exists() else "no"
        desc = info.get("description", "")
        status = info.get("status", "?")
        print(f"{name + marker:<20} {status:<10} {cached:<8} {desc}")

    print(f"\n* = default profile ({default})")


def show(name: str) -> None:
    """Print calibration values and degree ranges for a profile."""
    data = _load_calib_json(name)
    print(f"Profile: {name}")
    print(f"{'Joint':<18} {'Offset':>8} {'Min':>6} {'Max':>6} {'Width':>6} {'Deg Min':>8} {'Deg Max':>8}")
    print("-" * 75)
    for joint in JOINT_NAMES:
        if joint not in data:
            continue
        j = data[joint]
        mid = (j["range_min"] + j["range_max"]) / 2
        deg_min = (j["range_min"] - mid) * 360.0 / 4095
        deg_max = (j["range_max"] - mid) * 360.0 / 4095
        print(
            f"{joint:<18} {j['homing_offset']:>8} {j['range_min']:>6} {j['range_max']:>6} "
            f"{j['range_max'] - j['range_min']:>6} {deg_min:>+8.1f}° {deg_max:>+8.1f}°"
        )


def load(name: str, port: str = "/dev/ttyACM1") -> None:
    """Write calibration to motor EEPROM via FeetechMotorsBus."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    data = _load_calib_json(name)

    # Build motors dict and calibration dict
    motors = {}
    calibration = {}
    for joint in JOINT_NAMES:
        if joint not in data:
            continue
        j = data[joint]
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=j["id"], model="sts3215", norm_mode=norm)
        calibration[joint] = MotorCalibration(
            id=j["id"],
            drive_mode=j["drive_mode"],
            homing_offset=j["homing_offset"],
            range_min=j["range_min"],
            range_max=j["range_max"],
        )

    bus = FeetechMotorsBus(port=port, motors=motors)
    print(f"Loading calibration '{name}' to motors on {port}...")
    bus.connect()

    with bus.torque_disabled():
        bus.write_calibration(calibration)

    # Verify
    for joint, cal in calibration.items():
        print(
            f"  {joint:18s}  homing_offset={cal.homing_offset:5d}  "
            f"range=[{cal.range_min}, {cal.range_max}]"
        )

    bus.disconnect()
    print("Done. Motors calibrated.")


def export(name: str) -> None:
    """Copy calibration JSON from LeRobot cache to calibration/profiles/."""
    src = _calib_path(name)
    if not src.exists():
        print(f"Error: calibration file not found: {src}")
        return

    PROFILES_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = PROFILES_BACKUP_DIR / f"{name}.json"
    shutil.copy2(src, dst)
    print(f"Exported: {dst}")


def register(name: str, description: str = "", parent: str = "") -> None:
    """Add or update a profile entry in the registry."""
    registry = _load_registry()
    profiles = registry.setdefault("profiles", {})

    existing = profiles.get(name, {})
    profiles[name] = {
        "description": description or existing.get("description", ""),
        "created": existing.get("created", date.today().isoformat()),
        "status": existing.get("status", "active"),
    }
    if parent:
        profiles[name]["parent"] = parent

    # Set as default if it's the only profile
    if not registry.get("default") and len(profiles) == 1:
        registry["default"] = name

    _save_registry(registry)
    cached = "yes" if _calib_path(name).exists() else "no"
    print(f"Registered '{name}' (cached: {cached})")


def sync() -> None:
    """Re-scan calibration directories and update registry with any new profiles."""
    registry = _load_registry()
    profiles = registry.setdefault("profiles", {})
    added = []
    for d in (LEROBOT_CALIB_DIR, LEROBOT_CALIB_DIR_ALT):
        if d.exists():
            for f in sorted(d.glob("*.json")):
                if f.stem not in profiles:
                    profiles[f.stem] = {
                        "description": "",
                        "created": date.today().isoformat(),
                        "status": "active",
                    }
                    added.append(f.stem)
    _save_registry(registry)
    if added:
        print(f"Added {len(added)} new profiles: {', '.join(added)}")
    else:
        print("Registry up to date.")


def delete(name: str) -> None:
    """Delete a calibration profile from cache and registry."""
    registry = _load_registry()
    profiles = registry.get("profiles", {})

    if name not in profiles and not _calib_path(name).exists():
        print(f"Profile '{name}' not found.")
        return

    # Remove from registry
    if name in profiles:
        del profiles[name]
        if registry.get("default") == name:
            registry["default"] = next(iter(profiles), "")
        _save_registry(registry)

    # Remove from cache
    cache_file = _calib_path(name)
    if cache_file.exists():
        cache_file.unlink()

    # Remove from version-controlled backups
    backup_file = PROFILES_BACKUP_DIR / f"{name}.json"
    if backup_file.exists():
        backup_file.unlink()

    print(f"Deleted profile '{name}'.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "list": list_profiles,
        "show": show,
        "load": load,
        "export": export,
        "register": register,
        "sync": sync,
        "delete": delete,
    })
