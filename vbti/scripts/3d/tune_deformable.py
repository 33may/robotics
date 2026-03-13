"""
Utility functions to tune deformable body params in a USDA file.
Usage: import and call, or edit the __main__ block and run directly.
"""
from pathlib import Path
import re

DEFAULT_USD = "/home/may33/projects/ml_portfolio/robotics/vbti/data/so_v1/assets/duck/object_1_soft.usda"


def _set_param(usd_path: str, pattern: str, value: float):
    """Replace a numeric value for a given attribute pattern in a USDA file."""
    path = Path(usd_path)
    text = path.read_text()
    regex = re.compile(rf"({re.escape(pattern)}\s*=\s*)[\d.eE+\-]+")
    if not regex.search(text):
        print(f"[WARN] '{pattern}' not found in {usd_path}")
        return
    text = regex.sub(rf"\g<1>{value}", text)
    path.write_text(text)
    print(f"  {pattern} = {value}")


def tune(usd_path: str = DEFAULT_USD, **kwargs):
    """
    Set any deformable param by keyword. Examples:
        tune(youngs_modulus=2e5, elasticity_damping=0.3)
        tune(vertex_velocity_damping=0.1, solver_iterations=32)
    """
    param_map = {
        "youngs_modulus":           "physxDeformableBodyMaterial:youngsModulus",
        "poissons_ratio":           "physxDeformableBodyMaterial:poissonsRatio",
        "density":                  "physxDeformableBodyMaterial:density",
        "dynamic_friction":         "physxDeformableBodyMaterial:dynamicFriction",
        "elasticity_damping":       "physxDeformableBodyMaterial:elasticityDamping",
        "vertex_velocity_damping":  "physxDeformable:vertexVelocityDamping",
        "solver_iterations":        "physxDeformable:solverPositionIterationCount",
    }
    print(f"[tune] Updating {usd_path}")
    for key, val in kwargs.items():
        if key not in param_map:
            print(f"[WARN] Unknown param '{key}'. Valid: {list(param_map.keys())}")
            continue
        _set_param(usd_path, param_map[key], val)
    print("[done]")


def show(usd_path: str = DEFAULT_USD):
    """Print current deformable params."""
    text = Path(usd_path).read_text()
    params = [
        "physxDeformableBodyMaterial:youngsModulus",
        "physxDeformableBodyMaterial:poissonsRatio",
        "physxDeformableBodyMaterial:density",
        "physxDeformableBodyMaterial:dynamicFriction",
        "physxDeformableBodyMaterial:elasticityDamping",
        "physxDeformable:vertexVelocityDamping",
        "physxDeformable:solverPositionIterationCount",
    ]
    print(f"[params] {usd_path}")
    for p in params:
        match = re.search(rf"{re.escape(p)}\s*=\s*([\d.eE+\-]+)", text)
        if match:
            print(f"  {p.split(':')[1]:40s} = {match.group(1)}")


if __name__ == "__main__":
    show()
    # Example: uncomment and run to change params
    # tune(youngs_modulus=2e5, elasticity_damping=0.3, vertex_velocity_damping=0.1)
    # show()
