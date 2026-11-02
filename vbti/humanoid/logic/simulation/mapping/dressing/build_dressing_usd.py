"""build_dressing_usd.py — bake the dressing wrapper USD (MAY-173 dressing).

Authors full_warehouse_dressed.usda: /World references the untouched NVIDIA
full_warehouse.usd (its /Root default prim) and adds /World/Dressing — one
textured quad per layout placement (UsdPreviewSurface, no physics/colliders).
Geometry comes from layout_gen.quad_corners: positions in layout.yaml are the
final quad centers (2 cm off-wall included), so this script does no spatial
reasoning. Idempotent: re-running rewrites the same file.

Isaac env only (needs `pxr`). No SimulationApp — pure USD authoring, mirrors
build_camera_usd.py.

    conda run -n isaac python logic/simulation/mapping/dressing/build_dressing_usd.py \
        --layout assets/envs/warehouse_nvidia/dressing/layout.yaml \
        --scene assets/envs/warehouse_nvidia/Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
        --out assets/envs/warehouse_nvidia/dressing/full_warehouse_dressed.usda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from humanoid.logic.simulation.mapping.dressing.layout_gen import (  # noqa: E402
    load_layout,
    quad_corners,
)

_POSTER_ASPECT = 0.75  # quad height = width * aspect (matches 1024x768 texture)


def _author_quad(stage: Usd.Stage, path: str, corners: list[list[float]]) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    return mesh


def _author_material(stage: Usd.Stage, path: str, texture_rel: str) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"{path}/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{path}/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_rel)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))

    mat.CreateSurfaceOutput().ConnectToSource(
        pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def build(layout_path: Path, scene_path: Path, out_path: Path) -> tuple[int, int]:
    layout = load_layout(layout_path)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene_rel = os.path.relpath(scene_path.resolve(), out_path.resolve().parent)
    world.GetPrim().GetReferences().AddReference(scene_rel)

    UsdGeom.Scope.Define(stage, "/World/Dressing")
    UsdGeom.Scope.Define(stage, "/World/Dressing/Looks")

    n_plates = 0
    for p in layout["plates"]:
        name = f"plate_{p['number']:03d}"
        corners = quad_corners(p["pos"], p["normal"], p["size_m"])
        mesh = _author_quad(stage, f"/World/Dressing/{name}", corners)
        mat = _author_material(stage, f"/World/Dressing/Looks/{name}_mat",
                               f"./textures/{p['texture']}")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        n_plates += 1

    n_posters = 0
    for i, po in enumerate(layout["posters"]):
        name = f"poster_{i:02d}"
        corners = quad_corners(po["pos"], po["normal"], po["size_m"],
                               height=po["size_m"] * _POSTER_ASPECT)
        mesh = _author_quad(stage, f"/World/Dressing/{name}", corners)
        mat = _author_material(stage, f"/World/Dressing/Looks/{name}_mat",
                               f"./textures/{po['texture']}")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        n_posters += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_path))
    return n_plates, n_posters


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake the dressing wrapper USD.")
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True,
                    help="original full_warehouse.usd (stays untouched)")
    ap.add_argument("--out", type=Path, required=True, help="output .usda path")
    args = ap.parse_args()
    n_plates, n_posters = build(args.layout, args.scene, args.out)
    print(f"[build_dressing_usd] {n_plates} plates + {n_posters} posters -> {args.out}")


if __name__ == "__main__":
    main()
