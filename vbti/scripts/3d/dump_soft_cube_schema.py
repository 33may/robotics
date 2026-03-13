"""
Dump the schema/APIs of the working SoftCube prim.
Run in Isaac Sim Script Editor while the scene with the soft cube is loaded.
"""
import omni.usd

stage = omni.usd.get_context().get_stage()

# Try common paths - adjust if needed
candidates = [
    "/World/envs/env_0/SoftCube",
    "/World/SoftCube",
]

for path in candidates:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        print(f"\n{'='*60}")
        print(f"PRIM: {path}")
        print(f"Type: {prim.GetTypeName()}")
        print(f"APIs: {prim.GetAppliedSchemas()}")
        print(f"Attrs:")
        for attr in prim.GetAttributes():
            val = attr.Get()
            # Skip huge arrays
            if val is not None and not isinstance(val, (list, tuple)) or (isinstance(val, (list, tuple)) and len(val) < 20):
                print(f"  {attr.GetName()} = {val}")
            else:
                print(f"  {attr.GetName()} = <array len={len(val) if val else 0}>")

        # Dump children too
        for child in prim.GetAllChildren():
            print(f"\n  CHILD: {child.GetPath()}")
            print(f"  Type: {child.GetTypeName()}")
            print(f"  APIs: {child.GetAppliedSchemas()}")
            for attr in child.GetAttributes():
                val = attr.Get()
                if val is not None and not isinstance(val, (list, tuple)) or (isinstance(val, (list, tuple)) and len(val) < 20):
                    print(f"    {attr.GetName()} = {val}")
                else:
                    print(f"    {attr.GetName()} = <array len={len(val) if val else 0}>")
        break
else:
    print("SoftCube not found. Listing /World children:")
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for child in world.GetAllChildren():
            print(f"  {child.GetPath()} [{child.GetTypeName()}] APIs={child.GetAppliedSchemas()}")
