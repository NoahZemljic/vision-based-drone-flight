"""Make glTF-converted USD materials readable by Newton's importer.

IsaacLab's ``convert_mesh.py`` (omni asset converter) writes ``gltf/pbr.mdl`` shaders whose
colour lives in ``inputs:base_color_factor``; Newton's USD importer only reads
``diffuse_color_constant`` / ``baseColor`` / ``diffuseColor`` / ``displayColor`` and otherwise
assigns its default palette. This mirrors ``base_color_factor`` into a
``diffuse_color_constant`` input on every such shader (and ``roughness_factor`` /
``metallic_factor`` into the *_constant names), copies the base-colour texture out of the
``gltf_texture_lookup`` sub-shader into a direct ``diffuse_texture`` input (Newton never
follows MDL sub-shaders), and rewrites absolute texture asset paths
(the converter writes ``/tmp/.../textures/x.jpg``) to ``./textures/x.jpg`` next to the file.
Idempotent, edits the file in place. ``--zup`` additionally bakes the glTF Y-up -> Z-up
rotation (+90 deg about X) into the geometry Xform — the 3.0 converter no longer does this,
so a converted GLB lies on its side; irrelevant for the detection sampler (random attitudes),
required for pose-faithful uses (tracking viz).

    /workspace/isaaclab/_isaac_sim/python.sh tools/usd_expose_gltf_colors.py <file.usd> [--zup]
"""
import os
import sys

from pxr import Sdf, Usd, UsdShade

MIRROR = {"base_color_factor": ("diffuse_color_constant", Sdf.ValueTypeNames.Color3f),
          "roughness_factor": ("reflection_roughness_constant", Sdf.ValueTypeNames.Float),
          "metallic_factor": ("metallic_constant", Sdf.ValueTypeNames.Float)}


def main(path: str) -> None:
    stage = Usd.Stage.Open(path)
    n = 0
    t = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for src, (dst, tname) in MIRROR.items():
            inp = shader.GetInput(src)
            if not inp or inp.Get() is None:
                continue
            value = inp.Get()
            if src == "base_color_factor":
                value = tuple(float(v) for v in value)[:3]
            shader.CreateInput(dst, tname).Set(value)
            n += 1
        tex_inp = shader.GetInput("base_color_texture")
        if tex_inp and tex_inp.HasConnectedSource():
            sub = UsdShade.Shader(tex_inp.GetConnectedSource()[0].GetPrim())
            tex = sub.GetInput("texture")
            if tex and tex.Get():
                shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tex.Get())
                n += 1
        for inp in shader.GetInputs():
            if inp.GetTypeName() == "asset" and inp.Get():
                asset = str(inp.Get().path)
                if os.path.isabs(asset):
                    rel = os.path.join(".", "textures", os.path.basename(asset))
                    if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(path)), rel)):
                        inp.Set(Sdf.AssetPath(rel))
                        t += 1
    stage.GetRootLayer().Save()
    print(f"[usd_expose_gltf_colors] {path}: mirrored {n} inputs, relinked {t} textures")


def fix_zup(path: str) -> None:
    """Bake the glTF Y-up -> USD Z-up rotation (+90 deg about X) into the geometry Xform.

    IsaacLab 3.0's asset converter keeps glTF geometry in its Y-up frame (2.3 rotated it),
    so a converted GLB lies on its side. The op goes on <root>/geometry, not the rigid-body
    root, whose transform the sim overwrites. Idempotent.
    """
    from pxr import UsdGeom

    stage = Usd.Stage.Open(path)
    geo = stage.GetDefaultPrim().GetChild("geometry")
    if not geo:
        print(f"[usd_expose_gltf_colors] --zup: no <root>/geometry prim in {path}, skipped")
        return
    xf = UsdGeom.Xformable(geo)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateX:
            print(f"[usd_expose_gltf_colors] --zup: rotateX already present ({op.Get()} deg), skipped")
            return
    xf.AddRotateXOp().Set(90.0)
    stage.GetRootLayer().Save()
    print(f"[usd_expose_gltf_colors] --zup: baked +90 deg X on {geo.GetPath()}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--zup"]
    main(args[0])
    if "--zup" in sys.argv:
        fix_zup(args[0])
