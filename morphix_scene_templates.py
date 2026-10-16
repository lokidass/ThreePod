# morphix_scene_templates.py
# Tested, parameterized scene templates. Each returns a complete validated
# Scene IR (use morphix_scene_compile.compile_scene_ir to emit JS).
# Templates combine through the scene IR — never by concatenating raw JS.
#
# Attribution: original implementations informed by official Three.js
# examples (fog usage, PMREM environments, instancing patterns). No example
# source is copied; techniques are reimplemented against the vendored r152
# core. The fog benchmark (template 10) documents exactly which effects of
# webgpu_custom_fog_background are matched, approximated, or unsupported.

from morphix_scene_ir import (geometry, material, new_scene, node,
                              procedural, transform, validate_scene_ir)

TEMPLATE_NAMES = ("studio", "mountains", "forest", "sky_fog", "scifi",
                  "city", "water", "particles", "product", "fog_benchmark",
                  "fog_landscape")


def _finish(ir):
    errs = validate_scene_ir(ir)
    if errs:
        raise ValueError("template produced invalid IR: " + "; ".join(errs))
    return ir


def _sun(ir, shadow=True, bounds=15):
    ir["lights"].append({"id": "sun", "kind": "directional",
                         "color": "0xFFF2E0", "intensity": 1.6,
                         "position": [6, 10, 7],
                         "shadow": {"map": 1024, "left": -bounds,
                                    "right": bounds, "top": bounds,
                                    "bottom": -bounds} if shadow else None})
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0xBDD7F2", "intensity": 0.7,
                         "ground": "0x334455"})
    return ir


# 1. Professional studio environment.
def studio(seed=11, pedestal_color="0x2A3242"):
    """Soft studio: ground disc, backdrop gradient sky, 3-point lighting,
    product pedestal with a hero box. Shadows on, ACES tone mapping."""
    ir = new_scene("studio", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x3A4A63", "mid": "0x232C40",
        "bottom": "0x111622"}
    ir["camera"] = {"kind": "perspective", "fov": 45, "near": 0.1,
                    "far": 500, "position": [5, 3.2, 7], "target": [0, 0.8, 0]}
    _sun(ir)
    ir["lights"].append({"id": "rim", "kind": "directional",
                         "color": "0x88AAFF", "intensity": 0.8,
                         "position": [-6, 4, -6]})
    ir["nodes"].append(node(
        "floor", kind="mesh",
        geometry=geometry("CircleGeometry", {"r": 14, "seg": 64}),
        material=material("standard", color="0x39414F",
                          roughness=0.9, metalness=0.0),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    ir["nodes"].append(node(
        "pedestal", kind="mesh",
        geometry=geometry("CylinderGeometry",
                          {"rt": 1.1, "rb": 1.3, "h": 1.0, "seg": 48}),
        material=material("standard", color=pedestal_color,
                          roughness=0.35, metalness=0.1),
        transform=transform(position=[0, 0.5, 0]),
        cast_shadow=True, receive_shadow=True, printable=True))
    ir["nodes"].append(node(
        "hero", kind="mesh",
        geometry=geometry("BoxGeometry", {"w": 1.0, "h": 1.0, "d": 1.0}),
        material=material("standard", color="0xE8A33D",
                          roughness=0.4, metalness=0.2),
        transform=transform(position=[0, 1.7, 0], rotation=[0, 0.6, 0]),
        cast_shadow=True, printable=True))
    ir["animations"].append({"target": "hero", "kind": "rotate",
                             "speed": 0.4})
    return _finish(ir)


# 2. Procedural mountain landscape.
def mountains(seed=22, size=90, height=9):
    """Seeded fBm terrain + gradient sky + exp2 fog + sun. Deterministic:
    same seed renders identical pixels (tested)."""
    ir = new_scene("mountains", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x27435F", "mid": "0x9DB8D6",
        "bottom": "0x1A2433"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0x9DB8D6",
                                "density": 0.012}
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 600, "position": [28, 16, 34], "target": [0, 2, 0]}
    _sun(ir, bounds=45)
    ir["nodes"].append(node(
        "terrain", kind="mesh",
        geometry=procedural("terrain", {"size": size, "segments": 96,
                                        "height": height}),
        material=material("standard", color="0x5A6B4A",
                          roughness=0.95, metalness=0.0),
        receive_shadow=True))
    ir["nodes"].append(node(
        "sky", kind="mesh",
        geometry=procedural("gradient_sky",
                            {"top": "0x27435F", "mid": "0x9DB8D6",
                             "bot": "0x1A2433"}),
        material=material("basic", color="0xFFFFFF", fog=False)))
    return _finish(ir)


# 3. Forest with instanced trees.
def forest(seed=33, trees=2000, radius=26, clearing=4.0):
    """Two InstancedMeshes (trunks + canopies): thousands of trees in two
    draw calls. Seeded scatter with a central clearing."""
    ir = new_scene("forest", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x2E4A68", "mid": "0x8FA9C4",
        "bottom": "0x141C28"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0x8FA9C4",
                                "density": 0.02}
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 400, "position": [14, 7, 18], "target": [0, 2, 0]}
    _sun(ir, bounds=30)
    ir["nodes"].append(node(
        "ground", kind="mesh",
        geometry=geometry("CircleGeometry", {"r": radius + 6, "seg": 48}),
        material=material("standard", color="0x3D4A33",
                          roughness=1.0, metalness=0.0),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    dist = {"shape": "disc", "radius": radius, "clearing": clearing,
            "scale_jitter": 0.35}
    ir["nodes"].append(node(
        "trunks", kind="instanced",
        geometry=geometry("CylinderGeometry",
                          {"rt": 0.09, "rb": 0.14, "h": 1.6, "seg": 7}),
        material=material("standard", color="0x6B4A2F",
                          roughness=0.95, metalness=0.0),
        count=trees, distribution=dict(dist, y=0.8), cast_shadow=True))
    ir["nodes"].append(node(
        "canopies", kind="instanced",
        geometry=geometry("ConeGeometry", {"r": 1.0, "h": 2.6, "seg": 8}),
        material=material("standard", color="0x2F5B2F",
                          roughness=0.9, metalness=0.0),
        count=trees, distribution=dict(dist, y=2.6), cast_shadow=True))
    return _finish(ir)


# 4. Atmospheric fog and sky.
def sky_fog(seed=44, density=0.028):
    """Gradient sky dome + exp2 fog + drifting particle haze + moon disc."""
    ir = new_scene("sky_fog", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x1B2A45", "mid": "0x6E8FB4",
        "bottom": "0x0D1420"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0x6E8FB4",
                                "density": density}
    ir["camera"] = {"kind": "perspective", "fov": 60, "near": 0.1,
                    "far": 600, "position": [0, 3, 24], "target": [0, 4, 0]}
    ir["lights"].append({"id": "moon", "kind": "directional",
                         "color": "0xC9D9F2", "intensity": 0.9,
                         "position": [-8, 14, 6]})
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0x6E8FB4", "intensity": 0.6,
                         "ground": "0x1A2233"})
    ir["nodes"].append(node(
        "moon", kind="mesh",
        geometry=geometry("SphereGeometry", {"r": 2.2, "ws": 32, "hs": 32}),
        material=material("basic", color="0xE8F0FF", fog=False),
        transform=transform(position=[-30, 34, -80])))
    ir["nodes"].append(node(
        "haze", kind="points",
        geometry=procedural("particles", {"count": 1500}),
        material=material("points", color="0xBDD0E8", size=0.35),
        count=1500,
        distribution={"shape": "box", "hx": 40, "hz": 40}))
    return _finish(ir)


# 5. Sci-fi environment.
def scifi(seed=55):
    """Dark hangar: reflective floor, emissive strip lights, arch gates,
    flickering consoles. All self-contained (no external assets)."""
    ir = new_scene("scifi", seed=seed)
    ir["environment"]["background"] = {"kind": "color", "color": "0x05070C"}
    ir["environment"]["environment_map"] = {
        "kind": "procedural_equirect", "sky": "0x1E3A5C",
        "horizon": "0x3A6A9E", "ground": "0x05070C"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0x0A1220",
                                "density": 0.03}
    ir["camera"] = {"kind": "perspective", "fov": 60, "near": 0.1,
                    "far": 300, "position": [0, 2.2, 12],
                    "target": [0, 2.2, 0]}
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0x3A5A8C", "intensity": 0.5,
                         "ground": "0x05070C"})
    ir["lights"].append({"id": "key", "kind": "spot",
                         "color": "0xBFD9FF", "intensity": 4.0,
                         "position": [0, 9, 4], "distance": 40,
                         "angle": 0.6, "penumbra": 0.5})
    for i, z in enumerate((-8, 0, 8)):
        ir["lights"].append({"id": f"strip_light_{i}", "kind": "point",
                             "color": "0x33E0FF", "intensity": 1.2,
                             "position": [-5.5, 3.4, z], "distance": 14,
                             "decay": 2})
    ir["nodes"].append(node(
        "floor", kind="mesh",
        geometry=geometry("PlaneGeometry",
                          {"w": 16, "h": 40, "ws": 1, "hs": 1}),
        material=material("standard", color="0x0C1119",
                          roughness=0.25, metalness=0.8),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    for i, z in enumerate((-8, 0, 8)):
        ir["nodes"].append(node(
            f"arch_{i}", kind="mesh",
            geometry=geometry("TorusGeometry",
                              {"r": 5.2, "t": 0.35, "rs": 12, "ts": 40}),
            material=material("standard", color="0x1A2433",
                              roughness=0.5, metalness=0.7),
            transform=transform(position=[0, 0.4, z])))
        ir["nodes"].append(node(
            f"strip_{i}", kind="mesh",
            geometry=geometry("BoxGeometry",
                              {"w": 0.18, "h": 0.18, "d": 6.0}),
            material=material("basic", color="0x33E0FF"),
            transform=transform(position=[-5.4, 3.4, z])))
    ir["nodes"].append(node(
        "core", kind="mesh",
        geometry=geometry("SphereGeometry", {"r": 1.1, "ws": 32, "hs": 32}),
        material=material("standard", color="0x66D9FF",
                          roughness=0.2, metalness=0.1,
                          emissive="0x1A5A77", emissiveIntensity=1.2),
        transform=transform(position=[0, 2.4, -12])))
    ir["animations"].append({"target": "core", "kind": "bob",
                             "amplitude": 0.25, "frequency": 0.9})
    ir["animations"].append({"target": "strip_light_1", "kind": "flicker",
                             "base": 1.2, "amplitude": 0.5,
                             "frequency": 6.0})
    return _finish(ir)


# 6. Procedural city.
def city(seed=66, blocks=8, seed_height=14):
    """Seeded city blocks: one InstancedMesh of towers with a canvas window
    texture, ground grid, depth fog. Window lights via emissive map."""
    ir = new_scene("city", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x0E1626", "mid": "0x2C3E58",
        "bottom": "0x05080E"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0x2C3E58",
                                "density": 0.016}
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 600, "position": [34, 22, 40],
                    "target": [0, 6, 0]}
    ir["lights"].append({"id": "moon", "kind": "directional",
                         "color": "0x9FB8DD", "intensity": 0.7,
                         "position": [-20, 30, 10]})
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0x2C3E58", "intensity": 0.5,
                         "ground": "0x05080E"})
    ir["nodes"].append(node(
        "ground", kind="mesh",
        geometry=geometry("PlaneGeometry",
                          {"w": 120, "h": 120, "ws": 1, "hs": 1}),
        material=material("standard", color="0x0A0E16",
                          roughness=0.9, metalness=0.1),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    n = blocks * blocks
    ir["nodes"].append(node(
        "towers", kind="instanced",
        geometry=geometry("BoxGeometry", {"w": 1, "h": 1, "d": 1}),
        material=material("standard", color="0x1B2740",
                          roughness=0.6, metalness=0.3,
                          emissive="0xFFC873", emissiveIntensity=0.55),
        count=n,
        distribution={"shape": "box", "hx": blocks * 3.2,
                      "hz": blocks * 3.2, "y": seed_height / 2,
                      "scale_jitter": 0.0}))
    ir["nodes"][1]["distribution"]["height_var"] = seed_height
    return _finish(ir)


# 7. Water and reflective surfaces.
def water(seed=77, size=60):
    """Reflective lake: PMREM procedural environment, metalness-high water
    plane, drifting mist points, low sun. No external HDRI needed."""
    ir = new_scene("water", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x2C4A6E", "mid": "0xCF9A5A",
        "bottom": "0x1A2030"}
    ir["environment"]["environment_map"] = {
        "kind": "procedural_equirect", "sky": "0x2C4A6E",
        "horizon": "0xCF9A5A", "ground": "0x1A2030"}
    ir["environment"]["fog"] = {"kind": "exp2", "color": "0xCF9A5A",
                                "density": 0.008}
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 600, "position": [16, 6, 22], "target": [0, 0, 0]}
    ir["lights"].append({"id": "sun", "kind": "directional",
                         "color": "0xFFD9A0", "intensity": 1.8,
                         "position": [-14, 7, -18]})
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0xCF9A5A", "intensity": 0.5,
                         "ground": "0x1A2030"})
    ir["nodes"].append(node(
        "lake", kind="mesh",
        geometry=geometry("PlaneGeometry",
                          {"w": size, "h": size, "ws": 1, "hs": 1}),
        material=material("standard", color="0x1E3A55",
                          roughness=0.08, metalness=0.9,
                          envMapIntensity=1.2),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    ir["nodes"].append(node(
        "mist", kind="points",
        geometry=procedural("particles", {"count": 800}),
        material=material("points", color="0xE8D9C0", size=0.5),
        count=800, distribution={"shape": "box", "hx": 28, "hz": 28}))
    return _finish(ir)


# 8. Animated particle environment.
def particles(seed=88, count=4000):
    """Seeded starfield/ember cloud with slow drift + twinkle via two
    counter-rotating Points systems. Bounded by budgets."""
    ir = new_scene("particles", seed=seed)
    ir["environment"]["background"] = {"kind": "color", "color": "0x05070E"}
    ir["camera"] = {"kind": "perspective", "fov": 60, "near": 0.1,
                    "far": 400, "position": [0, 4, 26], "target": [0, 4, 0]}
    ir["lights"].append({"id": "fill", "kind": "ambient",
                         "color": "0x334455", "intensity": 0.8})
    ir["nodes"].append(node(
        "embers", kind="points",
        geometry=procedural("particles", {"count": count}),
        material=material("points", color="0xFFB066", size=0.14),
        count=count, distribution={"shape": "box", "hx": 24, "hz": 24}))
    ir["nodes"].append(node(
        "stars", kind="points",
        geometry=procedural("particles", {"count": count // 2}),
        material=material("points", color="0x9FC0FF", size=0.1),
        count=count // 2,
        distribution={"shape": "box", "hx": 60, "hz": 60}))
    ir["animations"].append({"target": "embers", "kind": "rotate",
                             "speed": 0.02})
    ir["animations"].append({"target": "stars", "kind": "rotate",
                             "speed": -0.008})
    return _finish(ir)


# 9. Complex product visualization.
def product(seed=99):
    """Turntable product shot: torus-knot hero, pedestal, 3-point studio
    lighting, environment reflections, slow rotation."""
    ir = new_scene("product", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x3A4A63", "mid": "0x1C2333",
        "bottom": "0x0B0E16"}
    ir["environment"]["environment_map"] = {
        "kind": "procedural_equirect", "sky": "0x3A4A63",
        "horizon": "0x8FA0C0", "ground": "0x0B0E16"}
    ir["camera"] = {"kind": "perspective", "fov": 42, "near": 0.1,
                    "far": 200, "position": [4.5, 2.8, 6.5],
                    "target": [0, 1.2, 0]}
    _sun(ir)
    ir["lights"].append({"id": "rim", "kind": "directional",
                         "color": "0x88AAFF", "intensity": 1.0,
                         "position": [-5, 3, -6]})
    ir["nodes"].append(node(
        "floor", kind="mesh",
        geometry=geometry("CircleGeometry", {"r": 10, "seg": 64}),
        material=material("standard", color="0x141A26",
                          roughness=0.4, metalness=0.4),
        transform=transform(rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    ir["nodes"].append(node(
        "pedestal", kind="mesh",
        geometry=geometry("CylinderGeometry",
                          {"rt": 1.0, "rb": 1.2, "h": 0.5, "seg": 48}),
        material=material("standard", color="0x222B3D",
                          roughness=0.3, metalness=0.6),
        transform=transform(position=[0, 0.25, 0]),
        cast_shadow=True, receive_shadow=True, printable=True))
    ir["nodes"].append(node(
        "hero", kind="mesh",
        geometry={"ctor": "TorusKnotGeometry",
                  "params": {"p": 2, "q": 3, "r": 0.62, "t": 0.2}},
        material=material("standard", color="0xD8B25A",
                          roughness=0.25, metalness=0.9),
        transform=transform(position=[0, 1.6, 0]),
        cast_shadow=True))
    ir["animations"].append({"target": "hero", "kind": "rotate",
                             "speed": 0.5})
    return _finish(ir)


# 10. Reference-inspired fog benchmark (original implementation).
#
# Benchmark against webgpu_custom_fog_background (three.js dev):
#   MATCHED      perspective hero subject, fog color 0x4080cc with the
#                reference near/far (2.7/4), ACES tone mapping, procedural
#                environment lighting, orbit-style composed camera.
#   APPROXIMATED post-processing TSL fog graph -> built-in linear Fog +
#                ACESFilmicToneMapping + sRGB output (same visual intent on
#                WebGL); HDRI environment -> procedural equirect + PMREM;
#                GLTF helmet -> original procedural hero (visor sphere +
#                shell + rim); OrbitControls -> fixed composed camera in
#                the offline harness (CDN controls in UI scene.html).
#   UNSUPPORTED  WebGPURenderer, TSL nodes (pass/rangeFogFactor),
#                RenderPipeline, UltraHDRLoader/GLTFLoader external assets
#                — all rejected loudly by the knowledge index on r152.
FOG_BENCHMARK_NOTES = {
    "reference": "three.js webgpu_custom_fog_background (dev)",
    "matched": ["perspective hero subject", "fog color 0x4080cc",
                "fog range 2.7/4", "ACES tone mapping",
                "procedural environment lighting", "composed camera"],
    "approximated": ["TSL post fog graph -> built-in linear Fog",
                     "HDRI -> procedural equirect + PMREM",
                     "GLTF helmet -> original procedural hero",
                     "OrbitControls -> fixed camera offline / CDN in UI"],
    "unsupported": ["WebGPURenderer", "TSL nodes", "RenderPipeline",
                    "external HDRI/GLTF loaders"],
}


def fog_benchmark(seed=152):
    """Original fog-benchmark scene: procedural hero (shell + visor + rim),
    linear fog 0x4080cc range 2.7/4, ACES, PMREM environment, backdrop
    pillars for depth readout. Deterministic (seeded)."""
    ir = new_scene("fog_benchmark", seed=seed)
    ir["environment"]["background"] = {"kind": "color", "color": "0x4080cc"}
    ir["environment"]["fog"] = {"kind": "linear", "color": "0x4080cc",
                                "near": 2.7, "far": 4.0}
    ir["environment"]["environment_map"] = {
        "kind": "procedural_equirect", "sky": "0x6E9BD6",
        "horizon": "0xCFE0F2", "ground": "0x2A3547"}
    ir["camera"] = {"kind": "perspective", "fov": 45, "near": 0.25,
                    "far": 20, "position": [-1.8, 0.6, 2.7],
                    "target": [0, -0.1, -0.2]}
    ir["lights"].append({"id": "key", "kind": "directional",
                         "color": "0xFFFFFF", "intensity": 1.6,
                         "position": [3, 5, 4]})
    ir["lights"].append({"id": "sky", "kind": "hemisphere",
                         "color": "0xBDD7F2", "intensity": 0.8,
                         "ground": "0x2A3547"})
    ir["nodes"].append(node(
        "shell", kind="mesh",
        geometry=geometry("SphereGeometry", {"r": 0.55, "ws": 48, "hs": 32}),
        material=material("standard", color="0x8A8F98",
                          roughness=0.45, metalness=0.75),
        transform=transform(position=[0, 0.1, -0.2], scale=[1, 1.15, 1]),
        cast_shadow=True))
    ir["nodes"].append(node(
        "visor", kind="mesh",
        geometry=geometry("SphereGeometry", {"r": 0.42, "ws": 48, "hs": 32}),
        material=material("standard", color="0x0E141C",
                          roughness=0.08, metalness=0.9),
        transform=transform(position=[0, 0.08, 0.12],
                            scale=[0.85, 0.62, 0.7])))
    ir["nodes"].append(node(
        "rim", kind="mesh",
        geometry=geometry("TorusGeometry",
                          {"r": 0.58, "t": 0.06, "rs": 12, "ts": 64}),
        material=material("standard", color="0x3A4048",
                          roughness=0.5, metalness=0.8),
        transform=transform(position=[0, -0.28, -0.2],
                            rotation=[1.5707963, 0, 0])))
    for i, x in enumerate((-1.6, 1.6)):
        ir["nodes"].append(node(
            f"pillar_{i}", kind="mesh",
            geometry=geometry("BoxGeometry",
                              {"w": 0.5, "h": 3.2, "d": 0.5}),
            material=material("standard", color="0x4A5568",
                              roughness=0.8, metalness=0.1),
            transform=transform(position=[x, 0.4, -2.2])))
    ir["nodes"].append(node(
        "floor", kind="mesh",
        geometry=geometry("PlaneGeometry",
                          {"w": 14, "h": 14, "ws": 1, "hs": 1}),
        material=material("standard", color="0x2E3A4E",
                          roughness=0.9, metalness=0.0),
        transform=transform(position=[0, -0.62, -1],
                            rotation=[-1.5707963, 0, 0]),
        receive_shadow=True))
    ir["animations"].append({"target": "shell", "kind": "rotate",
                             "speed": 0.25})
    return _finish(ir)


# 11. Custom-fog landscape benchmark: seeded terrain + instanced
# vegetation + custom height fog (the WebGL analogue of the TSL custom-fog
# landscape treatment). The height-fog injection is exercised in-browser.
def fog_landscape(seed=171, trees=800):
    """Terrain + instanced pines + height fog. Deterministic (seeded)."""
    ir = new_scene("fog_landscape", seed=seed)
    ir["environment"]["background"] = {
        "kind": "gradient", "top": "0x2C4258", "mid": "0xA8BECF",
        "bottom": "0x1C2634"}
    ir["environment"]["fog"] = {"kind": "custom", "color": "0xA8BECF",
                                "density": 0.035, "height_falloff": 0.12}
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 500, "position": [24, 13, 30], "target": [0, 2, 0]}
    _sun(ir, bounds=40)
    ir["nodes"].append(node(
        "terrain", kind="mesh",
        geometry=procedural("terrain", {"size": 80, "segments": 96,
                                        "height": 8}),
        material=material("standard", color="0x55663F",
                          roughness=0.95, metalness=0.0),
        receive_shadow=True))
    dist = {"shape": "disc", "radius": 30, "clearing": 5.0,
            "scale_jitter": 0.4, "ground": "terrain", "lift": 1.1}
    ir["nodes"].append(node(
        "pines", kind="instanced",
        geometry=geometry("ConeGeometry", {"r": 0.9, "h": 2.8, "seg": 8}),
        material=material("standard", color="0x2E5230",
                          roughness=0.9, metalness=0.0),
        count=trees, distribution=dist, cast_shadow=True))
    ir["nodes"].append(node(
        "sky", kind="mesh",
        geometry=procedural("gradient_sky",
                            {"top": "0x2C4258", "mid": "0xA8BECF",
                             "bot": "0x1C2634"}),
        material=material("basic", color="0xFFFFFF", fog=False)))
    return _finish(ir)


def build(name, **params):
    """Build a named template scene (KeyError on unknown names — never a
    silent substitution)."""
    fns = {"studio": studio, "mountains": mountains, "forest": forest,
           "sky_fog": sky_fog, "scifi": scifi, "city": city, "water": water,
           "particles": particles, "product": product,
           "fog_benchmark": fog_benchmark, "fog_landscape": fog_landscape}
    if name not in fns:
        raise KeyError(f"unknown scene template {name!r} "
                       f"(available: {sorted(fns)})")
    return fns[name](**params)
