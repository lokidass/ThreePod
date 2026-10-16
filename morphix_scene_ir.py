# morphix_scene_ir.py
# Scene-mode intermediate representation: the validated, serializable plan
# for a RICH interactive Three.js scene. This is deliberately SEPARATE from
# the printer Object IR (morphix_ir.py):
#
#   printer IR  -> canonical solids -> horizontal slicing -> deposition paths
#   scene IR    -> full scene graph  -> Three.js module  -> browser render
#
# A scene node is either printable solid geometry (a small, explicit subset
# of constructors with printable=True) or a non-printable scene effect
# (fog, particles, shaders, backgrounds, instancing beyond the build
# volume). The two are NEVER conflated: printable_nodes() is the only set
# the printer pipeline may ever consume, and every effect node carries
# printable=False by construction.
#
# The IR is JSON-serializable, strictly validated (validate_scene_ir returns
# error strings, never guesses), deterministic (procedural placement uses the
# declared integer seed), and reproducible (same IR -> same JS).

import copy
import json
import math

IR_VERSION = 1

# Node kinds the compiler knows how to emit.
VALID_NODE_KINDS = ("group", "mesh", "instanced", "points")

# Material kinds the compiler knows how to emit (r152 core only).
VALID_MATERIAL_KINDS = ("standard", "basic", "points", "shader")

# Light kinds (r152 core).
VALID_LIGHT_KINDS = ("ambient", "hemisphere", "directional", "point",
                     "spot")

# Fog kinds. "custom" is a documented onBeforeCompile injection (WebGL);
# TSL/post-processing fog graphs are WebGPU-only and rejected on r152.
VALID_FOG_KINDS = ("none", "linear", "exp2", "custom")

# Render backends.
VALID_BACKENDS = ("webgl", "webgpu")

# Geometry constructors that are ALSO printer-compatible solids when used
# with straightforward standard materials (the printer isolation contract).
PRINTABLE_GEOMETRY = ("BoxGeometry", "SphereGeometry", "CylinderGeometry",
                      "ConeGeometry", "TorusGeometry", "PlaneGeometry")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _is_vec3(v):
    return isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)


def _valid_id(cid):
    if not isinstance(cid, str) or not cid or len(cid) > 64:
        return False
    if not (cid[0].isalpha() or cid[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in cid)


# ------------------------------------------------------------------ builders
def transform(position=None, rotation=None, scale=None):
    """A validated transform dict (defaults: origin, no rotation, unit)."""
    t = {"position": list(position) if position else [0, 0, 0],
         "rotation": list(rotation) if rotation else [0, 0, 0],
         "scale": list(scale) if scale else [1, 1, 1]}
    return t


def material(kind="standard", color="0xFFFFFF", **kw):
    """A material definition. Extra keys depend on kind (roughness,
    metalness, size, transparent, opacity, emissive, shaders...)."""
    m = {"kind": kind, "color": color}
    m.update(kw)
    return m


def geometry(ctor, params=None):
    """A geometry constructor + parameters, e.g. BoxGeometry [w,h,d]."""
    return {"ctor": ctor, "params": dict(params or {})}


def procedural(generator, params=None):
    """A deterministic procedural generator (terrain, forest, city, ...).
    The compiler expands it with the IR seed; placement is reproducible."""
    return {"generator": generator, "params": dict(params or {})}


VALID_RELATIONS = ("on", "near", "inside", "above")

def node(nid, kind="mesh", geometry=None, material=None, transform=None,
         children=None, count=0, distribution=None, cast_shadow=False,
         receive_shadow=False, printable=False, required=False):
    """One scene-graph node. Groups carry children; instanced/points carry
    count + distribution; meshes carry geometry + material. ``required``
    marks elements the completion gate must see rendered (optional decor
    defaults False)."""
    return {
        "id": nid,
        "kind": kind,
        "geometry": geometry,
        "material": material,
        "transform": transform or globals()["transform"](),
        "children": list(children or []),
        "count": int(count),
        "distribution": dict(distribution or {}),
        "cast_shadow": bool(cast_shadow),
        "receive_shadow": bool(receive_shadow),
        "printable": bool(printable),
        "required": bool(required),
    }


def new_scene(name, seed=1, backend="webgl", three_revision="152"):
    """An empty scene IR with budgets and defaults filled in."""
    return {
        "ir_version": IR_VERSION,
        "kind": "scene",
        "meta": {"name": name, "seed": int(seed), "backend": backend,
                 "three_revision": str(three_revision), "mode": "scene"},
        "environment": {"background": {"kind": "color", "color": "0x111622"},
                        "fog": {"kind": "none"},
                        "environment_map": None},
        "camera": {"kind": "perspective", "fov": 55, "near": 0.1,
                   "far": 500, "position": [6, 5, 9], "target": [0, 0, 0]},
        "lights": [],
        "nodes": [],
        "relations": [],
        "effects": [],
        "animations": [],
        "interactions": [],
        "budgets": {"max_nodes": 512, "max_instances": 200000,
                    "max_particles": 60000, "max_lights": 8},
    }


# ------------------------------------------------------------------ queries
def printable_nodes(ir):
    """IDs the printer may consume: printable=True AND a printer-compatible
    solid constructor. Everything else (fog, particles, shaders, sky,
    instanced crowds) is a scene effect, never deposition geometry."""
    out = []
    for n in ir.get("nodes", []):
        if not isinstance(n, dict) or not n.get("printable"):
            continue
        g = n.get("geometry") or {}
        if g.get("ctor") in PRINTABLE_GEOMETRY:
            out.append(n.get("id"))
    return out


def instance_total(ir):
    """Total instanced/particle count (for budgets + run reports)."""
    total = 0
    for n in ir.get("nodes", []):
        if isinstance(n, dict) and n.get("kind") in ("instanced", "points"):
            try:
                total += max(0, int(n.get("count", 0)))
            except (TypeError, ValueError):
                pass
    return total


# ------------------------------------------------------------------ validate
def validate_scene_ir(ir):
    """Strict schema + semantic checks. Returns [error strings] ([] = valid).
    Never mutates, never guesses."""
    errs = []
    if not isinstance(ir, dict):
        return ["scene IR must be a JSON object"]
    for key in ("ir_version", "kind", "meta", "camera", "lights", "nodes",
                "budgets"):
        if key not in ir:
            errs.append(f"scene IR missing required field {key!r}")
    if errs:
        return errs
    if ir.get("ir_version") != IR_VERSION:
        errs.append(f"unsupported scene IR version {ir.get('ir_version')!r}")
    if ir.get("kind") != "scene":
        errs.append(f"scene IR kind must be 'scene' "
                    f"(got {ir.get('kind')!r})")
    meta = ir.get("meta") or {}
    if meta.get("backend") not in VALID_BACKENDS:
        errs.append(f"meta.backend must be one of {list(VALID_BACKENDS)} "
                    f"(got {meta.get('backend')!r})")
    if not isinstance(meta.get("seed"), int):
        errs.append("meta.seed must be an integer (deterministic placement)")
    if meta.get("backend") == "webgpu":
        errs.append("backend 'webgpu' requires a Three.js build with "
                    "WebGPURenderer/TSL support (vendored "
                    f"{meta.get('three_revision')!r} has neither); use "
                    f"backend 'webgl' or a newer pinned build")
    if meta.get("mode") != "scene":
        errs.append(f"meta.mode must be 'scene' (got {meta.get('mode')!r}); "
                    f"printer plans use the printer Object IR, never this")

    cam = ir.get("camera") or {}
    if cam.get("kind") not in ("perspective", "orthographic"):
        errs.append(f"camera.kind must be perspective|orthographic "
                    f"(got {cam.get('kind')!r})")
    if not _is_vec3(cam.get("position")) or not _is_vec3(cam.get("target")):
        errs.append("camera.position/target must be [x,y,z] numbers")

    env = ir.get("environment") or {}
    fog = (env.get("fog") or {})
    if fog.get("kind") not in VALID_FOG_KINDS:
        errs.append(f"fog.kind must be one of {list(VALID_FOG_KINDS)} "
                    f"(got {fog.get('kind')!r})")
    bg = (env.get("background") or {})
    if bg.get("kind") not in ("color", "gradient"):
        errs.append(f"background.kind must be color|gradient "
                    f"(got {bg.get('kind')!r})")

    lights = ir.get("lights")
    if not isinstance(lights, list):
        errs.append("lights must be a list")
        lights = []
    budgets = ir.get("budgets") or {}
    if len(lights) > int(budgets.get("max_lights", 8)):
        errs.append(f"{len(lights)} lights exceed max_lights "
                    f"{budgets.get('max_lights')}")
    for i, L in enumerate(lights):
        tag = f"light[{i}]"
        if not isinstance(L, dict):
            errs.append(f"{tag} must be an object")
            continue
        if L.get("kind") not in VALID_LIGHT_KINDS:
            errs.append(f"{tag}: kind must be one of "
                        f"{list(VALID_LIGHT_KINDS)} (got {L.get('kind')!r})")
        if not _valid_id(L.get("id", "")):
            errs.append(f"{tag}: invalid id (snake_case, <=64 chars)")

    nodes = ir.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errs.append("nodes must be a non-empty list")
        return errs
    if len(nodes) > int(budgets.get("max_nodes", 512)):
        errs.append(f"{len(nodes)} nodes exceed max_nodes "
                    f"{budgets.get('max_nodes')}")
    seen = set()
    for i, n in enumerate(nodes):
        tag = f"node[{i}]"
        if not isinstance(n, dict):
            errs.append(f"{tag} must be an object")
            continue
        cid = n.get("id", "")
        tag = f"node {cid!r}" if cid else tag
        if not _valid_id(cid):
            errs.append(f"{tag}: invalid id (snake_case, <=64 chars)")
        elif cid in seen:
            errs.append(f"{tag}: duplicate id")
        seen.add(cid)
        if n.get("kind") not in VALID_NODE_KINDS:
            errs.append(f"{tag}: kind must be one of "
                        f"{list(VALID_NODE_KINDS)} (got {n.get('kind')!r})")
        t = n.get("transform") or {}
        for f in ("position", "rotation", "scale"):
            if not _is_vec3(t.get(f)):
                errs.append(f"{tag}: transform.{f} must be [x,y,z] numbers")
        g = n.get("geometry")
        if n.get("kind") == "group":
            if g is not None:
                errs.append(f"{tag}: groups carry children, not geometry")
        else:
            if not isinstance(g, dict) or \
                    ("ctor" not in g and "generator" not in g):
                errs.append(f"{tag}: geometry needs ctor or generator")
        m = n.get("material")
        if n.get("kind") != "group":
            if not isinstance(m, dict) or \
                    m.get("kind") not in VALID_MATERIAL_KINDS:
                errs.append(f"{tag}: material.kind must be one of "
                            f"{list(VALID_MATERIAL_KINDS)}")
        if n.get("kind") in ("instanced", "points"):
            try:
                c = int(n.get("count", 0))
            except (TypeError, ValueError):
                c = -1
            if c <= 0:
                errs.append(f"{tag}: instanced/points need count > 0")
        kids = n.get("children")
        if not isinstance(kids, list) or \
                not all(isinstance(k, str) for k in kids):
            errs.append(f"{tag}: children must be a list of ids")
        if n.get("printable") and (g or {}).get("generator"):
            errs.append(f"{tag}: procedural generators are scene effects, "
                        f"never printable deposition geometry")
    ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        for k in n.get("children", []):
            if k not in ids:
                errs.append(f"node {n.get('id')!r}: unknown child {k!r}")
            if k == n.get("id"):
                errs.append(f"node {n.get('id')!r}: self-child cycle")
    light_ids = {L.get("id") for L in lights if isinstance(L, dict)}
    for i, r in enumerate(ir.get("relations", [])):
        if not isinstance(r, dict):
            errs.append(f"relation[{i}] must be an object")
            continue
        if r.get("rel") not in VALID_RELATIONS:
            errs.append(f"relation[{i}]: rel must be one of "
                        f"{list(VALID_RELATIONS)} (got {r.get('rel')!r})")
        for side in ("a", "b"):
            if r.get(side) not in ids:
                errs.append(f"relation[{i}]: unknown node {side}="
                            f"{r.get(side)!r}")
        try:
            tol = float(r.get("tol", 0.15))
        except (TypeError, ValueError):
            tol = float("nan")
        if not (tol == tol and tol >= 0):
            errs.append(f"relation[{i}]: tol must be a number >= 0")
    for i, a in enumerate(ir.get("animations", [])):
        if not isinstance(a, dict):
            errs.append(f"animation[{i}] must be an object")
            continue
        if a.get("kind") not in ("rotate", "bob", "flicker"):
            errs.append(f"animation[{i}]: unknown kind {a.get('kind')!r} "
                        f"(tested: rotate, bob, flicker)")
        if a.get("target") not in ids | light_ids:
            errs.append(f"animation[{i}]: target {a.get('target')!r} names "
                        f"neither a node nor a light")
    total = instance_total(ir)
    cap = int(budgets.get("max_instances", 200000)) + \
        int(budgets.get("max_particles", 60000))
    if total > cap:
        errs.append(f"instance total {total} exceeds budgets ({cap})")
    return errs


# ------------------------------------------------------------------ ser/de
def ir_to_json(ir):
    """Deterministic serialization (sorted keys) for reports + replay."""
    return json.dumps(ir, indent=2, sort_keys=True)


def ir_from_json(text):
    """Parse + deep-copy a scene IR (validation is a separate step)."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("scene IR must be a JSON object")
    return copy.deepcopy(obj)


def summarize(ir):
    """Compact, UI-friendly view for reports and the frontend."""
    return {
        "name": (ir.get("meta") or {}).get("name"),
        "backend": (ir.get("meta") or {}).get("backend"),
        "seed": (ir.get("meta") or {}).get("seed"),
        "nodes": len(ir.get("nodes", [])),
        "lights": len(ir.get("lights", [])),
        "instances": instance_total(ir),
        "printable": printable_nodes(ir),
        "required": [n.get("id") for n in ir.get("nodes", [])
                     if isinstance(n, dict) and n.get("required")],
        "relations": len(ir.get("relations", [])),
        "animations": len(ir.get("animations", [])),
        "effects": [e.get("kind") for e in ir.get("effects", [])
                    if isinstance(e, dict)],
    }
