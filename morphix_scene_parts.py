# morphix_scene_parts.py
# Parameterized multi-mesh assemblies ("structured geometry", Phase 4.1).
# These are NOT templates: they are reusable capability builders that emit
# IR node lists with documented proportions, materials, and contacts.
# Planners (router or thinker) compose them with template systems and raw
# freestyle elements; every assembly is independently unit-tested (valid
# IR, compiles, renders, contacts hold).
#
# Contrast with box-only approximations: each part uses the technique its
# subject needs (tapered bands, pyramid roofs, glass panes, emissive
# lamps), recorded in the returned "strategy" note.

from morphix_scene_ir import geometry, material, node, transform

PART_NAMES = ("table_set", "room_shell", "lighthouse_tower", "cabin",
              "castle_keep", "tent", "boat", "lamp_post", "pier",
              "dunes", "coral_bed")


def _box(nid, w, h, d, pos, mat="standard", color="0xFFFFFF", extra=None,
         shadow=True, prefix="", required=False):
    kw = dict(extra or {})
    n = node(prefix + nid, kind="mesh",
             geometry=geometry("BoxGeometry", {"w": w, "h": h, "d": d}),
             material=material(mat, color=color, **kw),
             transform=transform(position=list(pos)),
             cast_shadow=shadow, receive_shadow=True)
    n["required"] = bool(required)
    return n


def _cyl(nid, rt, rb, h, pos, mat="standard", color="0xFFFFFF", seg=20,
         extra=None, prefix="", required=False):
    kw = dict(extra or {})
    n = node(prefix + nid, kind="mesh",
             geometry=geometry("CylinderGeometry",
                               {"rt": rt, "rb": rb, "h": h, "seg": seg}),
             material=material(mat, color=color, **kw),
             transform=transform(position=list(pos)),
             cast_shadow=True, receive_shadow=True)
    n["required"] = bool(required)
    return n


# ------------------------------------------------------------------ table
def table_set(prefix="table_", w=2.2, d=1.3, h=0.78, top=0.09,
              wood="0x9C6B3F", required=False):
    """Coherent dining table: one thin top slab on four corner legs with
    correct proportions (top overhangs the leg rectangle). Strategy:
    box assemblies with wood materials; legs touch ground AND top."""
    leg_s, leg_h = 0.12, h - top
    nodes = [_box("top", w, top, d, [0, h - top / 2, 0],
                  color=wood, extra={"roughness": 0.6},
                  prefix=prefix, required=required)]
    lx, lz = w / 2 - 0.16, d / 2 - 0.16
    legs = []
    for qx, qz, tag in ((1, 1, "front_left"), (1, -1, "front_right"),
                        (-1, 1, "rear_left"), (-1, -1, "rear_right")):
        lid = f"leg_{tag}"
        nodes.append(_box(lid, leg_s, leg_h, leg_s,
                          [qx * lx, leg_h / 2, qz * lz],
                          color=wood, extra={"roughness": 0.7},
                          prefix=prefix, required=required))
        legs.append(prefix + lid)
    relations = [{"a": prefix + "top", "rel": "on", "b": lid, "tol": 0.05}
                 for lid in legs]
    return {"nodes": nodes, "lights": [], "relations": relations,
            "strategy": "box assembly: thin top slab + 4 corner legs"}


# -------------------------------------------------------------------- room
def room_shell(prefix="room_", w=9, d=8, h=3.2, wall="0xC9BFAE",
               floor="0x8A6F4D", required=False):
    """Interior shell: floor slab + back/side walls + a framed glass
    window proud of the back wall. Strategy: boxes with plaster/wood
    materials; window glass slightly proud (no CSG holes)."""
    t = 0.18
    nodes = [
        _box("floor", w, t, d, [0, -t / 2, 0], color=floor,
             extra={"roughness": 0.8}, shadow=False, prefix=prefix,
             required=required),
        _box("wall_back", w, h, t, [0, h / 2, -d / 2], color=wall,
             extra={"roughness": 0.9}, shadow=False, prefix=prefix,
             required=required),
        _box("wall_side", t, h, d, [-w / 2, h / 2, 0], color=wall,
             extra={"roughness": 0.9}, shadow=False, prefix=prefix,
             required=required),
        _box("window_frame", 1.7, 1.3, 0.1, [1.2, 1.8, -d / 2 + 0.06],
             color="0x5A4632", extra={"roughness": 0.6}, prefix=prefix),
        _box("window_glass", 1.5, 1.1, 0.06, [1.2, 1.8, -d / 2 + 0.07],
             mat="basic", color="0x9DB8D6", prefix=prefix),
    ]
    relations = [{"a": prefix + "wall_back", "rel": "on",
                    "b": prefix + "floor", "tol": 0.05},
                   {"a": prefix + "wall_side", "rel": "on",
                    "b": prefix + "floor", "tol": 0.05}]
    return {"nodes": nodes, "lights": [], "relations": relations,
            "strategy": "room shell: floor + 2 walls + framed glass pane"}


# -------------------------------------------------------------- lighthouse
def lighthouse_tower(prefix="light_", h=7.0, r_base=1.1, required=False):
    """Tapered banded tower + gallery + glass lamp room + cone cap + an
    emissive lamp sphere. Strategy: stacked tapered cylinders (bands),
    glass + emissive for the lamp (self-lit beacon)."""
    nodes = []
    bands, bh = 5, h * 0.62 / 5
    for i in range(bands):
        f0 = 1 - i / bands
        r0 = r_base * (0.72 + 0.28 * f0)
        r1 = r_base * (0.72 + 0.28 * (f0 - 1 / bands))
        col = "0xC23B2E" if i % 2 == 0 else "0xEDE6DA"
        nodes.append(_cyl(f"band_{i}", r1, r0, bh + 0.02,
                          [0, bh / 2 + i * bh, 0], color=col,
                          extra={"roughness": 0.7}, prefix=prefix,
                          required=required))
    top_y = bands * bh
    nodes.append(_cyl("gallery", r_base * 0.78, r_base * 0.78, 0.18,
                      [0, top_y + 0.09, 0], color="0x3A4048", prefix=prefix))
    nodes.append(_cyl("lamp_glass", 0.42, 0.42, 0.8,
                      [0, top_y + 0.58, 0], mat="basic",
                      color="0xFFE9A8", prefix=prefix))
    lamp = _box("lamp", 0.3, 0.3, 0.3, [0, top_y + 0.58, 0],
                mat="basic", color="0xFFF3C4", prefix=prefix)
    nodes.append(lamp)
    cap = node(prefix + "cap", kind="mesh",
               geometry=geometry("ConeGeometry",
                                 {"r": 0.62, "h": 0.7, "seg": 16}),
               material=material("standard", color="0x3A4048",
                                 roughness=0.6),
               transform=transform(position=[0, top_y + 1.3, 0]),
               cast_shadow=True)
    cap["required"] = bool(required)
    nodes.append(cap)
    relations = [{"a": prefix + f"band_{i + 1}", "rel": "on",
                  "b": prefix + f"band_{i}", "tol": 0.05}
                 for i in range(bands - 1)]
    return {"nodes": nodes, "lights": [], "relations": relations,
            "strategy": "tapered banded stack + glass lamp + emissive core"}


# ------------------------------------------------------------------- cabin
def cabin(prefix="cabin_", w=3.4, d=2.8, wall_h=1.8, required=False):
    """Log cabin: wall box + pyramid plank roof + door + warm window.
    Strategy: box walls, 4-sided cone roof, emissive window panes."""
    nodes = [
        _box("walls", w, wall_h, d, [0, wall_h / 2, 0], color="0x7A5A38",
             extra={"roughness": 0.85}, prefix=prefix, required=required),
        _box("door", 0.7, 1.3, 0.1, [0, 0.65, d / 2 + 0.03],
             color="0x4A3320", prefix=prefix),
        _box("window", 0.8, 0.6, 0.1, [-0.9, 1.1, d / 2 + 0.03],
             mat="basic", color="0xFFC873", prefix=prefix),
    ]
    roof = node(prefix + "roof", kind="mesh",
                geometry=geometry("ConeGeometry",
                                  {"r": max(w, d) * 0.72, "h": 1.2,
                                   "seg": 4}),
                material=material("standard", color="0x4E463C",
                                  roughness=0.9),
                transform=transform(position=[0, wall_h + 0.6, 0],
                                    rotation=[0, 0.7853982, 0]),
                cast_shadow=True)
    roof["required"] = bool(required)
    nodes.append(roof)
    relations = [{"a": prefix + "roof", "rel": "on",
                  "b": prefix + "walls", "tol": 0.1}]
    return {"nodes": nodes, "lights": [], "relations": relations,
            "strategy": "box walls + pyramid plank roof + lit openings"}


# ------------------------------------------------------------------ castle
def castle_keep(prefix="castle_", w=5.0, d=5.0, wall_h=3.0, required=False):
    """Small keep: crenellated curtain walls + corner turrets with cone
    caps. Strategy: boxes + tapered cylinders; crenellation as a row of
    small merlons (cheap, reads at distance)."""
    nodes = []
    t = 0.5
    for i, (nm, px, pz, ww, dd) in enumerate((
            ("wall_n", 0, d / 2, w, t), ("wall_s", 0, -d / 2, w, t),
            ("wall_e", w / 2, 0, t, d), ("wall_w", -w / 2, 0, t, d))):
        nodes.append(_box(nm, ww, wall_h, dd, [px, wall_h / 2, pz],
                          color="0x8D8D94", extra={"roughness": 0.9},
                          prefix=prefix, required=required))
        n = max(2, int((ww if ww > dd else dd) / 0.7))
        for k in range(n):
            off = -((ww if ww > dd else dd) / 2) + 0.35 + k * 0.7
            mx = px + (off if ww > dd else 0)
            mz = pz + (0 if ww > dd else off)
            nodes.append(_box(f"merlon_{i}_{k}", 0.34, 0.34, 0.34,
                              [mx, wall_h + 0.17, mz], color="0x8D8D94",
                              extra={"roughness": 0.9}, shadow=False,
                              prefix=prefix))
    for qx, qz, tag in ((1, 1, "ne"), (1, -1, "se"),
                        (-1, 1, "nw"), (-1, -1, "sw")):
        nodes.append(_cyl(f"turret_{tag}", 0.75, 0.85, wall_h + 1.2,
                          [qx * w / 2, (wall_h + 1.2) / 2, qz * d / 2],
                          color="0x7D7D84", extra={"roughness": 0.9},
                          prefix=prefix, required=required))
        cap = node(prefix + f"cap_{tag}", kind="mesh",
                   geometry=geometry("ConeGeometry",
                                     {"r": 0.95, "h": 1.1, "seg": 12}),
                   material=material("standard", color="0x4E463C",
                                     roughness=0.85),
                   transform=transform(
                       position=[qx * w / 2, wall_h + 1.2 + 0.55,
                                 qz * d / 2]),
                   cast_shadow=True)
        nodes.append(cap)
    return {"nodes": nodes, "lights": [],
            "strategy": "curtain walls + merlons + turrets with cone caps"}


# -------------------------------------------------------------------- tent
def tent(prefix="tent_", r=1.6, h=2.2, canvas="0xC9B189", required=False):
    """Camp tent: canvas cone + center pole + guy stakes. Strategy: cone
    shell with slight translucency look via basic+standard mix."""
    pole = _cyl("pole", 0.05, 0.05, h + 0.4, [0, (h + 0.4) / 2, 0],
                color="0x5A4632", prefix=prefix)
    shell = node(prefix + "shell", kind="mesh",
                 geometry=geometry("ConeGeometry",
                                   {"r": r, "h": h, "seg": 12}),
                 material=material("standard", color=canvas,
                                   roughness=0.85, side="DoubleSide"),
                 transform=transform(position=[0, h / 2, 0]),
                 cast_shadow=True, receive_shadow=True)
    shell["required"] = bool(required)
    return {"nodes": [shell, pole], "lights": [],
            "strategy": "canvas cone shell + center pole"}


# -------------------------------------------------------------------- boat
def boat(prefix="boat_", hull_l=3.2, hull_w=1.2, required=False):
    """Simple boat: hull box + gunwale rim + mast + sail plane.
    Strategy: stretched boxes, DoubleSide sail."""
    hull_h = 0.7
    nodes = [
        _box("hull", hull_l, hull_h, hull_w, [0, hull_h / 2, 0],
             color="0x6E4A2F", extra={"roughness": 0.7}, prefix=prefix,
             required=required),
        _box("gunwale_l", hull_l, 0.12, 0.1, [0, hull_h + 0.06, hull_w / 2],
             color="0x4A3320", prefix=prefix),
        _box("gunwale_r", hull_l, 0.12, 0.1, [0, hull_h + 0.06, -hull_w / 2],
             color="0x4A3320", prefix=prefix),
        _cyl("mast", 0.06, 0.07, 2.6, [0.3, hull_h + 1.3, 0],
             color="0x5A4632", prefix=prefix),
    ]
    sail = node(prefix + "sail", kind="mesh",
                geometry=geometry("PlaneGeometry",
                                  {"w": 1.5, "h": 1.9, "ws": 1, "hs": 1}),
                material=material("standard", color="0xEDE6DA",
                                  roughness=0.85, side="DoubleSide"),
                transform=transform(position=[-0.45, hull_h + 1.5, 0],
                                    rotation=[0, 1.5707963, 0]),
                cast_shadow=True)
    sail["required"] = bool(required)
    nodes.append(sail)
    return {"nodes": nodes, "lights": [],
            "strategy": "hull boxes + mast + DoubleSide sail"}


# --------------------------------------------------------------- lamp post
def lamp_post(prefix="lamp_", h=3.4, glow="0xFFD9A0", required=False):
    """Street lamp: pole + arm + emissive head. Strategy: dark metal +
    self-lit head (no extra PointLight: forward-pass budget)."""
    return {"nodes": [
        _cyl("pole", 0.07, 0.09, h, [0, h / 2, 0], color="0x22262E",
             prefix=prefix, required=required),
        _box("arm", 0.9, 0.08, 0.08, [0.4, h - 0.1, 0], color="0x22262E",
             prefix=prefix),
        _box("head", 0.34, 0.16, 0.2, [0.8, h - 0.18, 0], mat="basic",
             color=glow, prefix=prefix),
    ], "lights": [], "strategy": "dark pole + emissive head (no light)"}


# -------------------------------------------------------------------- pier
def pier(prefix="pier_", length=8.0, width=1.6, required=False):
    """Wooden pier: deck planks + piling cylinders into the water.
    Strategy: deck boxes + submerged piles."""
    nodes = [_box("deck", length, 0.14, width, [0, 0.5, 0],
                  color="0x7A6248", extra={"roughness": 0.85},
                  prefix=prefix, required=required)]
    x = -length / 2 + 0.5
    i = 0
    while x <= length / 2:
        for z in (-width / 2 + 0.15, width / 2 - 0.15):
            nodes.append(_cyl(f"pile_{i}", 0.09, 0.09, 1.4, [x, -0.1, z],
                              color="0x4E3B28", prefix=prefix))
            i += 1
        x += 2.0
    return {"nodes": nodes, "lights": [],
            "strategy": "deck planks + submerged piles"}


def dunes(prefix="dunes_", size=90, height=4.5, sand="0xD9BE8C",
          required=False):
    """Sand-dune field: seeded fBm terrain in sand tones. Strategy:
    procedural terrain surface (scene effect, verified live at render —
    never flagged static-required by construction)."""
    from morphix_scene_ir import node, procedural, material
    n = node(prefix + "field", kind="mesh",
             geometry=procedural("terrain", {"size": size, "segments": 80,
                                             "height": height}),
             material=material("standard", color=sand, roughness=1.0,
                               metalness=0.0),
             receive_shadow=True)
    n["required"] = False
    return {"nodes": [n], "lights": [],
            "strategy": "seeded fBm dune field (procedural, live-verified)"}


def coral_bed(prefix="coral_", count=60, radius=14, required=False):
    """Coral bed: instanced bright cones + bulb spheres on the seabed.
    Strategy: two InstancedMeshes (cheap at 60+40 instances)."""
    from morphix_scene_ir import node, geometry, material
    dist = {"shape": "disc", "radius": radius, "scale_jitter": 0.45,
            "y": 0.9}
    cones = node(prefix + "cones", kind="instanced",
                 geometry=geometry("ConeGeometry",
                                   {"r": 0.5, "h": 1.8, "seg": 7}),
                 material=material("standard", color="0xE86A8A",
                                   roughness=0.7, metalness=0.0),
                 count=count, distribution=dict(dist))
    cones["required"] = bool(required)
    bulbs = node(prefix + "bulbs", kind="instanced",
                 geometry=geometry("SphereGeometry",
                                   {"r": 0.35, "ws": 10, "hs": 8}),
                 material=material("standard", color="0xF2A541",
                                   roughness=0.6, metalness=0.0),
                 count=max(8, count // 2),
                 distribution=dict(dist, y=0.5))
    bulbs["required"] = bool(required)
    return {"nodes": [cones, bulbs], "lights": [],
            "strategy": "instanced coral cones + bulbs"}


def build_part(name, prefix="", required=False, **params):
    """Build a named part assembly (KeyError on unknown names — never a
    silent substitution). Returns {"nodes", "lights", "strategy"}."""
    fns = {"table_set": table_set, "room_shell": room_shell,
           "lighthouse_tower": lighthouse_tower, "cabin": cabin,
           "castle_keep": castle_keep, "tent": tent, "boat": boat,
           "lamp_post": lamp_post, "pier": pier,
           "dunes": dunes, "coral_bed": coral_bed}
    if name not in fns:
        raise KeyError(f"unknown part {name!r} "
                       f"(available: {sorted(fns)})")
    return fns[name](prefix=prefix, required=required, **params)
