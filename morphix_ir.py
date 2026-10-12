# morphix_ir.py
# Canonical Object IR — the one validated intermediate representation shared by
# text-to-3D and image-to-3D. Both sources must produce this structure before
# anything is sliced into printer paths:
#
#   text prompt ─┐
#                ├─► Object IR (validated) ─► printer primitives ─► layer slicing
#   photo ───────┘
#
# The IR is deliberately small and strict: a category, a coordinate convention,
# a flat list of components (each with a primitive, dimensions, position,
# material, parent, observed/confidence), documented assumptions, and the
# source of the reconstruction. The builder NEVER changes component dimensions
# while generating segments: the IR is frozen, converted to printer primitives
# once, and the deterministic slicer owns everything after that.
#
# Honesty rules:
#   - unknown categories are allowed but carry category + low confidence;
#   - templates are deterministic (no model): the same category always yields
#     the same component set and stable IDs;
#   - `observed` is True only when a component was seen in the input; assumed
#     or occluded geometry is marked False and recorded in `assumptions`.

import math

from morphix_path import MAX_LINEAR, MATERIAL_COLORS, BOUND_XZ, MAX_Y
from morphix_specs import _box_of, _cyl_of, _sphere_of, _role_of

IR_VERSION = 1
VALID_PRIMITIVES = ("box", "cylinder", "sphere")
VALID_SOURCES = ("text", "object-template", "image-vision", "image-manual",
                 "image-demo")

DEFAULT_COLORS = {"body": "0xFFAA33", "accent": "0x44CCFF",
                  "blue": "0x2A6FDB", "support": "0x888888",
                  "glass": "0x99DDFF", "tire": "0x222222",
                  "wood": "0x9C6B3F"}

_LABEL_RE = None  # lazy (re)use the validator's rule without importing it
_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


# ---------------------------------------------------------------- building
def _comp(cid, primitive, dims, position, material, color=None, role=None,
          parent=None, observed=True, confidence=1.0, constraints=None,
          taper=None):
    """One IR component with a stable ID. taper=[radiusTop, radiusBottom]
    preserves cone/spike profiles through conversions (slicer is
    taper-aware); omitted tapers print as straight cylinders."""
    comp = {
        "id": cid,
        "role": role if role is not None else _role_of(cid),
        "primitive": primitive,
        "dims": [round(float(v), 4) for v in dims],
        "position": [round(float(v), 4) for v in position],
        "material": material,
        "color": color or DEFAULT_COLORS.get(material, "0x888888"),
        "parent": parent,
        "observed": bool(observed),
        "confidence": round(float(confidence), 3),
        "constraints": list(constraints or []),
    }
    if taper is not None:
        comp["taper"] = [round(float(taper[0]), 4),
                         round(float(taper[1]), 4)]
    return comp


def _object_bbox(components: list) -> list:
    """Axis-aligned [w, h, d] of all components (centered primitives)."""
    if not components:
        return [0.0, 0.0, 0.0]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for c in components:
        w, h, d = c["dims"]
        pos = c["position"]
        for i, half in enumerate((w / 2.0, h / 2.0, d / 2.0)):
            lo[i] = min(lo[i], pos[i] - half)
            hi[i] = max(hi[i], pos[i] + half)
    return [round(hi[i] - lo[i], 4) for i in range(3)]


def _wrap(category: str, source: str, components: list, assumptions=None,
          confidence=0.9, observed=None) -> dict:
    return {
        "ir_version": IR_VERSION,
        "category": category,
        "source": source,
        "units": "scene",
        "up": "y",
        "front": "+x",
        "confidence": round(float(confidence), 3),
        "assumptions": list(assumptions or []),
        "observed": dict(observed or {}),
        "components": components,
        "constraints": [
            "y is up; the object rests on the ground plane y=0",
            "components are converted to printer primitives exactly once",
        ],
        "bbox": _object_bbox(components),
    }


# ---------------------------------------------------------------- validation
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _valid_id(cid) -> bool:
    if not isinstance(cid, str) or not cid or len(cid) > 40:
        return False
    if not (cid[0].isalpha() or cid[0] == "_"):
        return False
    return all(ch in _ID_CHARS for ch in cid)


def validate_ir(ir: dict) -> list:
    """Strict schema + geometric sanity. Returns a list of error strings
    ([] means the IR may be sliced). Never mutates and never guesses."""
    errs = []
    if not isinstance(ir, dict):
        return ["IR must be a JSON object"]
    for key in ("category", "source", "units", "up", "components", "bbox"):
        if key not in ir:
            errs.append(f"IR missing required field {key!r}")
    if errs:
        return errs
    if not isinstance(ir["category"], str) or not ir["category"].strip():
        errs.append("IR.category must be a non-empty string")
    if ir["source"] not in VALID_SOURCES:
        errs.append(f"IR.source must be one of {list(VALID_SOURCES)} "
                    f"(got {ir['source']!r})")
    if ir["units"] != "scene":
        errs.append(f"IR.units must be 'scene' (got {ir['units']!r})")
    if ir["up"] != "y":
        errs.append(f"IR.up must be 'y' (got {ir['up']!r})")
    comps = ir["components"]
    if not isinstance(comps, list) or not comps:
        errs.append("IR.components must be a non-empty list")
        return errs
    seen = set()
    for i, c in enumerate(comps):
        tag = f"component[{i}]"
        if not isinstance(c, dict):
            errs.append(f"{tag} must be an object")
            continue
        cid = c.get("id", "")
        tag = f"component {cid!r}" if cid else tag
        if not _valid_id(cid):
            errs.append(f"{tag}: invalid id (snake_case letters/digits/"
                        f"underscore, <=40 chars)")
        elif cid in seen:
            errs.append(f"{tag}: duplicate id")
        seen.add(cid)
        prim = c.get("primitive")
        if prim not in VALID_PRIMITIVES:
            errs.append(f"{tag}: primitive must be one of "
                        f"{list(VALID_PRIMITIVES)} (got {prim!r})")
        dims = c.get("dims")
        if not (isinstance(dims, list) and len(dims) == 3
                and all(_is_num(v) for v in dims)):
            errs.append(f"{tag}: dims must be [w,h,d] numbers")
        else:
            for name, v in zip(("w", "h", "d"), dims):
                if v <= 0:
                    errs.append(f"{tag}: dims.{name} must be > 0 (got {v})")
                elif v > MAX_LINEAR:
                    errs.append(f"{tag}: dims.{name}={v} exceeds the "
                                f"{MAX_LINEAR} scene limit")
            if prim == "cylinder" and abs(dims[0] - dims[2]) > 1e-3:
                errs.append(f"{tag}: cylinder footprint must be circular "
                            f"(w==d, got {dims[0]}x{dims[2]})")
            taper = c.get("taper")
            if taper is not None:
                if not (isinstance(taper, list) and len(taper) == 2
                        and all(_is_num(v) for v in taper)
                        and all(v > 0 for v in taper)):
                    errs.append(f"{tag}: taper must be [radiusTop, "
                                f"radiusBottom] positive numbers "
                                f"(got {taper!r})")
            if prim == "sphere" and not (
                    abs(dims[0] - dims[1]) <= 1e-3
                    and abs(dims[1] - dims[2]) <= 1e-3):
                errs.append(f"{tag}: sphere dims must be equal diameter "
                            f"(got {dims[0]}x{dims[1]}x{dims[2]})")
        pos = c.get("position")
        if not (isinstance(pos, list) and len(pos) == 3
                and all(_is_num(v) for v in pos)):
            errs.append(f"{tag}: position must be [x,y,z] numbers")
        else:
            if abs(pos[0]) > BOUND_XZ or abs(pos[2]) > BOUND_XZ:
                errs.append(f"{tag}: position {pos} out of bounds "
                            f"(+/-{BOUND_XZ})")
            if not (0 <= pos[1] <= MAX_Y):
                errs.append(f"{tag}: position y={pos[1]} out of range "
                            f"[0,{MAX_Y}]")
        if c.get("material") not in MATERIAL_COLORS:
            errs.append(f"{tag}: material must be one of "
                        f"{sorted(MATERIAL_COLORS)} (got {c.get('material')!r})")
        color = c.get("color")
        if not (isinstance(color, str) and len(color) == 8
                and color[:2] == "0x"
                and all(ch in "0123456789abcdefABCDEF" for ch in color[2:])):
            errs.append(f"{tag}: color must be 0xRRGGBB (got {color!r})")
        parent = c.get("parent")
        if parent is not None and parent not in seen:
            errs.append(f"{tag}: parent {parent!r} must reference an earlier "
                        f"component")
    if isinstance(ir.get("bbox"), list) and len(ir["bbox"]) == 3:
        calc = _object_bbox([c for c in comps if isinstance(c, dict)
                             and isinstance(c.get("dims"), list)])
        if all(_is_num(v) for v in calc):
            for a, b in zip(ir["bbox"], calc):
                if abs(a - b) > 0.5:
                    errs.append(f"IR.bbox {ir['bbox']} does not match "
                                f"component extents {calc}")
                    break
    return errs


# ---------------------------------------------------------------- conversion
def ir_to_actions(ir: dict) -> list:
    """Freeze the IR into printer primitives (add_box/add_cylinder/
    add_sphere). This is the ONLY translation step; the slicer never sees the
    IR again."""
    out = []
    for c in ir["components"]:
        act = {"label": c["id"], "position": list(c["position"]),
               "material": c["material"],
               "color": c.get("color")
               or DEFAULT_COLORS.get(c["material"], "0x888888")}
        w, h, d = c["dims"]
        if c["primitive"] == "box":
            act.update({"action": "add_box", "width": w, "height": h,
                        "depth": d})
        elif c["primitive"] == "sphere":
            act.update({"action": "add_sphere", "radius": round(w / 2.0, 4)})
        else:
            taper = c.get("taper")
            if isinstance(taper, (list, tuple)) and len(taper) == 2:
                # tapered spike/cone: profile preserved, not flattened
                act.update({"action": "add_cylinder",
                            "radiusTop": taper[0], "radiusBottom": taper[1],
                            "height": h})
            else:
                r = round(max(w, d) / 2.0, 4)
                act.update({"action": "add_cylinder", "radiusTop": r,
                            "radiusBottom": r, "height": h})
        out.append(act)
    return out


def actions_to_ir(actions: list, category: str, source: str = "text",
                  assumptions=None, confidence=0.7) -> dict:
    """Normalize a thought/repaired plan into the canonical IR (used for
    reporting text-to-3D). Invalid primitives are skipped with a note."""
    comps = []
    notes = list(assumptions or [])
    for a in actions:
        if not isinstance(a, dict):
            continue
        b = _box_of(a)
        c = _cyl_of(a)
        s = _sphere_of(a)
        label = a.get("label", "")
        if b:
            _cx, _cy, _cz, w, h, d = b
            comps.append(_comp(label, "box", [w, h, d], [_cx, _cy, _cz],
                               a.get("material", "body"), a.get("color"),
                               parent=a.get("on_top_of")))
        elif c:
            _cx, _cy, _cz, r, h = c
            try:
                rt = float(a.get("radiusTop", a.get("radius", r)))
                rb = float(a.get("radiusBottom", a.get("radius", r)))
            except (TypeError, ValueError):
                rt = rb = r
            comp = _comp(label, "cylinder", [2 * r, h, 2 * r],
                         [_cx, _cy, _cz], a.get("material", "body"),
                         a.get("color"), parent=a.get("on_top_of"))
            if abs(rt - rb) > 1e-9:
                # tapered profile (ears, tails, cones): record it so the
                # freeze step below preserves the spike, not a peg
                comp["taper"] = [round(rt, 4), round(rb, 4)]
            comps.append(comp)
        elif s:
            _cx, _cy, _cz, r = s
            comps.append(_comp(label, "sphere", [2 * r, 2 * r, 2 * r],
                               [_cx, _cy, _cz], a.get("material", "body"),
                               a.get("color"), parent=a.get("on_top_of")))
        else:
            notes.append(f"{label!r}: unsliceable primitive omitted from IR")
    return _wrap(category, source, comps, assumptions=notes,
                 confidence=confidence)


# ---------------------------------------------------------------- templates
def _car_template() -> dict:
    """Deterministic canonical car: body, raised cabin, windshield, two side
    windows, four distinct corner wheels, two headlights. Stable IDs; +x is
    the front, +y is up; wheels are vertical cylinders (the only axis the
    horizontal slicer supports), spread along both axes."""
    comps = []
    wr, wh, wy = 0.35, 0.5, 0.25          # wheel radius / height / centre y
    bx, by, bz = 0.0, 0.8, 0.0            # body centre (base y=0.5 on wheels)
    bw, bh, bd = 3.6, 0.6, 1.8
    for cid, x, z in (("wheel_front_left", 1.3, 0.75),
                      ("wheel_front_right", 1.3, -0.75),
                      ("wheel_rear_left", -1.3, 0.75),
                      ("wheel_rear_right", -1.3, -0.75)):
        comps.append(_comp(cid, "cylinder", [2 * wr, wh, 2 * wr], [x, wy, z],
                           "tire", role="wheel",
                           constraints=["axis:+y", "grounded"]))
    comps.append(_comp("car_body", "box", [bw, bh, bd], [bx, by, bz],
                       "body", role="body",
                       constraints=["rests_on:wheels"]))
    body_top = by + bh / 2.0
    cw, ch, cd = 1.8, 0.55, 1.6
    cy = body_top + ch / 2.0
    comps.append(_comp("cabin", "box", [cw, ch, cd], [bx, cy, bz], "accent",
                       role="cabin", parent="car_body",
                       constraints=["seated_on:car_body"]))
    comps.append(_comp("windshield", "box", [0.12, 0.4, 1.5],
                       [round(bx + cw / 2, 4), cy, bz], "glass",
                       role="windshield", parent="cabin",
                       constraints=["front_face:+x"]))
    comps.append(_comp("side_window_left", "box", [1.3, 0.3, 0.12],
                       [bx, cy, round(bz + cd / 2 + 0.01, 4)], "glass",
                       role="window", parent="cabin",
                       constraints=["side_face:+z"]))
    comps.append(_comp("side_window_right", "box", [1.3, 0.3, 0.12],
                       [bx, cy, round(bz - cd / 2 - 0.01, 4)], "glass",
                       role="window", parent="cabin",
                       constraints=["side_face:-z"]))
    for cid, z in (("headlight_left", 0.55), ("headlight_right", -0.55)):
        comps.append(_comp(cid, "box", [0.15, 0.2, 0.35],
                           [round(bx + bw / 2 - 0.08, 4),
                            round(by + 0.05, 4), z], "accent", role="light",
                           parent="car_body"))
    return _wrap("car", "object-template", comps,
                 assumptions=["wheels print as vertical cylinders (printer "
                              "axis convention)",
                              "hidden underside and far side are symmetric "
                              "assumptions"],
                 confidence=0.95,
                 observed={"category": True, "geometry": False})


def _chair_template() -> dict:
    """Deterministic canonical chair: one horizontal seat, one backrest
    rising from the seat rear (-X face), four grounded legs at the seat
    corners. Stable IDs; +x is the front, +y is up. No car components."""
    comps = []
    sw, sh, sd = 1.1, 0.12, 1.1
    sx, sy, sz = 0.0, 0.51, 0.0
    seat_top = sy + sh / 2.0
    comps.append(_comp("seat", "box", [sw, sh, sd], [sx, sy, sz], "wood",
                       role="seat", constraints=["horizontal", "supported"]))
    bw, bh, bd = 0.12, 0.9, 1.0
    bx = sx - sw / 2 + 0.06
    by = seat_top + bh / 2.0
    comps.append(_comp("backrest", "box", [bw, bh, bd], [bx, by, sz],
                       "wood", role="backrest", parent="seat",
                       constraints=["rear_face:-x", "upright",
                                    "rises_from:seat"]))
    lr, lh = 0.07, 0.45
    for cid, qx, qz in (("leg_front_left", 1, 1),
                        ("leg_front_right", 1, -1),
                        ("leg_rear_left", -1, 1),
                        ("leg_rear_right", -1, -1)):
        comps.append(_comp(cid, "cylinder", [2 * lr, lh, 2 * lr],
                           [sx + qx * (sw / 2 - 0.1), lh / 2,
                            sz + qz * (sd / 2 - 0.1)],
                           "wood", role="leg", parent="seat",
                           constraints=["axis:+y", "grounded"]))
    return _wrap("chair", "object-template", comps,
                 assumptions=["rear and underside assumed symmetric; "
                              "four-leg design (no casters, no armrests)",
                              "hidden underside and far side are symmetric "
                              "assumptions"],
                 confidence=0.9,
                 observed={"category": True, "geometry": False})


def _stool_template() -> dict:
    """Deterministic canonical stool: one horizontal seat on four grounded
    legs. No backrest (that would be a chair). Stable IDs; +y is up."""
    comps = []
    sw, sh, sd = 1.0, 0.12, 1.0
    sx, sy, sz = 0.0, 0.51, 0.0
    comps.append(_comp("seat", "box", [sw, sh, sd], [sx, sy, sz], "wood",
                       role="seat", constraints=["horizontal", "supported"]))
    lr, lh = 0.07, 0.45
    for cid, qx, qz in (("leg_front_left", 1, 1),
                        ("leg_front_right", 1, -1),
                        ("leg_rear_left", -1, 1),
                        ("leg_rear_right", -1, -1)):
        comps.append(_comp(cid, "cylinder", [2 * lr, lh, 2 * lr],
                           [sx + qx * (sw / 2 - 0.1), lh / 2,
                            sz + qz * (sd / 2 - 0.1)],
                           "wood", role="leg", parent="seat",
                           constraints=["axis:+y", "grounded"]))
    return _wrap("stool", "object-template", comps,
                 assumptions=["four-leg design; no backrest by definition",
                              "hidden underside and far side are symmetric "
                              "assumptions"],
                 confidence=0.9,
                 observed={"category": True, "geometry": False})


def _cat_template() -> dict:
    """Deterministic canonical sitting cat: body box, tapered neck grafting
    a head sphere above the body front, two tapered ears on the head top,
    four tapered grounded legs, tapered tail touching the body rear flank.
    Stable IDs; +x is the front, +y is up. No car components. Every contact
    is an exact touch (shared faces, zero interpenetration volume) so the
    as-built mesh checks pass on the canonical design itself."""
    comps = []
    bw, bh, bd = 1.3, 0.7, 0.7
    bx, by, bz = 0.0, 0.75, 0.0
    body_top = by + bh / 2
    comps.append(_comp("body", "box", [bw, bh, bd], [bx, by, bz], "body",
                       role="body", constraints=["supported"]))
    hr = 0.36
    neck_h, neck_rb, neck_rt = 0.3, 0.26, 0.2
    # 2-decimal derivation, identical to repair_cat_plan synthesis (0.45):
    # the head assembly must land on the same coordinates from either path.
    nx = round(bx + bw * 0.35, 2)
    comps.append(_comp("neck", "cylinder", [2 * neck_rb, neck_h, 2 * neck_rb],
                       [nx, round(body_top + neck_h / 2, 4), bz], "body",
                       role="neck", parent="body",
                       constraints=["axis:+y", "touches:body"],
                       taper=[neck_rt, neck_rb]))
    neck_top = body_top + neck_h
    hx, hy, hz = nx, round(neck_top + hr, 4), bz
    comps.append(_comp("head", "sphere", [2 * hr, 2 * hr, 2 * hr],
                       [hx, hy, hz], "body", role="head", parent="neck",
                       constraints=["touches:neck"]))
    er, eh = 0.12, 0.28
    ey = round(hy + hr + eh / 2, 4)
    for cid, sz in (("ear_left", 1), ("ear_right", -1)):
        comps.append(_comp(cid, "cylinder", [2 * er, eh, 2 * er],
                           [hx, ey, hz + sz * 0.16], "body",
                           role="ear", parent="head",
                           constraints=["axis:+y", "grounded:false",
                                        "touches:head"],
                           taper=[0.03, er]))
    lr_top, lr_bot, lh = 0.07, 0.10, 0.4
    for cid, qx, qz in (("leg_front_left", 1, 1),
                        ("leg_front_right", 1, -1),
                        ("leg_rear_left", -1, 1),
                        ("leg_rear_right", -1, -1)):
        comps.append(_comp(cid, "cylinder", [2 * lr_bot, lh, 2 * lr_bot],
                           [bx + qx * (bw / 2 - 0.15), lh / 2,
                            bz + qz * (bd / 2 - 0.1)],
                           "body", role="leg", parent="body",
                           constraints=["axis:+y", "grounded", "tapered"],
                           taper=[lr_top, lr_bot]))
    comps.append(_comp("tail", "cylinder", [0.18, 1.2, 0.18],
                       # exact touch: tail bottom radius 0.09 meets the rear
                       # face (same -0.74 as repair synthesis + fixtures).
                       [round(bx - bw / 2 - 0.09, 2), 0.6,
                        round(bz + 0.25, 4)],
                       "body", role="tail",
                       constraints=["axis:+y", "grounded", "tapered",
                                    "touches:body"],
                       taper=[0.04, 0.09]))
    return _wrap("cat", "object-template", comps,
                 assumptions=["sitting pose: neck grafts head above body "
                              "front, ears and tail tapered per profile, "
                              "contacts are exact face touches",
                              "hidden rear, far side and underside are "
                              "symmetric assumptions"],
                 confidence=0.9,
                 observed={"category": True, "geometry": False})


def _house_template() -> dict:
    comps = []
    fw, fh, fd = 4.0, 0.4, 3.2
    comps.append(_comp("foundation", "box", [fw, fh, fd], [0, fh / 2, 0],
                       "body", role="foundation",
                       constraints=["grounded"]))
    ww, wh, wd = 3.4, 1.6, 2.6
    wy = fh + wh / 2.0
    comps.append(_comp("walls", "box", [ww, wh, wd], [0, wy, 0], "body",
                       role="wall", parent="foundation",
                       constraints=["seated_on:foundation"]))
    comps.append(_comp("roof", "box", [3.8, 0.35, 3.0],
                       [0, round(fh + wh + 0.175, 4), 0], "accent",
                       role="roof", parent="walls",
                       constraints=["seated_on:walls"]))
    comps.append(_comp("door", "box", [0.6, 1.0, 0.1],
                       [0, fh + 0.5, round(fd / 2, 4)], "accent",
                       role="door", parent="walls",
                       constraints=["front_face:+z"]))
    for cid, x in (("window_left", -1.0), ("window_right", 1.0)):
        comps.append(_comp(cid, "box", [0.6, 0.5, 0.1],
                           [x, round(fh + 1.0, 4), round(fd / 2, 4)], "glass",
                           role="window", parent="walls",
                           constraints=["front_face:+z"]))
    return _wrap("house", "object-template", comps,
                 assumptions=["single-storey; roof is a flat cap"],
                 confidence=0.95,
                 observed={"category": True, "geometry": False})


def _generic_object_template() -> dict:
    """Documented fallback for an unrecognized category: one grounded base and
    one raised top — a real volumetric object, never a grid of strips."""
    comps = [
        _comp("object_base", "box", [1.6, 0.5, 1.6], [0, 0.25, 0], "body",
              constraints=["grounded"]),
        _comp("object_top", "box", [0.9, 0.7, 0.9], [0, 0.85, 0], "accent",
              parent="object_base", constraints=["seated_on:object_base"]),
    ]
    return _wrap("object", "object-template", comps,
                 assumptions=["category not recognized: a generic volumetric "
                              "object template is used"],
                 confidence=0.4,
                 observed={"category": False, "geometry": False})


def _table_template() -> dict:
    """Deterministic canonical table: one THIN rectangular tabletop resting on
    four separate corner legs. The tabletop is thin relative to its footprint
    (a table, never a thick block); every leg has the same length and touches
    the ground. Stable IDs: tabletop, leg_front_left, leg_front_right,
    leg_rear_left, leg_rear_right. +x is the length, +z the width; the
    material is wood-like. The four legs are listed before the tabletop so the
    slicer seats the top onto their supports deterministically."""
    tw, th, td = 2.0, 0.2, 1.2        # top width/length, thickness, depth
    leg_h, leg_s = 1.0, 0.2           # leg height and square cross-section
    ex = round(tw / 2 - leg_s / 2 - 0.05, 4)
    ez = round(td / 2 - leg_s / 2 - 0.05, 4)
    comps = []
    for cid, x, z, where in (
            ("leg_front_left", ex, ez, "front-left"),
            ("leg_front_right", ex, -ez, "front-right"),
            ("leg_rear_left", -ex, ez, "rear-left"),
            ("leg_rear_right", -ex, -ez, "rear-right")):
        comps.append(_comp(cid, "box", [leg_s, leg_h, leg_s],
                           [x, round(leg_h / 2, 4), z], "wood", role="leg",
                           constraints=["grounded", f"at:{where} corner"]))
    comps.append(_comp("tabletop", "box", [tw, th, td],
                       [0, round(leg_h + th / 2, 4), 0], "wood",
                       role="tabletop",
                       constraints=["rests_on:legs", "thin:tabletop"]))
    return _wrap("table", "object-template", comps,
                 assumptions=["overall length follows the photo silhouette "
                              "aspect; board thickness is a fixed thin slab",
                              "the four legs are symmetric and assumed "
                              "wood-like"],
                 confidence=0.95,
                 observed={"category": True, "geometry": False})


def _urn_template() -> dict:
    """Deterministic canonical urn: a stepped ROUND body (base, swelling
    belly, shoulder, neck, flared rim) built from stacked cylinders. The
    silhouette is taller than it is wide — recognizably an urn, never a table
    or a generic block. Stable IDs: urn_base, urn_belly, urn_shoulder,
    urn_neck, urn_rim."""
    comps = []
    y = 0.0

    def add(cid, radius, height, parent=None, base=False, rim=False):
        nonlocal y
        cons = ["grounded"] if base else [f"rests_on:{parent}"]
        if rim:
            cons.append("flare:top")
        comps.append(_comp(cid, "cylinder", [2 * radius, height, 2 * radius],
                           [0, round(y + height / 2, 4), 0], "body",
                           role="urn", parent=parent, constraints=cons))
        y = round(y + height, 4)

    add("urn_base", 0.42, 0.2, base=True)
    add("urn_belly", 0.60, 0.6, parent="urn_base")
    add("urn_shoulder", 0.44, 0.35, parent="urn_belly")
    add("urn_neck", 0.26, 0.3, parent="urn_shoulder")
    add("urn_rim", 0.36, 0.15, parent="urn_neck", rim=True)
    return _wrap("urn", "object-template", comps,
                 assumptions=["the round body is approximated by stacked "
                              "cylinders (the printer's horizontal slicer)",
                              "the hidden far side is a symmetry assumption"],
                 confidence=0.9,
                 observed={"category": True, "geometry": False})


def _box_sphere_template() -> dict:
    """Deterministic canonical 'box with a sphere on top': one grounded blue
    box and one sphere resting concentrically on its top face. Both poles of
    the sphere land on layer planes so the slicer realises a real round solid
    (a cap bead at each pole), never a flat ring. Stable IDs: box, sphere."""
    bw, bh, bd = 1.4, 1.0, 1.4
    r = 0.6
    comps = [
        _comp("box", "box", [bw, bh, bd], [0, bh / 2, 0], "blue",
              color=DEFAULT_COLORS["blue"], role="box",
              constraints=["grounded"]),
        _comp("sphere", "sphere", [2 * r, 2 * r, 2 * r], [0, bh + r, 0],
              "accent", color=DEFAULT_COLORS["accent"], role="sphere",
              parent="box", constraints=["on_top_of:box", "aligned:+y"]),
    ]
    return _wrap("box_sphere", "object-template", comps,
                 assumptions=["the sphere rests concentrically on the box top",
                              "beads are one layer tall so courses fuse; the "
                              "sphere is sliced to its true circular cross-"
                              "section on every layer"],
                 confidence=0.95,
                 observed={"category": True, "geometry": False})


TEMPLATES = {"car": _car_template, "house": _house_template,
             "table": _table_template, "urn": _urn_template,
             "chair": _chair_template, "stool": _stool_template,
             "cat": _cat_template,
             "box_sphere": _box_sphere_template,
             "object": _generic_object_template}

# Category synonyms -> template key.
_CATEGORY_ALIASES = {
    "car": "car", "cars": "car", "vehicle": "car", "automobile": "car",
    "sedan": "car", "truck": "car", "van": "car", "suv": "car",
    "chair": "chair", "armchair": "chair",
    "house": "house", "home": "house", "building": "house", "cabin": "house",
    "hut": "house", "cottage": "house",
    "table": "table", "desk": "table", "workbench": "table",
    "counter": "table", "countertop": "table",
    "urn": "urn", "vase": "urn", "pot": "urn", "amphora": "urn",
    "planter": "urn",
    "stool": "stool", "footstool": "stool",
    "cat": "cat", "kitten": "cat", "feline": "cat",
    "box_sphere": "box_sphere",
}

# Deterministic text routing: a recognized canonical prompt (here a box AND a
# round word) gets its fixed template without consulting a model. Anything
# else returns None and follows the normal model/plan path.
_BOX_WORDS = ("box", "cube", "cuboid", "block")
_ROUND_WORDS = ("sphere", "ball", "orb", "globe")


def category_for_prompt(prompt: str):
    p = (prompt or "").lower()
    if any(w in p for w in _BOX_WORDS) and any(w in p for w in _ROUND_WORDS):
        return "box_sphere"
    return None


def ir_for_prompt(prompt: str):
    """Canonical IR for a supported prompt, or None (never invents)."""
    cat = category_for_prompt(prompt)
    return ir_for_category(cat) if cat else None


def template_key(category: str) -> str:
    c = (category or "").strip().lower()
    return _CATEGORY_ALIASES.get(c, "object")


# Categories that have a real deterministic geometry template. Image
# reconstruction prints these; any other detected category is reported to the
# user for an explicit choice instead of silently becoming a generic block.
SUPPORTED_CATEGORIES = tuple(sorted(k for k in TEMPLATES if k != "object"))


def ir_for_category(category: str) -> dict:
    """Deterministic template for a category (unknown -> generic object).
    The returned dict is a fresh, independent copy every call."""
    key = template_key(category)
    ir = TEMPLATES.get(key, _generic_object_template)()
    label = (category or "").strip().lower() or "object"
    ir["category"] = label if category else "object"
    ir["template"] = key
    if key == "object" and label not in ("object", "unknown"):
        # recognized word we have no template for: keep the true label and
        # record the substitution honestly
        ir["assumptions"] = list(ir["assumptions"]) + [
            f"no template for {label!r}: generic object template used "
            f"(documented demonstration geometry)"]
    return ir


# ---------------------------------------------------------------- reporting
def ir_summary(ir: dict) -> dict:
    """Compact, JSON-safe view for the UI/report (no giant numbers)."""
    return {
        "category": ir.get("category"),
        "source": ir.get("source"),
        "confidence": ir.get("confidence"),
        "bbox": ir.get("bbox"),
        "up": ir.get("up", "y"),
        "assumptions": list(ir.get("assumptions", [])),
        "observed": dict(ir.get("observed", {})),
        "components": [
            {"id": c.get("id"), "role": c.get("role"),
             "primitive": c.get("primitive"), "dims": c.get("dims"),
             "position": c.get("position"), "material": c.get("material"),
             "parent": c.get("parent"), "observed": c.get("observed"),
             "confidence": c.get("confidence")}
            for c in ir.get("components", []) if isinstance(c, dict)
        ],
    }


def component_hierarchy(ir: dict) -> list:
    """Parent -> children view for the UI (roots first, one row per node)."""
    comps = [c for c in ir.get("components", []) if isinstance(c, dict)]
    ids = {c.get("id") for c in comps}
    children = {}
    for c in comps:
        children.setdefault(c.get("parent"), []).append(c)

    nodes = []

    def walk(node, depth):
        nodes.append({"id": node.get("id"), "role": node.get("role"),
                      "material": node.get("material"), "depth": depth,
                      "observed": node.get("observed"), "dims": node.get("dims")})
        for child in children.get(node.get("id"), []):
            walk(child, depth + 1)

    for root in [c for c in comps if c.get("parent") not in ids]:
        walk(root, 0)
    # any component whose parent is missing/forward-referenced is still shown
    shown = {n["id"] for n in nodes}
    for c in comps:
        if c.get("id") not in shown:
            walk(c, 0)
    return nodes


# ---------------------------------------------------------------- fidelity
# How far the reconstructed silhouette aspect may differ from the measured
# photo aspect (recon/photo) before the fit is rejected. The vision category
# gate imports this band so "accepted by vision" and "passes fidelity" agree.
SILHOUETTE_ASPECT_BAND = (0.5, 2.0)


def _add_shape_checks(tkey: str, comps: list, bbox: list, check) -> None:
    """Category-specific checks that separate a plausible reconstruction from a
    block that merely passed syntax/geometry validation.

    ``tkey`` is the template key; ``check(name, ok, detail)`` records a result.
    Categories without a specific shape (car/house/generic) add no extra
    checks here — their fidelity comes from the shared checks plus verify_solids
    and the per-category plan spec."""
    def dims(c):
        return [float(v) for v in (c.get("dims") or [0, 0, 0])]

    def pos(c):
        return [float(v) for v in (c.get("position") or [0, 0, 0])]

    if tkey == "table":
        tops = [c for c in comps if str(c.get("role")) == "tabletop"]
        legs = [c for c in comps if str(c.get("role")) == "leg"]
        check("tabletop_count", len(tops) == 1,
              f"{len(tops)} tabletop component(s)")
        check("leg_count", len(legs) == 4, f"{len(legs)} leg component(s)")
        if len(tops) == 1:
            w, h, d = dims(tops[0])
            check("tabletop_thin", h <= 0.3 * min(w, d),
                  f"thickness {h:.2f} vs footprint {w:.2f}x{d:.2f}")
        if legs:
            heights = [dims(c)[1] for c in legs]
            check("leg_lengths_consistent",
                  (max(heights) - min(heights)) <= 0.05,
                  f"leg heights {[round(x, 2) for x in heights]}")
            check("legs_grounded",
                  all(abs(pos(c)[1] - dims(c)[1] / 2) <= 0.05 for c in legs),
                  "every leg base at y=0")
            cx, cz = (pos(tops[0])[0], pos(tops[0])[2]) if tops else (0.0, 0.0)
            combos = {(pos(c)[0] >= cx, pos(c)[2] >= cz) for c in legs}
            check("legs_at_corners", len(combos) == 4,
                  f"{len(combos)}/4 distinct tabletop corners occupied")
            if tops:
                th = dims(tops[0])[1]
                top_base = pos(tops[0])[1] - th / 2
                leg_top = max(pos(c)[1] + dims(c)[1] / 2 for c in legs)
                check("tabletop_supported", abs(top_base - leg_top) <= 0.1,
                      f"tabletop base {top_base:.2f} vs leg tops {leg_top:.2f}")
    elif tkey == "urn":
        check("urn_round", any(c.get("primitive") == "cylinder" for c in comps),
              "urn body built from round cylinders")
        check("urn_taller_than_wide", bbox[1] > bbox[0],
              f"height {bbox[1]:.2f} vs width {bbox[0]:.2f}")
        bases = [pos(c)[1] - dims(c)[1] / 2 for c in comps]
        check("urn_grounded", bool(bases) and min(bases) <= 0.05,
              f"lowest base y={min(bases):.2f}" if bases else "no components")
    elif tkey == "chair":
        # A recognizable chair: one horizontal seat, one rear upright
        # backrest, four grounded corner legs, and no car-only components.
        seats = [c for c in comps if str(c.get("role")) == "seat"]
        backs = [c for c in comps if str(c.get("role")) == "backrest"]
        legs = [c for c in comps if str(c.get("role")) == "leg"]
        check("chair_seat_present", len(seats) == 1,
              f"{len(seats)} seat component(s)")
        check("chair_backrest_present", len(backs) == 1,
              f"{len(backs)} backrest component(s)")
        check("chair_leg_count", len(legs) == 4,
              f"{len(legs)} leg component(s)")
        check("chair_no_car_parts",
              not [c for c in comps
                   if str(c.get("role")) in ("wheel", "cabin", "windshield")
                   or str(c.get("material")) == "tire"],
              "no car-only components in a chair")
        if len(seats) == 1:
            sw, sh, sd = dims(seats[0])
            sx, sy, sz = pos(seats[0])
            check("chair_seat_horizontal", sh <= 0.3 * min(sw, sd),
                  f"seat {sw:.2f}x{sh:.2f}x{sd:.2f}")
            seat_top = sy + sh / 2
            if len(backs) == 1:
                bw, bh, bd = dims(backs[0])
                bx, by, bz = pos(backs[0])
                check("chair_backrest_rear", bx < sx,
                      f"backrest x={bx:.2f} vs seat x={sx:.2f}")
                check("chair_backrest_upright", bw <= 0.4 * bh,
                      f"backrest {bw:.2f}x{bh:.2f}x{bd:.2f}")
                check("chair_backrest_rises",
                      abs((by - bh / 2) - seat_top) <= 0.2,
                      f"backrest base {by - bh / 2:.2f} vs seat top "
                      f"{seat_top:.2f}")
            if legs:
                check("chair_legs_grounded",
                      all(abs(pos(c)[1] - dims(c)[1] / 2) <= 0.1
                          for c in legs),
                      "every leg base at y=0")
                leg_tops = [pos(c)[1] + dims(c)[1] / 2 for c in legs]
                check("chair_seat_supported",
                      any(abs((sy - sh / 2) - t) <= 0.15 for t in leg_tops),
                      f"seat base {sy - sh / 2:.2f} vs leg tops")
                combos = {(pos(c)[0] >= sx, pos(c)[2] >= sz) for c in legs}
                check("chair_legs_at_corners", len(combos) == 4,
                      f"{len(combos)}/4 distinct seat corners occupied")
    elif tkey == "stool":
        # A stool: seat plus four grounded corner legs, no backrest, no
        # car parts. Shares the chair's seat/leg geometry, minus the back.
        seats = [c for c in comps if str(c.get("role")) == "seat"]
        legs = [c for c in comps if str(c.get("role")) == "leg"]
        check("stool_seat_present", len(seats) == 1,
              f"{len(seats)} seat component(s)")
        check("stool_no_backrest",
              not [c for c in comps if str(c.get("role")) == "backrest"],
              "a backrest makes it a chair, not a stool")
        check("stool_leg_count", len(legs) == 4,
              f"{len(legs)} leg component(s)")
        check("stool_no_car_parts",
              not [c for c in comps
                   if str(c.get("role")) in ("wheel", "cabin", "windshield")
                   or str(c.get("material")) == "tire"],
              "no car-only components in a stool")
        if len(seats) == 1:
            sw, sh, sd = dims(seats[0])
            sx, sy, sz = pos(seats[0])
            check("stool_seat_horizontal", sh <= 0.3 * min(sw, sd),
                  f"seat {sw:.2f}x{sh:.2f}x{sd:.2f}")
            if legs:
                check("stool_legs_grounded",
                      all(abs(pos(c)[1] - dims(c)[1] / 2) <= 0.1
                          for c in legs),
                      "every leg base at y=0")
                leg_tops = [pos(c)[1] + dims(c)[1] / 2 for c in legs]
                check("stool_seat_supported",
                      any(abs((sy - sh / 2) - t) <= 0.15 for t in leg_tops),
                      f"seat base {sy - sh / 2:.2f} vs leg tops")
                combos = {(pos(c)[0] >= sx, pos(c)[2] >= sz) for c in legs}
                check("stool_legs_at_corners", len(combos) == 4,
                      f"{len(combos)}/4 distinct seat corners occupied")
    elif tkey == "cat":
        # A recognizable sitting cat: body, head sphere on the body
        # front-top, two tapered ears on the head, four grounded legs, one
        # tapered tail at the rear — and no car parts.
        bodies = [c for c in comps if str(c.get("role")) == "body"]
        heads = [c for c in comps if str(c.get("role")) == "head"]
        ears = [c for c in comps if str(c.get("role")) == "ear"]
        legs = [c for c in comps if str(c.get("role")) == "leg"]
        tails = [c for c in comps if str(c.get("role")) == "tail"]
        check("cat_body_present", len(bodies) == 1,
              f"{len(bodies)} body component(s)")
        check("cat_head_present", len(heads) == 1,
              f"{len(heads)} head component(s)")
        check("cat_ear_count", len(ears) == 2,
              f"{len(ears)} ear component(s)")
        check("cat_leg_count", len(legs) == 4,
              f"{len(legs)} leg component(s)")
        check("cat_tail_present", len(tails) == 1,
              f"{len(tails)} tail component(s)")
        check("cat_no_car_parts",
              not [c for c in comps
                   if str(c.get("role")) in ("wheel", "cabin", "windshield")
                   or str(c.get("material")) == "tire"],
              "no car-only components in a cat")
        if len(bodies) == 1:
            bw, bh, bd = dims(bodies[0])
            bx, by, bz = pos(bodies[0])
            check("cat_body_substantial", bw >= 1.0 and bh >= 0.4,
                  f"body {bw:.2f}x{bh:.2f}x{bd:.2f}")
            necks = [c for c in comps if str(c.get("role")) == "neck"]
            if necks:
                nw, nh, nd = dims(necks[0])
                nx, ny, nz = pos(necks[0])
                check("cat_neck_seated",
                      abs((ny - nh / 2) - (by + bh / 2)) <= 0.15,
                      f"neck base {ny - nh / 2:.2f} vs body top "
                      f"{by + bh / 2:.2f}")
                check("cat_neck_tapered",
                      (necks[0].get("taper") or [1, 1])[0]
                      < (necks[0].get("taper") or [1, 1])[1],
                      "neck tapers upward like the ears and tail")
            necks = [c for c in comps if str(c.get("role")) == "neck"]
            if len(heads) == 1:
                hw, hh, hd = dims(heads[0])
                hx, hy, hz = pos(heads[0])
                check("cat_head_round", abs(hw - hh) <= 0.05
                      and abs(hh - hd) <= 0.05,
                      f"head {hw:.2f}x{hh:.2f}x{hd:.2f}")
                if necks:
                    nw, nh, nd = dims(necks[0])
                    nx, ny, nz = pos(necks[0])
                    check("cat_neck_seated",
                          abs((ny - nh / 2) - (by + bh / 2)) <= 0.15,
                          f"neck base {ny - nh / 2:.2f} vs body top "
                          f"{by + bh / 2:.2f}")
                    check("cat_head_on_neck",
                          abs((hy - hh / 2) - (ny + nh / 2)) <= 0.2,
                          f"head base {hy - hh / 2:.2f} vs neck top "
                          f"{ny + nh / 2:.2f}")
                else:
                    check("cat_head_placed",
                          abs((hy - hh / 2) - (by + bh / 2)) <= 0.2
                          and hx > bx,
                          f"head sits on the body front-top")
            if legs:
                check("cat_legs_grounded",
                      all(abs(pos(c)[1] - dims(c)[1] / 2) <= 0.1
                          for c in legs),
                      "every leg base at y=0")
                combos = {(pos(c)[0] >= bx, pos(c)[2] >= bz) for c in legs}
                check("cat_legs_at_corners", len(combos) == 4,
                      f"{len(combos)}/4 distinct body corners occupied")
    elif tkey == "car":
        # A recognizable car: substantial body, seated cabin, front glass,
        # and four grounded wheels in four distinct corners. Anything less
        # (flat slab, ring wheels, floating parts, missing corners) is
        # reported here so acceptance can never celebrate it.
        bodies = [c for c in comps if str(c.get("role")) == "body"]
        cabins = [c for c in comps if str(c.get("role")) == "cabin"]
        shields = [c for c in comps if str(c.get("role")) == "windshield"]
        wheels = [c for c in comps if str(c.get("role")) == "wheel"]
        check("car_body_present", len(bodies) == 1,
              f"{len(bodies)} body component(s)")
        if bodies:
            bw, bh, bd = dims(bodies[0])
            bx, by, bz = pos(bodies[0])
            check("car_body_substantial", bw >= 2.0 and bh >= 0.3,
                  f"body {bw:.2f}x{bh:.2f}x{bd:.2f}")
            body_top = by + bh / 2
        else:
            bw, bh, bd, bx, by, bz, body_top = (0,) * 7
        check("car_wheel_count", len(wheels) == 4,
              f"{len(wheels)} wheel component(s)")
        if wheels:
            quads = {(pos(w)[0] >= bx, pos(w)[2] >= bz) for w in wheels}
            check("car_wheels_four_corners", len(quads) == 4,
                  f"{len(quads)}/4 body corners have a wheel")
            check("car_wheels_grounded",
                  all(abs(pos(w)[1] - dims(w)[1] / 2) <= 0.1 for w in wheels),
                  "every wheel base at y=0")
            check("car_wheels_filled",
                  all(dims(w)[0] / 2 >= 0.3 for w in wheels),
                  "every wheel wide enough for a filled disc layer")
        if cabins:
            cb = cabins[0]
            base = pos(cb)[1] - dims(cb)[1] / 2
            check("car_cabin_seated", abs(base - body_top) <= 0.15,
                  f"cabin base y={base:.2f} vs body top y={body_top:.2f}")
            check("car_cabin_proportioned",
                  dims(cb)[0] <= max(2.6, bw * 0.75),
                  f"cabin length {dims(cb)[0]:.2f} vs body {bw:.2f}")
        if shields and cabins:
            cb = cabins[0]
            cabin_front = pos(cb)[0] + dims(cb)[0] / 2
            check("car_windshield_front",
                  abs(pos(shields[0])[0] - cabin_front) <= 0.2,
                  f"windshield x={pos(shields[0])[0]:.2f} vs cabin front "
                  f"{cabin_front:.2f}")


def image_fidelity(ir: dict, stats: dict, observed_parts=None) -> dict:
    """Compare the reconstruction against measurable image properties. Only
    supported checks are marked ok/not-ok; checks that a single photograph
    cannot answer are reported as unsupported, never silently passed."""
    checks = []
    unsupported = []
    observed_parts = [str(p).lower() for p in (observed_parts or [])]

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    check("category_present", bool(ir.get("category")),
          f"category={ir.get('category')!r}")
    comps = ir.get("components", [])
    check("component_count", len(comps) >= 2,
          f"{len(comps)} components")
    check("not_a_voxel_grid",
          not all(str(c.get("id", "")).startswith("px_") for c in comps),
          "semantic components, not per-pixel columns")

    bbox = ir.get("bbox") or [0, 0, 0]
    recon_aspect = (bbox[0] / bbox[1]) if bbox[1] else 0.0
    obs_aspect = stats.get("aspect") if isinstance(stats, dict) else None
    if obs_aspect:
        ratio = recon_aspect / obs_aspect if obs_aspect else 0.0
        lo, hi = SILHOUETTE_ASPECT_BAND
        ok = lo <= ratio <= hi
        check("silhouette_aspect", ok,
              f"reconstructed {recon_aspect:.2f} vs photo {obs_aspect:.2f} "
              f"(x{ratio:.2f})")
    else:
        unsupported.append("silhouette aspect (no image statistics)")

    if observed_parts:
        roles = {str(c.get("role")) for c in comps}
        hit = sum(1 for p in observed_parts
                  if p in roles
                  or any(p in str(c.get("id", "")).lower() for c in comps))
        check("visible_parts_match", (hit / max(1, len(observed_parts))) >= 0.4,
              f"{hit}/{len(observed_parts)} visible parts mapped "
              f"({observed_parts})")

    # The category must select a real deterministic template (a generic
    # "object" template is an honest fallback for the no-vision demo path but
    # never counts as a verified semantic reconstruction of a photographed
    # object).
    tkey = template_key(ir.get("category"))
    src = str(ir.get("source", ""))
    check("category_template", tkey != "object" or src != "image-vision",
          f"category={ir.get('category')!r} -> template={tkey!r} "
          f"(source={src!r})")

    _add_shape_checks(tkey, comps, bbox, check)

    unsupported.append("exact depth and all hidden/occluded surfaces "
                       "(a single photograph does not determine them)")
    return {"checks": checks,
            "verified": all(c["ok"] for c in checks),
            "unsupported": unsupported,
            "reconstructed_aspect": round(recon_aspect, 3),
            "photo_aspect": obs_aspect}


def summary_of(ir: dict) -> dict:
    """Compact IR view plus the component hierarchy for reports/UI."""
    s = ir_summary(ir)
    s["hierarchy"] = component_hierarchy(ir)
    return s
