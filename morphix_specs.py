# morphix_specs.py
# Geometric fidelity for printer mode: object specs, plan gates, and
# post-print verification. The pipeline can validate ACTIONS all day and
# still print a yellow strip for a "car" prompt — these checks compare
# intended geometry against planned and committed geometry instead.
#
# Separation of statuses (all reported, never conflated):
#   plan accepted -> paths compiled -> actions validated -> segments
#   committed -> components verified -> goal covered -> quality accepted.
# A component is complete only when its committed deposits verify against
# its spec; action counts alone never complete anything.
#
# Specs are matched by prompt keyword (car/housePrimer...). Unknown objects
# fall back to generic dimensional sanity + support (documented limit).

import math

from morphix_path import (LAYER_HEIGHT, MATERIAL_COLORS, GROUND_TOL,
                          BEAD_WIDTH)

TOL_POS = 0.5    # world-space position tolerance
TOL_DIM = 0.25   # bounding-box dimension tolerance

ROLE_KEYWORDS = {
    "body": ("body", "chassis"),
    "cabin": ("cabin", "cockpit", "cab"),
    "windshield": ("windshield",),
    "wheel": ("wheel", "tire"),
    "window": ("window", "glass"),
    "door": ("door",),
    "light": ("light", "headlight", "lamp"),
    "roof": ("roof",),
    "wall": ("wall",),
    "foundation": ("foundation", "base", "chassis_base"),
    "trunk": ("trunk",),
    "seat": ("seat",),
    "tabletop": ("tabletop", "table_top"),
    "leg": ("leg",),
    "seat": ("seat",),
    "backrest": ("backrest", "back_rest"),
    "armrest": ("armrest", "arm_rest"),
    "head": ("head",),
    "neck": ("neck",),
    "ear": ("ear",),
    "tail": ("tail",),
    # container/ball roles are checked last so more specific roles win.
    "box": ("box", "cube", "cuboid", "block"),
    "sphere": ("sphere", "ball", "orb", "globe"),
}

# Documented defaults for a missing body FIELD (absent keys only — invalid
# values are an explicit error, never silently replaced).
BODY_DEFAULTS = {"width": 3.0, "height": 0.6, "depth": 1.6,
                 "position": [0, 0.3, 0], "color": "0xFF0000",
                 "material": "body"}

CAR_SPEC = {
    "name": "car",
    "keywords": ("car", "vehicle", "sedan", "truck"),
    "body": {"dims": ((2.8, 5.0), (0.5, 1.2), (1.2, 2.4)),
             "grounded": True},
    "cabin": {"dims": ((1.2, 2.6), (0.35, 0.9), (1.0, 2.2)),
              "on": "body", "lift": (-0.1, 0.4)},
    "wheel": {"count": 4, "radius": (0.3, 0.5), "grounded": True,
              "spread_x": 1.0, "z_band": (0.4, 1.4), "min_sep": 0.4},
    "windshield": {"thin": 0.2, "on": "cabin", "face_tol": 0.4},
    "window": {"thin": 0.18, "on": "cabin", "face_tol": 0.35},
    "door": {"thin": 0.18, "on": "body", "face_tol": 0.35},
    "optional": ("window", "door"),
}

HOUSE_SPEC = {
    "name": "house",
    "keywords": ("house", "building", "cabin", "hut", "cottage"),
    "foundation": {"grounded": True},
    "wall": {"on": "foundation", "lift": (-0.1, 0.5)},
    "roof": {"on": "wall", "lift": (-0.2, 1.2)},
    "window": {"thin": 0.25, "on": "wall", "face_tol": 0.5},
    "door": {"thin": 0.3, "on": "wall", "face_tol": 0.5},
    "optional": ("window", "door"),
}

# A box with a sphere resting concentrically on top. Unlike car/house this one
# matches only when BOTH a box word and a round word appear, so a lone "ball"
# prompt still falls through to the generic path.
BOX_SPHERE_SPEC = {
    "name": "box_sphere",
    "all_keywords": (("box", "cube", "cuboid", "block"),
                     ("sphere", "ball", "orb", "globe")),
    "box": {"dims": ((0.2, 8.0), (0.2, 8.0), (0.2, 8.0)), "grounded": True},
    "sphere": {"radius": (0.1, 4.0), "on": "box"},
}

# A table: exactly one thin tabletop on exactly four grounded legs of equal
# length, sitting at the tabletop's corners. Used by semantic image
# reconstruction (and any "table/desk" prompt): a table must never print as a
# thick block, and an urn must never print as a table.
TABLE_SPEC = {
    "name": "table",
    "keywords": ("table", "desk", "workbench", "countertop"),
    "tabletop": {"thin_ratio": 0.3},
    "leg": {"count": 4},
}

# A chair: exactly one horizontal seat, one rear backrest, and four
# grounded legs. Used by text prompts ("chair") and semantic image
# reconstruction. A chair must never print with car parts.
CHAIR_SPEC = {
    "name": "chair",
    "keywords": ("chair", "armchair"),
    "seat": {"dims": ((0.6, 2.0), (0.08, 0.3), (0.6, 2.0))},
    "backrest": {"dims": ((0.08, 0.4), (0.5, 1.6), (0.5, 2.0))},
    "leg": {"count": 4, "height": (0.2, 1.5)},
    "armrest": {"dims": ((0.08, 0.35), (0.25, 1.0), (0.4, 1.4))},
    "optional": ("armrest",),
}

# A stool: a seat on four grounded legs, and nothing else. In particular a
# stool has NO backrest — a backrest makes it a chair, which is a different
# supported category, not a stool variant.
STOOL_SPEC = {
    "name": "stool",
    "keywords": ("stool", "footstool"),
    "seat": {"dims": ((0.5, 1.6), (0.08, 0.3), (0.5, 1.6))},
    "leg": {"count": 4, "height": (0.15, 1.0)},
    "optional": (),
}

# A sitting cat: body, head, two ears, four legs and a tail. Only
# box/cylinder/sphere primitives (what the slicer supports); taper carries
# the ear/tail spikes (see IR taper field).
CAT_SPEC = {
    "name": "cat",
    "keywords": ("cat", "kitten", "feline"),
    "body": {"dims": ((1.0, 2.2), (0.5, 1.0), (0.5, 1.2))},
    "head": {"radius": (0.25, 0.5)},
    "ear": {"count": 2},
    "leg": {"count": 4, "height": (0.2, 0.8)},
    "tail": {},
    "optional": (),
}

SPECS = (CAR_SPEC, HOUSE_SPEC, BOX_SPHERE_SPEC, TABLE_SPEC, CHAIR_SPEC,
         STOOL_SPEC, CAT_SPEC)


def spec_for_prompt(prompt: str):
    """(spec|None) by keyword. Unknown objects use generic sanity only.

    A spec may declare `all_keywords` (a tuple of keyword groups) when it
    only applies when *every* group is present — "box + sphere" must not
    swallow a bare 'a ball' prompt."""
    p = (prompt or "").lower()
    for spec in SPECS:
        groups = spec.get("all_keywords")
        if groups is not None:
            if all(any(k in p for k in group) for group in groups):
                return spec
            continue
        if any(k in p for k in spec["keywords"]):
            return spec
    return None


def _role_of(label: str):
    lab = (label or "").lower()
    for role, keys in ROLE_KEYWORDS.items():
        if any(k in lab for k in keys):
            return role
    return ""


def _box_of(act: dict):
    """(cx, cy, cz, w, h, d) or None for non-box actions."""
    if not isinstance(act, dict) or act.get("action") != "add_box":
        return None
    try:
        p = act["position"]
        return (float(p[0]), float(p[1]), float(p[2]), float(act["width"]),
                float(act["height"]), float(act["depth"]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _cyl_of(act: dict):
    """(cx, cy, cz, r, h) or None for non-cylinder actions."""
    if not isinstance(act, dict) or act.get("action") != "add_cylinder":
        return None
    try:
        p = act["position"]
        r = max(float(act.get("radiusTop", act.get("radius", 0))),
                float(act.get("radiusBottom", act.get("radius", 0))))
        return (float(p[0]), float(p[1]), float(p[2]), r,
                float(act["height"]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _sphere_of(act: dict):
    """(cx, cy, cz, r) or None for non-sphere actions."""
    if not isinstance(act, dict) or act.get("action") != "add_sphere":
        return None
    try:
        p = act["position"]
        return (float(p[0]), float(p[1]), float(p[2]), float(act["radius"]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _car_part_errors(actions: list) -> list:
    """Category-leakage circuit breaker shared by every non-car spec: a
    wheel/cabin/windshield role or tire material outside a car plan is a
    hard error, never a warning."""
    errs = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        role = _role_of(a.get("label", ""))
        if role in ("wheel", "cabin", "windshield") \
                or a.get("material") == "tire":
            errs.append(
                f"component {a.get('label')!r} is a car part "
                f"(role {role!r}) in a non-car plan: category leakage")
    return errs


def check_plan(spec: dict, actions: list) -> tuple:
    """Fidelity gate on the THOUGHT plan (before slicing). Returns
    (errors, warnings). ERRORS block compilation (missing roles, invalid
    data, unplaceable counts); WARNINGS print anyway but loudly (ugly
    proportions, misplaced panels) and veto quality acceptance. [] means
    the plan is geometrically worth printing."""
    errs, warns = [], []
    if spec is None:
        return errs, warns
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        m = a.get("material", "body")
        if m not in MATERIAL_COLORS:
            errs.append(f"component {a.get('label', i)!r}: unknown material "
                        f"{m!r} (use {sorted(MATERIAL_COLORS)})")

    def warn(msg):
        warns.append(msg)
    boxes = [(a, _box_of(a)) for a in actions if _box_of(a)]
    cyls = [(a, _cyl_of(a)) for a in actions if _cyl_of(a)]
    spheres = [(a, _sphere_of(a)) for a in actions if _sphere_of(a)]
    by_role = {}
    for a, b in boxes:
        by_role.setdefault(_role_of(a.get("label", "")), []).append((a, b))
    for a, c in cyls:
        by_role.setdefault(_role_of(a.get("label", "")), []).append((a, c))
    for a, s in spheres:
        by_role.setdefault(_role_of(a.get("label", "")), []).append((a, s))

    def need(role):
        items = by_role.get(role, [])
        if not items:
            errs.append(f"plan has no {role} component")
        return items

    if spec["name"] == "car":
        bodies = need("body")
        for a, (cx, cy, cz, w, h, d) in bodies:
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["body"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"body {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside car-body range "
                     f"{lo_l}-{hi_l} x {lo_h}-{hi_h} x {lo_w}-{hi_w}")
        cabins = by_role.get("cabin", [])
        if not cabins:
            # The 1.5b thinker systematically omits cabins; a missing cabin
            # warns loudly and the car prints without one (goal coverage
            # reports it) instead of failing the whole run.
            warn("plan has no cabin component: car will print without one")
        for a, (cx, cy, cz, w, h, d) in cabins:
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["cabin"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"cabin {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside cabin range")
            ok = False
            for _b, (bx, by, bz, bw, bh, bd) in bodies:
                lo, hi = spec["cabin"]["lift"]
                if (abs((cy - h / 2) - (by + bh / 2)) <= max(hi, -lo) + 0.3
                        and abs(cx - bx) <= bw / 2 and abs(cz - bz) <= bd / 2):
                    ok = True
            if bodies and not ok:
                warn(f"cabin {a.get('label')!r} is not on top of the body "
                     f"(compiler will seat it onto support)")
        wheels = need("wheel")
        if wheels and len(wheels) != spec["wheel"]["count"]:
            errs.append(f"car needs exactly {spec['wheel']['count']} wheels "
                        f"(got {len(wheels)})")
        pts = []
        for a, c in wheels:
            if len(c) == 6:
                warn(f"wheel {a.get('label')!r} is a box, not a "
                     f"wheel cylinder (prints as a block)")
                continue
            _cx, _cy, _cz, r, h = c
            lo_r, hi_r = spec["wheel"]["radius"]
            if not (lo_r <= r <= hi_r):
                warn(f"wheel {a.get('label')!r} radius {r:.2f} "
                     f"outside {lo_r}-{hi_r}")
            pts.append((_cx, _cz))
        if len(pts) >= 2:
            xs = sorted({round(x, 1) for x, _z in pts})
            if len(xs) < 2 or max(xs) - min(xs) < spec["wheel"]["spread_x"]:
                warn("wheels are not spread front/rear along the body")
            zs = [z for _x, z in pts]
            lo_z, hi_z = spec["wheel"]["z_band"]
            if not (any(z > 0 for z in zs) and any(z < 0 for z in zs)):
                warn("wheels are not on both left and right sides")
            if any(not (lo_z <= abs(z) <= hi_z) for z in zs):
                warn("wheels are not at the body flanks")
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if math.dist(pts[i], pts[j]) < spec["wheel"]["min_sep"]:
                        warn("two wheels share (nearly) one position")
                        break
        for role in ("windshield", "window", "door"):
            for a, b in by_role.get(role, []):
                _check_panel(spec, role, a, b, by_role, warns)
    elif spec["name"] == "house":
        if not by_role.get("foundation"):
            errs.append("plan has no foundation component")
        for a, (cx, cy, cz, w, h, d) in by_role.get("wall", []):
            ok = any(abs((cy - h / 2) - (fy + fh / 2)) <= 0.8
                     and abs(cx - fx) <= fw / 2 + 0.5
                     and abs(cz - fz) <= fd / 2 + 0.5
                     for _f, (fx, fy, fz, fw, fh, fd)
                     in by_role.get("foundation", []))
            if by_role.get("foundation") and not ok:
                warn(f"wall {a.get('label')!r} is not on the foundation "
                     f"(compiler will seat it onto support)")
        for role in ("window", "door"):
            for a, b in by_role.get(role, []):
                _check_panel(spec, role, a, b, by_role, warns)
    elif spec["name"] == "box_sphere":
        box_items = need("box")
        sphere_items = need("sphere")
        for a, (cx, cy, cz, w, h, d) in box_items:
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["box"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"box {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside box range")
            if cy - h / 2 > GROUND_TOL:
                warn(f"box {a.get('label')!r} is not grounded on the build "
                     f"plate (base y={cy - h / 2:.2f})")
        if len(sphere_items) > 1:
            warn("plan has more than one sphere: only the first is required")
        if box_items and sphere_items:
            _b, (bx, by, bz, bw, bh, bd) = box_items[0]
            _s, (sx, sy, sz, sr) = sphere_items[0]
            lo_r, hi_r = spec["sphere"]["radius"]
            if not (lo_r <= sr <= hi_r):
                warn(f"sphere radius {sr:.2f} outside {lo_r}-{hi_r}")
            box_top = by + bh / 2
            box_base = by - bh / 2
            if (abs(sx - bx) > bw / 2 + TOL_POS
                    or abs(sz - bz) > bd / 2 + TOL_POS):
                warn(f"sphere is not aligned over the box centre "
                     f"(sphere ({sx:.2f},{sz:.2f}) vs box ({bx:.2f},{bz:.2f}))")
            if sy - sr > box_top + TOL_POS:
                warn(f"sphere floats above the box "
                     f"(base y={sy - sr:.2f} > top y={box_top:.2f})")
            if sy + sr < box_base:
                errs.append("sphere is below the box, not on top of it")
    elif spec["name"] == "table":
        tops = need("tabletop")
        legs = need("leg")
        if legs and len(legs) != spec["leg"]["count"]:
            errs.append(f"table needs exactly {spec['leg']['count']} legs "
                        f"(got {len(legs)})")
        tops_h = [b[1] + b[4] / 2 for _a, b in legs]
        for a, (cx, cy, cz, w, h, d) in tops:
            if h > spec["tabletop"]["thin_ratio"] * min(w, d):
                warn(f"tabletop {a.get('label')!r} is too thick "
                     f"({h:.2f} vs {min(w, d):.2f} footprint): a tabletop "
                     f"must be a thin slab")
            base = cy - h / 2
            if tops_h and not any(abs(base - th) <= 0.1 for th in tops_h):
                warn(f"tabletop {a.get('label')!r} is not resting on the legs "
                     f"(base y={base:.2f})")
        heights = [b[4] for _a, b in legs]
        if heights and (max(heights) - min(heights)) > 0.05:
            warn("table legs have different lengths")
        for a, (cx, cy, cz, w, h, d) in legs:
            if cy - h / 2 > GROUND_TOL:
                warn(f"leg {a.get('label')!r} does not reach the ground "
                     f"(base y={cy - h / 2:.2f})")
    elif spec["name"] == "chair":
        # Car-only components in a chair plan are a category leak, not a
        # style choice: hard error, never a warning.
        errs.extend(_car_part_errors(actions))
        seats = [(a, b) for a, b in boxes
                 if _role_of(a.get("label", "")) == "seat"]
        if not seats:
            errs.append("plan has no seat component")
        for a, (cx, cy, cz, w, h, d) in seats:
            if a.get("action") != "add_box":
                errs.append(f"seat {a.get('label')!r} must be an add_box "
                            f"(got {a.get('action')!r})")
                continue
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["seat"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"seat {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside chair-seat range")
            if h > min(w, d):
                errs.append(f"seat {a.get('label')!r} is not a horizontal "
                            f"surface (height {h:.2f} exceeds footprint)")
        backs = [(a, b) for a, b in boxes
                 if _role_of(a.get("label", "")) == "backrest"]
        if not backs:
            errs.append("plan has no backrest component")
        for a, (cx, cy, cz, w, h, d) in backs:
            if a.get("action") != "add_box":
                errs.append(f"backrest {a.get('label')!r} must be an add_box "
                            f"(got {a.get('action')!r})")
                continue
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["backrest"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"backrest {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside backrest range")
            if seats:
                sx, sy, sz, sw, sh, sd = seats[0][1]
                if cx >= sx:
                    warn(f"backrest {a.get('label')!r} is not at the rear "
                         f"of the seat")
                if abs((cy - h / 2) - (sy + sh / 2)) > 0.2:
                    warn(f"backrest {a.get('label')!r} does not rise from "
                         f"the seat top")
            if not (w <= 0.4 * h):
                warn(f"backrest {a.get('label')!r} is not an upright thin "
                     f"panel")
        leg_items = ([(a, _box_of(a)) for a in actions
                      if _role_of(a.get("label", "")) == "leg"
                      and _box_of(a) is not None]
                     + [(a, _cyl_of(a)) for a in actions
                        if _role_of(a.get("label", "")) == "leg"
                        and _cyl_of(a) is not None])
        if len(leg_items) != spec["leg"]["count"]:
            errs.append(f"chair needs exactly {spec['leg']['count']} legs "
                        f"(got {len(leg_items)})")
        else:
            pts = [(c[0], c[2]) for _a, c in leg_items]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if math.dist(pts[i], pts[j]) < 0.25:
                        warn("two legs share (nearly) one position")
                        break
            lo_h, hi_h = spec["leg"]["height"]
            for a, c in leg_items:
                h = c[4]
                if not (lo_h <= h <= hi_h):
                    warn(f"leg {a.get('label')!r} height {h:.2f} outside "
                         f"{lo_h}-{hi_h}")
                if len(c) == 6:
                    w, d = c[3], c[5]
                    if not (0.05 <= w <= 0.5 and 0.05 <= d <= 0.5):
                        warn(f"leg {a.get('label')!r} footprint "
                             f"{w:.2f}x{d:.2f} outside leg range")
                else:
                    if not (0.03 <= c[3] <= 0.3):
                        warn(f"leg {a.get('label')!r} radius {c[3]:.2f} "
                             f"outside leg range")
            for a, c in leg_items:
                base = c[1] - c[4] / 2
                if base > GROUND_TOL:
                    warn(f"leg {a.get('label')!r} does not reach the ground "
                         f"(base y={base:.2f})")
            if seats:
                sx, sy, sz, sw, sh, sd = seats[0][1]
                tops = [c[1] + c[4] / 2 for _a, c in leg_items]
                if tops and not any(abs((sy - sh / 2) - t) <= 0.15
                                    for t in tops):
                    warn("seat does not rest on the legs")
        if seats:
            sx, sy, sz, sw, sh, sd = seats[0][1]
            for a, (cx, cy, cz, w, h, d) in [
                    (a, b) for a, b in boxes
                    if _role_of(a.get("label", "")) == "armrest"]:
                if a.get("action") != "add_box":
                    errs.append(f"armrest {a.get('label')!r} must be an "
                                f"add_box (got {a.get('action')!r})")
                    continue
                (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = \
                    spec["armrest"]["dims"]
                if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                        and lo_w <= d <= hi_w):
                    warn(f"armrest {a.get('label')!r} dimensions "
                         f"{w:.2f}x{h:.2f}x{d:.2f} outside armrest range")
                if abs((cy - h / 2) - (sy + sh / 2)) > 0.2:
                    warn(f"armrest {a.get('label')!r} does not rise from "
                         f"the seat top")
                if abs(abs(cx - sx) - sw / 2) > 0.25:
                    warn(f"armrest {a.get('label')!r} is not at a seat side")
    elif spec["name"] == "stool":
        # A stool is a seat on legs — and nothing else. A backrest makes it
        # a chair (different supported category); car parts are leakage.
        errs.extend(_car_part_errors(actions))
        seats = [(a, b) for a, b in boxes
                 if _role_of(a.get("label", "")) == "seat"]
        if not seats:
            errs.append("plan has no seat component")
        for a, (cx, cy, cz, w, h, d) in seats:
            if a.get("action") != "add_box":
                errs.append(f"seat {a.get('label')!r} must be an add_box "
                            f"(got {a.get('action')!r})")
                continue
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["seat"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"seat {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside stool-seat range")
            if h > min(w, d):
                errs.append(f"seat {a.get('label')!r} is not a horizontal "
                            f"surface (height {h:.2f} exceeds footprint)")
        for a in actions:
            if isinstance(a, dict) \
                    and _role_of(a.get("label", "")) == "backrest":
                errs.append(
                    f"component {a.get('label')!r} is a backrest: a stool "
                    f"has no backrest (prompt 'chair' for the chair design)")
        leg_items = ([(a, _box_of(a)) for a in actions
                      if _role_of(a.get("label", "")) == "leg"
                      and _box_of(a) is not None]
                     + [(a, _cyl_of(a)) for a in actions
                        if _role_of(a.get("label", "")) == "leg"
                        and _cyl_of(a) is not None])
        if len(leg_items) != spec["leg"]["count"]:
            errs.append(f"stool needs exactly {spec['leg']['count']} legs "
                        f"(got {len(leg_items)})")
        else:
            pts = [(c[0], c[2]) for _a, c in leg_items]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if math.dist(pts[i], pts[j]) < 0.25:
                        warn("two legs share (nearly) one position")
                        break
            lo_h, hi_h = spec["leg"]["height"]
            for a, c in leg_items:
                if not (lo_h <= c[4] <= hi_h):
                    warn(f"leg {a.get('label')!r} height {c[4]:.2f} "
                         f"outside {lo_h}-{hi_h}")
                if c[1] - c[4] / 2 > GROUND_TOL:
                    warn(f"leg {a.get('label')!r} does not reach the ground "
                         f"(base y={c[1] - c[4] / 2:.2f})")
            if seats:
                sx, sy, sz, sw, sh, sd = seats[0][1]
                tops = [c[1] + c[4] / 2 for _a, c in leg_items]
                if tops and not any(abs((sy - sh / 2) - t) <= 0.15
                                    for t in tops):
                    warn("seat does not rest on the legs")
    elif spec["name"] == "cat":
        # A cat is not a stack of boxes: body, head, ears, legs and tail
        # are all required, and car parts are leakage, not style.
        errs.extend(_car_part_errors(actions))
        bodies = [(a, b) for a, b in boxes
                  if _role_of(a.get("label", "")) == "body"]
        if not bodies:
            errs.append("plan has no body component")
        for a, (cx, cy, cz, w, h, d) in bodies:
            if a.get("action") != "add_box":
                errs.append(f"body {a.get('label')!r} must be an add_box "
                            f"(got {a.get('action')!r})")
                continue
            (lo_l, hi_l), (lo_h, hi_h), (lo_w, hi_w) = spec["body"]["dims"]
            if not (lo_l <= w <= hi_l and lo_h <= h <= hi_h
                    and lo_w <= d <= hi_w):
                warn(f"body {a.get('label')!r} dimensions "
                     f"{w:.2f}x{h:.2f}x{d:.2f} outside cat-body range")
        heads = [a for a in actions
                 if _role_of(a.get("label", "")) == "head"
                 and _sphere_of(a) is not None]
        if not heads:
            errs.append("plan has no head component (a sphere)")
        for a in actions:
            if _role_of(a.get("label", "")) == "head" \
                    and _sphere_of(a) is None:
                errs.append(f"head {a.get('label')!r} must be an add_sphere "
                            f"(got {a.get('action')!r})")
        necks = [a for a in actions
                 if _role_of(a.get("label", "")) == "neck"]
        for a in necks:
            c = _cyl_of(a)
            if c is None:
                errs.append(f"neck {a.get('label')!r} must be an "
                            f"add_cylinder (got {a.get('action')!r})")
                continue
            if bodies:
                _b, (bx, by, bz, bw, bh, bd) = bodies[0]
                if abs((c[1] - c[4] / 2) - (by + bh / 2)) > 0.15:
                    warn(f"neck {a.get('label')!r} is not seated on the "
                         f"body top")
        for a in heads:
            _hx, hy, _hz, hr = _sphere_of(a)
            lo_r, hi_r = spec["head"]["radius"]
            if not (lo_r <= hr <= hi_r):
                warn(f"head {a.get('label')!r} radius {hr:.2f} outside "
                     f"{lo_r}-{hi_r}")
            if bodies:
                _b, (bx, by, bz, bw, bh, bd) = bodies[0]
                neck_tops = []
                for n in necks:
                    nc = _cyl_of(n)
                    if nc is not None:
                        neck_tops.append(nc[1] + nc[4] / 2)
                if neck_tops:
                    ref = max(neck_tops)
                    if abs((hy - hr) - ref) > 0.2:
                        warn(f"head {a.get('label')!r} is not seated on "
                             f"the neck top")
                elif abs((_hx) - (bx + bw / 2)) > bw / 2 + TOL_POS \
                        or abs(hy - (by + bh / 2)) > bh / 2 + hr + TOL_POS:
                    warn(f"head {a.get('label')!r} is not at the body "
                         f"front-top")
        ears = [a for a in actions
                if _role_of(a.get("label", "")) == "ear"]
        if len(ears) != spec["ear"]["count"]:
            errs.append(f"cat needs exactly {spec['ear']['count']} ears "
                        f"(got {len(ears)})")
        for a in ears:
            c = _cyl_of(a)
            if c is None:
                errs.append(f"ear {a.get('label')!r} must be an add_cylinder "
                            f"(got {a.get('action')!r})")
                continue
            rt = float(a.get("radiusTop", c[3]))
            rb = float(a.get("radiusBottom", a.get("radius", c[3])))
            if not (rt < rb):
                warn(f"ear {a.get('label')!r} is not tapered to a point")
        if len(ears) == 2:
            p0 = _cyl_of(ears[0])
            p1 = _cyl_of(ears[1])
            if p0 is not None and p1 is not None:
                if abs(p0[0] - p1[0]) < 0.15 and abs(p0[2] - p1[2]) < 0.15:
                    warn("ears share (nearly) one position")
        if heads:
            _hx, hy, _hz, hr = _sphere_of(heads[0])
            head_top = hy + hr
            for a in ears:
                c = _cyl_of(a)
                if c is None:
                    continue
                base = c[1] - c[4] / 2
                if abs(base - head_top) > 0.25:
                    warn(f"ear {a.get('label')!r} is disconnected from "
                         f"the head (base y={base:.2f} vs head top "
                         f"y={head_top:.2f})")
        legs = ([(a, _box_of(a)) for a in actions
                 if _role_of(a.get("label", "")) == "leg"
                 and _box_of(a) is not None]
                + [(a, _cyl_of(a)) for a in actions
                   if _role_of(a.get("label", "")) == "leg"
                   and _cyl_of(a) is not None])
        if len(legs) != spec["leg"]["count"]:
            errs.append(f"cat needs exactly {spec['leg']['count']} legs "
                        f"(got {len(legs)})")
        else:
            pts = [(c[0], c[2]) for _a, c in legs]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if math.dist(pts[i], pts[j]) < 0.25:
                        warn("two legs share (nearly) one position")
                        break
            lo_h, hi_h = spec["leg"]["height"]
            for a, c in legs:
                if not (lo_h <= c[4] <= hi_h):
                    warn(f"leg {a.get('label')!r} height {c[4]:.2f} "
                         f"outside {lo_h}-{hi_h}")
                if c[1] - c[4] / 2 > GROUND_TOL:
                    warn(f"leg {a.get('label')!r} does not reach the ground "
                         f"(base y={c[1] - c[4] / 2:.2f})")
        tails = [a for a in actions
                 if _role_of(a.get("label", "")) == "tail"]
        if not tails:
            errs.append("plan has no tail component")
        for a in tails:
            c = _cyl_of(a)
            if c is None:
                errs.append(f"tail {a.get('label')!r} must be an "
                            f"add_cylinder (got {a.get('action')!r})")
                continue
            rt = float(a.get("radiusTop", c[3]))
            rb = float(a.get("radiusBottom", a.get("radius", c[3])))
            if not (rt < rb):
                warn(f"tail {a.get('label')!r} is not tapered upward")
            if bodies:
                _b, (bx, by, bz, bw, bh, bd) = bodies[0]
                if not (c[0] < bx):
                    warn(f"tail {a.get('label')!r} is not at the body rear")
                if c[1] - c[4] / 2 > GROUND_TOL and not (
                        abs(c[0] - (bx - bw / 2)) <= bw / 2 + TOL_POS):
                    warn(f"tail {a.get('label')!r} floats unattached")
    return errs, warns


def _check_panel(spec, role, act, boxv, by_role, warns):
    """Thin parts (windows/doors) must sit on their host's faces."""
    host = spec[role]["on"]
    thin = spec[role]["thin"]
    tol = spec[role]["face_tol"]
    cx, cy, cz, w, h, d = boxv
    if min(w, d) > thin:
        warns.append(f"{role} {act.get('label')!r} is not a thin panel "
                     f"({w:.2f}x{d:.2f})")
        return
    hosts = by_role.get(host, [])
    if not hosts:
        return  # host missing is reported by its own check
    ok = False
    for _h, (hx, hy, hz, hw, hh, hd) in hosts:
        on_face = (abs(abs(cx - hx) - hw / 2) <= tol
                   or abs(abs(cz - hz) - hd / 2) <= tol)
        in_range = abs(cy - hy) <= (hh + h) / 2 + tol
        if on_face and in_range:
            ok = True
    if not ok:
        warns.append(f"{role} {act.get('label')!r} is not on the {host} "
                     f"surface")


def verify_component(spec: dict, comp: dict, records: list) -> dict:
    """Post-print truth: compare committed deposits against the planned
    component geometry. Returns {verified, checks:[{name, ok, detail}]}."""
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    recs = [r for r in records
            if r.get("component") == comp.get("component")
            and r.get("action") in ("deposit_segment", "deposit_point")]
    planned = comp.get("planned_deposits", 0)
    check("has_deposits", len(recs) > 0, f"{len(recs)} deposits")
    check("deposit_count", len(recs) >= max(1, planned // 2),
          f"{len(recs)}/{planned}")
    layers = sorted({r.get("layer") for r in recs
                     if isinstance(r.get("layer"), int)})
    check("layer_coverage", layers == list(comp.get("layers", [])),
          f"layers {layers} vs planned {comp.get('layers', [])}")
    if recs:
        xs, ys, zs = [], [], []
        for r in recs:
            pts = [r["from"], r["to"]] if "from" in r else [r["at"]]
            for x, y, z in pts:
                xs.append(x)
                ys.append(y)
                zs.append(z)
        bb = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        exp = comp.get("expect_bbox")
        if exp:
            # x/z must cover the planned footprint. For y the deposits sit
            # on layer planes from the bottom course to the top course, so
            # the deposit span must cover the full solid height (minus at
            # most one bead: the top course plane sits BEAD_HEIGHT below the
            # solid top). A half-printed component has a short span and fails
            # here even when every committed action was syntactically valid.
            ok = (abs(bb[0] - exp[0]) <= TOL_DIM
                  and abs(bb[2] - exp[2]) <= TOL_DIM
                  and abs(bb[1] - exp[1]) <= LAYER_HEIGHT + TOL_DIM)
            check("bbox", ok,
                  f"got {tuple(round(v, 2) for v in bb)} vs solid "
                  f"{(exp[0], exp[1], exp[2])}")
        exp_c = comp.get("expect_center")
        if exp_c:
            got_c = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                     (min(zs) + max(zs)) / 2)
            check("position", all(abs(a - b) <= TOL_POS
                                  for a, b in zip(got_c, exp_c)),
                  f"got {tuple(round(v, 2) for v in got_c)} vs {exp_c}")
    verified = all(c["ok"] for c in checks)
    return {"verified": verified, "checks": checks}


# ------------------------------------------------ solid geometry checks
# Independent of deposit counts: a box must be the right size and place; a
# sphere must be the real round slice of a solid (per-layer radius follows
# r(y)=sqrt(R^2-(y-c)^2)) centred on the axis and resting on the box; every
# deposit must carry the component's declared material.

def verify_solids(actions: list, committed: list) -> list:
    """Returns a flat list of {component, name, ok, detail} checks."""
    out = []

    def ck(comp, name, ok, detail=""):
        out.append({"component": comp, "name": name,
                    "ok": bool(ok), "detail": str(detail)})

    by_label = {}
    for a in actions:
        if isinstance(a, dict) and a.get("label"):
            by_label[a["label"]] = a
    recs_by = {}
    for r in committed:
        if r.get("action") in ("deposit_segment", "deposit_point"):
            recs_by.setdefault(r.get("component"), []).append(r)

    def points_of(recs):
        pts = []
        for r in recs:
            if r.get("action") == "deposit_segment":
                pts += [r["from"], r["to"]]
            else:
                pts.append(r["at"])
        return pts

    for label, a in by_label.items():
        name = a.get("action")
        recs = recs_by.get(label, [])
        if not recs:
            continue
        want_mat = a.get("material")
        if want_mat:
            bad = [r for r in recs if r.get("material") != want_mat]
            ck(label, "material", not bad,
               f"{len(recs) - len(bad)}/{len(recs)} deposits use {want_mat!r}")
        pts = points_of(recs)
        xs = [p[0] for p in pts]
        zs = [p[2] for p in pts]
        cx0, cz0 = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        pos = a.get("position", [0, 0, 0])
        try:
            px, py, pz = (float(pos[i]) for i in range(3))
        except (TypeError, ValueError, IndexError):
            px = py = pz = None
        if name == "add_box":
            w, h, d = (float(a["width"]), float(a["height"]),
                       float(a["depth"]))
            ck(label, "box_extent",
               abs((max(xs) - min(xs)) - w) <= TOL_DIM
               and abs((max(zs) - min(zs)) - d) <= TOL_DIM,
               f"span=({max(xs) - min(xs):.2f},{max(zs) - min(zs):.2f}) "
               f"want=({w:.2f},{d:.2f})")
            if px is not None:
                ck(label, "box_centre",
                   abs(cx0 - px) <= TOL_POS and abs(cz0 - pz) <= TOL_POS,
                   f"centre=({cx0:.2f},{cz0:.2f}) want=({px:.2f},{pz:.2f})")
        elif name == "add_sphere":
            R = float(a["radius"])
            ck(label, "sphere_centre",
               abs(cx0 - px) <= TOL_POS and abs(cz0 - pz) <= TOL_POS,
               f"centre=({cx0:.2f},{cz0:.2f}) want=({px:.2f},{pz:.2f})")
            widest = max(max(xs) - min(xs), max(zs) - min(zs))
            ck(label, "sphere_diameter", abs(widest - 2 * R) <= TOL_DIM,
               f"widest={widest:.2f} want={2 * R:.2f}")
            layer_res = {}
            for r in recs:
                pts2 = ([r["from"], r["to"]]
                        if r.get("action") == "deposit_segment" else [r["at"]])
                for p in pts2:
                    layer_res.setdefault(r.get("layer"), []).append(
                        (math.hypot(p[0] - px, p[2] - pz), p[1]))
            worst, ok = 0.0, True
            for _k, vals in layer_res.items():
                y = vals[0][1]
                want_r = math.sqrt(max(0.0, R * R - (y - py) ** 2))
                got_r = max(v[0] for v in vals)
                if want_r < BEAD_WIDTH:
                    if got_r > max(BEAD_WIDTH, R * 0.6):
                        ok = False
                    worst = max(worst, abs(got_r - min(want_r, BEAD_WIDTH)))
                else:
                    worst = max(worst, abs(got_r - want_r))
                    if abs(got_r - want_r) > max(TOL_DIM, LAYER_HEIGHT):
                        ok = False
            ck(label, "sphere_round", ok, f"max radius error {worst:.3f}")
            ck(label, "sphere_layers", len(layer_res) >= 3,
               f"{len(layer_res)} sliced layers")

    # -- contact: every sphere must rest on the box beneath it
    boxes = {l: _box_of(a) for l, a in by_label.items()
             if a.get("action") == "add_box" and l in recs_by}
    for l, a in by_label.items():
        s = _sphere_of(a)
        if not s or l not in recs_by:
            continue
        sx, sy, sz, sr = s
        best = None
        for bl, b in boxes.items():
            if not b:
                continue
            bx, by, bz, bw, bh, bd = b
            if (abs(sx - bx) <= bw / 2 + TOL_POS
                    and abs(sz - bz) <= bd / 2 + TOL_POS):
                top = by + bh / 2
                if best is None or top > best:
                    best = top
        if best is None:
            continue
        gap = (sy - sr) - best
        ck(l, "contact_box", -LAYER_HEIGHT - TOL_POS <= gap <= TOL_POS,
           f"sphere base y={sy - sr:.2f} vs box top y={best:.2f} "
           f"(gap {gap:+.2f})")
    return out


def expected_of(act: dict) -> dict:
    """Expected bbox + center from a thought primitive (for verification)."""
    b = _box_of(act)
    if b:
        _cx, _cy, _cz, w, h, d = b
        return {"expect_bbox": (w, h, d), "expect_center": (_cx, _cy, _cz)}
    c = _cyl_of(act)
    if c:
        _cx, _cy, _cz, r, h = c
        return {"expect_bbox": (2 * r, h, 2 * r),
                "expect_center": (_cx, _cy, _cz)}
    s = _sphere_of(act)
    if s:
        _cx, _cy, _cz, r = s
        return {"expect_bbox": (2 * r, 2 * r, 2 * r),
                "expect_center": (_cx, _cy, _cz)}
    return {}


# ------------------------------------------------ car plan repair
# Deterministic normalization of a thought car plan into the canonical
# component set (body/cabin/windshield/2 windows/4 wheels with stable IDs).
# Conventions (front = +X, left = +Z). Derived parts are computed from body
# dimensions; valid model-proposed geometry is always preserved; every
# addition, drop, or default is logged with its reason. Fatal schema
# problems (no body, invalid body dims, non-box structural parts) raise
# ValueError with an explicit message instead of guessing.

def repair_car_plan(actions: list) -> tuple:
    """Returns (repaired_actions, repairs). repairs is a list of
    {"component", "reason"} in canonical order."""
    from morphix_path import MAX_LINEAR
    if not isinstance(actions, list) or not actions:
        raise ValueError("car plan must be a non-empty action list")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("car plan entries must be JSON objects")
    repairs = []

    def log(component, reason):
        repairs.append({"component": component, "reason": reason})

    # -- body: exactly one box; missing FIELDS take documented defaults,
    # -- invalid values are an explicit error.
    body_cands = [a for a in actions
                  if _role_of(a.get("label", "")) == "body"]
    for cand in body_cands:
        if cand.get("action") != "add_box":
            raise ValueError(
                f"body must be an add_box (got {cand.get('action')!r} "
                f"for {cand.get('label')!r})")
    if not body_cands:
        raise ValueError("car plan has no body component")
    body = dict(body_cands[0])
    for extra in body_cands[1:]:
        log(extra.get("label", "?"), "duplicate body dropped (kept first)")
    for key, default in BODY_DEFAULTS.items():
        if key not in body:
            body[key] = (list(default) if isinstance(default, list)
                         else default)
            log(body.get("label", "body"),
                f"missing {key}: documented default {default!r}")
    for key in ("width", "height", "depth"):
        v = body.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"body {body.get('label')!r}: {key} must be a number "
                f"(got {v!r})")
        if not math.isfinite(v) or v <= 0:
            raise ValueError(
                f"body {body.get('label')!r}: {key} must be finite and > 0 "
                f"(got {v!r})")
        if v > MAX_LINEAR:
            raise ValueError(
                f"body {body.get('label')!r}: {key}={v} exceeds the "
                f"{MAX_LINEAR} scene limit")
    try:
        bx, by, bz = (float(body["position"][i]) for i in range(3))
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError(f"body {body.get('label')!r}: position must be "
                         f"[x,y,z] numbers")
    bw, bh, bd = float(body["width"]), float(body["height"]), \
        float(body["depth"])
    body_top = by + bh / 2.0

    def keep_first(role, kind="add_box"):
        """First valid existing part of a role; extras dropped with a log."""
        items = [a for a in actions
                 if _role_of(a.get("label", "")) == role
                 and a.get("action") == kind]
        for extra in items[1:]:
            log(extra.get("label", "?"),
                f"duplicate {role} dropped (kept first)")
        return items[0] if items else None

    out = [body]

    # -- cabin: synthesized from the body when missing. Proportions are
    # -- clamped so the cabin reads as a cabin, never a second body shell:
    # -- at most ~55% of body length and 0.7 of body width, seated exactly.
    cabin = keep_first("cabin")
    if cabin is None:
        ch = round(min(0.9, max(0.35, bh * 0.8)), 2)
        cl = round(min(bw * 0.55, 2.6, bw - 0.3), 2)
        cw = round(min(max(0.8, bd - 0.2), 2.2), 2)
        cabin = {"action": "add_box", "width": cl, "height": ch,
                 "depth": cw, "position": [round(bx, 2),
                                           round(body_top + ch / 2, 2),
                                           round(bz, 2)],
                 "color": "0x44CCFF", "label": "cabin",
                 "material": "body"}
        log("cabin", f"missing: synthesized {cl}x{ch}x{cw} on body top "
                     f"y={body_top:.2f}")
    out.append(cabin)
    cb = _box_of(cabin)
    ccx, ccy, ccz, cw, ch, cd = cb

    # -- windshield: thin glass at the cabin front face.
    shield = keep_first("windshield")
    if shield is None:
        shield = {"action": "add_box", "width": 0.12,
                  "height": round(ch * 0.7, 2), "depth": round(cd * 0.9, 2),
                  "position": [round(ccx + cw / 2, 2), round(ccy, 2),
                               round(ccz, 2)],
                  "color": "0x99DDFF", "label": "windshield",
                  "material": "glass"}
        log("windshield", "missing: synthesized glass at cabin front face")
    elif shield["action"] != "add_box":
        raise ValueError(f"windshield {shield.get('label')!r} must be an "
                         f"add_box panel (got {shield['action']!r})")
    out.append(shield)

    # -- side windows: keep existing, synthesize missing sides only.
    want_sides = {"side_window_left": 1, "side_window_right": -1}
    have = {}
    for a in actions:
        if _role_of(a.get("label", "")) != "window":
            continue
        if a.get("action") != "add_box":
            raise ValueError(
                f"window {a.get('label')!r} must be an add_box panel "
                f"(got {a.get('action')!r})")
        try:
            side = 1 if float(a["position"][2]) >= ccz else -1
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"window {a.get('label')!r}: position must be "
                             f"[x,y,z] numbers")
        if side in have:
            log(a.get("label", "?"),
                "duplicate side window dropped (kept first per side)")
            continue
        have[side] = a
    for label, side in (("side_window_left", 1), ("side_window_right", -1)):
        if side in have:
            out.append(have[side])
        else:
            out.append({
                "action": "add_box", "width": round(cw * 0.5, 2),
                "height": round(ch * 0.5, 2), "depth": 0.1,
                "position": [round(ccx, 2), round(ccy + 0.05, 2),
                             round(ccz + side * (cd / 2 + 0.01), 2)],
                "color": "0x99DDFF", "label": label, "material": "glass"})
            log(label, f"missing: synthesized glass on cabin "
                       f"{'left' if side > 0 else 'right'} face")

    # -- wheels: keep valid distinct grounded cylinders, drop the rest,
    # -- fill slots. Wheel height equals wheel diameter (wh = 2*wr) with the
    # -- centre at y=wr, so every synthesized wheel rests exactly on y=0
    # -- instead of floating, and r>=0.3 guarantees a filled disc per layer
    # -- rather than an empty printed ring. Slot order is stable:
    # -- front-left, front-right, rear-left, rear-right.
    wr = round(min(0.45, max(0.3, bh * 0.5)), 2)
    wh = round(wr * 2.0, 2)
    hx = round(max(0.5, bw / 2 - 0.5), 2)
    hz = round(max(0.3, bd / 2 - 0.15), 2)
    slots = {"wheel_front_left": (hx, wr, hz),
             "wheel_front_right": (hx, wr, -hz),
             "wheel_rear_left": (-hx, wr, hz),
             "wheel_rear_right": (-hx, wr, -hz)}
    taken = []
    kept = {}
    for a in actions:
        if _role_of(a.get("label", "")) != "wheel":
            continue
        c = _cyl_of(a)
        lo_r, hi_r = CAR_SPEC["wheel"]["radius"]
        if c is None or not (lo_r <= c[3] <= hi_r):
            log(a.get("label", "?"),
                f"invalid wheel (not a cylinder or radius outside "
                f"{lo_r}-{hi_r}): "
                "dropped, slot refilled deterministically")
            continue
        _cx, _cy, _cz, _r, _h = c
        base = _cy - _h / 2.0
        if not (-0.1 <= base <= 0.05):
            # A wheel that floats (or sinks) prints as a hovering ring, not a
            # wheel: drop it and refill the slot grounded instead. Geometry
            # below the bead is never fudged to hide this.
            log(a.get("label", "?"),
                f"wheel base y={base:.2f} is not on the ground: dropped, "
                f"slot refilled grounded")
            continue
        if any(math.dist((_cx, _cz), (tx, tz)) < 0.4 for tx, tz in taken):
            log(a.get("label", "?"),
                "duplicate wheel position: dropped, slot refilled")
            continue
        taken.append((_cx, _cz))
        # occupy the nearest free slot so synthesized wheels fill the rest
        best, bestd = None, None
        for sid, (sx, _sy, sz) in slots.items():
            if sid in kept:
                continue
            d = math.dist((_cx, _cz), (bx + sx, bz + sz))
            if bestd is None or d < bestd:
                best, bestd = sid, d
        if best is None:
            log(a.get("label", "?"), "no free wheel slot: dropped")
            continue
        if a.get("label") != best:
            log(a.get("label", "?"),
                f"assigned stable ID {best} (geometry preserved)")
            a = dict(a, label=best)
        kept[best] = a
    for sid, (sx, _sy, sz) in slots.items():
        if sid in kept:
            out.append(kept[sid])
        else:
            out.append({
                "action": "add_cylinder", "radiusTop": wr,
                "radiusBottom": wr, "height": wh,
                "position": [round(bx + sx, 2), wr,
                             round(bz + sz, 2)],
                "color": "0x222222", "label": sid, "material": "tire"})
            log(sid, f"missing: synthesized wheel at "
                     f"({bx + sx:.2f}, {wr}, {bz + sz:.2f})")

    # -- everything else (doors, extras) passes through in plan order.
    role_covered = {"body", "cabin", "windshield", "window", "wheel"}
    for a in actions:
        if _role_of(a.get("label", "")) not in role_covered:
            out.append(a)
    return out, repairs


# Canonical chair dimensions (scene units, Y up, front +X, rear -X):
# seat 1.1 x 0.12 x 1.1 centred [0, 0.51, 0]; legs 0.45 tall at the four
# seat corners; backrest panel 0.12 x 0.9 x 1.0 rising from the seat rear.
SEAT_DEFAULTS = {"width": 1.1, "height": 0.12, "depth": 1.1,
                 "position": [0, 0.51, 0], "color": "0x9C6B3F",
                 "material": "wood"}
LEG_SLOTS = (("leg_front_left", 1, 1), ("leg_front_right", 1, -1),
             ("leg_rear_left", -1, 1), ("leg_rear_right", -1, -1))


def repair_chair_plan(actions: list) -> tuple:
    """Returns (repaired_actions, repairs) for the chair category.

    Car-only components (body/cabin/windshield/wheel roles, tire material)
    are REMOVED with a loud logged reason — a chair must never print with
    car parts, no matter what the thinker proposed. Missing seat, backrest
    or legs are synthesized from the canonical dimensions; valid model
    geometry is preserved. Fatal schema problems raise ValueError."""
    from morphix_path import MAX_LINEAR
    if not isinstance(actions, list) or not actions:
        raise ValueError("chair plan must be a non-empty action list")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("chair plan entries must be JSON objects")
    repairs = []

    def log(component, reason):
        repairs.append({"component": component, "reason": reason})

    # -- category firewall: drop car parts, loudly. This is the repair half
    # -- of the leakage fix (check_plan errors if any survive it).
    clean = []
    for a in actions:
        role = _role_of(a.get("label", ""))
        if role in ("wheel", "cabin", "windshield") \
                or a.get("material") == "tire":
            log(a.get("label", "?"),
                f"car-only component removed from chair plan "
                f"(role {role!r})")
            continue
        clean.append(a)
    actions = clean

    def numbox(a, what):
        for key in ("width", "height", "depth"):
            v = a.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"{what} {a.get('label')!r}: {key} must be a number "
                    f"(got {v!r})")
            if not math.isfinite(v) or v <= 0:
                raise ValueError(
                    f"{what} {a.get('label')!r}: {key} must be finite and > 0 "
                    f"(got {v!r})")
            if v > MAX_LINEAR:
                raise ValueError(
                    f"{what} {a.get('label')!r}: {key}={v} exceeds the "
                    f"{MAX_LINEAR} scene limit")
        try:
            return (float(a["position"][i]) for i in range(3))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"{what} {a.get('label')!r}: position must be "
                             f"[x,y,z] numbers")

    # -- seat: exactly one box; synthesized when missing.
    seats = [a for a in actions
             if _role_of(a.get("label", "")) == "seat"]
    for s in seats:
        if s.get("action") != "add_box":
            raise ValueError(
                f"seat must be an add_box (got {s.get('action')!r} "
                f"for {s.get('label')!r})")
    for extra in seats[1:]:
        log(extra.get("label", "?"), "duplicate seat dropped (kept first)")
    if seats:
        seat = dict(seats[0])
        numbox(seat, "seat")
    else:
        seat = {"action": "add_box", "label": "seat",
                **{k: (list(v) if isinstance(v, list) else v)
                   for k, v in SEAT_DEFAULTS.items()}}
        log("seat", "missing: synthesized canonical seat")
    sx, sy, sz = (float(seat["position"][i]) for i in range(3))
    sw, sh, sd = (float(seat["width"]), float(seat["height"]),
                  float(seat["depth"]))
    seat_base, seat_top = sy - sh / 2, sy + sh / 2
    out = [seat]

    # -- backrest: one box at the seat rear; synthesized when missing.
    backs = [a for a in actions
             if _role_of(a.get("label", "")) == "backrest"]
    for b in backs:
        if b.get("action") != "add_box":
            raise ValueError(
                f"backrest must be an add_box (got {b.get('action')!r} "
                f"for {b.get('label')!r})")
    for extra in backs[1:]:
        log(extra.get("label", "?"),
            "duplicate backrest dropped (kept first)")
    if backs:
        back = dict(backs[0])
        numbox(back, "backrest")
    else:
        back = {"action": "add_box", "width": 0.12, "height": 0.9,
                "depth": round(min(sd - 0.1, 1.0), 2),
                "position": [round(sx - sw / 2 + 0.06, 2),
                             round(seat_top + 0.45, 2), round(sz, 2)],
                "color": "0x9C6B3F", "label": "backrest",
                "material": "wood"}
        log("backrest", "missing: synthesized canonical backrest at seat rear")
    out.append(back)

    # -- legs: keep valid distinct supports, drop the rest, fill to four
    # -- corner slots sized from the seat so their tops meet its base.
    leg_h = round(max(0.2, seat_base), 2)
    taken, kept = [], {}
    for a in actions:
        if _role_of(a.get("label", "")) != "leg":
            continue
        c = _box_of(a) or _cyl_of(a)
        if c is None:
            log(a.get("label", "?"),
                "invalid leg (not a box/cylinder with numeric geometry): "
                "dropped, slot refilled deterministically")
            continue
        _cx, _cy, _cz = c[0], c[1], c[2]
        base = _cy - c[4] / 2
        if not (-0.1 <= base <= 0.05):
            log(a.get("label", "?"),
                f"leg base y={base:.2f} is not on the ground: dropped, "
                f"slot refilled grounded")
            continue
        if any(math.dist((_cx, _cz), t) < 0.25 for t in taken):
            log(a.get("label", "?"),
                "duplicate leg position: dropped, slot refilled")
            continue
        taken.append((_cx, _cz))
        best, bestd = None, None
        for sid, qx, qz in LEG_SLOTS:
            if sid in kept:
                continue
            d = math.dist((_cx, _cz),
                          (sx + qx * (sw / 2 - 0.1),
                           sz + qz * (sd / 2 - 0.1)))
            if bestd is None or d < bestd:
                best, bestd = sid, d
        if best is None:
            log(a.get("label", "?"), "no free leg slot: dropped")
            continue
        if a.get("label") != best:
            log(a.get("label", "?"),
                f"assigned stable ID {best} (geometry preserved)")
            a = dict(a, label=best)
        kept[best] = a
    for sid, qx, qz in LEG_SLOTS:
        if sid in kept:
            out.append(kept[sid])
        else:
            out.append({
                "action": "add_cylinder", "radiusTop": 0.07,
                "radiusBottom": 0.07, "height": leg_h,
                "position": [round(sx + qx * (sw / 2 - 0.1), 2),
                             round(leg_h / 2, 2),
                             round(sz + qz * (sd / 2 - 0.1), 2)],
                "color": "0x9C6B3F", "label": sid, "material": "wood"})
            log(sid, "missing: synthesized grounded leg")

    # -- anything else (armrests, stretchers) passes through in plan order;
    # -- the compiler, not the repair, is the authority on printability.
    role_covered = {"seat", "backrest", "leg"}
    for a in actions:
        if _role_of(a.get("label", "")) not in role_covered:
            out.append(a)
    return out, repairs


def repair_stool_plan(actions: list) -> tuple:
    """Returns (repaired_actions, repairs) for the stool category: one seat
    on four grounded legs. A backrest makes it a chair, not a stool, so
    backrest parts are removed loudly; car parts are removed loudly."""
    from morphix_path import MAX_LINEAR
    if not isinstance(actions, list) or not actions:
        raise ValueError("stool plan must be a non-empty action list")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("stool plan entries must be JSON objects")
    repairs = []

    def log(component, reason):
        repairs.append({"component": component, "reason": reason})

    clean = []
    for a in actions:
        role = _role_of(a.get("label", ""))
        if role in ("wheel", "cabin", "windshield") \
                or a.get("material") == "tire":
            log(a.get("label", "?"),
                f"car-only component removed from stool plan "
                f"(role {role!r})")
            continue
        if role == "backrest":
            log(a.get("label", "?"),
                "backrest removed from stool plan (a stool has no "
                "backrest; prompt 'chair' for the chair design)")
            continue
        clean.append(a)
    actions = clean

    seats = [a for a in actions
             if _role_of(a.get("label", "")) == "seat"]
    for s in seats:
        if s.get("action") != "add_box":
            raise ValueError(
                f"seat must be an add_box (got {s.get('action')!r} "
                f"for {s.get('label')!r})")
    for extra in seats[1:]:
        log(extra.get("label", "?"), "duplicate seat dropped (kept first)")
    if seats:
        seat = dict(seats[0])
        for key in ("width", "height", "depth"):
            v = seat.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"seat {seat.get('label')!r}: {key} must be a number "
                    f"(got {v!r})")
            if not math.isfinite(v) or v <= 0 or v > MAX_LINEAR:
                raise ValueError(
                    f"seat {seat.get('label')!r}: bad {key} {v!r}")
        try:
            sx, sy, sz = (float(seat["position"][i]) for i in range(3))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"seat {seat.get('label')!r}: position must be "
                             f"[x,y,z] numbers")
    else:
        seat = {"action": "add_box", "label": "seat",
                **{k: (list(v) if isinstance(v, list) else v)
                   for k, v in SEAT_DEFAULTS.items()}}
        sx, sy, sz = (float(seat["position"][i]) for i in range(3))
        log("seat", "missing: synthesized canonical seat")
    sw, sh, sd = (float(seat["width"]), float(seat["height"]),
                  float(seat["depth"]))
    seat_base = sy - sh / 2
    out = [seat]

    leg_h = round(max(0.15, seat_base), 2)
    taken, kept = [], {}
    for a in actions:
        if _role_of(a.get("label", "")) != "leg":
            continue
        c = _box_of(a) or _cyl_of(a)
        if c is None:
            log(a.get("label", "?"),
                "invalid leg (not a box/cylinder with numeric geometry): "
                "dropped, slot refilled deterministically")
            continue
        _cx, _cy, _cz = c[0], c[1], c[2]
        base = _cy - c[4] / 2
        if not (-0.1 <= base <= 0.05):
            log(a.get("label", "?"),
                f"leg base y={base:.2f} is not on the ground: dropped, "
                f"slot refilled grounded")
            continue
        if any(math.dist((_cx, _cz), t) < 0.25 for t in taken):
            log(a.get("label", "?"),
                "duplicate leg position: dropped, slot refilled")
            continue
        taken.append((_cx, _cz))
        best, bestd = None, None
        for sid, qx, qz in LEG_SLOTS:
            if sid in kept:
                continue
            d = math.dist((_cx, _cz),
                          (sx + qx * (sw / 2 - 0.1),
                           sz + qz * (sd / 2 - 0.1)))
            if bestd is None or d < bestd:
                best, bestd = sid, d
        if best is None:
            log(a.get("label", "?"), "no free leg slot: dropped")
            continue
        if a.get("label") != best:
            log(a.get("label", "?"),
                f"assigned stable ID {best} (geometry preserved)")
            a = dict(a, label=best)
        kept[best] = a
    for sid, qx, qz in LEG_SLOTS:
        if sid in kept:
            out.append(kept[sid])
        else:
            out.append({
                "action": "add_cylinder", "radiusTop": 0.07,
                "radiusBottom": 0.07, "height": leg_h,
                "position": [round(sx + qx * (sw / 2 - 0.1), 2),
                             round(leg_h / 2, 2),
                             round(sz + qz * (sd / 2 - 0.1), 2)],
                "color": "0x9C6B3F", "label": sid, "material": "wood"})
            log(sid, "missing: synthesized grounded leg")

    role_covered = {"seat", "leg"}
    for a in actions:
        if _role_of(a.get("label", "")) not in role_covered:
            out.append(a)
    return out, repairs


# Canonical sitting-cat dimensions (scene units, Y up, front +X): body box
# with legs beneath, head sphere touching the body front-top, tapered ears
# on the head top, tapered tail grounded at the body rear.
CAT_BODY_DEFAULTS = {"width": 1.3, "height": 0.7, "depth": 0.7,
                     "position": [0, 0.75, 0], "color": "0xFFAA33",
                     "material": "body"}


def repair_cat_plan(actions: list) -> tuple:
    """Returns (repaired_actions, repairs) for the cat category.

    Car-only components are REMOVED loudly (a cat is never a stack of car
    parts); missing body/head/ears/legs/tail are synthesized from the
    canonical sitting-cat dimensions; valid model geometry is preserved."""
    from morphix_path import MAX_LINEAR
    if not isinstance(actions, list) or not actions:
        raise ValueError("cat plan must be a non-empty action list")
    if any(not isinstance(a, dict) for a in actions):
        raise ValueError("cat plan entries must be JSON objects")
    repairs = []

    def log(component, reason):
        repairs.append({"component": component, "reason": reason})

    clean = []
    for a in actions:
        role = _role_of(a.get("label", ""))
        if role in ("wheel", "cabin", "windshield") \
                or a.get("material") == "tire":
            log(a.get("label", "?"),
                f"car-only component removed from cat plan "
                f"(role {role!r})")
            continue
        clean.append(a)
    actions = clean

    def numbox(a, what):
        for key in ("width", "height", "depth"):
            v = a.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"{what} {a.get('label')!r}: {key} must be a number "
                    f"(got {v!r})")
            if not math.isfinite(v) or v <= 0 or v > MAX_LINEAR:
                raise ValueError(
                    f"{what} {a.get('label')!r}: bad {key} {v!r}")
        try:
            return (float(a["position"][i]) for i in range(3))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"{what} {a.get('label')!r}: position must be "
                             f"[x,y,z] numbers")

    # -- body: exactly one box; missing FIELDS take documented defaults.
    # A wrong-primitive body (the small thinker often emits a sphere or a
    # bare field set) is dropped LOUDLY and replaced by the canonical body
    # below — same drop-and-refill philosophy as the legs — instead of
    # killing the whole run over one part's opcode.
    bodies = [a for a in actions
              if _role_of(a.get("label", "")) == "body"]
    kept_bodies = []
    for b in bodies:
        if b.get("action") != "add_box":
            log(b.get("label", "?"),
                f"body is {b.get('action')!r}, not a printable box: "
                f"dropped, canonical body synthesized below")
            continue
        kept_bodies.append(b)
    bodies = kept_bodies
    for extra in bodies[1:]:
        log(extra.get("label", "?"), "duplicate body dropped (kept first)")
    if bodies:
        body = dict(bodies[0])
        numbox(body, "body")
    else:
        body = {"action": "add_box", "label": "body",
                **{k: (list(v) if isinstance(v, list) else v)
                   for k, v in CAT_BODY_DEFAULTS.items()}}
        log("body", "missing: synthesized canonical body")
    bx, by, bz = (float(body["position"][i]) for i in range(3))
    bw, bh, bd = (float(body["width"]), float(body["height"]),
                  float(body["depth"]))
    body_top, body_front = by + bh / 2, bx + bw / 2
    out = [body]

    # -- neck: a short tapered connector grafting the head onto the body.
    # -- Synthesized from the body when missing; a thinker-provided neck is
    # -- kept after a cylinder check. The head is then seated on the neck
    # -- top (or bridged to a kept head floating above the body).
    necks = [a for a in actions
             if _role_of(a.get("label", "")) == "neck"]
    kept_necks = []
    for n in necks:
        if _cyl_of(n) is None:
            log(n.get("label", "?"),
                f"neck is {n.get('action')!r}, not a printable cylinder: "
                f"dropped, canonical neck synthesized below")
            continue
        kept_necks.append(n)
    necks = kept_necks
    for extra in necks[1:]:
        log(extra.get("label", "?"), "duplicate neck dropped (kept first)")
    # -- neck + head: the head is grafted onto the body through a short
    # -- tapered neck (smoother than a ball perched on a box, still pure
    # -- stacked primitives). Radii scale with head size; positions are
    # -- derived, never guessed.
    hr_new = round(min(0.5, max(0.25, bw * 0.28)), 2)
    neck_h = round(min(0.5, max(0.2, bh * 0.43)), 2)
    heads = [a for a in actions
             if _role_of(a.get("label", "")) == "head"]
    kept_heads = []
    for h in heads:
        if _sphere_of(h) is None:
            log(h.get("label", "?"),
                f"head is {h.get('action')!r}, not a printable sphere: "
                f"dropped, canonical head synthesized below")
            continue
        kept_heads.append(h)
    heads = kept_heads
    for extra in heads[1:]:
        log(extra.get("label", "?"), "duplicate head dropped (kept first)")
    head_kept = dict(heads[0]) if heads else None
    hr = _sphere_of(head_kept)[3] if head_kept else hr_new

    def _neck_at(nx, base_y, h, color="0xFFAA33", label="neck"):
        rb = round(hr * 0.72, 2)
        rt = round(hr * 0.55, 2)
        return {"action": "add_cylinder", "radiusTop": rt,
                "radiusBottom": rb, "height": round(h, 2),
                "position": [round(nx, 2), round(base_y + h / 2, 2),
                             round(bz, 2)],
                "color": color, "label": label, "material": "body"}

    if necks:
        if _cyl_of(necks[0]) is None:
            log(necks[0].get("label", "?"),
                "invalid neck geometry: dropped, slot refilled below")
            necks = []
    if necks:
        out.append(dict(necks[0]))
        neck_top = None
        try:
            _nx, _ny, _nz, _nr, _nh = _cyl_of(necks[0])
            neck_top = _ny + _nh / 2
        except (TypeError, ValueError, IndexError):
            neck_top = None
    elif head_kept is None:
        # standard neck from the body; the head below sits on its top
        neck = _neck_at(round(bx + bw * 0.35, 2), by + bh / 2, neck_h)
        out.append(neck)
        neck_top = by + bh / 2 + neck_h
        log("neck", "missing: synthesized tapered neck on body top")
    else:
        _hx0, _hy0, _hz0, _hr0 = _sphere_of(head_kept)
        gap = (_hy0 - _hr0) - (by + bh / 2)
        if gap > 0.1:
            neck = _neck_at(_hx0, by + bh / 2, round(gap, 2))
            out.append(neck)
            neck_top = by + bh / 2 + round(gap, 2)
            log("neck", "missing: synthesized bridge neck under kept head")
        else:
            neck_top = None
            log("neck", "head sits directly on body; no neck needed")

    # -- head: one sphere; synthesized on the neck top when missing.
    if head_kept is not None:
        head = head_kept
    else:
        if neck_top is None:
            neck_top = by + bh / 2 + neck_h
        head = {"action": "add_sphere", "radius": hr,
                "position": [round(bx + bw * 0.35, 2),
                             round(neck_top + hr, 2), round(bz, 2)],
                "color": "0xFFAA33", "label": "head",
                "material": "body"}
        log("head", "missing: synthesized head on neck top")
    out.append(head)
    _hx, hy, _hz, hr = _sphere_of(head)

    # -- ears: two tapered cylinders on the head top; synthesized as a pair.
    # Wrong-primitive ears are dropped loudly and the slots refilled, exactly
    # like invalid legs — never a whole-run failure over one opcode.
    ears = [a for a in actions
            if _role_of(a.get("label", "")) == "ear"]
    kept_ears = []
    for e in ears:
        if _cyl_of(e) is None:
            log(e.get("label", "?"),
                f"ear is {e.get('action')!r}, not a printable cylinder: "
                f"dropped, canonical ear synthesized below")
            continue
        kept_ears.append(e)
    ears = kept_ears
    for extra in ears[2:]:
        log(extra.get("label", "?"), "extra ear dropped (a cat has two)")
    slots = [("ear_left", 1), ("ear_right", -1)]
    taken, kept = [], {}
    for a in ears[:2]:
        c = _cyl_of(a)
        _cx, _cy, _cz, _r, _h = c
        if any(math.dist((_cx, _cz), t) < 0.15 for t in taken):
            log(a.get("label", "?"),
                "duplicate ear position: dropped, slot refilled")
            continue
        taken.append((_cx, _cz))
        best = "ear_left" if _cz >= _hz else "ear_right"
        if best in kept:
            best = "ear_right" if best == "ear_left" else "ear_left"
        if best in kept:
            log(a.get("label", "?"), "no free ear slot: dropped")
            continue
        if a.get("label") != best:
            log(a.get("label", "?"),
                f"assigned stable ID {best} (geometry preserved)")
            a = dict(a, label=best)
        kept[best] = a
    for sid, side in slots:
        if sid in kept:
            out.append(kept[sid])
        else:
            er, eh = 0.12, 0.28
            out.append({
                "action": "add_cylinder", "radiusTop": 0.03,
                "radiusBottom": er, "height": eh,
                "position": [round(_hx, 2),
                             round(hy + hr + eh / 2, 2),
                             round(_hz + side * 0.16, 2)],
                "color": "0xFFAA33", "label": sid, "material": "body"})
            log(sid, "missing: synthesized tapered ear on head top")
    # -- legs: four grounded supports under the body.
    leg_h = round(max(0.2, by - bh / 2), 2)
    taken, kept = [], {}
    for a in actions:
        if _role_of(a.get("label", "")) != "leg":
            continue
        c = _box_of(a) or _cyl_of(a)
        if c is None:
            log(a.get("label", "?"),
                "invalid leg (not a box/cylinder with numeric geometry): "
                "dropped, slot refilled deterministically")
            continue
        _cx, _cy, _cz = c[0], c[1], c[2]
        base = _cy - c[4] / 2
        if not (-0.1 <= base <= 0.05):
            log(a.get("label", "?"),
                f"leg base y={base:.2f} is not on the ground: dropped, "
                f"slot refilled grounded")
            continue
        if any(math.dist((_cx, _cz), t) < 0.25 for t in taken):
            log(a.get("label", "?"),
                "duplicate leg position: dropped, slot refilled")
            continue
        taken.append((_cx, _cz))
        best, bestd = None, None
        for lid, qx, qz in (("leg_front_left", 1, 1),
                            ("leg_front_right", 1, -1),
                            ("leg_rear_left", -1, 1),
                            ("leg_rear_right", -1, -1)):
            if lid in kept:
                continue
            d = math.dist((_cx, _cz),
                          (bx + qx * (bw / 2 - 0.15),
                           bz + qz * (bd / 2 - 0.1)))
            if bestd is None or d < bestd:
                best, bestd = lid, d
        if best is None:
            log(a.get("label", "?"), "no free leg slot: dropped")
            continue
        if a.get("label") != best:
            log(a.get("label", "?"),
                f"assigned stable ID {best} (geometry preserved)")
            a = dict(a, label=best)
        kept[best] = a
    for lid, qx, qz in (("leg_front_left", 1, 1),
                        ("leg_front_right", 1, -1),
                        ("leg_rear_left", -1, 1),
                        ("leg_rear_right", -1, -1)):
        if lid in kept:
            out.append(kept[lid])
        else:
            out.append({
                "action": "add_cylinder", "radiusTop": 0.07,
                "radiusBottom": 0.10, "height": leg_h,
                "position": [round(bx + qx * (bw / 2 - 0.15), 2),
                             round(leg_h / 2, 2),
                             round(bz + qz * (bd / 2 - 0.1), 2)],
                "color": "0xFFAA33", "label": lid, "material": "body"})
            log(lid, "missing: synthesized grounded tapered leg")
    # -- tail: one tapered cylinder grounded at the body rear.
    # A wrong-primitive tail is dropped loudly and synthesized, like legs.
    tails = [a for a in actions
             if _role_of(a.get("label", "")) == "tail"]
    kept_tails = []
    for t in tails:
        if _cyl_of(t) is None:
            log(t.get("label", "?"),
                f"tail is {t.get('action')!r}, not a printable cylinder: "
                f"dropped, canonical tail synthesized below")
            continue
        kept_tails.append(t)
    tails = kept_tails
    # Positional validation: a "tail" nowhere near the body rear is a
    # mislabeled part (observed: an ear spike tagged tail_a1) hijacking
    # the tail slot and blocking the real tail synthesis. Drop it loudly
    # so the canonical rear tail below is built instead.
    rear_tails = []
    for t in tails:
        try:
            tx = float(t["position"][0])
        except (TypeError, ValueError, IndexError, KeyError):
            tx = bx
        if tx < bx:
            rear_tails.append(t)
        else:
            log(t.get("label", "?"),
                f"tail at x={tx:.2f} is not at the body rear "
                f"(body x={bx:.2f}): dropped, canonical tail "
                f"synthesized below")
    tails = rear_tails
    for extra in tails[1:]:
        log(extra.get("label", "?"), "duplicate tail dropped (kept first)")
    if tails:
        out.append(dict(tails[0]))
    else:
        # grounded at the rear flank, touching (not floating beside) it
        out.append({
            "action": "add_cylinder", "radiusTop": 0.04,
            "radiusBottom": 0.09, "height": 1.2,
            "position": [round(bx - bw / 2 - 0.09, 2), 0.6,
                         round(bz + 0.25, 2)],
            "color": "0xFFAA33", "label": "tail", "material": "body"})
        log("tail", "missing: synthesized tapered tail at body rear")

    role_covered = {"body", "neck", "head", "ear", "leg", "tail"}
    for a in actions:
        if _role_of(a.get("label", "")) not in role_covered:
            out.append(a)
    return out, repairs


# Category registry: which repair owns which plan. Repair runs ONLY against
# the selected category's specification — a missing car wheel is never
# inserted into a chair/table/house/stool plan, and car repair never touches
# a non-car plan (it raises on a missing body instead of inventing one).
REPAIR_FOR_SPEC = {
    "car": repair_car_plan,
    "chair": repair_chair_plan,
    "stool": repair_stool_plan,
    "cat": repair_cat_plan,
}


def repair_plan_for_spec(spec_name: str, actions: list) -> tuple:
    """Dispatch to the selected category's repair. Unknown categories get
    their actions back untouched (no repair, no invention)."""
    fn = REPAIR_FOR_SPEC.get(spec_name or "")
    if fn is None:
        return list(actions), []
    return fn(actions)
