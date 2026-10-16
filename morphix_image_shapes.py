# morphix_image_shapes.py
# Shape-first parametric builders for image -> 3D, a software silhouette
# projector, and a bounded fitter that matches viewpoint + proportions to
# the measured reference silhouette.
#
# Build order (recorded per component as "stage"):
#   1 silhouette       overall proportions + viewpoint, fitted to the mask
#   2 main_volume      the dominant mass as a real cross-section (car side
#                      profile, house walls, animal torso, revolved vessel)
#   3 structure        major parts placed relative to the main volume
#                      (wheels, cabin/greenhouse, legs, roof, head)
#   4 secondary        small parts (lights, door, windows, ears, tail)
#   5 surface          measured paint per material slot
#   6 camera_lighting  verification camera from the fitted viewpoint
#
# Geometry uses the printer's sliceable vocabulary: add_profile (extruded
# cross-section — a car body is its side profile with wheel arches, wheels
# are horizontal-axis circles, a roof is a gable triangle), add_lathe
# (revolved half-silhouette), boxes and cylinders where they are the right
# shape (table legs, a chimney). A complex subject never defaults to a box.
#
# Conventions: y up, ground y=0, front = +x, left = +z, units = scene.
# Nothing here reads the reference's absolute pixel sizes or colors as
# constants: proportions come from a preset family fitted to the measured
# silhouette, colors from the segmentation clusters.

import math

import numpy as np
from PIL import Image, ImageDraw

STAGES = ("silhouette", "main_volume", "structure", "secondary", "surface",
          "camera_lighting")
FOV = 30.0  # projector + verification camera field of view (degrees)


# ------------------------------------------------------------------ helpers
def _r(v, n=4):
    return round(float(v), n)


def _centre_profile(pts):
    """Centre a polygon on its bbox. Returns (local_pts, (cu, cy))."""
    us = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cu, cy = (min(us) + max(us)) / 2, (min(ys) + max(ys)) / 2
    return [[_r(u - cu), _r(y - cy)] for u, y in pts], (cu, cy)


def profile_comp(cid, pts_world, z_or_x, depth, material, color, stage,
                 role=None, depth_top=None, axis="z", required=True,
                 observed=True):
    """Profile component from WORLD (u, y) polygon points. For axis z, u is
    world x and the extrusion is centred at world z = z_or_x; for axis x, u
    is world z and the extrusion is centred at world x = z_or_x."""
    local, (cu, cy) = _centre_profile(pts_world)
    us = [p[0] for p in local]
    ys = [p[1] for p in local]
    dmax = max(depth, depth_top or depth)
    if axis == "z":
        pos = [_r(cu), _r(cy), _r(z_or_x)]
        dims = [_r(max(us) - min(us)), _r(max(ys) - min(ys)), _r(dmax)]
    else:
        pos = [_r(z_or_x), _r(cy), _r(cu)]
        dims = [_r(dmax), _r(max(ys) - min(ys)), _r(max(us) - min(us))]
    c = {"id": cid, "role": role or cid.split("_")[0], "primitive": "profile",
         "profile": local, "depth": _r(depth), "axis": axis, "dims": dims,
         "position": pos, "material": material, "color": color,
         "stage": stage, "required": bool(required),
         "observed": bool(observed), "confidence": 0.7, "parent": None,
         "constraints": []}
    if depth_top is not None and abs(depth_top - depth) > 1e-6:
        c["depth_top"] = _r(depth_top)
    return c


def box_comp(cid, w, h, d, pos, material, color, stage, role=None,
             required=True, observed=True):
    return {"id": cid, "role": role or cid.split("_")[0], "primitive": "box",
            "dims": [_r(w), _r(h), _r(d)], "position": [_r(v) for v in pos],
            "material": material, "color": color, "stage": stage,
            "required": bool(required), "observed": bool(observed),
            "confidence": 0.7, "parent": None, "constraints": []}


def cyl_comp(cid, r, h, pos, material, color, stage, role=None, rt=None,
             required=True, observed=True):
    c = {"id": cid, "role": role or cid.split("_")[0],
         "primitive": "cylinder", "dims": [_r(2 * r), _r(h), _r(2 * r)],
         "position": [_r(v) for v in pos], "material": material,
         "color": color, "stage": stage, "required": bool(required),
         "observed": bool(observed), "confidence": 0.7, "parent": None,
         "constraints": []}
    if rt is not None and abs(rt - r) > 1e-6:
        c["taper"] = [_r(rt), _r(r)]
        c["dims"] = [_r(2 * max(r, rt)), _r(h), _r(2 * max(r, rt))]
    return c


def lathe_comp(cid, pts_world, cx, cz, material, color, stage, role=None,
               required=True, observed=True):
    """Lathe from WORLD [[r, y], ...] (y ascending)."""
    ys = [p[1] for p in pts_world]
    cy = (min(ys) + max(ys)) / 2
    local = [[_r(max(0.0, r)), _r(y - cy)] for r, y in pts_world]
    R = max(p[0] for p in local)
    return {"id": cid, "role": role or cid.split("_")[0],
            "primitive": "lathe", "profile": local,
            "dims": [_r(2 * R), _r(max(ys) - min(ys)), _r(2 * R)],
            "position": [_r(cx), _r(cy), _r(cz)], "material": material,
            "color": color, "stage": stage, "required": bool(required),
            "observed": bool(observed), "confidence": 0.7, "parent": None,
            "constraints": []}


def circle(cx, cy, r, n=24):
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def clip_below(pts, ycut, keep="below"):
    """Sutherland-Hodgman clip of a polygon against y = ycut."""
    def inside(p):
        return p[1] <= ycut + 1e-12 if keep == "below" else p[1] >= ycut - 1e-12
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ia, ib = inside(a), inside(b)
        if ia:
            out.append(a)
        if ia != ib:
            t = (ycut - a[1]) / (b[1] - a[1])
            out.append((a[0] + (b[0] - a[0]) * t, ycut))
    return out


# ------------------------------------------------------------------ colors
def _luma(hx):
    return sum(int(hx[i:i + 2], 16) for i in (2, 4, 6)) / 3.0


def _blueish(hx):
    r, g, b = (int(hx[i:i + 2], 16) for i in (2, 4, 6))
    return b > r + 25 and b >= g - 10


def paint_from_clusters(clusters, family):
    """Measured colors per role from the subject's color clusters (sorted by
    area). body = largest; glass = a blue-ish or very dark cluster in the
    upper half; light = the brightest small cluster. Defaults are used and
    REPORTED when a role has no measurement."""
    notes = []
    body = clusters[0]["color"] if clusters else "0x888888"
    glass = None
    for c in clusters[1:]:
        if c["cy"] <= 0.55 and (_blueish(c["color"]) or c["luma"] < 60
                                or c["luma"] > 150):
            glass = c["color"]
            break
    if glass is None:
        glass = "0x2A3540" if _luma(body) > 90 else "0x6F8796"
        notes.append(f"glass color not measured: default {glass}")
    light = None
    small = [c for c in clusters if c["fraction"] <= 0.12 and c["luma"] > 170]
    if small:
        light = max(small, key=lambda c: c["luma"])["color"]
    if light is None:
        light = "0xF3EED9"
        notes.append("light color not measured: default warm white")
    second = clusters[1]["color"] if len(clusters) > 1 else body
    return {"body": body, "glass": glass, "light": light, "second": second,
            "notes": notes}


# ------------------------------------------------------------------ builders
CAR_PRESETS = {
    # hr: height/length  belt: beltline/height  nose: nose top/beltline
    # rf/rb: roof front/rear x (fraction of L)  wsb/rbb: windshield / rear
    # glass base x  clear: body bottom/L  wr: wheel radius/L  wb: wheelbase/L
    "sports": dict(hr=0.25, belt=0.56, nose=0.72, rf=0.03, rb=-0.20,
                   wsb=0.22, rbb=-0.36, clear=0.030, wr=0.072, wb=0.60,
                   width=0.42),
    "sedan": dict(hr=0.29, belt=0.56, nose=0.82, rf=0.07, rb=-0.20,
                  wsb=0.24, rbb=-0.33, clear=0.035, wr=0.070, wb=0.60,
                  width=0.40),
    "suv": dict(hr=0.34, belt=0.60, nose=0.9, rf=0.12, rb=-0.34,
                wsb=0.26, rbb=-0.44, clear=0.045, wr=0.080, wb=0.60,
                width=0.40),
    "hatch": dict(hr=0.32, belt=0.56, nose=0.82, rf=0.08, rb=-0.36,
                  wsb=0.23, rbb=-0.46, clear=0.035, wr=0.072, wb=0.62,
                  width=0.41),
    "tall": dict(hr=0.44, belt=0.55, nose=0.92, rf=0.13, rb=-0.30,
                 wsb=0.24, rbb=-0.40, clear=0.050, wr=0.082, wb=0.60,
                 width=0.42),
    "boxy": dict(hr=0.31, belt=0.55, nose=0.97, rf=0.10, rb=-0.14,
                 wsb=0.20, rbb=-0.22, clear=0.040, wr=0.072, wb=0.60,
                 width=0.40),
}
CAR_LENGTH = 6.0


def build_car(p, paint):
    """Vehicle: side-profile body with wheel arches (main volume), tapered
    greenhouse + roof (structure), 4 horizontal-axis wheels (structure),
    headlights (secondary)."""
    L = CAR_LENGTH
    H = p["hr"] * L
    W = p["width"] * L
    rw = p["wr"] * L
    # >= one full course: a base inside the first course band is seated
    # to the ground by the slicer (which would open a gap above it)
    clear = max(0.22, p["clear"] * L)
    belt = p["belt"] * H
    nose = max(clear + 0.25, p["nose"] * belt)
    xf, xr = p["wb"] * L / 2, -p["wb"] * L / 2
    ra = rw * 1.12
    pts = [(-L / 2 + 0.12, clear)]

    def arch(xw):
        a = math.asin(max(-1.0, min(1.0, (clear - rw) / ra)))
        n = 10
        out = []
        for i in range(n + 1):
            th = (math.pi - a) + (a - (math.pi - a)) * i / n
            out.append((xw + ra * math.cos(th), rw + ra * math.sin(th)))
        return out
    pts += arch(xr)
    pts += arch(xf)
    pts += [(L / 2 - 0.12, clear), (L / 2, clear + 0.18), (L / 2, nose - 0.08),
            (L / 2 - 0.15, nose), (p["wsb"] * L, belt),
            (p["rbb"] * L, belt), (-L / 2 + 0.15, belt * 0.97),
            (-L / 2, belt * 0.88), (-L / 2, clear + 0.18)]
    comps = [profile_comp("car_body", pts, 0.0, W, "body", paint["body"],
                          "main_volume", role="body", depth_top=W * 0.96)]
    # greenhouse: windshield rake to the roof, rear glass down to the deck
    gh = [(p["wsb"] * L, belt), (p["rf"] * L, H),
          ((p["rf"] + p["rb"]) * L / 2, H), (p["rb"] * L, H),
          (p["rbb"] * L, belt)]
    roof_t = max(0.2, 0.18 * (H - belt))
    glass = clip_below(gh, H - roof_t, "below")
    roof = clip_below(gh, H - roof_t, "above")
    # glass tapers 0.86W -> 0.70W over its own height; the roof continues
    # the same taper so it sits flush (no overhang)
    gt = W * 0.86 - (W * 0.86 - W * 0.70) * 1.0
    comps.append(profile_comp("cabin_glass", glass, 0.0, W * 0.86, "glass",
                              paint["glass"], "structure", role="cabin",
                              depth_top=gt))
    if len(roof) >= 3:
        comps.append(profile_comp("cabin_roof", roof, 0.0, gt, "body",
                                  paint["body"], "structure", role="roof",
                                  depth_top=gt * 0.94))
    tw = 0.16 * W
    for name, xw, zs in (("front_left", xf, 1), ("front_right", xf, -1),
                         ("rear_left", xr, 1), ("rear_right", xr, -1)):
        comps.append(profile_comp(f"wheel_{name}", circle(xw, rw, rw),
                                  zs * (W / 2 - tw / 2 - 0.02), tw, "tire",
                                  "0x1A1A1C", "structure", role="wheel"))
    hl_y = nose - 0.18
    for zs, side in ((1, "left"), (-1, "right")):
        comps.append(box_comp(f"headlight_{side}", 0.16, 0.18, 0.42,
                              (L / 2 - 0.08, hl_y, zs * W * 0.32), "accent",
                              paint["light"], "secondary", role="light",
                              required=False))
    return comps


TABLE_PRESETS = {
    "dining": dict(hr=0.42, wr=0.55, top=0.045, leg=0.045, inset=0.06),
    "coffee": dict(hr=0.26, wr=0.55, top=0.06, leg=0.06, inset=0.05),
    "desk": dict(hr=0.45, wr=0.45, top=0.04, leg=0.04, inset=0.03),
    "square": dict(hr=0.6, wr=1.0, top=0.05, leg=0.06, inset=0.06),
}
TABLE_LENGTH = 3.6


def build_table(p, paint):
    L = TABLE_LENGTH
    H, W = p["hr"] * L, p["wr"] * L
    top = max(0.2, p["top"] * L)
    leg = max(0.2, p["leg"] * L)
    ins = p["inset"] * L
    comps = [box_comp("tabletop", L, top, W, (0, H - top / 2, 0), "wood",
                      paint["body"], "main_volume", role="tabletop")]
    lh = H - top
    for name, sx, sz in (("front_left", 1, 1), ("front_right", 1, -1),
                         ("rear_left", -1, 1), ("rear_right", -1, -1)):
        comps.append(box_comp(f"leg_{name}", leg, lh, leg,
                              (sx * (L / 2 - ins - leg / 2), lh / 2,
                               sz * (W / 2 - ins - leg / 2)), "wood",
                              paint["second"] if _luma(paint["second"]) > 30
                              else paint["body"], "structure", role="leg"))
    return comps


# Proportions relative to the seat width (a real dining chair is about as
# deep and as high at the seat as it is wide, with a back of similar height).
CHAIR_PRESETS = {
    "dining": dict(seat_h=1.0, back_h=1.0, depth=1.0, leg=0.1,
                   seat_t=0.12, back_t=0.12),
    "tall_back": dict(seat_h=0.95, back_h=1.35, depth=0.95, leg=0.09,
                      seat_t=0.11, back_t=0.11),
    "low_back": dict(seat_h=1.0, back_h=0.7, depth=1.05, leg=0.11,
                     seat_t=0.13, back_t=0.13),
}
CHAIR_WIDTH = 1.6


def build_chair(p, paint):
    S = CHAIR_WIDTH
    D = p["depth"] * S
    sh, bh = p["seat_h"] * S * 1.0, p["back_h"] * S
    leg = max(0.2, p["leg"] * S)
    st = max(0.2, p["seat_t"] * S)
    bt = max(0.2, p["back_t"] * S)
    comps = [box_comp("seat", D, st, S, (0, sh - st / 2, 0), "wood",
                      paint["body"], "main_volume", role="seat")]
    lh = sh - st
    for name, sx, sz in (("front_left", 1, 1), ("front_right", 1, -1),
                         ("rear_left", -1, 1), ("rear_right", -1, -1)):
        comps.append(box_comp(f"leg_{name}", leg, lh, leg,
                              (sx * (D / 2 - leg / 2), lh / 2,
                               sz * (S / 2 - leg / 2)), "wood",
                              paint["body"], "structure", role="leg"))
    comps.append(box_comp("backrest", bt, bh, S, (-D / 2 + bt / 2,
                                                  sh + bh / 2, 0), "wood",
                          paint["body"], "structure", role="backrest"))
    return comps


STOOL_PRESETS = {
    "bar": dict(h=1.0, seat_r=0.22, legs=4, seat_t=0.07, spread=0.26),
    "low": dict(h=0.55, seat_r=0.25, legs=3, seat_t=0.08, spread=0.28),
}
STOOL_HEIGHT = 2.6


def build_stool(p, paint):
    H = p["h"] * STOOL_HEIGHT
    R = p["seat_r"] * STOOL_HEIGHT
    st = max(0.2, p["seat_t"] * STOOL_HEIGHT)
    comps = [cyl_comp("seat", R, st, (0, H - st / 2, 0), "wood",
                      paint["body"], "main_volume", role="seat")]
    n = int(p["legs"])
    sp = p["spread"] * STOOL_HEIGHT
    for i in range(n):
        a = 2 * math.pi * i / n + math.pi / 4
        comps.append(cyl_comp(f"leg_{i}", 0.12, H - st,
                              (sp * math.cos(a), (H - st) / 2,
                               sp * math.sin(a)), "wood", paint["body"],
                              "structure", role="leg"))
    return comps


HOUSE_PRESETS = {
    "cottage": dict(wall=0.45, depth=0.7, roof=0.45, over=0.05),
    "tall": dict(wall=0.65, depth=0.7, roof=0.4, over=0.05),
    "wide": dict(wall=0.35, depth=0.6, roof=0.3, over=0.06),
}
HOUSE_LENGTH = 4.4


def build_house(p, paint, roof_color=None):
    L = HOUSE_LENGTH
    Hw, D = p["wall"] * L, p["depth"] * L
    Hr = p["roof"] * L
    ov = p["over"] * L
    comps = [box_comp("walls", L, Hw, D, (0, Hw / 2, 0), "body",
                      paint["body"], "main_volume", role="wall")]
    # gable roof: triangle across the depth (z), ridge along the length (x)
    tri = [(-D / 2 - ov, Hw), (D / 2 + ov, Hw), (0.0, Hw + Hr)]
    comps.append(profile_comp("roof", tri, 0.0, L + 2 * ov, "wood",
                              roof_color or paint["second"], "structure",
                              role="roof", axis="x"))
    comps.append(box_comp("door", 0.7, min(1.4, Hw * 0.75), 0.2,
                          (0.0, min(1.4, Hw * 0.75) / 2, D / 2 + 0.1),
                          "accent", "0x5A3A22", "secondary", role="door"))
    for side, x in (("left", -L * 0.3), ("right", L * 0.3)):
        comps.append(box_comp(f"window_{side}", 0.7, 0.6, 0.2,
                              (x, Hw * 0.6, D / 2 + 0.1), "glass",
                              paint["glass"], "secondary", role="window",
                              required=False))
    comps.append(box_comp("chimney", 0.4, Hr * 0.9, 0.4,
                          (L * 0.28, Hw + Hr * 0.45 + 0.2, -D * 0.18),
                          "body", paint["body"], "secondary",
                          role="chimney", required=False))
    return comps


QUADRUPED_PRESETS = {
    "cat": dict(body_h=0.30, leg_h=0.26, head=0.17, tail=0.5, neck=0.05,
                depth=0.32),
    "dog": dict(body_h=0.32, leg_h=0.36, head=0.17, tail=0.35, neck=0.1,
                depth=0.30),
    "horse": dict(body_h=0.36, leg_h=0.55, head=0.15, tail=0.35, neck=0.3,
                  depth=0.26),
    "cow": dict(body_h=0.42, leg_h=0.40, head=0.15, tail=0.3, neck=0.08,
                depth=0.34),
}
ANIMAL_LENGTH = 3.0


def ellipsoid_comp(cid, cx, cy, cz, rx, ry, rz, material, color, stage,
                   role=None, n=36, required=True):
    """A true ellipsoid as an inflate grid (half-thickness of the ellipse
    cross-section at every x, y): rounded from every side, sliceable."""
    cell = 2 * max(rx, ry) / n
    cols, rows = int(round(2 * rx / cell)), int(round(2 * ry / cell))
    half = []
    for j in range(rows):
        y = -ry + (j + 0.5) * cell
        row = []
        for i in range(cols):
            x = -rx + (i + 0.5) * cell
            v = 1.0 - (x / rx) ** 2 - (y / ry) ** 2
            row.append(round(rz * math.sqrt(v), 4) if v > 0 else 0.0)
        half.append(row)
    return {"id": cid, "role": role or cid.split("_")[0],
            "primitive": "inflate",
            "grid": {"rows": rows, "cols": cols, "cell": round(cell, 5),
                     "half": half},
            "dims": [_r(cols * cell), _r(rows * cell), _r(2 * rz)],
            "position": [_r(cx), _r(cy), _r(cz)], "material": material,
            "color": color, "stage": stage, "required": bool(required),
            "observed": True, "confidence": 0.7, "parent": None,
            "constraints": []}


def build_quadruped(p, paint):
    """Torso as an elliptical side profile, head as a revolved sphere, four
    legs, ears and a tail. Proportions per preset, fitted to the mask."""
    L = ANIMAL_LENGTH
    bh, lh = p["body_h"] * L, p["leg_h"] * L
    D = p["depth"] * L
    bl = L * 0.62
    cy = lh + bh / 2
    # torso: a true ellipsoid (rounded from every side), not a flat disc
    comps = [ellipsoid_comp("torso", 0.0, cy, 0.0, bl / 2, bh / 2, D / 2,
                            "body", paint["body"], "main_volume",
                            role="body")]
    hr = p["head"] * L
    hx, hy = bl / 2 + hr * 0.4, cy + bh / 2 + p["neck"] * L
    head = [(hr * math.sin(t), hy + -hr * math.cos(t))
            for t in [math.pi * i / 10 for i in range(11)]]
    comps.append(lathe_comp("head", head, hx, 0.0, "body", paint["body"],
                            "structure", role="head"))
    if p["neck"] > 0.07:
        comps.append(profile_comp(
            "neck", [(bl / 2 - hr * 0.6, cy), (bl / 2 + hr * 0.2, cy),
                     (hx + hr * 0.3, hy), (hx - hr * 0.6, hy)], 0.0,
            D * 0.5, "body", paint["body"], "structure", role="neck"))
    legr = max(0.11, 0.035 * L)
    for name, sx, sz in (("front_left", 1, 1), ("front_right", 1, -1),
                         ("rear_left", -1, 1), ("rear_right", -1, -1)):
        comps.append(cyl_comp(f"leg_{name}", legr, lh + bh * 0.25,
                              (sx * bl * 0.36, (lh + bh * 0.25) / 2,
                               sz * D * 0.3), "body", paint["body"],
                              "structure", role="leg"))
    for side, zs in (("left", 1), ("right", -1)):
        comps.append(cyl_comp(f"ear_{side}", hr * 0.38, hr * 0.7,
                              (hx - hr * 0.1, hy + hr * 0.95,
                               zs * hr * 0.5), "body", paint["body"],
                              "secondary", role="ear", rt=0.05,
                              required=False))
    tl = p["tail"] * L
    tail = [(-bl / 2 + 0.05, cy + bh * 0.2), (-bl / 2 - tl * 0.6, cy + bh * 0.5),
            (-bl / 2 - tl, cy + bh * 0.9), (-bl / 2 - tl + 0.12, cy + bh),
            (-bl / 2 - tl * 0.5, cy + bh * 0.7), (-bl / 2 + 0.05, cy + bh * 0.45)]
    comps.append(profile_comp("tail", tail, 0.0, 0.2, "body", paint["body"],
                              "secondary", role="tail", required=False))
    return comps


# ------------------------------------------------- measured-silhouette build
def mask_outline(mask: np.ndarray, max_pts: int = 48):
    """Outer contour of a binary mask as (x, y_up) pixel points, simplified.
    OpenCV when available, else a column-scan upper/lower envelope."""
    try:
        import cv2
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            eps = 0.004 * peri
            ap = cv2.approxPolyDP(c, eps, True)
            while len(ap) > max_pts:
                eps *= 1.4
                ap = cv2.approxPolyDP(c, eps, True)
            return [(float(p[0][0]), float(-p[0][1])) for p in ap]
    except ImportError:
        pass
    ys, xs = np.where(mask)
    top, bot = [], []
    for x in range(xs.min(), xs.max() + 1, max(1, (xs.max() - xs.min()) // 24)):
        col = np.where(mask[:, x])[0]
        if len(col):
            top.append((float(x), float(-col.min())))
            bot.append((float(x), float(-col.max())))
    return top + bot[::-1]


def build_extrusion(mask, paint, length=3.4, depth_ratio=0.35):
    """Unfamiliar object seen roughly side-on: its measured outline, scaled
    to `length`, extruded by an ASSUMED depth (reported)."""
    pts = mask_outline(mask)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # largest side -> `length`, height capped to the printable layer budget
    s = min(length / max(1e-6, max(xs) - min(xs)),
            5.6 / max(1e-6, max(ys) - min(ys)))
    length = s * (max(xs) - min(xs))
    y0 = min(ys)
    world = [((x - (min(xs) + max(xs)) / 2) * s, (y - y0) * s) for x, y in pts]
    return [profile_comp("silhouette_body", world, 0.0, length * depth_ratio,
                         "body", paint["body"], "main_volume", role="body")]


INFLATE_SIZE = 3.6      # largest side of an inflated subject (scene units)
INFLATE_THICKNESS = 0.32  # peak half-thickness / sqrt(width*height): ASSUMED


def build_inflate(mask, paint, size=INFLATE_SIZE, max_cols=110):
    """Unfamiliar subject: its measured silhouette INFLATED into a rounded
    solid (front/back symmetric about z=0). Half-thickness follows a
    circular profile of the distance to the outline, smoothed, so the
    surface curves like a real body instead of a flat cut-out. The peak
    thickness is an assumption (a single photo has no depth) and is
    reported as such. The grid also carries the photo uv frame so the
    presentation can project the photograph onto the front surface."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    H, W = crop.shape
    step = max(1, int(math.ceil(max(W, H) / float(max_cols))))
    s = min(size / max(W, H), 5.6 / H)          # px -> world
    dist = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    dmax = float(dist.max()) or 1.0
    world_w, world_h = W * s, H * s
    tpeak = INFLATE_THICKNESS * math.sqrt(world_w * world_h)
    r = np.clip(dist / dmax, 0.0, 1.0)
    t = tpeak * np.sqrt(np.clip(1.0 - (1.0 - r) ** 2, 0.0, 1.0))
    t = gaussian_filter(t, 0.8) * crop
    rows = []
    for j in range(H - 1, -1, -step):            # row 0 = bottom of subject
        rows.append([round(float(t[j, i]), 4) for i in range(0, W, step)])
    cols = len(rows[0])
    cell = step * s
    hh, mh = mask.shape
    grid = {"rows": len(rows), "cols": cols, "cell": round(cell, 5),
            "half": rows,
            # photo frame (normalized image coords) of column 0 / row 0
            "uv": {"u0": (x0 + 0.5) / mh, "du": step / float(mh),
                   "v0": 1.0 - (y1 + 0.5) / hh, "dv": step / float(hh)}}
    h_w = len(rows) * cell
    c = {"id": "silhouette_body", "role": "body", "primitive": "inflate",
         "grid": grid,
         "dims": [_r(cols * cell), _r(h_w), _r(2 * tpeak)],
         "position": [0.0, _r(h_w / 2), 0.0], "material": "body",
         "color": paint["body"], "stage": "main_volume", "required": True,
         "observed": True, "confidence": 0.5, "parent": None,
         "constraints": [], "texture": "photo"}
    return [c]


def build_revolve(mask, paint, height=3.0, samples=20):
    """Round object seen from the side (vase, bottle, unknown symmetric):
    revolve the measured half-width per row about the vertical axis."""
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    cx = (xs.min() + xs.max()) / 2.0
    s = height / float(y1 - y0 + 1)
    prof = []
    for i in range(samples + 1):
        row = int(round(y1 - (y1 - y0) * i / samples))
        cols = np.where(mask[row, :])[0]
        r = (max(abs(cols.max() + 0.5 - cx), abs(cols.min() - 0.5 - cx))
             if len(cols) else 0.0)
        prof.append([r * s, (y1 - row) * s])
    prof[0][0] = max(prof[0][0], 0.15)
    return [lathe_comp("revolved_body", prof, 0.0, 0.0, "body",
                       paint["body"], "main_volume", role="body")]


BUILDERS = {"vehicle": (build_car, CAR_PRESETS),
            "table": (build_table, TABLE_PRESETS),
            "chair": (build_chair, CHAIR_PRESETS),
            "stool": (build_stool, STOOL_PRESETS),
            "house": (build_house, HOUSE_PRESETS),
            "quadruped": (build_quadruped, QUADRUPED_PRESETS)}

# Continuous parameters refined by the fitter (per family) and their bounds.
REFINE = {
    "vehicle": {"hr": (0.2, 0.5), "belt": (0.45, 0.68), "rf": (-0.1, 0.2),
                "rb": (-0.42, -0.1), "wsb": (0.12, 0.34),
                "rbb": (-0.5, -0.25), "wr": (0.06, 0.09),
                "nose": (0.55, 1.0), "width": (0.34, 0.46)},
    "table": {"hr": (0.18, 0.8), "wr": (0.35, 1.1), "top": (0.03, 0.1),
              "inset": (0.0, 0.16), "leg": (0.03, 0.1)},
    "chair": {"seat_h": (0.7, 1.35), "back_h": (0.5, 1.7),
              "depth": (0.7, 1.35), "leg": (0.07, 0.16)},
    "stool": {"h": (0.4, 1.2), "seat_r": (0.15, 0.32)},
    "house": {"wall": (0.25, 0.8), "depth": (0.4, 1.0), "roof": (0.15, 0.6)},
    "quadruped": {"body_h": (0.22, 0.48), "leg_h": (0.15, 0.6),
                  "head": (0.1, 0.22), "neck": (0.0, 0.35)},
}


# ------------------------------------------------------------------ projector
def component_faces(c):
    """World-space polygons (lists of xyz) whose union is the solid."""
    prim = c["primitive"]
    cx, cy, cz = c["position"]
    faces = []
    if prim == "profile":
        pts = c["profile"]
        ys = [p[1] for p in pts]
        half = (max(ys) - min(ys)) / 2 or 1e-9
        d0, d1 = c["depth"], c.get("depth_top", c["depth"])

        def dep(y):
            return d0 + (d1 - d0) * ((y + half) / (2 * half))
        if c.get("axis", "z") == "z":
            f = [(cx + u, cy + y, cz + dep(y) / 2) for u, y in pts]
            b = [(cx + u, cy + y, cz - dep(y) / 2) for u, y in pts]
        else:
            f = [(cx + dep(y) / 2, cy + y, cz + u) for u, y in pts]
            b = [(cx - dep(y) / 2, cy + y, cz + u) for u, y in pts]
        faces += [f, b]
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            faces.append([f[i], f[j], b[j], b[i]])
    elif prim == "box":
        w, h, d = c["dims"]
        xs, ys, zs = (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2), \
            (cz - d / 2, cz + d / 2)
        v = [(x, y, z) for x in xs for y in ys for z in zs]
        for q in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6),
                  (0, 2, 6, 4), (1, 3, 7, 5)):
            faces.append([v[i] for i in q])
    elif prim == "inflate":
        g = c["grid"]
        cell, rows, cols = g["cell"], g["rows"], g["cols"]
        hw, hh = cols * cell / 2, rows * cell / 2
        st = 2 if max(rows, cols) > 40 else 1
        for j in range(0, rows - st, st):
            for i in range(0, cols - st, st):
                q = [g["half"][j][i], g["half"][j][i + st],
                     g["half"][j + st][i + st], g["half"][j + st][i]]
                if min(q) <= 0:
                    continue
                xa = cx - hw + (i + 0.5) * cell
                xb = cx - hw + (i + st + 0.5) * cell
                ya = cy - hh + (j + 0.5) * cell
                yb = cy - hh + (j + st + 0.5) * cell
                pts = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
                for sgn in (1, -1):
                    faces.append([(x, y, cz + sgn * d)
                                  for (x, y), d in zip(pts, q)])
    elif prim in ("cylinder", "lathe", "sphere"):
        if prim == "cylinder":
            h = c["dims"][1]
            rt, rb = (c.get("taper") or [c["dims"][0] / 2, c["dims"][0] / 2])
            prof = [(rb, -h / 2), (rt, h / 2)]
        elif prim == "sphere":
            R = c["dims"][0] / 2
            prof = [(R * math.sin(math.pi * i / 8), -R * math.cos(math.pi * i / 8))
                    for i in range(9)]
        else:
            prof = [tuple(p) for p in c["profile"]]
        n = 20
        rings = [[(cx + r * math.cos(2 * math.pi * k / n), cy + y,
                   cz + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
                 for r, y in prof]
        for a, b in zip(rings, rings[1:]):
            for k in range(n):
                faces.append([a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]])
        faces += [rings[0], rings[-1]]
    return faces


def scene_bounds(comps):
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for c in comps:
        for f in component_faces(c):
            for p in f:
                for i in range(3):
                    lo[i] = min(lo[i], p[i])
                    hi[i] = max(hi[i], p[i])
    return lo, hi


def camera_for(comps, yaw_deg, elev_deg, fov=FOV, aspect=4 / 3.0,
               margin=1.18):
    """Fixed-rule camera: looks at the bbox centre from (yaw, elevation) at
    the distance that frames the bounding sphere. yaw 0 = front (+x)."""
    lo, hi = scene_bounds(comps)
    t = [(a + b) / 2 for a, b in zip(lo, hi)]
    R = 0.5 * math.sqrt(sum((b - a) ** 2 for a, b in zip(lo, hi)))
    fv = math.radians(fov)
    fh = 2 * math.atan(math.tan(fv / 2) * aspect)
    D = margin * R / math.sin(min(fv, fh) / 2)
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    pos = [t[0] + D * math.cos(e) * math.cos(y), t[1] + D * math.sin(e),
           t[2] + D * math.cos(e) * math.sin(y)]
    return {"position": [_r(v) for v in pos], "target": [_r(v) for v in t],
            "fov": fov, "yaw": yaw_deg, "elevation": elev_deg}


def _basis(cam):
    pos = np.array(cam["position"], float)
    tgt = np.array(cam["target"], float)
    f = tgt - pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return pos, f, r, u


def project(comps, cam, width=160, height=120):
    """Software render: (mask bool[h,w], ids int[h,w] (-1 empty)) with a
    painter's-order id buffer so visible area per component is measurable."""
    pos, f, r, u = _basis(cam)
    F = (height / 2.0) / math.tan(math.radians(cam["fov"]) / 2)
    faces = []
    for ci, c in enumerate(comps):
        for face in component_faces(c):
            P = np.array(face, float) - pos
            zc = P @ f
            if np.any(zc <= 1e-6):
                continue
            xs = width / 2.0 + (P @ r) / zc * F
            ys = height / 2.0 - (P @ u) / zc * F
            faces.append((float(zc.mean()), ci,
                          [(float(a), float(b)) for a, b in zip(xs, ys)]))
    faces.sort(key=lambda t: -t[0])
    img = Image.new("I", (width, height), 0)
    dr = ImageDraw.Draw(img)
    for _z, ci, poly in faces:
        if len(poly) >= 3:
            dr.polygon(poly, fill=ci + 1)
    ids = np.asarray(img, dtype=np.int32) - 1
    return ids >= 0, ids


# ------------------------------------------------------------------ fitting
def _norm_mask(m, size=64, frame=None):
    """Crop to the mask's bbox (or to ``frame``'s bbox) and resample."""
    ys, xs = np.where(m if frame is None else frame)
    if len(xs) == 0:
        return np.zeros((size, size), bool), None
    c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(c.astype(np.uint8) * 255).resize((size, size),
                                                           Image.BILINEAR)
    return np.asarray(img) >= 128, (xs.max() - xs.min() + 1) / float(
        ys.max() - ys.min() + 1)


class RefMask:
    """Reference silhouette + optional orientation anchors measured from
    the photo: ``anchors`` maps a component role to the normalized (x, y)
    centroid (within the subject bbox) where it was observed."""

    def __init__(self, mask, anchors=None, ignore=None):
        frame = mask if ignore is None else (mask | ignore)
        self.norm, self.aspect = _norm_mask(mask, frame=frame)
        self.anchors = dict(anchors or {})
        # occluded region (bbox-normalized): neither rewarded nor penalised
        self.ignore = None if ignore is None else _norm_mask(
            ignore, frame=frame)[0]


def bright_spots(rgb, mask, luma_min=215.0, max_frac=0.06):
    """Normalized centroid of small, very bright subject regions (lamps)
    or None. A large bright area is paint/glare, not a lamp."""
    lum = rgb.mean(2)
    spot = mask & (lum >= luma_min)
    n = int(spot.sum())
    if n == 0 or n > max_frac * mask.sum() or n < 3:
        return None
    ys, xs = np.where(mask)
    sy, sx = np.where(spot)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return (float((sx.mean() - x0) / max(1, x1 - x0)),
            float((sy.mean() - y0) / max(1, y1 - y0)))


def contact_positions(mask):
    """Normalized x-centres of ground-contact runs of a mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    x0, x1, y1 = xs.min(), xs.max(), ys.max()
    band = max(1, int(round(0.06 * (ys.max() - ys.min() + 1))))
    low = []
    for x in range(x0, x1 + 1):
        col = np.where(mask[:, x])[0]
        low.append(bool(len(col) and col.max() >= y1 - band))
    runs, start = [], None
    for i, v in enumerate(low + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= max(1, 0.015 * len(low)):
                runs.append(((start + i - 1) / 2.0) / max(1, len(low) - 1))
            start = None
    return runs


def _anchor_penalty(mask, ids, comps, ref):
    """Orientation evidence beyond the silhouette: model parts that the
    reference shows at a measured place must project there (lamps), and
    ground contacts (wheels/legs) must line up. 0 = consistent."""
    pen = 0.0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 1.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    want = ref.anchors.get("light")
    if want is not None:
        sel = np.zeros_like(mask)
        for i, c in enumerate(comps):
            if c.get("role") == "light":
                sel |= ids == i
        if sel.sum() == 0:
            pen += 0.12
        else:
            sy, sx = np.where(sel)
            gx = (sx.mean() - x0) / max(1, x1 - x0)
            gy = (sy.mean() - y0) / max(1, y1 - y0)
            pen += 0.35 * abs(gx - want[0]) + 0.15 * abs(gy - want[1])
    rc = ref.anchors.get("contacts")
    if rc:
        mc = contact_positions(mask)
        if mc and len(rc) >= 2:
            d = sum(min(abs(a - b) for b in mc) for a in rc) / len(rc)
            pen += 0.4 * d
    return pen


def silhouette_score(mask, ref: RefMask):
    """(score, iou, aspect_ratio): IoU of bbox-normalized silhouettes,
    penalised by the log aspect mismatch. 1.0 = identical shape."""
    nm, asp = _norm_mask(mask)
    if asp is None or ref.aspect is None:
        return -1.0, 0.0, None
    from morphix_image_seg import tolerant
    keep = ~ref.ignore if ref.ignore is not None else np.ones_like(nm)
    a, b = tolerant(nm), tolerant(ref.norm)
    inter = np.logical_and(a, b)[keep].sum()
    union = np.logical_or(a, b)[keep].sum()
    iou = inter / union if union else 0.0
    ar = asp / ref.aspect
    return float(iou - 0.8 * abs(math.log(ar))), float(iou), float(ar)


def yaw_candidates(view_label=None):
    """All yaws (15° steps). A vision viewpoint label only adds a small
    preference — it never restricts the search (small vision models
    misreport views; the silhouette decides)."""
    return [float(a) for a in range(-180, 180, 15)]


VIEW_YAWS = {"front": (0,), "rear": (180, -180), "side": (90, -90),
             "angle": (45, -45), "angle_rear": (135, -135)}


def _view_bonus(yaw, label):
    if not label or label not in VIEW_YAWS:
        return 0.0
    d = min(abs(((yaw - y) + 180) % 360 - 180) for y in VIEW_YAWS[label])
    return 0.03 if d <= 30 else 0.0


# Vision label words -> preferred preset (a small bonus; geometry decides).
STYLE_HINTS = {"vehicle": {"suv": "suv", "crossover": "suv",
                           "sports": "sports", "supercar": "sports",
                           "coupe": "sports", "roadster": "sports",
                           "sedan": "sedan", "hatchback": "hatch",
                           "wagon": "hatch", "classic": "boxy",
                           "van": "tall", "minivan": "tall", "truck": "tall",
                           "jeep": "tall",
                           "vintage": "boxy", "cartoon": "boxy"},
               "table": {"coffee": "coffee", "desk": "desk",
                         "dining": "dining"},
               "quadruped": {"cat": "cat", "kitten": "cat", "dog": "dog",
                             "puppy": "dog", "horse": "horse", "cow": "cow"}}
MIN_GAIN = 0.004  # a refinement must measurably improve the fit


def fit(family, ref_mask, paint, view_label=None, budget=900, emit=None,
        anchors=None, ignore=None, label=None):
    """Bounded fit of preset + continuous params + camera (yaw, elevation)
    to the reference silhouette. Returns dict with comps, params, preset,
    camera, score, iou, aspect_ratio, evaluations, log."""
    builder, presets = BUILDERS[family]
    ref = RefMask(ref_mask, anchors, ignore)
    evals = {"n": 0}
    log = []

    def score_of(params, yaw, elev):
        evals["n"] += 1
        comps = builder(params, paint)
        cam = camera_for(comps, yaw, elev)
        m, ids = project(comps, cam)
        s, iou, ar = silhouette_score(m, ref)
        s -= _anchor_penalty(m, ids, comps, ref)
        s -= 0.004 * elev  # photos are rarely shot from high above
        return s + _view_bonus(yaw, view_label), iou, ar, comps, cam

    best = None
    yaws = yaw_candidates(view_label)
    words = (label or "").lower().split()
    hinted = {STYLE_HINTS.get(family, {}).get(w) for w in words} - {None}
    for name, pr in presets.items():
        for yaw in yaws:
            for elev in (0.0, 12.0, 24.0):
                s, iou, ar, comps, cam = score_of(dict(pr), yaw, elev)
                s += 0.02 if name in hinted else 0.0
                if best is None or s > best["score"]:
                    best = {"score": s, "iou": iou, "aspect_ratio": ar,
                            "params": dict(pr), "preset": name, "yaw": yaw,
                            "elev": elev, "comps": comps, "camera": cam}
    log.append(f"stage 1 silhouette: preset {best['preset']!r} yaw "
               f"{best['yaw']:.0f} elev {best['elev']:.0f} -> IoU "
               f"{best['iou']:.3f} aspect x{best['aspect_ratio']:.2f} "
               f"({evals['n']} evaluations)")
    # coordinate refinement: continuous params then camera, step halving
    bounds = REFINE.get(family, {})
    step = 0.5
    for _round in range(4):
        improved = False
        for key, (lo, hi) in bounds.items():
            for sgn in (1, -1):
                if evals["n"] >= budget:
                    break
                p = dict(best["params"])
                p[key] = min(hi, max(lo, p[key] + sgn * step * (hi - lo) / 4))
                s, iou, ar, comps, cam = score_of(p, best["yaw"], best["elev"])
                if s > best["score"] + MIN_GAIN:
                    best.update(score=s, iou=iou, aspect_ratio=ar, params=p,
                                comps=comps, camera=cam)
                    improved = True
        for dyaw, delev in ((7.5 * step * 2, 0), (-7.5 * step * 2, 0),
                            (0, 6 * step * 2), (0, -6 * step * 2)):
            if evals["n"] >= budget:
                break
            yaw = best["yaw"] + dyaw
            elev = min(40.0, max(0.0, best["elev"] + delev))
            s, iou, ar, comps, cam = score_of(best["params"], yaw, elev)
            if s > best["score"] + MIN_GAIN:
                best.update(score=s, iou=iou, aspect_ratio=ar, yaw=yaw,
                            elev=elev, comps=comps, camera=cam)
                improved = True
        if not improved:
            step /= 2
        if step < 0.06 or evals["n"] >= budget:
            break
    log.append(f"stage 2-3 volume+structure refined: IoU {best['iou']:.3f} "
               f"aspect x{best['aspect_ratio']:.2f} yaw {best['yaw']:.1f} "
               f"elev {best['elev']:.1f} ({evals['n']} evaluations)")
    best["evaluations"] = evals["n"]
    best["log"] = log
    return best


def fit_generic(kind, ref_mask, paint):
    """Measured-silhouette builders (no category template)."""
    if kind == "revolve":
        comps = build_revolve(ref_mask, paint)
        yaws = (0.0,)
    elif kind == "inflate":
        comps = build_inflate(ref_mask, paint)
        yaws = (90.0,)               # front surface faces +z (the photo)
    else:
        comps = build_extrusion(ref_mask, paint)
        yaws = (90.0, -90.0)
    ref = RefMask(ref_mask)
    best = None
    for yaw in yaws:
        for elev in (0.0, 10.0, 20.0):
            cam = camera_for(comps, yaw, elev)
            m, _ = project(comps, cam)
            s, iou, ar = silhouette_score(m, ref)
            if best is None or s > best["score"]:
                best = {"score": s, "iou": iou, "aspect_ratio": ar,
                        "yaw": yaw, "elev": elev, "camera": cam}
    best.update(comps=comps, params={"method": kind}, preset=kind,
                evaluations=len(yaws) * 3,
                log=[f"measured-silhouette {kind}: IoU {best['iou']:.3f}"])
    return best
