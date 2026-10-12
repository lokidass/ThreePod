# morphix_mesh.py
# Committed deposits -> triangle soup -> mesh validity + intent checks.
#
# Built FROM the committed scene (PathState records), never from a reference
# mesh: each deposit_segment becomes the oriented bead box it renders as,
# each deposit_point a small filler cube. Checks then compare this
# as-built mesh against the fixture intent (bbox, contacts, materials).
# Anything that passes here actually printed; anything merely planned
# but never committed cannot hide.
import math

BEAD_EPS = 1e-6


def _frame(d):
    dx, dy, dz = d
    if abs(dx) < 0.9:
        up = (1.0, 0.0, 0.0)
    else:
        up = (0.0, 1.0, 0.0)
    n1 = (dy * up[2] - dz * up[1],
          dz * up[0] - dx * up[2],
          dx * up[1] - dy * up[0])
    n = math.sqrt(sum(v * v for v in n1)) or 1.0
    n1 = (n1[0] / n, n1[1] / n, n1[2] / n)
    n2 = (dy * n1[2] - dz * n1[1],
          dz * n1[0] - dx * n1[2],
          dx * n1[1] - dy * n1[0])
    return n1, n2


def _box_tris(cx, cy, cz, hx, hy, hz):
    c = [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    # index: sx*4 + sy*2 + sz with -1 -> 0
    def v(sx, sy, sz):
        return c[(0 if sx < 0 else 1) * 4 + (0 if sy < 0 else 1) * 2
                 + (0 if sz < 0 else 1)]
    faces = [
        (v(-1, -1, -1), v(-1, 1, -1), v(-1, 1, 1), v(-1, -1, 1)),   # -x
        (v(1, -1, -1), v(1, -1, 1), v(1, 1, 1), v(1, 1, -1)),       # +x
        (v(-1, -1, -1), v(-1, -1, 1), v(1, -1, 1), v(1, -1, -1)),   # -y
        (v(-1, 1, -1), v(1, 1, -1), v(1, 1, 1), v(-1, 1, 1)),       # +y
        (v(-1, -1, -1), v(1, -1, -1), v(1, 1, -1), v(-1, 1, -1)),   # -z
        (v(-1, -1, 1), v(-1, 1, 1), v(1, 1, 1), v(1, -1, 1)),       # +z
    ]
    tris = []
    for a, b, cc, d in faces:
        tris.append((a, b, cc))
        tris.append((a, cc, d))
    return tris


def _tri_area(t):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz), (nx, ny, nz)


def deposits_to_mesh(records):
    """records -> {"tris": [(tri, meta)], "boxes": [boxdict]}.
    meta: component, material, layer, kind."""
    tris, boxes = [], []
    for r in records:
        kind = r.get("action")
        if kind == "deposit_segment":
            frm, to = r["from"], r["to"]
            L = math.dist(frm, to)
            if L < BEAD_EPS:
                continue
            d = ((to[0] - frm[0]) / L, (to[1] - frm[1]) / L,
                 (to[2] - frm[2]) / L)
            n1, n2 = _frame(d)
            hw = r.get("bead_width", 0.2) / 2.0
            hh = r.get("bead_height", 0.2) / 2.0
            mid = ((frm[0] + to[0]) / 2, (frm[1] + to[1]) / 2,
                   (frm[2] + to[2]) / 2)
            # 8 corners: mid +/- L/2*d +/- hw*n1 +/- hh*n2.
            # Index bit order is (s0, s1, s2).
            corners = []
            for s0 in (-1, 1):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        corners.append((
                            mid[0] + s0 * L / 2 * d[0] + s1 * hw * n1[0]
                            + s2 * hh * n2[0],
                            mid[1] + s0 * L / 2 * d[1] + s1 * hw * n1[1]
                            + s2 * hh * n2[1],
                            mid[2] + s0 * L / 2 * d[2] + s1 * hw * n1[2]
                            + s2 * hh * n2[2]))

            def _c(a, b, c):
                return corners[a * 4 + b * 2 + c]

            faces = [
                (_c(0, 0, 0), _c(0, 1, 0), _c(0, 1, 1), _c(0, 0, 1)),
                (_c(1, 0, 0), _c(1, 0, 1), _c(1, 1, 1), _c(1, 1, 0)),
                (_c(0, 0, 0), _c(0, 0, 1), _c(1, 0, 1), _c(1, 0, 0)),
                (_c(0, 1, 0), _c(1, 1, 0), _c(1, 1, 1), _c(0, 1, 1)),
                (_c(0, 0, 0), _c(1, 0, 0), _c(1, 1, 0), _c(0, 1, 0)),
                (_c(0, 0, 1), _c(0, 1, 1), _c(1, 1, 1), _c(1, 0, 1)),
            ]
            box = {"min": [min(p[i] for p in corners) for i in range(3)],
                   "max": [max(p[i] for p in corners) for i in range(3)],
                   "component": r.get("component"),
                   "material": r.get("material"), "layer": r.get("layer"),
                   "kind": "segment",
                   "from": list(frm), "to": list(to)}
            boxes.append(box)
            meta = {"component": r.get("component"),
                    "material": r.get("material"), "layer": r.get("layer"),
                    "kind": "segment"}
            for a, b, cc, dd in faces:
                tris.append(((a, b, cc), meta))
                tris.append(((a, cc, dd), meta))
        elif kind == "deposit_point":
            p = r.get("at", r.get("position"))
            s = r.get("size", 0.2) / 2.0
            box = {"min": [p[0] - s, p[1] - s, p[2] - s],
                   "max": [p[0] + s, p[1] + s, p[2] + s],
                   "component": r.get("component"),
                   "material": r.get("material"), "layer": r.get("layer"),
                   "kind": "point", "at": list(p)}
            boxes.append(box)
            meta = {"component": r.get("component"),
                    "material": r.get("material"), "layer": r.get("layer"),
                    "kind": "point"}
            cx, cy, cz = p
            for t in _box_tris(cx, cy, cz, s, s, s):
                tris.append((t, meta))
    return {"tris": tris, "boxes": boxes}


def mesh_bbox(mesh):
    xs, ys, zs = [], [], []
    for t, _ in mesh["tris"]:
        for p in t:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    if not xs:
        return None
    return ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def _aabbs_overlap_volume(a, b, eps=1e-9):
    overlap = 1.0
    for i in range(3):
        o = min(a["max"][i], b["max"][i]) - max(a["min"][i], b["min"][i])
        if o <= eps:
            return 0.0
        overlap *= o
    return overlap


def check_mesh_validity(mesh):
    """Generic mesh sanity. Returns list of error strings (empty = valid)."""
    errs = []
    if not mesh["tris"]:
        return ["mesh has no triangles"]
    bad_finite = 0
    bad_normal = 0
    degenerate = 0
    area = 0.0
    for t, _ in mesh["tris"]:
        if any(not math.isfinite(v) for p in t for v in p):
            bad_finite += 1
            continue
        a, n = _tri_area(t)
        area += a
        if a < 1e-12:
            degenerate += 1
        nl = math.sqrt(sum(v * v for v in n))
        if nl < 1e-12 or not math.isfinite(nl):
            bad_normal += 1
    if bad_finite:
        errs.append(f"{bad_finite} triangles with non-finite vertices")
    if degenerate:
        errs.append(f"{degenerate} degenerate (zero-area) triangles")
    if bad_normal:
        errs.append(f"{bad_normal} triangles with invalid normals")
    if area <= 0:
        errs.append("mesh has no surface area")
    return errs


def check_no_duplicate_deposits(records):
    """Exact-duplicate committed deposits (same endpoints/size/material)."""
    seen = {}
    dups = 0
    for r in records:
        k = r.get("action")
        if k == "deposit_segment":
            key = ("seg", tuple(r["from"]), tuple(r["to"]),
                   r.get("bead_width"), r.get("bead_height"),
                   r.get("material"))
        elif k == "deposit_point":
            key = ("pt", tuple(r.get("at", r.get("position"))),
                   r.get("size"), r.get("material"))
        else:
            continue
        if key in seen:
            dups += 1
        seen[key] = True
    if dups:
        return [f"{dups} duplicate committed deposits"]
    return []


def _endpoints_shared(a, b, tol=1e-6):
    for p in (a.get("from"), a.get("to")):
        if p is None:
            continue
        for q in (b.get("from"), b.get("to")):
            if q is None:
                continue
            if math.dist(p, q) <= tol:
                return True
    return False


def check_same_material_overlaps(mesh, bead=0.2, eps=1e-9):
    """Interpenetration between same-material segment boxes.

    Fused deposition legitimately touches: stacked courses share faces,
    scanlines terminate into perimeters, contour joints share endpoints,
    thin walls fuse across their thickness, and first courses fuse into
    supports. All of those contacts are thin in at least two axes (at most
    a bead of engagement). Embedded glass is exempt by design (different
    materials). Only segment boxes are checked — join-point cubes
    intentionally overlap bead ends. Flags pairs whose overlap extends
    beyond one bead width in at least TWO axes without a shared endpoint:
    a component printed inside another, or a shifted duplicate layer."""
    errs = []
    segs = [b for b in mesh["boxes"] if b["kind"] == "segment"]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if segs[i]["material"] != segs[j]["material"]:
                continue
            if _endpoints_shared(segs[i], segs[j]):
                continue
            wide = 0
            spans = []
            for k in range(3):
                o = min(segs[i]["max"][k], segs[j]["max"][k]) - max(
                    segs[i]["min"][k], segs[j]["min"][k])
                if o <= eps:
                    break
                spans.append(o)
                if o > bead + eps:
                    wide += 1
            else:
                if wide >= 2:
                    errs.append(
                        f"segments interpenetrate "
                        f"({ 'x'.join(f'{s:.2f}' for s in spans)} overlap): "
                        f"{segs[i]['component']}@{segs[i]['layer']} vs "
                        f"{segs[j]['component']}@{segs[j]['layer']}")
                    if len(errs) >= 5:
                        errs.append("(+ more)")
                        return errs
    return errs


def component_meshes(mesh):
    out = {}
    for t, m in mesh["tris"]:
        out.setdefault(m["component"], []).append((t, m))
    return out


def component_aabb(mesh, component):
    xs, ys, zs = [], [], []
    for t, m in mesh["tris"]:
        if m["component"] != component:
            continue
        for p in t:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    if not xs:
        return None
    return ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def aabbs_touch_or_overlap(a, b, tol=0.05):
    """True if two AABBs touch or interpenetrate (contact, not coincidence)."""
    for i in range(3):
        if min(a[1][i], b[1][i]) - max(a[0][i], b[0][i]) < -tol:
            return False
    return True


def check_round_fill(records, component, r_bottom, r_top=None,
                     y_bottom=None, y_top=None, layer_h=0.2,
                     fill_floor=0.3, spherical=False):
    """A round layer must be a filled disc, not an empty ring — but only
    where the cross-section is wide enough to fill (radius >= fill_floor,
    mirroring the compiler's infill rule). Narrow courses (taper tips,
    thin legs, small apex rings) legitimately print as lone perimeters.
    r_bottom/r_top + y_bottom/y_top describe the taper profile so each
    layer is judged against its OWN local radius: a neck narrowing
    0.26 -> 0.20 must not be failed for lacking wheel-size scanlines.
    Constant-radius parts pass r_top=None / no y range (every recorded
    layer is then expected to carry fill). Spheres pass spherical=True:
    their cross-section is r(y)=sqrt(R^2-(y-c)^2), so an apex course is
    judged against its own small local radius (exempt below fill_floor),
    never against the full nominal radius."""
    if r_top is None:
        r_top = r_bottom
    by_layer = {}
    for r in records:
        if r.get("action") != "deposit_segment":
            continue
        if r.get("component") != component:
            continue
        k = r.get("layer")
        if not isinstance(k, int):
            continue
        L = math.dist(r["from"], r["to"])
        if L > by_layer.get(k, 0.0):
            by_layer[k] = L
    errs = []
    for k in sorted(by_layer):
        if y_bottom is None or y_top is None or y_top <= y_bottom:
            expected = r_bottom
        elif spherical:
            c = (y_bottom + y_top) / 2.0
            expected = math.sqrt(max(
                0.0, r_bottom * r_bottom - (k * layer_h - c) ** 2))
        else:
            t = min(1.0, max(0.0, (k * layer_h - y_bottom)
                             / (y_top - y_bottom)))
            expected = r_bottom + (r_top - r_bottom) * t
        if expected < fill_floor:
            continue
        if by_layer[k] < 1.2 * expected:
            errs.append(f"{component} L{k}: widest bead "
                        f"{by_layer[k]:.2f} < 1.2r ({1.2 * expected:.2f}); "
                        f"layer prints as an empty ring")
    return errs


def check_box_fill(records, component, width, depth):
    """A solid box layer needs infill, not just a perimeter: every deposited
    layer must carry well beyond the perimeter length. Small footprints
    (fully covered by one bead course) are exempt."""
    if min(width, depth) <= 0.8:
        return []
    perim = 2.0 * (width + depth)
    by_layer = {}
    for r in records:
        if r.get("action") != "deposit_segment":
            continue
        if r.get("component") != component:
            continue
        by_layer[r.get("layer")] = by_layer.get(r.get("layer"), 0.0) + \
            math.dist(r["from"], r["to"])
    errs = []
    for layer in sorted(by_layer):
        if by_layer[layer] < 1.5 * perim:
            errs.append(f"{component} L{layer}: deposited {by_layer[layer]:.1f} "
                        f"< 1.5x perimeter ({1.5 * perim:.1f}); layer is "
                        f"strips, not solid fill")
    return errs
