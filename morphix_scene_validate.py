# morphix_scene_validate.py
# Geometric + visual validation of RENDERED scenes (Phase 4.1, items 5-6).
# Inspects actual live geometry — world bboxes introspected from the browser
# scene graph — instead of trusting descriptions, then checks the rendered
# pixels against the plan (framing, lighting, fog presence).
# actual live geometry — world bboxes introspected from the browser scene
# graph — instead of trusting descriptions. Checks, per required node:
# presence, finite + non-degenerate bboxes, scene bounds, declared
# relations (on/near/inside/above), and suspicious floating/buried
# placement. Intentional overlaps are NOT errors: only declared relations
# and gross anomalies (infinite, empty, light-years away) fail.
#
# Findings: [{check, ok, detail, severity}] with severity error|warning.
# verified == no error-severity failures. Warnings (suspicious floating,
# sparse scatter) are reported, never hidden, never fatal.

import math

WORLD_BOUND = 250.0
MIN_VOLUME = 1e-9


def _finite(bbox):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for pt in bbox for v in pt)


def _volume(bbox):
    (x0, y0, z0), (x1, y1, z1) = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) * max(0.0, z1 - z0)


def _overlap_2d(a, b):
    (ax0, _, az0), (ax1, _, az1) = a
    (bx0, _, bz0), (bx1, _, bz1) = b
    return min(ax1, bx1) - max(ax0, bx0) > 0 \
        and min(az1, bz1) - max(az0, bz0) > 0


def _dist(a, b):
    ca = [(x + y) / 2 for x, y in zip(a[0], a[1])]
    cb = [(x + y) / 2 for x, y in zip(b[0], b[1])]
    return math.dist(ca, cb)


def validate_assembly(ir, graph):
    """Validate rendered geometry against the plan. ``graph`` maps IR ids
    to {present, bbox, visible} from live introspection. Returns
    {"checks": [...], "verified": bool}."""
    checks = []

    def check(name, ok, detail, severity="error"):
        checks.append({"name": name, "ok": bool(ok),
                       "detail": str(detail), "severity": severity})

    nodes = {n["id"]: n for n in ir.get("nodes", []) if isinstance(n, dict)}
    required = [nid for nid, n in nodes.items() if n.get("required")]
    # Structurally supported elements (satisfied `on` relations) never
    # trigger the floating suspicion below: a tabletop on legs is elevated
    # by design, not by accident.
    supported = set()
    for rel in ir.get("relations", []):
        if not isinstance(rel, dict) or rel.get("rel") != "on":
            continue
        ga, gb = graph.get(rel.get("a")), graph.get(rel.get("b"))
        if not ga or not gb or not ga.get("bbox") or not gb.get("bbox"):
            continue
        ba, bb = ga["bbox"], gb["bbox"]
        tol = rel.get("tol", 0.15)
        if abs(ba[0][1] - bb[1][1]) <= tol and _overlap_2d(ba, bb):
            supported.add(rel["a"])
    for nid in required:
        g = graph.get(nid)
        check(f"present:{nid}", g is not None and g.get("present", False),
              "rendered in scene" if g and g.get("present") else
              "MISSING from rendered scene")
    for nid, g in graph.items():
        if not isinstance(g, dict) or not g.get("present"):
            continue
        n = nodes.get(nid, {})
        bbox = g.get("bbox")
        if g.get("empty") or bbox is None:
            if n.get("kind") in ("mesh",) and nid in required:
                check(f"nonempty:{nid}", False,
                      "required mesh rendered no geometry")
            continue
        if not _finite(bbox):
            check(f"finite:{nid}", False, f"non-finite bbox {bbox}")
            continue
        check(f"finite:{nid}", True, "bbox finite")
        if _volume(bbox) <= MIN_VOLUME and n.get("kind") == "mesh":
            check(f"solid:{nid}", False,
                  f"degenerate zero-volume bbox {bbox}")
        else:
            check(f"solid:{nid}", True, "non-degenerate")
        flat = [v for pt in bbox for v in pt]
        if any(abs(v) > WORLD_BOUND for v in flat):
            check(f"in_bounds:{nid}", False,
                  f"outside scene bounds (+/-{WORLD_BOUND})")
        if n.get("kind") == "mesh" and bbox[0][1] > 0.5 \
                and nid in required and nid not in supported:
            check(f"grounded:{nid}", True,
                  f"base y={bbox[0][1]:.2f} above ground with no support "
                  f"relation (suspicious floating?)", severity="warning")
    for i, rel in enumerate(ir.get("relations", [])):
        if not isinstance(rel, dict):
            continue
        a, b, kind = rel.get("a"), rel.get("b"), rel.get("rel")
        tol = rel.get("tol", 0.15)
        ga, gb = graph.get(a), graph.get(b)
        if not ga or not ga.get("present") or not gb \
                or not gb.get("present") or not ga.get("bbox") \
                or not gb.get("bbox"):
            check(f"relation:{a}:{kind}:{b}", False,
                  "endpoint missing from render")
            continue
        ba, bb = ga["bbox"], gb["bbox"]
        if kind == "on":
            ok = abs(ba[0][1] - bb[1][1]) <= tol and _overlap_2d(ba, bb)
            check(f"relation:{a}:{kind}:{b}", ok,
                  f"{a} base {ba[0][1]:.2f} vs {b} top {bb[1][1]:.2f}")
        elif kind == "near":
            ok = _dist(ba, bb) <= tol
            check(f"relation:{a}:{kind}:{b}", ok,
                  f"centers {_dist(ba, bb):.2f} apart (tol {tol})")
        elif kind == "inside":
            ok = all(bb[0][j] - tol <= ba[0][j] and ba[1][j] <= bb[1][j] + tol
                     for j in range(3))
            check(f"relation:{a}:{kind}:{b}", ok, "containment")
        elif kind == "above":
            ok = ba[0][1] >= bb[1][1] - tol
            check(f"relation:{a}:{kind}:{b}", ok,
                  f"{a} base {ba[0][1]:.2f} vs {b} top {bb[1][1]:.2f}")
        else:
            check(f"relation:{a}:{kind}:{b}", False,
                  f"unknown relation {kind!r}")
    errors = [c for c in checks
              if not c["ok"] and c["severity"] == "error"]
    return {"checks": checks, "verified": not errors}


# ------------------------------------------------- visual verification
def project_point(eye, target, fov_deg, aspect, width, height, pt):
    """Project a world point to pixel coords (or None if behind camera).
    Pure math over the IR camera — unit-testable, no browser needed."""
    import numpy as np
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return None
    fwd = fwd / n
    up = np.array([0.0, 1.0, 0.0])
    if abs(abs(float(fwd @ up)) - 1.0) < 1e-3:
        up = np.array([0.0, 0.0, 1.0])
    s = np.cross(fwd, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, fwd)
    d = np.array(pt, float) - eye
    z = float(d @ fwd)
    if z <= 0.05:
        return None
    t = math.tan(math.radians(fov_deg) / 2)
    nx = float(d @ s) / (z * t * aspect)
    ny = float(d @ u) / (z * t)
    return (int((nx * 0.5 + 0.5) * width),
            int((0.5 - ny * 0.5) * height))


def subject_rect(graph, ids, camera, width=640, height=480):
    """Pixel rect covering required subject ids (or None when no corner
    projects). Uses live world bboxes, not planned transforms."""
    xs, ys = [], []
    fov = camera.get("fov", 55)
    aspect = width / height
    for nid in ids:
        g = graph.get(nid) or {}
        bbox = g.get("bbox")
        if not bbox:
            continue
        (x0, y0, z0), (x1, y1, z1) = bbox
        for x in (x0, x1):
            for y in (y0, y1):
                for z in (z0, z1):
                    p = project_point(camera["position"], camera["target"],
                                      fov, aspect, width, height, (x, y, z))
                    if p is not None:
                        xs.append(min(max(p[0], 0), width - 1))
                        ys.append(min(max(p[1], 0), height - 1))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def verify_visual(ir, png_path, graph, camera=None):
    """Check the rendered pixels against the plan. Returns
    {"checks": [...], "verified": bool}. Checks: required subjects frame
    inside the viewport with real extent; subject region not pitch black;
    declared fog visibly present. All deterministic; no model opinion."""
    from PIL import Image
    import numpy as np
    checks = []

    def check(name, ok, detail, severity="error"):
        checks.append({"name": name, "ok": bool(ok),
                       "detail": str(detail), "severity": severity})

    try:
        img = np.asarray(Image.open(png_path).convert("RGB"), dtype=float)
    except Exception as e:
        check("screenshot_readable", False, f"cannot read {png_path}: {e}")
        return {"checks": checks, "verified": False}
    H, W, _ = img.shape
    cam = camera or ir.get("camera", {})
    required = [n["id"] for n in ir.get("nodes", [])
                if isinstance(n, dict) and n.get("required")]
    rect = subject_rect(graph, required, cam, W, H)
    if rect is None:
        check("subject_framed", False,
              "no required-subject corner projects into the viewport")
    else:
        x0, y0, x1, y1 = rect
        area = max(0, x1 - x0) * max(0, y1 - y0)
        check("subject_framed", area >= 40 * 40,
              f"subject rect {x1 - x0}x{y1 - y0}px")
        if area > 0:
            crop = img[y0:y1 + 1, x0:x1 + 1] / 255.0
            check("subject_lit", crop.mean() >= 0.06,
                  f"subject brightness {crop.mean():.3f}")
    fog = (ir.get("environment") or {}).get("fog") or {}
    if fog.get("kind", "none") != "none" and fog.get("color"):
        h = str(fog["color"])
        target = np.array([int(h[2:4], 16), int(h[4:6], 16),
                           int(h[6:8], 16)]) / 255.0
        d = np.abs(img / 255.0 - target).max(axis=2)
        frac = float((d < 0.12).mean())
        check("fog_visible", frac >= 0.05,
              f"{frac:.3f} of pixels near fog color")
    errors = [c for c in checks
              if not c["ok"] and c["severity"] == "error"]
    return {"checks": checks, "verified": not errors}
