# morphix_image3d.py
# Open-ended image -> 3D reconstruction (any subject, not a fixed category
# list). One validated semantic IR, shape-first geometry, printer slicing,
# browser rendering, measured comparison with the reference, bounded repair.
#
#   photo ─► segmentation (subject vs backdrop/floor/platform/shadow)
#         ─► vision adapter (normalized replies -> subject + confidence)
#         ─► builder choice:
#               family builder  (vehicle, table, chair, stool, house,
#                                quadruped) fitted to the silhouette
#               measured silhouette (revolve / extrude) for ANY other
#                                subject, or when vision cannot name it
#         ─► semantic IR v2 (validated) ─► printer primitives ─► slicing
#         ─► render (shaded + per-component id pass) ─► measured checks
#         ─► bounded repair ─► verdict: reconstructed | approximation |
#                                       failed (with diagnostics)
#
# A subject the system has no dedicated builder for is NEVER refused and
# NEVER swapped for an unrelated template: it is built from its own
# measured outline and reported as a silhouette approximation. A vision
# description is never proof of correct geometry: every verdict needs the
# rendered result to agree with the reference by measurement.

import json
import math
import os
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

import morphix_image_seg as SEG
import morphix_image_shapes as SH
import morphix_image_vision as VIS

IR_VERSION = 2
MAX_REPAIRS = 2
RENDER_W, RENDER_H = 640, 480

# Critical thresholds (measured on the bbox-normalized silhouettes).
IOU_MIN = 0.62          # silhouette agreement render vs reference
ASPECT_TOL = 0.25       # |render/ref aspect - 1|
COLOR_DE_MAX = 32.0     # large color region ΔE (lighting tolerant)
PLAN_SHIFT_MAX = 0.12   # component centroid: render vs plan (bbox units)

ID_COLORS = ["0xE6194B", "0x3CB44B", "0xFFE119", "0x4363D8", "0xF58231",
             "0x911EB4", "0x46F0F0", "0xF032E6", "0xBCF60C", "0xFABEBE",
             "0x008080", "0xE6BEFF", "0x9A6324", "0xFFFAC8", "0x800000",
             "0xAAFFC3", "0x808000", "0xFFD8B1", "0x000075", "0x808080"]


class ReconstructionError(RuntimeError):
    def __init__(self, stage, message, info=None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.info = dict(info or {})


# ------------------------------------------------------------------ subject
def _manual_observation(category: str) -> dict:
    cat, term, fam = VIS.resolve_category(category)
    cat = cat or category.strip().lower()
    return {"model": None, "available": False, "status": "manual",
            "category": cat, "family": fam or VIS.FAMILY.get(cat),
            "label": category.strip(), "confidence": 0.8,
            "decided_by": "category supplied by the user",
            "viewpoint": {"label": None, "raw": "", "facing": None},
            "parts": [], "colors": [], "environment_words": [],
            "replies": [], "evidence": [], "dissent": []}


def identify(path, seg, category=None, model="", ask=None, emit=None):
    """Subject observation. Never raises for vision problems: a failed
    vision stage yields an UNIDENTIFIED subject (built from its measured
    silhouette) with the failure kind recorded."""
    if category and category.strip():
        return _manual_observation(category)
    if not model:
        return {"model": None, "available": False, "status": "unavailable",
                "category": None, "family": None, "label": None,
                "confidence": 0.0,
                "decided_by": "no vision model configured or installed",
                "failure": {"kind": "unavailable",
                            "message": "no vision model available"},
                "viewpoint": {"label": None}, "parts": [], "colors": [],
                "environment_words": [], "replies": [], "evidence": []}
    ask = ask or VIS.ollama_ask(path, model)
    try:
        return VIS.observe(ask, model, aspect=seg["subject"]["aspect"],
                           emit=emit)
    except VIS.VisionFailure as e:
        return {"model": model, "available": e.kind != "unavailable",
                "status": e.kind, "category": None, "family": None,
                "label": None, "confidence": 0.0,
                "decided_by": str(e),
                "failure": {"kind": e.kind, "message": str(e),
                            "decision": e.info.get("decision")},
                "viewpoint": {"label": None}, "parts": [], "colors": [],
                "environment_words": [], "replies": e.info.get("replies", []),
                "evidence": e.info.get("evidence", [])}


def choose_method(obs, seg):
    """Builder for the subject: its family's parametric builder when one
    exists, otherwise its measured silhouette (revolve when the outline is
    mirror-symmetric and not much wider than tall, else extrude)."""
    fam = obs.get("family")
    if fam in SH.BUILDERS:
        return "family", fam
    sym = seg["subject"]["symmetry"]
    asp = seg["subject"]["aspect"]
    if fam == "vessel" or (sym >= 0.9 and asp <= 1.4):
        return "silhouette", "revolve"
    return "silhouette", "inflate"


# ------------------------------------------------------------------ anchors
def reference_anchors(seg, family):
    anchors = {}
    if family == "vehicle":
        b = SH.bright_spots(seg["_rgb"], seg["_mask"])
        if b:
            anchors["light"] = b
    c = SH.contact_positions(seg["_mask"])
    if len(c) >= 2 and family in ("vehicle", "table", "chair", "stool",
                                  "quadruped"):
        anchors["contacts"] = c
    return anchors


# ------------------------------------------------------------------ IR
def palette_for(comps):
    """Filament color loaded per material slot: the color of the largest
    component printed with that slot (volume-weighted)."""
    best = {}
    for c in comps:
        w, h, d = c["dims"]
        v = w * h * d
        m = c["material"]
        if m not in best or v > best[m][0]:
            best[m] = (v, c["color"])
    return {m: col for m, (_v, col) in best.items()}


def build_ir(obs, seg, method, kind, fitres, environment):
    """The ONE validated intermediate representation for this image."""
    sub = seg["subject"]
    identified = bool(obs.get("category"))
    source = ("image-manual" if obs.get("status") == "manual"
              else "image-vision" if identified and method == "family"
              else "image-silhouette")
    comps = fitres["comps"]
    occluded, assumptions = [], []
    cam = fitres["camera"]
    yaw = cam["yaw"]
    if method == "family":
        assumptions.append("proportions fitted to the measured silhouette "
                           f"(preset {fitres['preset']!r}); far side "
                           "assumed mirror-symmetric")
        if abs(yaw) < 120:
            occluded.append("rear")
        if abs(yaw) > 60:
            occluded.append("front")
        occluded.append("left side" if yaw < 0 else "right side")
        occluded.append("underside")
    else:
        assumptions.append(
            "no dedicated builder for this subject: geometry is its "
            f"measured outline ({kind}); "
            + ("depth along the view axis is ASSUMED (35% of length)"
               if kind == "extrude" else
               "outline inflated into a rounded solid; thickness and the "
               "unseen side are ASSUMED (the far side mirrors the "
               "photographed side, photo texture on both)"
               if kind == "inflate" else
               "assumed rotationally symmetric about the vertical axis"))
        occluded += ["back side", "depth profile"]
    if sub["touches_border"]:
        occluded.append("cropped at the frame border")
    if seg.get("occluder"):
        occluded.append(f"part of the subject hidden behind a foreground "
                        f"object ({seg['occluder']['color']}); hidden geometry "
                        f"assumed from the fitted shape")
    env = seg.get("environment") or {}
    env_elems = []
    if seg.get("occluder"):
        env_elems.append({"kind": "occluder", **seg["occluder"],
                          "build": False, "printed": False})
    for key in ("backdrop", "floor", "platform", "surface_light", "shadow"):
        if env.get(key):
            env_elems.append({"kind": key, **{k: v for k, v in env[key].items()
                                               if k != "mask"},
                              "build": environment == "preserve"
                              and key in ("platform", "floor"),
                              "printed": False})
    for cl in env.get("clutter", []):
        env_elems.append({"kind": "clutter", **cl, "build": False,
                          "printed": False})
    ir = {
        "ir_version": IR_VERSION,
        "kind": "image-reconstruction",
        "category": obs.get("category") or "unidentified object",
        "source": source,
        "units": "scene", "up": "y", "front": "+x",
        "confidence": round(float(obs.get("confidence") or 0.0), 3),
        "subject": {
            "category": obs.get("category"),
            "label": obs.get("label"),
            "family": obs.get("family"),
            "identified": identified,
            "confidence": round(float(obs.get("confidence") or 0.0), 3),
            "decided_by": obs.get("decided_by"),
            "vision_status": obs.get("status"),
        },
        "silhouette": {
            "aspect": sub["aspect"], "bbox": sub["bbox"],
            "coverage": sub["coverage"],
            "ground_contacts": sub["ground_contacts"],
            "symmetry": sub["symmetry"],
            "segmentation": seg["method"],
        },
        "method": {"builder": method, "kind": kind,
                   "preset": fitres.get("preset"),
                   "params": fitres.get("params"),
                   "fit_iou": round(fitres["iou"], 3),
                   "fit_aspect_ratio": round(fitres["aspect_ratio"] or 0, 3),
                   "build_order": list(SH.STAGES),
                   "log": fitres.get("log", [])},
        "viewpoint": {"yaw_deg": round(cam["yaw"], 1),
                      "elevation_deg": round(cam["elevation"], 1),
                      "vision_label": (obs.get("viewpoint") or {}).get("label"),
                      "basis": "silhouette fit (vision label advisory)"},
        "camera": cam,
        "appearance": {
            "body_color": sub["body_color"],
            "clusters": sub["colors"],
            "palette": palette_for(comps),
            "vision_colors": obs.get("colors", []),
        },
        "visible_surfaces": sorted({c["role"] for c in comps
                                    if c.get("observed", True)}),
        "observed_parts": obs.get("parts", []),
        "environment": {"mode": environment, "elements": env_elems},
        "uncertainty": {"occluded": occluded,
                        "assumptions": assumptions,
                        "ambiguous": obs.get("dissent", []),
                        "segmentation_notes": seg.get("notes", [])},
        "components": comps,
        "assumptions": assumptions,
        "observed": {"visible_parts": obs.get("parts", [])},
        "constraints": ["y is up; the object rests on the ground plane y=0",
                        "environment elements are never printed"],
    }
    from morphix_ir import _object_bbox
    ir["bbox"] = _object_bbox(comps)
    return ir


def validate_image_ir(ir) -> list:
    """Object IR schema (printer contract) + the semantic contract."""
    from morphix_ir import validate_ir
    errs = list(validate_ir(ir))
    sub = ir.get("subject") or {}
    if "identified" not in sub or not isinstance(sub.get("confidence"),
                                                 (int, float)):
        errs.append("semantic: subject.identified/confidence required")
    sil = ir.get("silhouette") or {}
    if not isinstance(sil.get("aspect"), (int, float)):
        errs.append("semantic: silhouette.aspect (measured) required")
    if not isinstance(ir.get("viewpoint", {}).get("yaw_deg"), (int, float)):
        errs.append("semantic: viewpoint.yaw_deg required")
    if not (ir.get("appearance") or {}).get("palette"):
        errs.append("semantic: appearance.palette required")
    env = ir.get("environment") or {}
    if env.get("mode") not in ("object-only", "preserve"):
        errs.append("semantic: environment.mode must be object-only|preserve")
    if any(e.get("printed") for e in env.get("elements", [])):
        errs.append("semantic: environment elements must never be printed")
    if not isinstance(ir.get("uncertainty"), dict):
        errs.append("semantic: uncertainty required")
    roles = {c.get("stage") for c in ir.get("components", [])}
    if "main_volume" not in roles:
        errs.append("semantic: no main_volume component (shape before "
                    "details)")
    return errs


# ------------------------------------------------------------------ render
RENDER_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>morphix verify</title>
<style>html,body{margin:0;padding:0;background:#000}canvas{display:block}</style>
</head><body>
<script type="importmap">{"imports":{"three":"__THREE__"}}</script>
<script type="module">
import * as THREE from 'three';
const W = __W__, H = __H__;
const DATA = __DATA__;
const MODE = DATA.mode;
const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(1);
renderer.setSize(W, H);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(parseInt(DATA.bg, 16));
const cam = new THREE.PerspectiveCamera(DATA.camera.fov, W / H, 0.05, 500);
cam.position.set(...DATA.camera.position);
cam.lookAt(...DATA.camera.target);
if (MODE !== "ids") {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404850, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(cam.position.x + 4, cam.position.y + 10, cam.position.z + 3);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.5);
  fill.position.set(-6, 4, -6);
  scene.add(fill);
}
function mat(hex) {
  return MODE === "ids"
    ? new THREE.MeshBasicMaterial({color: hex, side: THREE.DoubleSide})
    : new THREE.MeshStandardMaterial({color: hex, roughness: 0.55,
        metalness: 0.1, side: THREE.DoubleSide});
}
const mats = DATA.colors.map((c) => mat(parseInt(c, 16)));
for (const d of DATA.deposits) {
  if (d[0] === 0) {
    const [, fx, fy, fz, tx, ty, tz, w, h, ci] = d;
    const len = Math.hypot(tx - fx, ty - fy, tz - fz);
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, len), mats[ci]);
    m.position.set((fx + tx) / 2, (fy + ty) / 2, (fz + tz) / 2);
    m.lookAt(tx, ty, tz);
    scene.add(m);
  } else {
    const [, x, y, z, s, ci] = d;
    const m = new THREE.Mesh(new THREE.SphereGeometry(s / 2, 10, 8), mats[ci]);
    m.position.set(x, y, z);
    scene.add(m);
  }
}
function profileGeometry(c) {
  const pts = c.profile, n = pts.length;
  const ys = pts.map((q) => q[1]);
  const half = (Math.max(...ys) - Math.min(...ys)) / 2 || 1e-9;
  const d0 = c.depth, d1 = (c.depth_top ?? c.depth);
  const dep = (y) => d0 + (d1 - d0) * ((y + half) / (2 * half));
  const ax = c.axis === "x";
  const v = (u, y, s) => { const d = dep(y) / 2 * s; return ax ? [d, y, u] : [u, y, d]; };
  const tris = THREE.ShapeUtils.triangulateShape(
    pts.map((q) => new THREE.Vector2(q[0], q[1])), []);
  const pos = [];
  const push = (a) => pos.push(a[0], a[1], a[2]);
  for (const t of tris) {
    for (const s of [1, -1]) for (const k of (s > 0 ? [0, 1, 2] : [0, 2, 1]))
      push(v(pts[t[k]][0], pts[t[k]][1], s));
  }
  for (let i = 0; i < n; i++) {
    const a = pts[i], b = pts[(i + 1) % n];
    const A = v(a[0], a[1], 1), B = v(b[0], b[1], 1), C = v(b[0], b[1], -1), D = v(a[0], a[1], -1);
    push(A); push(D); push(C); push(A); push(C); push(B);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}
function inflateGeometry(c) {
  const g = c.grid, cell = g.cell, R = g.rows, C = g.cols, N = R * C;
  const hw = C * cell / 2, hh = R * cell / 2, pos = [], idx = [];
  for (let s of [1, -1]) for (let j = 0; j < R; j++) for (let i = 0; i < C; i++)
    pos.push(-hw + (i + 0.5) * cell, -hh + (j + 0.5) * cell, s * g.half[j][i]);
  const ins = (i, j) => i >= 0 && j >= 0 && i < C - 1 && j < R - 1 && g.half[j][i] > 0 &&
    g.half[j][i + 1] > 0 && g.half[j + 1][i + 1] > 0 && g.half[j + 1][i] > 0;
  for (let j = 0; j < R - 1; j++) for (let i = 0; i < C - 1; i++) {
    if (!ins(i, j)) continue;
    const a = j * C + i, b = a + 1, cc = a + C + 1, d = a + C;
    idx.push(a, b, cc, a, cc, d, a + N, cc + N, b + N, a + N, d + N, cc + N);
    for (const [p, q, ni, nj] of [[a, b, i, j - 1], [b, cc, i + 1, j], [cc, d, i, j + 1], [d, a, i - 1, j]])
      if (!ins(ni, nj)) idx.push(p, p + N, q + N, p, q + N, q);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx); geo.computeVertexNormals();
  return geo;
}
function buildComp(c, material) {
  let g;
  if (c.primitive === "inflate") g = inflateGeometry(c);
  else if (c.primitive === "box") g = new THREE.BoxGeometry(...c.dims);
  else if (c.primitive === "cylinder") {
    const t = c.taper || [c.dims[0] / 2, c.dims[0] / 2];
    g = new THREE.CylinderGeometry(t[0], t[1], c.dims[1], 28);
  } else if (c.primitive === "sphere") g = new THREE.SphereGeometry(c.dims[0] / 2, 24, 16);
  else if (c.primitive === "lathe")
    g = new THREE.LatheGeometry(c.profile.map((q) => new THREE.Vector2(q[0], q[1])), 40);
  else g = profileGeometry(c);
  const m = new THREE.Mesh(g, material);
  m.position.set(...c.position);
  return m;
}
for (const c of DATA.comps) scene.add(buildComp(c, mats[c.ci]));
for (const e of DATA.env) {
  // DISPLAY-ONLY environment context (never printed, never verified)
  const m = new THREE.Mesh(
    e.kind === "platform" ? new THREE.CylinderGeometry(e.r, e.r, 0.06, 48)
                          : new THREE.BoxGeometry(e.size, 0.02, e.size),
    new THREE.MeshStandardMaterial({color: parseInt(e.color, 16), roughness: 0.9}));
  m.position.set(e.x, e.kind === "platform" ? -0.03 : -0.07, e.z);
  scene.add(m);
}
renderer.render(scene, cam);
window.__done = true;
</script></body></html>
"""


def _vendor_three():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "tests", "visual_geometry", "vendor",
                        "three.module.js")


def page_html(data, three_url):
    return (RENDER_TEMPLATE.replace("__THREE__", three_url)
            .replace("__W__", str(RENDER_W)).replace("__H__", str(RENDER_H))
            .replace("__DATA__", json.dumps(data)))


def make_texture(image_path, seg, out_png, max_w=1024):
    """Photo texture for inflated subjects: the reference itself, with
    background seen THROUGH the subject (eye sockets, gaps) shaded as dark
    cavities and the outside ring in-painted from the subject so edges do
    not fringe with backdrop colour. Returns the path."""
    img = Image.open(image_path).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)),
                         Image.LANCZOS)
    a = np.asarray(img).astype(np.float32)
    m = np.asarray(Image.fromarray(seg["_mask"].astype(np.uint8) * 255)
                   .resize(img.size, Image.BILINEAR)) >= 128
    bd = (seg.get("environment") or {}).get("backdrop") or {}
    if bd.get("color"):
        bgc = _hex_to_rgb(bd["color"])
        # only LARGE connected see-through regions are cavities (eye
        # sockets, gaps); isolated bright highlights are kept as they are
        from scipy.ndimage import binary_opening, label
        cav = m & (np.abs(a - bgc).sum(2) < 60)
        cav = binary_opening(cav, iterations=2)
        lab, n = label(cav)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = sizes >= 0.004 * max(1, int(m.sum()))
            cav = keep[lab]
            body = _hex_to_rgb(seg["subject"]["body_color"] or "0x808080")
            a[cav] = body * 0.16
    out = a.astype(np.uint8)
    try:
        import cv2
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(m, iterations=14) & ~m
        out = cv2.inpaint(out, ring.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    except ImportError:
        pass
    Image.fromarray(out).save(out_png)
    return out_png


def render_pages(pages, timeout_ms=60000, extra_files=()):
    """pages: [(name, data, out_png)] rendered headlessly (one browser,
    local HTTP server so ES modules load; vendored three.js r152, no
    network). Returns {name: {"png", "ok", "errors"}}."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    tmp = tempfile.mkdtemp(prefix="morphix_i3d_")
    shutil.copy(_vendor_three(), os.path.join(tmp, "three.module.js"))
    for f in extra_files:
        shutil.copy(f, os.path.join(tmp, os.path.basename(f)))
    for i, (name, data, _png) in enumerate(pages):
        with open(os.path.join(tmp, f"p{i}.html"), "w",
                  encoding="utf-8") as f:
            f.write(data if isinstance(data, str)
                    else page_html(data, "./three.module.js"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              partial(Quiet, directory=tmp))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    out = {}
    try:
        with sync_playwright() as p:
            br = p.chromium.launch(args=["--use-gl=angle",
                                         "--use-angle=swiftshader",
                                         "--no-sandbox"])
            try:
                pg = br.new_page(viewport={"width": RENDER_W,
                                           "height": RENDER_H})
                for i, (name, _data, png) in enumerate(pages):
                    errors = []

                    def on_err(e, errs=errors):
                        errs.append(str(e))
                    pg.on("pageerror", on_err)
                    pg.goto(f"{base}/p{i}.html", wait_until="load",
                            timeout=timeout_ms)
                    try:
                        pg.wait_for_function("window.__done === true",
                                             timeout=timeout_ms)
                        pg.screenshot(path=png)
                        out[name] = {"png": png, "ok": True, "errors": errors}
                    except Exception as e:  # render never finished
                        out[name] = {"png": None, "ok": False,
                                     "errors": errors + [str(e)]}
                    pg.remove_listener("pageerror", on_err)
            finally:
                br.close()
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _deposit_rows(records, color_index):
    rows = []
    for r in records:
        ci = color_index(r)
        if r.get("action") == "deposit_segment":
            rows.append([0, *[round(v, 4) for v in r["from"]],
                         *[round(v, 4) for v in r["to"]],
                         r["bead_width"], r["bead_height"], ci])
        elif r.get("action") == "deposit_point":
            rows.append([1, *[round(v, 4) for v in r["at"]], r["size"], ci])
    return rows


def _bg_for(body_hex):
    return "0xD8DDE2" if SH._luma(body_hex) < 110 else "0x1C2128"


def _env_display(ir):
    """Display-only context meshes for environment-preserving mode."""
    out = []
    if ir["environment"]["mode"] != "preserve":
        return out
    lo_x = -ir["bbox"][0] / 2
    for e in ir["environment"]["elements"]:
        if not e.get("build"):
            continue
        if e["kind"] == "platform":
            r = max(ir["bbox"][0], ir["bbox"][2]) * 0.62
            out.append({"kind": "platform", "r": round(r, 3),
                        "color": e["color"], "x": 0.0, "z": 0.0})
        elif e["kind"] == "floor":
            out.append({"kind": "floor", "size": round(abs(lo_x) * 6, 2),
                        "color": e["color"], "x": 0.0, "z": 0.0})
    return out


def render_set(ir, records, out_dir, tag=""):
    """Shaded print render, id pass, smooth IR preview (BufferGeometry),
    and (preserve mode) a display render with the environment."""
    comps = ir["components"]
    idx = {c["id"]: i for i, c in enumerate(comps)}
    palette = ir["appearance"]["palette"]
    mats = sorted(palette)
    bg = _bg_for(ir["appearance"]["body_color"] or "0x888888")
    cam = ir["camera"]
    shaded = {"mode": "shaded", "bg": bg, "camera": cam,
              "colors": [palette[m] for m in mats],
              "deposits": _deposit_rows(
                  records, lambda r: mats.index(r.get("material"))
                  if r.get("material") in mats else 0),
              "comps": [], "env": []}
    ids = {"mode": "ids", "bg": "0x000000", "camera": cam,
           "colors": [ID_COLORS[i % len(ID_COLORS)] for i in range(len(comps))],
           "deposits": _deposit_rows(
               records, lambda r: idx.get(r.get("component"), 0)),
           "comps": [], "env": []}
    preview = {"mode": "shaded", "bg": bg, "camera": cam,
               "colors": [c["color"] for c in comps], "deposits": [],
               "comps": [dict(c, ci=i) for i, c in enumerate(comps)],
               "env": []}
    pages = [("render", shaded, os.path.join(out_dir, f"render{tag}.png")),
             ("ids", ids, os.path.join(out_dir, f"ids{tag}.png")),
             ("preview", preview, os.path.join(out_dir, f"preview{tag}.png"))]
    # realistic presentation of the same IR at the fitted photo camera
    import morphix_image_display as DSP
    tex = os.path.join(out_dir, "texture.png")
    has_tex = os.path.exists(tex) and any(
        c.get("texture") == "photo" for c in comps)
    pages.append(("model", DSP.display_html(
        ir, verify=True, three_url="./three.module.js", addons_url="./",
        width=RENDER_W, height=RENDER_H, bg=bg,
        texture="texture.png" if has_tex else None),
        os.path.join(out_dir, f"model{tag}.png")))
    # fixed three-quarter overview of the PRINT (display only, not compared):
    # shows the full proportions a head-on reference view hides
    pages.append(("overview", dict(shaded, camera=SH.camera_for(
        comps, 40.0, 22.0)), os.path.join(out_dir, f"overview{tag}.png")))
    env = _env_display(ir)
    if env:
        disp = dict(shaded, env=env, bg=(ir["environment"]["elements"][0]
                                         .get("color", bg)))
        pages.append(("display", disp,
                      os.path.join(out_dir, f"display{tag}.png")))
    res = render_pages(pages, extra_files=[tex] if has_tex else ())
    res["_bg"] = bg
    return res


# ------------------------------------------------------------------ checks
def _hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (2, 4, 6)], float)


def render_mask(png, bg_hex, tol=40):
    a = np.asarray(Image.open(png).convert("RGB")).astype(float)
    return np.abs(a - _hex_to_rgb(bg_hex)).sum(2) > tol, a


def id_map(png, n):
    a = np.asarray(Image.open(png).convert("RGB")).astype(float)
    cols = np.array([_hex_to_rgb(ID_COLORS[i % len(ID_COLORS)])
                     for i in range(n)])
    d = ((a[:, :, None, :] - cols[None, None, :, :]) ** 2).sum(3)
    ids = d.argmin(2)
    ids[d.min(2) > 60 ** 2] = -1
    ids[a.sum(2) < 30] = -1
    return ids


def _bbox(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _centroids(ids, n, ref_box):
    x0, y0, x1, y1 = ref_box
    out = {}
    for i in range(n):
        ys, xs = np.where(ids == i)
        if len(xs) >= 6:
            out[i] = ((xs.mean() - x0) / max(1, x1 - x0),
                      (ys.mean() - y0) / max(1, y1 - y0), len(xs))
    return out


def family_rules(family, comps, cent, yaw):
    """Semantic placement rules on the RENDER (not on component names)."""
    checks = []

    def ck(name, ok, detail, sev="critical"):
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "severity": sev})
    role = {i: c["role"] for i, c in enumerate(comps)}
    vis = {i: v for i, v in cent.items()}
    by = {}
    for i, v in vis.items():
        by.setdefault(role[i], []).append(v)
    if family == "vehicle":
        wheels = by.get("wheel", [])
        side_on = 25 <= abs(yaw) <= 155
        need = 2 if side_on else 1
        ck("wheels_visible", len(wheels) >= need,
           f"{len(wheels)} wheels visible (need >= {need} at yaw {yaw:.0f})")
        if wheels:
            low = all(w[1] >= 0.5 for w in wheels)
            ck("wheels_at_bottom", low,
               "wheel centroids y=" + ", ".join(f"{w[1]:.2f}" for w in wheels)
               + " (must be in the lower half)")
        if side_on and len(wheels) >= 2:
            span = max(w[0] for w in wheels) - min(w[0] for w in wheels)
            ck("front_rear_axles", span >= 0.3,
               f"wheel x-span {span:.2f} of body width (>= 0.30)")
        body = by.get("body", [])
        cab = by.get("cabin", []) + by.get("roof", [])
        ck("body_present", bool(body), "body visible" if body else "no body")
        if body and cab:
            ok = min(c[1] for c in cab) < body[0][1]
            ck("cabin_above_body", ok,
               f"cabin y {min(c[1] for c in cab):.2f} vs body y "
               f"{body[0][1]:.2f}")
        elif body:
            ck("cabin_above_body", False, "cabin not visible")
    elif family == "table":
        top = by.get("tabletop", [])
        legs = by.get("leg", [])
        ck("top_present", bool(top), "tabletop visible" if top else "no top")
        ck("legs_visible", len(legs) >= 2, f"{len(legs)} legs visible")
        if top and legs:
            ck("legs_below_top", all(l[1] > top[0][1] for l in legs),
               "legs under the top")
    elif family == "chair":
        seat, back, legs = by.get("seat", []), by.get("backrest", []), \
            by.get("leg", [])
        ck("seat_present", bool(seat), "seat visible" if seat else "no seat")
        ck("backrest_present", bool(back), "backrest visible" if back
           else "no backrest", "major")
        ck("legs_visible", len(legs) >= 2, f"{len(legs)} legs visible")
        if seat and back:
            ck("backrest_above_seat", back[0][1] < seat[0][1],
               "backrest above seat")
    elif family == "stool":
        ck("seat_present", bool(by.get("seat")), "seat")
        ck("legs_visible", len(by.get("leg", [])) >= 2, "legs")
    elif family == "house":
        walls, roof = by.get("wall", []), by.get("roof", [])
        ck("walls_present", bool(walls), "walls")
        ck("roof_present", bool(roof), "roof")
        if walls and roof:
            ck("roof_above_walls", roof[0][1] < walls[0][1], "roof on top")
    elif family == "quadruped":
        body, head, legs = by.get("body", []), by.get("head", []), \
            by.get("leg", [])
        ck("body_present", bool(body), "torso")
        ck("head_present", bool(head), "head")
        ck("legs_visible", len(legs) >= 2, f"{len(legs)} legs visible")
        if body and head:
            ck("head_above_or_ahead", head[0][1] <= body[0][1],
               "head at or above torso height")
    return checks


def verify(ir, seg, rendered, obs, ask=None):
    """Measured comparison of the rendered result with the reference."""
    checks = []

    def ck(name, ok, detail, sev="critical", value=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "severity": sev, "value": value})
    r = rendered.get("render") or {}
    if not r.get("ok"):
        ck("render_completed", False, f"render failed: {r.get('errors')}")
        return {"checks": checks, "critical_ok": False, "metrics": {}}
    ck("render_completed", not r["errors"],
       "no page errors" if not r["errors"] else f"errors: {r['errors'][:3]}",
       "major")
    bg = rendered["_bg"]
    m, rgb = render_mask(r["png"], bg)
    ref = seg["_mask"]
    if m.sum() < 50:
        ck("render_nonempty", False, "rendered subject is empty")
        return {"checks": checks, "critical_ok": False, "metrics": {}}
    ign = seg.get("_ignore")
    if ign is None:
        iou = SEG.mask_iou_normalized(m, ref)
        ar = SEG.aspect_of(m) / SEG.aspect_of(ref)
    else:  # occluded reference: compare outside the occluder only
        rr = SH.RefMask(ref & ~ign, ignore=ign)
        _s, iou, ar = SH.silhouette_score(m, rr)
    ck("silhouette_agreement", iou >= IOU_MIN,
       f"bbox-normalized IoU {iou:.3f} (>= {IOU_MIN})", value=round(iou, 3))
    ck("aspect_agreement", abs(ar - 1) <= ASPECT_TOL,
       f"render/reference aspect x{ar:.2f} (within ±{ASPECT_TOL:.0%})",
       value=round(ar, 3))
    fx0, fy0, fx1, fy1 = _bbox(m)
    frac = m.mean()
    ck("foreground_separation", 0.02 <= frac <= 0.8
       and fx0 > 0 and fy0 > 0 and fx1 < RENDER_W - 1 and fy1 < RENDER_H - 1,
       f"subject covers {frac:.3f} of the frame, fully inside it")
    comps = ir["components"]
    cent = {}
    idr = rendered.get("ids") or {}
    if idr.get("ok"):
        ids = id_map(idr["png"], len(comps))
        cent = _centroids(ids, len(comps), (fx0, fy0, fx1, fy1))
        # required components that the PLAN says are visible must render
        pm, pids = SH.project(comps, ir["camera"], 160, 120)
        pbox = _bbox(pm)
        planned = _centroids(pids, len(comps), pbox)
        missing, shifted = [], []
        min_area = 0.02 * max(1, int(pm.sum()))
        for i, c in enumerate(comps):
            # only parts the plan shows with a meaningful visible area must
            # render (a far wheel hidden behind the near one does not count)
            if not c.get("required") or i not in planned                     or planned[i][2] < min_area:
                continue
            if i not in cent:
                missing.append(c["id"])
                continue
            dx = abs(cent[i][0] - planned[i][0])
            dy = abs(cent[i][1] - planned[i][1])
            if max(dx, dy) > PLAN_SHIFT_MAX:
                shifted.append(f"{c['id']} ({dx:.2f},{dy:.2f})")
        ck("required_components_rendered", not missing,
           "all planned-visible required parts render" if not missing
           else f"missing in render: {missing}")
        ck("relative_placement", not shifted,
           "parts render where the plan places them" if not shifted
           else f"displaced: {shifted}")
        fam = (ir["subject"] or {}).get("family")
        if ir["method"]["builder"] == "family":
            checks.extend(family_rules(fam, comps, cent,
                                       ir["camera"]["yaw"]))
        # reference anchors: lamps / ground contacts agree with the photo
        _ign = seg.get("_ignore")
        anchors = reference_anchors(
            seg if _ign is None else {**seg, "_mask": seg["_mask"] & ~_ign},
            fam)
        if "light" in anchors:
            lights = [cent[i] for i, c in enumerate(comps)
                      if c["role"] == "light" and i in cent]
            if lights:
                gx = sum(l[0] for l in lights) / len(lights)
                ok = abs(gx - anchors["light"][0]) <= 0.25
                ck("front_matches_reference", ok,
                   f"lamps at x={gx:.2f} vs reference bright spots "
                   f"x={anchors['light'][0]:.2f}", "major")
        if "contacts" in anchors:
            mc = SH.contact_positions(m)
            if mc:
                d = sum(min(abs(a - b) for b in mc)
                        for a in anchors["contacts"]) / len(anchors["contacts"])
                ck("ground_contacts_match", d <= 0.12,
                   f"mean contact offset {d:.3f} (reference "
                   f"{[round(v, 2) for v in anchors['contacts']]}, render "
                   f"{[round(v, 2) for v in mc]})", "major")
    else:
        ck("id_pass", False, f"id render failed: {idr.get('errors')}")
    # the realistic presentation must show the same shape as the photo
    md = rendered.get("model") or {}
    if md.get("ok"):
        mm, _ = render_mask(md["png"], bg)
        if ign is None:
            miou = SEG.mask_iou_normalized(mm, ref)
        else:
            _s, miou, _a = SH.silhouette_score(
                mm, SH.RefMask(ref & ~ign, ignore=ign))
        ck("display_silhouette_agreement", miou >= IOU_MIN,
           f"realistic render IoU {miou:.3f} (>= {IOU_MIN})",
           value=round(miou, 3))
        ck("display_render_clean", not md["errors"],
           "no page errors" if not md["errors"] else str(md["errors"][:2]),
           "major")
    else:
        ck("display_render_completed", False,
           f"realistic render failed: {md.get('errors')}")
    # large color region consistency (body paint)
    body_ids = [i for i, c in enumerate(comps) if c["stage"] == "main_volume"]
    sel = np.zeros(m.shape, bool)
    if idr.get("ok"):
        for i in body_ids:
            sel |= ids == i
    sel &= m
    if sel.sum() > 30 and seg["subject"]["body_color"]:
        med = np.median(rgb[sel], axis=0)
        got = SEG._hex(med)
        de = SEG.lab_distance(got, seg["subject"]["body_color"])
        ck("color_region_consistency", de <= COLOR_DE_MAX,
           f"rendered body {got} vs reference body "
           f"{seg['subject']['body_color']} ΔE {de:.1f} (<= {COLOR_DE_MAX})",
           "major", round(de, 1))
    # the paint must agree with a colour vision states for the subject
    for w in obs.get("subject_colors", []) if sel.sum() > 30 else []:
        de = SEG.lab_distance(SEG._hex(np.median(rgb[sel], axis=0)),
                              VIS.NAMED_COLORS[w])
        ck("stated_color_consistency", de <= 45,
           f"rendered body vs stated {w!r}: ΔE {de:.0f} (<= 45)", "major")
    # environment separation: object-only builds contain no environment
    if ir["environment"]["mode"] == "object-only":
        printed_env = [c["id"] for c in comps if c.get("role") in
                       ("platform", "floor", "backdrop")]
        ck("environment_excluded", not printed_env,
           "no environment geometry in the object" if not printed_env
           else f"environment printed: {printed_env}")
    # recognizability: deterministic composite + ADVISORY vision opinion
    crit = [c for c in checks if c["severity"] == "critical"]
    recog = all(c["ok"] for c in crit)
    advisory = None
    if ask is not None and obs.get("category"):
        try:
            raw = ask("", "What is the main object in this image?")
            kind = VIS.classify_reply(raw)
            ev = VIS.extract_subject(raw, kind)
            got = (ev or {}).get("category")
            agree = bool(ev) and ((ev.get("family") or got)
                                  == (obs.get("family") or obs["category"]))
            advisory = {"reply": raw[:300], "kind": kind, "category": got,
                        "agrees": agree,
                        "note": "advisory only: a vision opinion never "
                                "proves geometry"}
        except Exception as e:
            advisory = {"error": str(e)}
    ck("subject_recognizable", recog,
       "all critical shape/placement checks pass" if recog else
       "critical checks failed: " + ", ".join(c["name"] for c in crit
                                               if not c["ok"]))
    return {"checks": checks,
            "critical_ok": all(c["ok"] for c in checks
                               if c["severity"] == "critical"),
            "major_failed": [c["name"] for c in checks
                             if c["severity"] == "major" and not c["ok"]],
            "metrics": {"iou": round(iou, 3), "aspect_ratio": round(ar, 3)},
            "vision_on_render": advisory}


def comparison_sheet(ref_path, seg, rendered, out_png):
    """Side-by-side: reference | segmentation | printed render | preview."""
    tiles = []
    ref = Image.open(ref_path).convert("RGB")
    ref.thumbnail((320, 240))
    tiles.append(ref)
    mk = Image.fromarray(seg["_mask"].astype(np.uint8) * 255).convert("RGB")
    mk.thumbnail((320, 240))
    tiles.append(mk)
    for k in ("model", "render", "preview", "overview", "display"):
        p = (rendered.get(k) or {}).get("png")
        if p and os.path.exists(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((320, 240))
            tiles.append(im)
    sheet = Image.new("RGB", (320 * len(tiles), 240), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * 320 + (320 - t.width) // 2, (240 - t.height) // 2))
    sheet.save(out_png)
    return out_png


# ------------------------------------------------------------------ run
def _print(ir, out_dir, emit, segment_cap=6000):
    import morphix_pipeline as P
    from morphix_ir import ir_to_actions
    actions = ir_to_actions(ir)
    label = ir["subject"].get("label") or ir["category"]
    rep = P.run_path_print(
        f"reconstruct the photographed {label}", out_dir=out_dir, emit=emit,
        segment_cap=segment_cap, layer_cap=32, plan_actions=actions,
        plan_status=f"IMAGE {ir['category']}", spec_name="image",
        plan_source=ir["source"], stream_scene=False,
        palette=ir["appearance"]["palette"])
    with open(os.path.join(out_dir, "path.json"), encoding="utf-8") as f:
        records = json.load(f).get("records", [])
    return rep, actions, records


def _fit(method, kind, seg, paint, obs, budget=900, exclude=()):
    if method == "family":
        ign = seg.get("_ignore")
        vis = seg["_mask"] if ign is None else seg["_mask"] & ~ign
        anchors = reference_anchors({**seg, "_mask": vis}, kind)
        presets = SH.BUILDERS[kind][1]
        if exclude:
            keep = {k: v for k, v in presets.items() if k not in exclude}
            if keep:
                SH.BUILDERS[kind] = (SH.BUILDERS[kind][0], keep)
        try:
            return SH.fit(kind, vis, paint,
                          view_label=(obs.get("viewpoint") or {}).get("label"),
                          budget=budget, anchors=anchors, ignore=ign,
                          label=obs.get("label"))
        finally:
            SH.BUILDERS[kind] = (SH.BUILDERS[kind][0], presets)
    return SH.fit_generic(kind, seg["_mask"], paint)


def run(image_path, out_dir, category=None, vision_model="",
        environment="object-only", emit=None, ask=None, render=True,
        max_repairs=MAX_REPAIRS):
    """Full reconstruction. Returns the report dict (also written to
    out_dir/report.json). Never raises for vision or quality problems —
    those are verdicts; raises ReconstructionError only when nothing can
    be built (no separable subject, unprintable geometry)."""
    from morphix_image import available_vision_model
    from morphix_config import VISION_MODEL
    emit = emit or (lambda e: None)
    t0 = time.monotonic()
    if environment not in ("object-only", "preserve"):
        raise ReconstructionError("input", f"environment must be "
                                  f"object-only|preserve (got {environment!r})")
    os.makedirs(out_dir, exist_ok=True)
    trace = os.path.join(out_dir, "trace")
    os.makedirs(trace, exist_ok=True)
    shutil.copyfile(image_path, os.path.join(trace, "reference" +
                                             os.path.splitext(image_path)[1]))
    timings = {}

    seg = SEG.segment(image_path)
    SEG.save_mask(seg, os.path.join(trace, "mask.png"))
    _dump(os.path.join(trace, "segmentation.json"), SEG.public(seg))
    timings["segment_s"] = round(time.monotonic() - t0, 2)
    emit({"type": "segmentation", "ok": seg["ok"], "method": seg["method"],
          "subject": (seg.get("subject") or {}).get("bbox")})
    if not seg["ok"]:
        raise ReconstructionError("segmentation", seg["error"],
                                  {"segmentation": SEG.public(seg)})

    model = vision_model or VISION_MODEL or (available_vision_model()
                                             if ask is None else "test")
    obs = identify(image_path, seg, category=category, model=model, ask=ask,
                   emit=emit)
    _dump(os.path.join(trace, "vision.json"), obs)
    timings["vision_s"] = round(time.monotonic() - t0 - timings["segment_s"], 2)
    emit({"type": "subject", "category": obs.get("category"),
          "family": obs.get("family"), "status": obs.get("status"),
          "confidence": obs.get("confidence")})

    method, kind = choose_method(obs, seg)
    paint = SH.paint_from_clusters(seg["subject"]["colors"], obs.get("family"))
    attempts = []
    fitres = _fit(method, kind, seg, paint, obs)
    # Occlusion: a uniform block glued to the subject's edge is accepted as
    # an occluder ONLY if ignoring it measurably improves the fit. Its
    # pixels then count neither for nor against the shape, and paint is
    # re-measured on the visible subject.
    seg["occluder"] = None
    if method == "family":
        for cand in SEG.occluder_candidates(seg):
            vis = seg["_mask"] & ~cand["mask"]
            vis_paint = SH.paint_from_clusters(
                SEG.subject_colors(seg["_rgb"], vis), obs.get("family"))
            alt = SH.fit(kind, vis, vis_paint,
                         view_label=(obs.get("viewpoint") or {}).get("label"),
                         anchors=reference_anchors({**seg, "_mask": vis},
                                                   kind),
                         ignore=cand["mask"])
            gain = alt["iou"] - fitres["iou"]
            # corroboration beyond fit gain: the subject colour vision
            # states (measured rest of the mask matches it, the block does
            # not), or a co-object vision names whose colour matches the
            # block. Vision alone never decides; the block must already be
            # a geometric occluder candidate.
            why = []
            if gain >= 0.06:
                why.append(f"fit IoU +{gain:.3f} when ignored")
            rest_col = SEG._hex(seg["_rgb"][vis].mean(0))
            for w in obs.get("subject_colors", []):
                named = VIS.NAMED_COLORS[w]
                d_c = SEG.lab_distance(cand["color"], named)
                d_r = SEG.lab_distance(vis_paint["body"], named)
                if d_c > 40 and d_r < d_c - 15:
                    why.append(f"subject stated {w!r}: block {cand['color']} "
                               f"ΔE {d_c:.0f}, rest {vis_paint['body']} "
                               f"ΔE {d_r:.0f}")
            for co in obs.get("co_objects", []):
                if co.get("color") and SEG.lab_distance(
                        cand["color"], VIS.NAMED_COLORS[co["color"]]) < 35:
                    why.append(f"vision names a {co['color']} {co['noun']} "
                               f"beside the subject")
            attempts.append({"stage": "occlusion", "candidate": cand["color"],
                             "fraction": cand["fraction"],
                             "iou_without": round(alt["iou"], 3),
                             "iou_with": round(fitres["iou"], 3),
                             "evidence": why, "accepted": bool(why)})
            if why:
                seg["occluder"] = {k: v for k, v in cand.items()
                                   if k != "mask"}
                seg["_ignore"] = cand["mask"]
                seg["subject"]["colors"] = SEG.subject_colors(seg["_rgb"],
                                                              vis)
                seg["subject"]["body_color"] =                     seg["subject"]["colors"][0]["color"]
                seg.setdefault("notes", []).append(
                    f"occluder {cand['color']} ({cand['fraction']:.0%} of the "
                    f"mask) excluded: " + "; ".join(why))
                paint, fitres = vis_paint, alt
                break
    # Pre-print gate on the plan's projected silhouette: a family builder
    # that cannot reproduce the outline falls back to the subject's own
    # measured silhouette (reported), never to an unrelated shape.
    if method == "family" and fitres["iou"] < IOU_MIN:
        alt_kind = "revolve" if seg["subject"]["symmetry"] >= 0.9 else "inflate"
        alt = SH.fit_generic(alt_kind, seg["_mask"], paint)
        attempts.append({"stage": "plan", "action": f"family fit IoU "
                         f"{fitres['iou']:.3f} < {IOU_MIN}; measured-"
                         f"silhouette {alt_kind} IoU {alt['iou']:.3f}"})
        if alt["iou"] > fitres["iou"]:
            method, kind, fitres = "silhouette", alt_kind, alt
    report = None
    verification = None
    for attempt in range(max_repairs + 1):
        if any(c.get("texture") == "photo" for c in fitres["comps"]):
            make_texture(image_path, seg, os.path.join(trace, "texture.png"))
        ir = build_ir(obs, seg, method, kind, fitres, environment)
        errs = validate_image_ir(ir)
        if errs:
            raise ReconstructionError("ir_validation", "; ".join(errs[:6]),
                                      {"ir": ir})
        _dump(os.path.join(trace, f"image_ir{'' if not attempt else attempt}.json"), ir)
        emit({"type": "ir", "category": ir["category"],
              "builder": method, "kind": kind,
              "components": [c["id"] for c in ir["components"]]})
        try:
            report, actions, records = _print(ir, out_dir, emit)
        except Exception as e:
            raise ReconstructionError("printing", str(e), {"ir": ir})
        _dump(os.path.join(trace, "component_plan.json"), actions)
        if not render:
            verification = {"checks": [], "critical_ok": None,
                            "note": "browser verification disabled"}
            break
        tag = "" if attempt == 0 else f"_r{attempt}"
        try:
            rendered = render_set(ir, records, trace, tag)
        except Exception as e:
            verification = {"checks": [{"name": "render_completed",
                                        "ok": False, "severity": "critical",
                                        "detail": f"browser unavailable: {e}"}],
                            "critical_ok": False}
            break
        render_ask = ask
        if render_ask is None and obs.get("model") and obs.get("available"):
            p = (rendered.get("render") or {}).get("png")
            render_ask = VIS.ollama_ask(p, obs["model"]) if p else None
        verification = verify(ir, seg, rendered, obs, ask=render_ask)
        verification["attempt"] = attempt
        verification["renders"] = {k: os.path.basename(v["png"])
                                   for k, v in rendered.items()
                                   if isinstance(v, dict) and v.get("png")}
        comparison_sheet(image_path, seg, rendered,
                         os.path.join(trace, f"compare{tag}.png"))
        _dump(os.path.join(trace, f"verification{tag}.json"), verification)
        emit({"type": "verify", "attempt": attempt,
              "critical_ok": verification["critical_ok"],
              "metrics": verification["metrics"]})
        if verification["critical_ok"] or attempt == max_repairs:
            break
        failed = [c["name"] for c in verification["checks"]
                  if c["severity"] == "critical" and not c["ok"]]
        # Bounded repair: (1) refit with a larger budget excluding the
        # failing preset, (2) the subject's measured silhouette.
        if attempt == 0 and method == "family":
            fitres = _fit(method, kind, seg, paint, obs, budget=1800,
                          exclude=(fitres["preset"],))
            attempts.append({"stage": "render", "failed": failed,
                             "action": f"refit excluding preset; new preset "
                                       f"{fitres['preset']!r} IoU "
                                       f"{fitres['iou']:.3f}"})
        else:
            alt_kind = ("revolve" if seg["subject"]["symmetry"] >= 0.9
                        and seg["subject"]["aspect"] <= 1.4 else "inflate")
            if method == "silhouette" and kind == alt_kind:
                attempts.append({"stage": "render", "failed": failed,
                                 "action": "no further repair available"})
                break
            method, kind = "silhouette", alt_kind
            fitres = SH.fit_generic(kind, seg["_mask"], paint)
            attempts.append({"stage": "render", "failed": failed,
                             "action": f"rebuilt from measured silhouette "
                                       f"({kind})"})

    identified = bool(obs.get("category"))
    crit = verification.get("critical_ok")
    if crit is False:
        status = "failed"
    elif method == "family" and identified:
        status = "reconstructed"
    else:
        status = "approximation"
    report["mode"] = "image-reconstruct"
    report["ir_full"] = ir
    report["image"] = {
        "path": os.path.basename(image_path),
        "status": status,
        "subject": ir["subject"],
        "builder": {"method": method, "kind": kind,
                    "preset": fitres.get("preset")},
        "vision": {k: obs.get(k) for k in ("model", "status", "category",
                                           "label", "confidence",
                                           "decided_by", "failure",
                                           "viewpoint", "parts")},
        "segmentation": SEG.public(seg),
        "environment": ir["environment"],
        "repairs": attempts,
        "timings": {**timings, "total_s": round(time.monotonic() - t0, 1)},
        "trace_dir": "trace",
    }
    report["visual_verification"] = verification
    report["reconstruction_status"] = status
    report["recognized"] = status == "reconstructed"
    warnings = list(report.get("warnings", []))
    if obs.get("failure"):
        warnings.append(f"vision {obs['failure']['kind']}: subject not "
                        f"identified; built from its measured silhouette")
    if method == "silhouette":
        warnings.append("silhouette approximation: " + "; ".join(
            ir["uncertainty"]["assumptions"]))
    if verification.get("major_failed"):
        warnings.append("minor visual mismatches: "
                        + ", ".join(verification["major_failed"]))
    if status == "failed":
        warnings.append("reconstruction FAILED visual verification after "
                        f"{len(attempts)} repair(s): " + ", ".join(
                            c["name"] for c in verification["checks"]
                            if c["severity"] == "critical" and not c["ok"]))
    report["warnings"] = warnings
    report["quality_accepted"] = bool(report.get("ok") and report.get(
        "complete") and status != "failed" and crit is not False)
    report["success"] = report["quality_accepted"]
    _dump(os.path.join(out_dir, "report.json"), report)
    _dump(os.path.join(out_dir, "object.json"), ir)
    import morphix_image_display as DSP
    with open(os.path.join(out_dir, "model.html"), "w", encoding="utf-8") as f:
        f.write(DSP.display_html(ir, texture="trace/texture.png" if any(
            c.get("texture") == "photo" for c in ir["components"]) else None))
    print(f"RECONSTRUCT: {ir['category']} [{method}/{kind}] -> {status} "
          f"(IoU {verification.get('metrics', {}).get('iou')})")
    return report


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Open-ended image -> 3D")
    ap.add_argument("image")
    ap.add_argument("--out", default="./out_recon")
    ap.add_argument("--category", default="")
    ap.add_argument("--vision-model", default="")
    ap.add_argument("--environment", default="object-only",
                    choices=("object-only", "preserve"))
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    rep = run(a.image, a.out, category=a.category or None,
              vision_model=a.vision_model, environment=a.environment,
              render=not a.no_render)
    print(json.dumps({"status": rep["reconstruction_status"],
                      "success": rep["success"],
                      "subject": rep["image"]["subject"],
                      "metrics": rep["visual_verification"].get("metrics"),
                      "warnings": rep["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
