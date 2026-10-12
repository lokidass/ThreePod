# tests/visual_geometry/render.py
# Deterministic multi-view rendering + tolerant image comparison.
#
# Every render uses fixed viewport (640x480), fixed cameras, fixed lights,
# fixed background, antialiasing OFF and a locally vendored pinned three.js
# (vendor/three.module.js) — no CDN, no UI, no OrbitControls. Renders are
# therefore a pure function of the committed scene JS. Comparison uses
# SSIM (>= threshold passes), never exact pixel equality: lighting and
# software-rasterizer differences must not cause false failures.
# On mismatch the actual render and an amplified diff image are saved
# under .actual/ for diagnosis. Baselines are NEVER overwritten by tests;
# only regen_baselines.py --approve writes baselines/.
import json
import math
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH, HEIGHT = 640, 480
VIEWS = ("front", "side", "threequarter", "rear", "top")
BASELINE_DIR = os.path.join(HERE, "baselines")
ACTUAL_DIR = os.path.join(HERE, ".actual")
SSIM_THRESHOLD = 0.985
# Mid-gray background (#3d465e, also hardcoded in the harness template):
# dark tires and bright bodies must both read against it.

# fixed camera directions (unit-ish); distance fitted per fixture bbox
CAM_DIRS = {
    "front": (1.0, 0.35, 0.0),
    "side": (0.0, 0.35, 1.0),
    "threequarter": (1.0, 0.55, 1.0),
    "rear": (-1.0, 0.35, 0.5),
    "top": (0.001, 1.0, 0.001),
}

HARNESS_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>regression</title>
<style>html,body{margin:0;padding:0;background:#3d465e}canvas{display:block}</style>
</head><body>
<script type="importmap">{"imports":{"three":"__VENDOR__"}}</script>
<script type="module">
import * as THREE from 'three';
const renderer = new THREE.WebGLRenderer({antialias:false});
renderer.setPixelRatio(1);
renderer.setSize(__W__, __H__);
renderer.setClearColor(0x3d465e, 1);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x3d465e);
const camera = new THREE.PerspectiveCamera(40, __W__/__H__, 0.1, 500);
camera.position.set(__POS__);
camera.lookAt(__TGT__);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.9));
const dir1 = new THREE.DirectionalLight(0xffffff, 1.4);
dir1.position.set(6, 10, 7);
scene.add(dir1);
const dir2 = new THREE.DirectionalLight(0x8899ff, 0.4);
dir2.position.set(-6, 4, -6);
scene.add(dir2);
// ---- committed scene blocks (identical JS the printer committed) ----
__BLOCKS__
// ---- deterministic settle then signal ----
let frames = 0;
window.__done = false;
function tick() {
  renderer.render(scene, camera);
  if (++frames >= 3) window.__done = true;
  else requestAnimationFrame(tick);
}
tick();
</script></body></html>
"""


def _camera_for(view, bbox):
    (lo, hi) = bbox
    cx, cy, cz = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                  (lo[2] + hi[2]) / 2)
    size = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.5)
    dist = size * 2.0 + 0.5
    dx, dy, dz = CAM_DIRS[view]
    n = (dx * dx + dy * dy + dz * dz) ** 0.5
    pos = (cx + dx / n * dist, cy + dy / n * dist, cz + dz / n * dist)
    return pos, (cx, cy, cz)


def _view_matrix(eye, tgt):
    import numpy as _np
    eye = _np.array(eye, float)
    tgt = _np.array(tgt, float)
    fwd = tgt - eye
    fwd = fwd / _np.linalg.norm(fwd)
    up = _np.array([0.0, 1.0, 0.0])
    if abs(abs(float(fwd @ up)) - 1.0) < 1e-3:
        up = _np.array([0.0, 0.0, 1.0])
    s = _np.cross(fwd, up)
    s = s / _np.linalg.norm(s)
    u = _np.cross(s, fwd)
    m = _np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def project_aabb(aabb, view, frame_bbox):
    """Project a world-space AABB to a pixel rect (x0, y0, x1, y1) under the
    deterministic view camera. Mirrors the harness (fov 40, 640x480)."""
    import numpy as _np
    pos, tgt = _camera_for(view, frame_bbox)
    vm = _view_matrix(pos, tgt)
    fov = math.radians(40.0)
    aspect = WIDTH / HEIGHT
    (lo, hi) = aabb
    pts = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                v = _np.array([x, y, z, 1.0])
                c = vm @ v
                depth = -c[2]
                if depth <= 0.05:
                    continue
                ndc_x = (c[0] / depth) / (math.tan(fov / 2) * aspect)
                ndc_y = (c[1] / depth) / math.tan(fov / 2)
                pts.append((ndc_x, ndc_y))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 0.05
    x0 = max(0, int((min(xs) * 0.5 + 0.5 - pad) * WIDTH))
    x1 = min(WIDTH, int((max(xs) * 0.5 + 0.5 + pad) * WIDTH))
    # NDC y up vs image y down
    y0 = max(0, int((0.5 - max(ys) - pad) * HEIGHT))
    y1 = min(HEIGHT, int((0.5 - min(ys) + pad) * HEIGHT))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def crop_ssim(a_path, b_path, rect):
    """SSIM over a pixel-rect crop. Small regions are noisier, so callers
    use a lower threshold than full-frame."""
    from PIL import Image as _I
    import tempfile as _tf
    a = _I.open(a_path).convert("L").crop(rect)
    b = _I.open(b_path).convert("L").crop(rect)
    if a.size != b.size or a.size[0] < 8 or a.size[1] < 8:
        return 0.0, "empty crop"
    tmp = _tf.gettempdir()
    crop_ssim.counter = getattr(crop_ssim, "counter", 0) + 1
    tag = f"vgcrop_{os.getpid()}_{crop_ssim.counter}"
    pa, pb = os.path.join(tmp, tag + "_a.png"), os.path.join(tmp, tag + "_b.png")
    try:
        a.save(pa)
        b.save(pb)
        return ssim_score(pa, pb), ""
    finally:
        for p in (pa, pb):
            try:
                os.unlink(p)
            except OSError:
                pass


def write_harness(path, scene_js, view, bbox, vendor_url):
    pos, tgt = _camera_for(view, bbox)
    html = HARNESS_TEMPLATE.replace("__VENDOR__", vendor_url)
    html = html.replace("__W__", str(WIDTH)).replace("__H__", str(HEIGHT))
    html = html.replace("__POS__", f"{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}")
    html = html.replace("__TGT__", f"{tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}")
    html = html.replace("__BLOCKS__", scene_js)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve_dir(directory):
    """Local HTTP server (file:// breaks ES module CORS). Returns (server, url)."""
    handler = partial(_QuietHandler, directory=directory)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


def screenshot_pages(pages, timeout_ms=30000):
    """pages: [(url, out_png)]. One shared browser; waits for window.__done
    (deterministic settle) instead of blind sleeps."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader",
            "--disable-gpu-sandbox", "--no-sandbox"])
        try:
            page = browser.new_page(
                viewport={"width": WIDTH, "height": HEIGHT})
            for url, out_png in pages:
                page.goto(url, wait_until="load", timeout=timeout_ms)
                page.wait_for_function("window.__done === true",
                                       timeout=timeout_ms)
                page.screenshot(path=out_png)
        finally:
            browser.close()


def _to_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def _gaussian(arr, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)


def ssim_score(a_path, b_path):
    """Mean SSIM in [0,1] between two grayscale renders (numpy+scipy)."""
    a = _to_gray(a_path)
    b = _to_gray(b_path)
    assert a.shape == b.shape, f"size mismatch {a.shape} vs {b.shape}"
    mu_a = _gaussian(a)
    mu_b = _gaussian(b)
    sig_a2 = _gaussian(a * a) - mu_a * mu_a
    sig_b2 = _gaussian(b * b) - mu_b * mu_b
    sig_ab = _gaussian(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / den))


def write_diff(a_path, b_path, out_path, amplify=4.0):
    a = _to_gray(a_path)
    b = _to_gray(b_path)
    diff = np.abs(a - b) * amplify
    Image.fromarray(np.clip(diff, 0, 255).astype("uint8")).save(out_path)


def compare_to_baseline(name, view, actual_png, threshold=SSIM_THRESHOLD):
    """Returns (ok, info). Saves actual + amplified diff on mismatch."""
    os.makedirs(ACTUAL_DIR, exist_ok=True)
    base = os.path.join(BASELINE_DIR, f"{name}_{view}.png")
    info = {"fixture": name, "view": view, "threshold": threshold}
    if not os.path.isfile(base):
        info.update(ok=False,
                    error="missing baseline (run regen_baselines.py --approve)")
        return False, info
    score = ssim_score(base, actual_png)
    info["ssim"] = round(score, 5)
    if score >= threshold:
        info["ok"] = True
        return True, info
    import shutil
    dst = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
    if os.path.abspath(actual_png) != os.path.abspath(dst):
        shutil.copy(actual_png, dst)
    # else the failing render is already in place: no self-copy.
    diff_path = os.path.join(ACTUAL_DIR, f"{name}_{view}.diff.png")
    write_diff(base, actual_png, diff_path)
    info.update(ok=False, diff=diff_path,
                error=f"SSIM {score:.4f} < {threshold}")
    return False, info


def load_manifest():
    path = os.path.join(BASELINE_DIR, "manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(entries):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    with open(os.path.join(BASELINE_DIR, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
