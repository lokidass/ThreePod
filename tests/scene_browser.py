# tests/scene_browser.py
# Browser harness for SCENE-mode validation (Phase E). Distinct from the
# printer-geometry harness in visual_geometry/render.py:
#   * loads a compiled Scene IR module (own camera, lights, fog, effects),
#   * waits for window.__scene_ready (NOT __done),
#   * captures window.__scene_error + window.__scene_stats + console errors,
#   * screenshots at a fixed 640x480, pixelRatio 1, antialias off.
# One shared headless Chromium per session (SwiftShader software GL).
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "visual_geometry"))

WIDTH, HEIGHT = 640, 480

VENDOR_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "visual_geometry", "vendor", "three.module.js")

PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>scene</title>
<style>html,body{margin:0;padding:0;background:#111622}canvas{display:block}</style>
</head><body>
<script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script>
<script type="module">
const __W__ = 640, __H__ = 480;
__SCENE_JS__
</script></body></html>
"""


def _compile(ir):
    import morphix_scene_compile as C
    js = C.compile_scene_ir(ir)
    errs = C.validate_scene_js(js)
    if errs:
        raise ValueError("scene JS invalid: " + "; ".join(errs[:5]))
    return js.replace("__W__", str(WIDTH)).replace("__H__", str(HEIGHT))


def render_scenes(pages):
    """Render compiled scene pages.

    pages: [(name, scene_ir_or_js, out_png)] -> {name: result dict} with
    keys: png, ready, scene_error, console_errors, stats, elapsed_s.
    A single shared browser session serves all pages.
    """
    import morphix_scene_compile as C
    from render import serve_dir, screenshot_pages
    import time

    srv_dir = tempfile.mkdtemp(prefix="morphix_scene_")
    os.makedirs(os.path.join(srv_dir, "vendor"), exist_ok=True)
    shutil.copy(VENDOR_SRC, os.path.join(srv_dir, "vendor/three.module.js"))
    jobs, results = [], {}
    for name, ir_or_js, out_png in pages:
        if isinstance(ir_or_js, dict):
            js = _compile(ir_or_js)
        else:
            js = ir_or_js
            errs = C.validate_scene_js(js)
            if errs:
                results[name] = {"png": None, "ready": False,
                                 "scene_error": "static: " + "; ".join(errs),
                                 "console_errors": [], "stats": {},
                                 "elapsed_s": 0.0}
                continue
        html_path = os.path.join(srv_dir, f"{name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(PAGE_TEMPLATE.replace("__SCENE_JS__", js))
        jobs.append((name, html_path, out_png, js))
    if not jobs:
        shutil.rmtree(srv_dir, ignore_errors=True)
        return results

    from playwright.sync_api import sync_playwright
    srv, base = serve_dir(srv_dir)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-gl=angle", "--use-angle=swiftshader",
                "--disable-gpu-sandbox", "--no-sandbox"])
            try:
                page = browser.new_page(
                    viewport={"width": WIDTH, "height": HEIGHT})
                for name, html_path, out_png, _js in jobs:
                    rel = os.path.relpath(
                        html_path, srv_dir).replace(os.sep, "/")
                    console_errors = []
                    page.on("console", lambda m, _c=console_errors:
                            _c.append(m.text) if m.type == "error" else None)
                    page.on("pageerror", lambda e, _c=console_errors:
                            _c.append(str(e)))
                    t0 = time.monotonic()
                    page.goto(base + "/" + rel, wait_until="load",
                              timeout=30000)
                    try:
                        page.wait_for_function(
                            "window.__scene_ready === true", timeout=60000)
                        ready = True
                    except Exception:
                        ready = False
                    scene_error = page.evaluate(
                        "window.__scene_error || null")
                    stats = page.evaluate(
                        "window.__scene_stats || null")
                    try:
                        graph = collect_graph(page)
                    except Exception:
                        graph = {}
                    page.screenshot(path=out_png)
                    results[name] = {
                        "png": out_png, "ready": ready,
                        "scene_error": scene_error,
                        "console_errors": list(console_errors),
                        "stats": stats or {},
                        "graph": graph,
                        "elapsed_s": round(time.monotonic() - t0, 2)}
            finally:
                browser.close()
    finally:
        srv.shutdown()
        shutil.rmtree(srv_dir, ignore_errors=True)
    return results


def ssim(a_path, b_path):
    """Mean SSIM between two renders (self-consistency checks)."""
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "visual_geometry"))
    from render import ssim_score
    return ssim_score(a_path, b_path)


def collect_graph(page):
    """Live scene introspection: {ir_id: {present, bbox, visible, ...}}
    with world-space bboxes from rendered geometry. Calls the compiled
    module's own __introspectScene hook (THREE stays module-scoped)."""
    return page.evaluate("window.__introspectScene()")
