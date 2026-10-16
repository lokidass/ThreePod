# tests/test_scene_runtime.py
# Priority 1: one authoritative Three.js runtime. The served scene.html
# (what the UI iframe actually loads) must declare THREE exactly once —
# a duplicated `import * as THREE` is a fatal module SyntaxError that
# kills the whole viewer. Consecutive generations must all work.
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _png(name):
    os.makedirs(ACTUAL, exist_ok=True)
    return os.path.join(ACTUAL, f"_rt_{name}.png")


def test_served_scene_html_declares_three_exactly_once():
    import morphix_scene_compile as C
    import morphix_scene_templates as T
    html = C.scene_html(C.compile_scene_ir(T.build("fog_benchmark")))
    imports = re.findall(r"import\s+\*\s+as\s+THREE\b", html)
    assert len(imports) == 1, f"THREE imported {len(imports)}x in scene.html"
    assert html.count("const THREE") == 0
    assert "var THREE" not in html


def test_served_scene_page_renders_without_syntax_error():
    """End-to-end through the REAL served wrapper (not the test-only page
    template): write scene.html, serve it, render it."""
    import morphix_scene_compile as C
    import morphix_scene_templates as T
    from scene_browser import render_scenes
    import re as _re

    # render through scene_html by serving its exact bytes
    html = C.scene_html(C.compile_scene_ir(T.build("studio")))
    assert len(_re.findall(r"import\s+\*\s+as\s+THREE\b", html)) == 1
    with open(_png("studio_page.html"), "w", encoding="utf-8") as f:
        f.write(html)
    # serve the directory containing the page + vendor, screenshot it
    import shutil
    import tempfile
    from render import serve_dir
    srv_dir = tempfile.mkdtemp(prefix="morphix_rt_")
    os.makedirs(os.path.join(srv_dir, "vendor"), exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "visual_geometry", "vendor", "three.module.js"),
                os.path.join(srv_dir, "vendor", "three.module.js"))
    # rewrite the unpkg import to the local vendor for the offline harness
    html_local = html.replace("https://unpkg.com/three@0.152.2/build/three.module.js",
                              "/vendor/three.module.js")
    with open(os.path.join(srv_dir, "page.html"), "w",
              encoding="utf-8") as f:
        f.write(html_local)
    from playwright.sync_api import sync_playwright
    srv, base = serve_dir(srv_dir)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-gl=angle", "--use-angle=swiftshader",
                "--disable-gpu-sandbox", "--no-sandbox"])
            try:
                page = browser.new_page(viewport={"width": 640,
                                                  "height": 480})
                console_errors = []
                page.on("console", lambda m:
                        console_errors.append(m.text)
                        if m.type == "error" else None)
                page.on("pageerror", lambda e:
                        console_errors.append(str(e)))
                page.goto(base + "/page.html", wait_until="load",
                          timeout=30000)
                page.wait_for_function("window.__scene_ready === true",
                                       timeout=60000)
                assert console_errors == [], console_errors
                page.screenshot(path=_png("studio_served"))
            finally:
                browser.close()
    finally:
        srv.shutdown()
        shutil.rmtree(srv_dir, ignore_errors=True)


def test_consecutive_generations_do_not_collide():
    """Three different scenes back-to-back in one browser session: no
    duplicate declarations, renderers, or leaked loops — each page is an
    independent realm that reaches ready with zero errors."""
    import morphix_scene_templates as T
    from scene_browser import render_scenes
    pages = [(n, T.build(n), _png(f"regen_{n}"))
             for n in ("studio", "city", "particles")]
    res = render_scenes(pages)
    assert len(res) == 3
    for name, r in res.items():
        assert r["ready"], (name, r["scene_error"])
        assert r["scene_error"] is None, (name, r["scene_error"])
        assert r["console_errors"] == [], (name, r["console_errors"])


def test_printer_blocks_cannot_smuggle_imports():
    from morphix_guard import validate_js
    box = {"action": "add_box", "width": 1, "height": 1, "depth": 1,
           "position": [0, 0.5, 0], "color": "0xFFAA33", "label": "probe"}
    sneaky = ("import * as THREE from 'three';\n"
              "const geo_probe = new THREE.BoxGeometry(1,1,1);\n"
              "scene.add(new THREE.Mesh(geo_probe));")
    assert any("import" in e for e in validate_js(box, sneaky))
