# tests/visual_geometry/regen_baselines.py
# Explicit baseline (re)generation. NEVER runs as part of the test suite.
#
#   python tests/visual_geometry/regen_baselines.py          # report-only:
#       renders every fixture x view into .actual/, diffs against current
#       baselines, prints SSIM table. Baselines untouched.
#   python tests/visual_geometry/regen_baselines.py --approve
#       copies .actual renders (+ manifest) into baselines/. Only run this
#       after HUMAN review of the diffs. There is no automatic overwrite.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from fixtures import FIXTURES
from helpers import build_scene
from render import (VIEWS, BASELINE_DIR, ACTUAL_DIR, SSIM_THRESHOLD,
                    write_harness, serve_dir, screenshot_pages, ssim_score,
                    save_manifest)
from morphix_mesh import mesh_bbox, deposits_to_mesh


def main():
    approve = "--approve" in sys.argv
    os.makedirs(ACTUAL_DIR, exist_ok=True)
    srv, base_url = serve_dir(os.path.dirname(os.path.abspath(__file__)))
    manifest = {}
    rows = []
    try:
        for name in sorted(FIXTURES):
            scene_js, records, layers, plan, script = build_scene(name)
            mesh = deposits_to_mesh(records)
            bbox = mesh_bbox(mesh)
            (lo, hi) = bbox
            pad = 0.3
            frame = ([lo[0] - pad, lo[1] - pad, lo[2] - pad],
                     [hi[0] + pad, hi[1] + pad, hi[2] + pad])
            jobs = []
            for view in VIEWS:
                html = os.path.join(ACTUAL_DIR, f"{name}_{view}.html")
                png = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
                write_harness(html, scene_js, view, frame,
                              "/vendor/three.module.js")
                rel = os.path.relpath(
                    html, os.path.dirname(os.path.abspath(__file__)))
                jobs.append((base_url + "/" + rel.replace(os.sep, "/"),
                             png))
            screenshot_pages(jobs)
            manifest[name] = {"views": {}, "threshold": SSIM_THRESHOLD}
            for view in VIEWS:
                actual = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
                base = os.path.join(BASELINE_DIR, f"{name}_{view}.png")
                if os.path.isfile(base):
                    score = ssim_score(base, actual)
                    rows.append((name, view, score,
                                 "SAME" if score >= SSIM_THRESHOLD
                                 else "DIFF"))
                else:
                    rows.append((name, view, None, "NEW (no baseline)"))
                manifest[name]["views"][view] = os.path.basename(actual)
    finally:
        srv.shutdown()
    print(f"{'fixture':12s} {'view':14s} {'ssim':>8s}  status")
    for name, view, score, status in rows:
        shown = f"{score:.4f}" if score is not None else "—"
        print(f"{name:12s} {view:14s} {shown:>8s}  {status}")
    if approve:
        for name in sorted(FIXTURES):
            for view in VIEWS:
                src = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
                dst = os.path.join(BASELINE_DIR, f"{name}_{view}.png")
                os.makedirs(BASELINE_DIR, exist_ok=True)
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
        save_manifest(manifest)
        print(f"approved {len(rows)} renders -> baselines/")
    else:
        print("report-only: baselines untouched "
              "(re-run with --approve after review)")


if __name__ == "__main__":
    main()
