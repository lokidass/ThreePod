# tests/visual_geometry/test_visual.py
# Image-based rendering regression: every fixture rendered from fixed
# viewpoints vs approved baselines with SSIM (never pixel equality).
# Needs headless chromium (mark: visual). Baselines are read-only here.
import os

import pytest

pytestmark = pytest.mark.visual

from fixtures import FIXTURES
from helpers import build_scene
from render import (VIEWS, BASELINE_DIR, ACTUAL_DIR, SSIM_THRESHOLD,
                    write_harness, serve_dir, screenshot_pages,
                    compare_to_baseline, load_manifest, project_aabb,
                    crop_ssim)
from morphix_mesh import (deposits_to_mesh, mesh_bbox, component_aabb)

CROP_THRESHOLD = 0.95


def _frame_of(records):
    bbox = mesh_bbox(deposits_to_mesh(records))
    (lo, hi) = bbox
    pad = 0.3
    return ([lo[0] - pad, lo[1] - pad, lo[2] - pad],
            [hi[0] + pad, hi[1] + pad, hi[2] + pad])


@pytest.fixture(scope="module")
def rendered():
    """Render every fixture x view ONCE in a single browser session.
    Returns {name: {"frame": bbox, "components": {label: aabb}}}."""
    os.makedirs(ACTUAL_DIR, exist_ok=True)
    srv, base = serve_dir(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.dirname(os.path.abspath(__file__))
    info = {}
    jobs = []
    try:
        for name in sorted(FIXTURES):
            scene_js, records, layers, plan, script = build_scene(name)
            mesh = deposits_to_mesh(records)
            frame = _frame_of(records)
            comps = {}
            for a in plan:
                bb = component_aabb(mesh, a["label"])
                assert bb is not None, (name, a["label"])
                comps[a["label"]] = bb
            info[name] = {"frame": frame, "components": comps}
            for view in VIEWS:
                html = os.path.join(ACTUAL_DIR, f"{name}_{view}.html")
                png = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
                write_harness(html, scene_js, view, frame,
                              "/vendor/three.module.js")
                rel = os.path.relpath(html, here).replace(os.sep, "/")
                jobs.append((base + "/" + rel, png))
        screenshot_pages(jobs)
    finally:
        srv.shutdown()
    return info


@pytest.mark.visual
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_views_match_baselines(rendered, name):
    manifest = load_manifest()
    assert manifest.get(name), f"no manifest entry for {name}"
    frame = rendered[name]["frame"]
    for view in VIEWS:
        actual = os.path.join(ACTUAL_DIR, f"{name}_{view}.png")
        ok, info = compare_to_baseline(name, view, actual)
        assert ok, (f"{name}/{view}: {info.get('error')} "
                    f"(diff: {info.get('diff')})")
        # per-component crops localize the verdict to the failing part:
        # a missing wheel must fail its own region, not hide in the average
        for label, bb in rendered[name]["components"].items():
            rect = project_aabb(bb, view, frame)
            if rect is None:
                continue
            base = os.path.join(BASELINE_DIR, f"{name}_{view}.png")
            score, err = crop_ssim(base, actual, rect)
            assert not err, (name, view, label, err)
            assert score >= CROP_THRESHOLD, (
                f"{name}/{view}/{label}: crop SSIM {score:.4f} < "
                f"{CROP_THRESHOLD}")


def test_baselines_cover_all_fixtures_and_views():
    from render import BASELINE_DIR
    manifest = load_manifest()
    for name in sorted(FIXTURES):
        assert name in manifest, f"baseline manifest missing {name}"
        for view in VIEWS:
            path = os.path.join(BASELINE_DIR, f"{name}_{view}.png")
            assert os.path.isfile(path), f"missing baseline {name}/{view}"
            assert os.path.getsize(path) > 1024, f"empty baseline {path}"


@pytest.mark.visual
def test_missing_wheels_fail_their_own_crop(rendered):
    # A car without rear wheels is schema-valid and renders fine everywhere
    # except where the wheels should be: the per-component crop gate must
    # catch what the full-frame average (0.99+) hides.
    from fixtures import get_fixture
    from morphix_path import (compile_plan, merge_layers, optimize_toolpath,
                              PathState, validate_geometry, render_deposit)
    from morphix_mesh import deposits_to_mesh, mesh_bbox, component_aabb
    fx = get_fixture("car")
    plan = [a for a in fx["plan"] if a["label"] not in (
        "wheel_rear_left", "wheel_rear_right")]
    compiled, order, errs, keep, _ = compile_plan(plan, {})
    assert errs == {}
    script, layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    opt, _ = optimize_toolpath(script)
    st = PathState(total_layers=layers)
    finals = []
    for a in opt:
        r = st.apply(a)
        if r.get("action") in ("deposit_segment", "deposit_point"):
            finals.append(render_deposit(r))
    from render import (BASELINE_DIR, write_harness, serve_dir,
                        screenshot_pages, ssim_score)
    here = os.path.dirname(os.path.abspath(__file__))
    from helpers import build_scene
    _, recs, _, _, _ = build_scene("car")
    mesh = deposits_to_mesh(recs)
    (lo, hi) = mesh_bbox(mesh)
    pad = 0.3
    frame = ([lo[0] - pad, lo[1] - pad, lo[2] - pad],
             [hi[0] + pad, hi[1] + pad, hi[2] + pad])
    srv, base = serve_dir(here)
    try:
        html = os.path.join(ACTUAL_DIR, "neg_nowheels.html")
        png = os.path.join(ACTUAL_DIR, "neg_nowheels_side.png")
        write_harness(html, "\n\n".join(finals), "side", frame,
                      "/vendor/three.module.js")
        rel = os.path.relpath(html, here).replace(os.sep, "/")
        screenshot_pages([(base + "/" + rel, png)])
    finally:
        srv.shutdown()
    bb = component_aabb(mesh, "wheel_rear_right")
    rect = project_aabb(bb, "side", frame)
    assert rect is not None
    base_png = os.path.join(BASELINE_DIR, "car_side.png")
    score, err = crop_ssim(base_png, png, rect)
    assert not err
    assert score < CROP_THRESHOLD, (
        f"gate blind: missing-wheel crop SSIM {score:.4f} >= {CROP_THRESHOLD}")
    full_score = ssim_score(base_png, png)
    assert full_score > score, "crop must be more sensitive than full frame"
