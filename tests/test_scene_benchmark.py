# tests/test_scene_benchmark.py
# Reference benchmarks (Phase E): the fog-background scene inspired by
# three.js webgpu_custom_fog_background, and the custom-fog landscape
# (terrain + vegetation + height fog). Validation is NOT screenshot
# comparison alone: render success, zero errors, measured fog presence,
# subject visibility, determinism, documented match/approximate/unsupported
# coverage, and performance budgets.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _png(name):
    os.makedirs(ACTUAL, exist_ok=True)
    return os.path.join(ACTUAL, f"_bench_{name}.png")


@pytest.fixture(scope="module")
def benched():
    import morphix_scene_templates as T
    from scene_browser import render_scenes
    pages = [("fog", T.build("fog_benchmark"), _png("fog")),
             ("land", T.build("fog_landscape"), _png("land"))]
    return render_scenes(pages)


def _frac_near(png, rgb, tol=0.12):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("RGB"), dtype=float) / 255.0
    d = np.abs(a - np.array(rgb)).max(axis=2)
    return float((d < tol).mean()), float(a.mean())


def test_benchmarks_render_cleanly(benched):
    for name, r in benched.items():
        assert r["ready"], (name, r["scene_error"])
        assert r["scene_error"] is None, (name, r["scene_error"])
        assert r["console_errors"] == [], (name, r["console_errors"])


def test_fog_treatment_is_visible_not_subtle(benched):
    # The reference is DEFINED by its fog: a large share of pixels must
    # sit near the declared fog color, while the hero subject stays
    # visible (not everything fogged out, not a black frame).
    fog_frac, bright = _frac_near(_png("fog"), (0x40 / 255, 0x80 / 255,
                                                 0xCC / 255))
    assert fog_frac >= 0.25, fog_frac       # fog dominates, like reference
    assert fog_frac <= 0.95, fog_frac       # subject still visible
    assert bright >= 0.2, bright
    land_frac, land_bright = _frac_near(_png("land"), (0xA8 / 255, 0xBE / 255,
                                                        0xCF / 255))
    assert land_frac >= 0.2, land_frac
    assert land_bright >= 0.2, land_bright


def test_benchmarks_use_reference_fog_range():
    # The TSL rangeFogFactor(2.7, 4) numbers are carried over exactly.
    import morphix_scene_templates as T
    fog = T.build("fog_benchmark")["environment"]["fog"]
    assert (fog["color"], fog["near"], fog["far"]) == ("0x4080cc", 2.7, 4.0)


def test_coverage_notes_are_explicit():
    import morphix_scene_templates as T
    notes = T.FOG_BENCHMARK_NOTES
    assert notes["reference"].startswith("three.js ")
    for key in ("matched", "approximated", "unsupported"):
        assert notes[key], key
    # every "unsupported" item must actually be rejected by the index —
    # the docs and the gate cannot drift apart.
    import morphix_three_knowledge as K
    for api in ("WebGPURenderer", "RenderPipeline"):
        assert api in notes["unsupported"]
        ok, _ = K.api_supported(api)
        assert not ok


def test_benchmarks_are_deterministic():
    import morphix_scene_templates as T
    from scene_browser import render_scenes, ssim
    again = render_scenes([("fog", T.build("fog_benchmark"), _png("fog_b"))])
    assert again["fog"]["ready"]
    assert ssim(_png("fog"), _png("fog_b")) >= 0.999


def test_performance_budgets_are_measured(benched):
    # Measured on SwiftShader software GL (floor, not ceiling): real GPUs
    # are orders of magnitude faster. Budgets guard regressions, and the
    # numbers are reported, not assumed.
    for name, r in benched.items():
        assert r["stats"]["drawCalls"] <= 8, (name, r["stats"])
        assert r["elapsed_s"] < 60, (name, r["elapsed_s"])
    assert benched["land"]["stats"]["triangles"] >= 20000  # real terrain
