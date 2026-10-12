# tests/test_scene_templates.py
# Template library: every template builds valid IR, compiles to valid JS,
# and initializes + renders in headless Chromium with zero console errors.
# One shared browser session per module (batched render). Determinism is
# measured (same scene twice), not assumed.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _png(name):
    os.makedirs(ACTUAL, exist_ok=True)
    return os.path.join(ACTUAL, f"_scene_{name}.png")


@pytest.fixture(scope="module")
def rendered():
    import morphix_scene_templates as T
    from scene_browser import render_scenes
    pages = [(n, T.build(n), _png(n)) for n in T.TEMPLATE_NAMES]
    return render_scenes(pages), {n: T.build(n) for n in T.TEMPLATE_NAMES}


def test_all_templates_build_valid_ir():
    import morphix_scene_ir as S
    import morphix_scene_templates as T
    for name in T.TEMPLATE_NAMES:
        ir = T.build(name)
        assert S.validate_scene_ir(ir) == [], name
        assert S.summarize(ir)["nodes"] >= 2, name


def test_unknown_template_is_an_error_not_a_substitution():
    import morphix_scene_templates as T
    with pytest.raises(KeyError):
        T.build("dragon_c°.castle")


def test_all_templates_render_without_errors(rendered):
    res, _irs = rendered
    for name, r in res.items():
        assert r["ready"], (name, r["scene_error"], r["console_errors"])
        assert r["scene_error"] is None, (name, r["scene_error"])
        assert r["console_errors"] == [], (name, r["console_errors"])
        assert r["stats"].get("drawCalls", 0) >= 1, (name, r["stats"])
        assert os.path.getsize(r["png"]) > 1024, name


def test_instancing_keeps_draw_calls_flat(rendered):
    # 4000 forest trees + 1500 haze points render in a handful of calls:
    # instancing (not per-tree meshes) is what makes this possible.
    res, irs = rendered
    import morphix_scene_ir as S
    assert S.instance_total(irs["forest"]) == 4000
    assert res["forest"]["stats"]["drawCalls"] <= 6, \
        res["forest"]["stats"]
    assert S.instance_total(irs["city"]) == 64
    assert res["city"]["stats"]["drawCalls"] <= 6


def test_renders_are_deterministic():
    # Same IR rendered twice must be pixel-identical (seeded placement).
    import morphix_scene_templates as T
    from scene_browser import render_scenes, ssim
    again = render_scenes([("fog_benchmark", T.build("fog_benchmark"),
                            _png("fog_benchmark_b"))])
    assert again["fog_benchmark"]["ready"]
    score = ssim(_png("fog_benchmark"), _png("fog_benchmark_b"))
    assert score >= 0.999, score
