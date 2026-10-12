# tests/test_scene_compile.py
# Scene IR -> JS compiler: deterministic emission, backend gating, API
# validation, required scene markers. No browser needed here.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _basic_ir():
    import morphix_scene_ir as S
    ir = S.new_scene("probe", seed=3)
    ir["nodes"].append(S.node(
        "hero", kind="mesh",
        geometry=S.geometry("BoxGeometry", {"w": 1, "h": 1, "d": 1}),
        material=S.material("standard", color="0xFFAA33")))
    return ir


def test_compile_is_deterministic():
    import morphix_scene_compile as C
    assert C.compile_scene_ir(_basic_ir()) == C.compile_scene_ir(_basic_ir())


def test_compile_rejects_invalid_ir_and_backends():
    import morphix_scene_compile as C
    with pytest.raises(ValueError):
        C.compile_scene_ir({"kind": "scene"})
    ir = _basic_ir()
    ir["meta"]["backend"] = "webgpu"
    with pytest.raises(ValueError) as ei:
        C.compile_scene_ir(ir)
    assert "webgl" in str(ei.value).lower()


def test_compile_rejects_unknown_constructors():
    import morphix_scene_compile as C
    import morphix_scene_ir as S
    ir = _basic_ir()
    ir["nodes"][0]["geometry"] = S.geometry("BogusGeometry", {})
    with pytest.raises(ValueError) as ei:
        C.compile_scene_ir(ir)
    assert "BogusGeometry" in str(ei.value)


def test_validate_scene_js_requires_more_than_parse():
    import morphix_scene_compile as C
    assert C.validate_scene_js(C.compile_scene_ir(_basic_ir())) == []
    # parseable but not a scene
    assert C.validate_scene_js("const x = 1;") != []
    # WebGPU-isms fail the WebGL gate
    assert C.validate_scene_js("import { pass } from 'three/tsl';") != []


def test_sky_helper_emitted_once_with_many_users():
    import morphix_scene_compile as C
    import morphix_scene_templates as T
    js = C.compile_scene_ir(T.build("mountains"))
    assert js.count("function __installGradient") == 1
    assert C.validate_scene_js(js) == []


def test_light_animation_targets_resolve():
    import morphix_scene_compile as C
    import morphix_scene_ir as S
    ir = _basic_ir()
    ir["animations"].append({"target": "ghost", "kind": "rotate"})
    with pytest.raises(ValueError) as ei:
        C.compile_scene_ir(ir)
    assert "ghost" in str(ei.value)


def test_scene_html_wraps_compiled_js():
    import morphix_scene_compile as C
    html = C.scene_html(C.compile_scene_ir(_basic_ir()))
    assert "importmap" in html and "__scene_ready" in html
