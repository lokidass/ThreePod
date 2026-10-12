# tests/test_scene_plan.py
# Scene planner: strict plan parsing, deterministic IR composition from
# tested templates, graceful failure. Thinker access is mocked.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _good_raw():
    return ('{"objective": "misty night forest", "style": "cinematic", '
            '"backend": "webgl", "templates": ["forest", "sky_fog"], '
            '"seed": 5, "animations": true}')


def test_parse_accepts_valid_plan():
    import morphix_scene_plan as P
    plan, errors = P.parse_plan_response(_good_raw())
    assert errors == []
    assert plan["templates"] == ["forest", "sky_fog"]
    assert plan["backend"] == "webgl"


def test_parse_rejects_garbage():
    import morphix_scene_plan as P
    assert P.parse_plan_response("just words")[0] is None
    plan, errors = P.parse_plan_response(
        '{"objective": "x", "style": "y", "backend": "webgpu", '
        '"templates": ["dragon"], "seed": "NaN"}')
    assert plan is None
    assert any("webgl" in e for e in errors)
    assert any("dragon" in e for e in errors)


def test_plan_to_ir_composes_systems():
    import morphix_scene_ir as S
    import morphix_scene_plan as P
    plan, _ = P.parse_plan_response(_good_raw())
    ir = P.plan_to_scene_ir(plan)
    assert S.validate_scene_ir(ir) == []
    kinds = [n["kind"] for n in ir["nodes"]]
    assert "instanced" in kinds and "points" in kinds  # forest + haze
    assert ir["meta"]["seed"] == 5
    # base (forest) owns the world; the sky system only added nodes
    assert ir["camera"]["position"] == [14, 7, 18]
    assert S.instance_total(ir) == 4000 + 1500


def test_plan_to_ir_is_deterministic_and_guarded():
    import morphix_scene_compile as C
    import morphix_scene_plan as P
    plan, _ = P.parse_plan_response(_good_raw())
    a = C.compile_scene_ir(P.plan_to_scene_ir(plan))
    b = C.compile_scene_ir(P.plan_to_scene_ir(plan))
    assert a == b
    with pytest.raises(ValueError):
        P.plan_to_scene_ir({"templates": ["nope"], "seed": 1})
    with pytest.raises(KeyError):
        __import__("morphix_scene_templates").build("nope")


def test_plan_scene_uses_thinker_then_validates(monkeypatch):
    import morphix_config
    import morphix_scene_plan as P
    # plan_scene imports ollama_generate from morphix_config at call time.
    monkeypatch.setattr(morphix_config, "ollama_generate",
                        lambda *a, **k: _good_raw())
    plan = P.plan_scene("misty forest", "fake-model")
    assert plan["templates"][0] == "forest"


def test_plan_scene_retries_then_raises(monkeypatch):
    import morphix_config
    import morphix_scene_plan as P
    from morphix_config import PipelineError
    monkeypatch.setattr(morphix_config, "ollama_generate",
                        lambda *a, **k: "just words, no json")
    with pytest.raises(PipelineError):
        P.plan_scene("misty forest", "fake-model", retries=1)
