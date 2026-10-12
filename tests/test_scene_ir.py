# tests/test_scene_ir.py
# Scene-mode IR: schema, validation, serialization, printable/effect
# separation, budgets. Deterministic; no models, no browser.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _basic_scene():
    import morphix_scene_ir as S
    ir = S.new_scene("probe", seed=7)
    ir["nodes"].append(S.node(
        "hero", kind="mesh",
        geometry=S.geometry("BoxGeometry", {"w": 1, "h": 1, "d": 1}),
        material=S.material("standard", color="0xFFAA33"),
        printable=True))
    ir["lights"].append({"id": "sun", "kind": "directional",
                         "color": "0xFFFFFF", "intensity": 1.4,
                         "position": [6, 10, 7]})
    return ir


def test_valid_scene_passes():
    import morphix_scene_ir as S
    assert S.validate_scene_ir(_basic_scene()) == []


def test_serialization_round_trip_is_stable():
    import morphix_scene_ir as S
    ir = _basic_scene()
    back = S.ir_from_json(S.ir_to_json(ir))
    assert back == ir
    assert S.ir_to_json(back) == S.ir_to_json(ir)


def test_printable_tagging_is_explicit():
    import morphix_scene_ir as S
    ir = _basic_scene()
    ir["nodes"].append(S.node(
        "mist", kind="points",
        geometry=S.procedural("particles", {"count": 100}),
        material=S.material("points", color="0xFFFFFF"),
        count=100, printable=False))
    assert S.validate_scene_ir(ir) == []
    assert S.printable_nodes(ir) == ["hero"]   # effects never leak in
    assert S.instance_total(ir) == 100


def test_rejects_unknown_ids_cycles_and_children():
    import morphix_scene_ir as S
    ir = _basic_scene()
    ir["nodes"][0]["children"] = ["ghost"]
    assert any("ghost" in e for e in S.validate_scene_ir(ir))
    ir["nodes"][0]["children"] = ["hero"]
    assert any("cycle" in e for e in S.validate_scene_ir(ir))


def test_rejects_bad_backend_and_mode():
    import morphix_scene_ir as S
    ir = _basic_scene()
    ir["meta"]["backend"] = "webgpu"           # r152 has no WebGPU
    assert any("WebGPU" in e for e in S.validate_scene_ir(ir))
    ir["meta"]["backend"] = "webgl"
    ir["meta"]["mode"] = "printer"             # wrong IR family
    assert any("printer" in e for e in S.validate_scene_ir(ir))


def test_rejects_budget_overruns_and_generators_marked_printable():
    import morphix_scene_ir as S
    ir = _basic_scene()
    ir["nodes"].append(S.node(
        "crowd", kind="instanced",
        geometry=S.geometry("ConeGeometry", {"r": 0.5, "h": 2}),
        material=S.material("standard"), count=999999999,
        printable=False))
    assert any("budgets" in e for e in S.validate_scene_ir(ir))
    ir["nodes"][-1]["count"] = 10
    ir["nodes"][-1]["geometry"] = S.procedural("forest", {"trees": 10})
    ir["nodes"][-1]["printable"] = True        # generators are effects
    assert any("never printable" in e for e in S.validate_scene_ir(ir))


def test_summarize_reports_counts():
    import morphix_scene_ir as S
    s = S.summarize(_basic_scene())
    assert s["nodes"] == 1 and s["lights"] == 1
    assert s["printable"] == ["hero"]
    assert s["backend"] == "webgl"
