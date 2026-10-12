# tests/test_chair.py
# Regression suite for object-category isolation, driven by the observed
# failure where the prompt `chair` printed car parts (car_body, cabin_top,
# wheel_0) copied from the shared print-prompt example.
# No live model inference: deterministic fixtures, repair, and frozen plans.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphix_specs import (CHAIR_SPEC, spec_for_prompt, check_plan,
                           repair_plan_for_spec, repair_car_plan)
from morphix_ir import (ir_for_category, ir_to_actions, validate_ir,
                        _add_shape_checks)


def _chair_thinker_plan():
    """A plausible thinker chair plan with one leaked car wheel."""
    return [
        {"action": "add_box", "width": 1.1, "height": 0.12, "depth": 1.1,
         "position": [0, 0.51, 0], "color": "0x9C6B3F", "label": "seat_a1",
         "material": "wood"},
        {"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.3,
         "height": 0.4, "position": [2, 0.2, 0], "color": "0x222222",
         "label": "wheel_x", "material": "tire"},
    ]


def _shape_fails(ir):
    fails = []
    _add_shape_checks("chair", ir.get("components", []),
                      ir.get("bbox") or [0, 0, 0],
                      lambda n, ok, d: None if ok
                      else fails.append(f"{n}: {d}"))
    return fails


def test_chair_plan_has_seat_backrest_and_four_legs():
    out, _ = repair_plan_for_spec("chair", _chair_thinker_plan())
    roles = {}
    for a in out:
        lab = a.get("label", "")
        for role in ("seat", "backrest", "leg"):
            if role in lab:
                roles.setdefault(role, []).append(a)
    assert len(roles.get("seat", [])) == 1
    assert len(roles.get("backrest", [])) == 1
    assert len(roles.get("leg", [])) == 4
    errs, warns = check_plan(CHAIR_SPEC, out)
    assert errs == [], errs
    assert warns == [], warns


def test_car_plan_keeps_body_cabin_windows_wheels():
    ir = ir_for_category("car")
    assert validate_ir(ir) == []
    roles = [c.get("role") for c in ir["components"]]
    assert "body" in roles and "cabin" in roles
    assert roles.count("wheel") == 4
    assert "windshield" in roles and roles.count("window") == 2


def test_table_plan_keeps_tabletop_and_legs():
    from morphix_specs import TABLE_SPEC
    plan = [
        {"action": "add_box", "width": 2.0, "height": 0.2, "depth": 1.2,
         "position": [0, 1.1, 0], "color": "0x9C6B3F", "label": "tabletop",
         "material": "wood"},
    ] + [{"action": "add_box", "width": 0.2, "height": 1.0, "depth": 0.2,
           "position": [x, 0.5, z], "color": "0x9C6B3F",
           "label": f"leg_{i}", "material": "wood"}
          for i, (x, z) in enumerate([(0.9, 0.5), (-0.9, 0.5),
                                      (0.9, -0.5), (-0.9, -0.5)])]
    errs, warns = check_plan(TABLE_SPEC, plan)
    assert errs == [], errs


def test_car_repair_cannot_touch_chair_plan():
    # car repair on a chair plan raises (no body) instead of inserting
    # wheels/cabins into it.
    with pytest.raises(ValueError, match="no body"):
        repair_car_plan(_chair_thinker_plan())
    # and the dispatcher routes chair plans to chair repair: no tire, no
    # wheel, no cabin survives it.
    out, repairs = repair_plan_for_spec("chair", _chair_thinker_plan())
    assert not any(a.get("material") == "tire" for a in out)
    assert not any("wheel" in a.get("label", "")
                   or "cabin" in a.get("label", "") for a in out)
    assert any("car-only component removed" in r["reason"] for r in repairs)


def test_runs_do_not_share_component_specifications(tmp_path):
    import morphix_pipeline as P
    box = {"action": "add_box", "width": 1.0, "height": 0.4, "depth": 1.0,
           "position": [0, 0.2, 0], "color": "0xFF0000", "label": "box_a1"}
    r1 = P.run_path_print("box", out_dir=str(tmp_path) + "_1",
                          plan_actions=[dict(box)], spec_name=None,
                          stream_scene=False)
    chair = ir_to_actions(ir_for_category("chair"))
    r2 = P.run_path_print("chair", out_dir=str(tmp_path) + "_2",
                          plan_actions=chair, spec_name="chair",
                          stream_scene=False)
    labels1 = {c.get("component", "") for c in r1["components"]}
    labels2 = {c.get("component", "") for c in r2["components"]}
    assert labels1.isdisjoint(labels2)
    assert r2["success"] is True


def test_invalid_chair_plan_fails_before_printing(tmp_path):
    import morphix_pipeline as P
    from morphix_config import PipelineError
    junk = [
        {"action": "add_box", "width": 1.0, "height": 0.5, "depth": 1.0,
         "position": [0, 0.25, 0], "color": "0xFF0000", "label": "box_a1"},
        {"action": "add_box", "width": 1.0, "height": 0.5, "depth": 1.0,
         "position": [0, 0.25, 3.0], "color": "0xFF0000", "label": "box_a2"},
        {"action": "add_box", "width": 1.0, "height": 0.5, "depth": 1.0,
         "position": [0, 0.25, -3.0], "color": "0xFF0000", "label": "box_a3"},
    ]
    with pytest.raises(PipelineError, match="fidelity gate"):
        P.run_path_print("chair", out_dir=str(tmp_path),
                         plan_actions=junk, spec_name="chair",
                         stream_scene=False)
    # refused before compiling: no paths were ever generated
    assert not os.path.isfile(os.path.join(str(tmp_path), "scene.js"))


def test_valid_chair_slices_and_executes_segment_by_segment(tmp_path):
    import morphix_pipeline as P
    chair = ir_to_actions(ir_for_category("chair"))
    rep = P.run_path_print("chair", out_dir=str(tmp_path),
                           plan_actions=chair, spec_name="chair",
                           stream_scene=False)
    assert rep["success"] is True
    assert rep["deposits"] > 20
    layers = sorted({b.get("layer", -1) for b in
                     json.load(open(os.path.join(str(tmp_path), "path.json"),
                                    encoding="utf-8"))["records"]})
    assert layers == list(range(max(layers) + 1))


def test_final_validator_rejects_unrelated_boxes_as_chair():
    boxes = [
        {"action": "add_box", "width": 1.0, "height": 0.5, "depth": 1.0,
         "position": [i * 2.0, 0.25, 0], "color": "0xFF0000",
         "label": f"box_{i}"} for i in range(3)]
    errs, _ = check_plan(CHAIR_SPEC, boxes)
    assert any("seat" in e for e in errs)
    assert any("backrest" in e for e in errs)
    assert any("leg" in e for e in errs)


def test_rejected_chair_actions_do_not_mutate_committed_geometry(tmp_path):
    import morphix_pipeline as P
    chair = ir_to_actions(ir_for_category("chair"))

    def echo(ctx):
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    calls = {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"action": "deposit_segment",
                               "from": [9, 9, 9], "to": [9.2, 9, 9],
                               "bead_width": 0.2, "bead_height": 0.2,
                               "material": "body", "layer": 0,
                               "label": "evil_01"})
        return echo(ctx)

    rep_f = P.run_path_print("chair", out_dir=str(tmp_path) + "_f",
                             plan_actions=chair, spec_name="chair",
                             stream_scene=False, guided=True, build_one=flaky)
    rep_c = P.run_path_print("chair", out_dir=str(tmp_path) + "_c",
                             plan_actions=chair, spec_name="chair",
                             stream_scene=False, guided=True, build_one=echo)
    assert rep_f["success"] is True
    js_f = open(os.path.join(str(tmp_path) + "_f", "scene.js")).read()
    js_c = open(os.path.join(str(tmp_path) + "_c", "scene.js")).read()
    assert js_f == js_c
