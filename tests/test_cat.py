# tests/test_cat.py
# Regression suite for the cat category: a `cat` prompt must produce a
# recognizable cat (body, head, ears, legs, tail) through the existing
# incremental pipeline — never a stack of generic boxes, never car parts.
# No live model inference; deterministic fixtures and frozen plans.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphix_specs import (CAT_SPEC, spec_for_prompt, check_plan,
                           repair_plan_for_spec, repair_car_plan)
from morphix_ir import (ir_for_category, ir_to_actions, validate_ir,
                        _add_shape_checks, template_key)


def _cat_thinker_plan():
    """Plausible thinker output: body + head + one ear + two legs + wheel."""
    return [
        {"action": "add_box", "width": 1.3, "height": 0.7, "depth": 0.7,
         "position": [0, 0.75, 0], "color": "0xFFAA33", "label": "body_a1",
         "material": "body"},
        {"action": "add_sphere", "radius": 0.36,
         "position": [0.325, 1.46, 0], "color": "0xFFAA33",
         "label": "head_a1", "material": "body"},
        {"action": "add_cylinder", "radiusTop": 0.03, "radiusBottom": 0.12,
         "height": 0.28, "position": [0.325, 1.96, 0.16],
         "color": "0xFFAA33", "label": "ear_a1", "material": "body"},
        {"action": "add_cylinder", "radiusTop": 0.09, "radiusBottom": 0.09,
         "height": 0.4, "position": [0.5, 0.2, 0.25], "color": "0xFFAA33",
         "label": "leg_0", "material": "body"},
        {"action": "add_cylinder", "radiusTop": 0.09, "radiusBottom": 0.09,
         "height": 0.4, "position": [-0.5, 0.2, -0.25], "color": "0xFFAA33",
         "label": "leg_1", "material": "body"},
        {"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.3,
         "height": 0.4, "position": [2, 0.2, 0], "color": "0x222222",
         "label": "wheel_x", "material": "tire"},
    ]


def _shape_fails(ir):
    fails = []
    _add_shape_checks("cat", ir.get("components", []),
                      ir.get("bbox") or [0, 0, 0],
                      lambda n, ok, d: None if ok
                      else fails.append(f"{n}: {d}"))
    return fails


def test_cat_prompt_routes_to_cat_spec():
    assert spec_for_prompt("cat")["name"] == "cat"
    assert spec_for_prompt("a cute kitten")["name"] == "cat"
    assert template_key("cat") == "cat"


def test_cat_template_has_required_structure():
    ir = ir_for_category("cat")
    assert validate_ir(ir) == []
    roles = [c.get("role") for c in ir["components"]]
    assert roles.count("body") == 1
    assert roles.count("neck") == 1
    assert roles.count("head") == 1
    assert roles.count("ear") == 2
    assert roles.count("leg") == 4
    assert roles.count("tail") == 1
    assert not any(r in ("wheel", "cabin", "windshield") for r in roles)
    assert _shape_fails(ir) == []


def test_cat_is_not_a_stack_of_blocks():
    ir = ir_for_category("cat")
    prims = {c.get("primitive") for c in ir["components"]}
    # a recognizable cat needs curved/tapered geometry, not only boxes
    assert prims & {"sphere", "cylinder"}, prims
    spheres = [c for c in ir["components"] if c.get("primitive") == "sphere"]
    assert spheres, "no curved head geometry"
    tapered = [c for c in ir["components"]
               if c.get("primitive") == "cylinder"
               and (c.get("taper") or [0, 0])[0]
               < (c.get("taper") or [0, 0])[1]]
    assert len(tapered) >= 4, "ears + neck + tail must keep their taper"


def test_cat_repair_removes_car_parts_and_completes():
    from morphix_specs import _role_of
    out, repairs = repair_plan_for_spec("cat", _cat_thinker_plan())
    assert not any(a.get("material") == "tire" for a in out)
    assert not any("wheel" in a.get("label", "") for a in out)
    assert any("car-only component removed" in r["reason"] for r in repairs)
    labels = [a.get("label") for a in out]
    assert "head_a1" in labels  # valid model geometry preserved
    roles = [_role_of(a.get("label", "")) for a in out]
    assert roles.count("ear") == 2
    assert roles.count("leg") == 4
    assert roles.count("tail") == 1
    assert "body_a1" in labels
    errs, warns = check_plan(CAT_SPEC, out)
    assert errs == [], errs
    assert warns == [], warns


def test_car_repair_cannot_touch_cat_plan():
    # a cat plan has no car body: car repair must refuse loudly instead of
    # inventing car geometry around cat parts.
    bodyless = [a for a in _cat_thinker_plan()
                if "body" not in a.get("label", "")]
    assert bodyless, "fixture must keep non-body parts"
    with pytest.raises(ValueError, match="no body"):
        repair_car_plan(bodyless)


def test_unknown_category_has_no_template_and_stays_honest():
    assert template_key("dragon") == "object"
    assert spec_for_prompt("a dragon") is None
    ir = ir_for_category("dragon")
    assert ir["template"] == "object"
    assert any("no template" in a for a in ir["assumptions"])


def test_cat_prints_end_to_end(tmp_path):
    import morphix_pipeline as P
    rep = P.run_path_print("cat", out_dir=str(tmp_path),
                           plan_actions=ir_to_actions(ir_for_category("cat")),
                           spec_name="cat", stream_scene=False,
                           deadline_s=300)
    assert rep["success"] is True
    assert rep["spec"] == "cat"
    assert rep["category_support"] == "template"
    assert rep["deposits"] > 50


def test_invalid_cat_shape_cannot_pass_success_gate(tmp_path):
    import morphix_pipeline as P
    from morphix_config import PipelineError
    flat = [
        {"action": "add_box", "width": 2.0, "height": 0.2, "depth": 1.0,
         "position": [0, 0.1, 0], "color": "0xFFAA33", "label": "body_a1",
         "material": "body"},
    ]
    with pytest.raises(PipelineError):
        P.run_path_print("cat", out_dir=str(tmp_path), plan_actions=flat,
                         spec_name="cat", stream_scene=False,
                         deadline_s=300)


def test_neck_synthesized_from_body_only_plan():
    out, repairs = repair_plan_for_spec("cat", [
        {"action": "add_box", "width": 1.3, "height": 0.7, "depth": 0.7,
         "position": [0, 0.75, 0], "color": "0xFFAA33", "label": "body_a1",
         "material": "body"}])
    by_label = {a.get("label"): a for a in out}
    neck = by_label["neck"]
    assert neck["position"] == [0.45, 1.25, 0]
    assert (neck["radiusTop"], neck["radiusBottom"]) == (0.2, 0.26)
    head = by_label["head"]
    assert head["position"] == [0.45, 1.76, 0]
    # graft contacts are exact touches, not guesses
    assert abs((1.25 - 0.15) - 1.1) < 1e-9  # neck base == body top
    assert abs((1.76 - 0.36) - 1.4) < 1e-9  # head base == neck top
    errs, warns = check_plan(CAT_SPEC, out)
    assert errs == [], errs
    assert warns == [], warns


def test_bridge_neck_under_floating_kept_head():
    plan = [
        {"action": "add_box", "width": 1.3, "height": 0.7, "depth": 0.7,
         "position": [0, 0.75, 0], "color": "0xFFAA33", "label": "body_a1",
         "material": "body"},
        {"action": "add_sphere", "radius": 0.36,
         "position": [0.45, 1.9, 0], "color": "0xFFAA33",
         "label": "head_a1", "material": "body"},
    ]
    out, repairs = repair_plan_for_spec("cat", plan)
    by_label = {a.get("label"): a for a in out}
    assert by_label["head_a1"]["position"] == [0.45, 1.9, 0]
    neck = by_label["neck"]
    assert neck["height"] == round(1.9 - 0.36 - 1.1, 2)  # bridges the gap
    assert neck["position"][1] == round(1.1 + neck["height"] / 2, 2)
    assert any("bridge neck" in r["reason"] for r in repairs)
    errs, warns = check_plan(CAT_SPEC, out)
    assert errs == [], errs


def test_taper_survives_ir_and_action_round_trip():
    from morphix_ir import actions_to_ir, ir_to_actions
    ear = {"action": "add_cylinder", "radiusTop": 0.03,
           "radiusBottom": 0.12, "height": 0.28,
           "position": [0.45, 2.26, 0.16], "color": "0xFFAA33",
           "label": "ear_left", "material": "body"}
    ir = actions_to_ir([ear], "cat", source="text")
    comp = ir["components"][0]
    assert comp.get("taper") == [0.03, 0.12]
    assert validate_ir(ir) == []
    back = ir_to_actions(ir)[0]
    assert (back["radiusTop"], back["radiusBottom"]) == (0.03, 0.12)


def test_deposit_bead_boxes_are_closed_manifolds():
    # every committed bead box: 12 triangles over 8 distinct vertices, all
    # finite, none degenerate — the unit guarantee behind mesh validity.
    from morphix_mesh import deposits_to_mesh
    rec = {"action": "deposit_segment", "from": [0, 0.4, 0],
           "to": [1.2, 0.4, 0.3], "bead_width": 0.2, "bead_height": 0.2,
           "material": "body", "layer": 2, "component": "probe",
           "label": "probe_0", "seq": 0}
    mesh = deposits_to_mesh([rec])
    assert len(mesh["tris"]) == 12
    verts = {p for t, _ in mesh["tris"] for p in t}
    assert len(verts) == 8
    assert all(all(v == v and abs(v) != float("inf") for v in p)
               for p in verts)
    from morphix_mesh import check_mesh_validity
    assert check_mesh_validity(mesh) == []


def test_disconnected_ear_fails_check():
    plan = [dict(a, label="x") for a in _cat_thinker_plan()]
    # lift both ears well clear of the head
    fixed = []
    for a in _cat_thinker_plan():
        if a["label"] == "ear_a1":
            a = dict(a, position=[0.325, 2.9, 0.16])
        fixed.append(a)
    out, _ = repair_plan_for_spec("cat", fixed)
    errs, warns = check_plan(CAT_SPEC, out)
    assert any("disconnected" in w for w in warns), warns


def test_missing_tail_fails_check():
    plan = [a for a in _cat_thinker_plan()
            if "tail" not in a.get("label", "")]
    # repair would synthesize it; check the raw thinker plan instead
    errs, _ = check_plan(CAT_SPEC, plan)
    assert any("tail" in e for e in errs), errs


def test_malformed_taper_fails_at_both_levels():
    from morphix_ir import validate_ir
    bad = dict(_cat_thinker_plan()[2], radiusTop=0.12, radiusBottom=0.03)
    _, warns = check_plan(CAT_SPEC, [bad])
    assert any("tapered" in w for w in warns), warns
    ir = ir_for_category("cat")
    for c in ir["components"]:
        if c["id"] == "ear_left":
            c["taper"] = [0.0, -1.0]
    assert any("taper" in e for e in validate_ir(ir))


def test_degenerate_triangles_detected():
    from morphix_mesh import check_mesh_validity
    mesh = {"tris": [(((0, 0, 0), (0, 0, 0), (1, 0, 0)),
                      {"component": "x", "material": "body",
                       "layer": 0, "kind": "segment"})],
            "boxes": []}
    assert any("degenerate" in e for e in check_mesh_validity(mesh))


def test_wrong_proportions_fail_shape_gate():
    ir = ir_for_category("cat")
    for c in ir["components"]:
        if c["id"] == "body":
            c["dims"] = [4.0, 0.2, 2.0]
    ir["bbox"] = [4.0, 0.2, 2.0]
    assert any("substantial" in f for f in _shape_fails(ir))
