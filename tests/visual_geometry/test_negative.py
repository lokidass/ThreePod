# tests/visual_geometry/test_negative.py
# Every known Morphix failure mode, as a fixture that MUST fail the
# relevant gate. If any of these passes, the gate is blind.
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from fixtures import get_fixture
from morphix_mesh import (deposits_to_mesh, check_round_fill,
                           check_box_fill, check_no_duplicate_deposits,
                           check_same_material_overlaps,
                           check_mesh_validity)
from morphix_specs import SPECS, check_plan, repair_plan_for_spec
from morphix_specs import verify_component  # noqa: used by coverage test
from morphix_path import (compile_plan, merge_layers, optimize_toolpath,
                          PathState, validate_geometry)
from morphix_validator import validate_actions
from quality import quality_gate


def _execute(plan):
    compiled, order, errs, keep, warns = compile_plan(plan, {})
    assert errs == {}, errs
    script, layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    opt, _ = optimize_toolpath(script)
    st = PathState(total_layers=layers)
    recs = []
    for a in opt:
        assert validate_actions([a]) + st.check(a) == [], a
        r = st.apply(a)
        if r.get("action") in ("deposit_segment", "deposit_point"):
            recs.append(r)
    return recs


def test_chair_from_car_parts_fails_category_gate():
    spec = next(s for s in SPECS if s["name"] == "chair")
    plan = [
        {"action": "add_box", "width": 1.1, "height": 0.12, "depth": 1.1,
         "position": [0, 0.51, 0], "color": "0x9C6B3F", "label": "seat_a1",
         "material": "wood"},
        {"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.3,
         "height": 0.4, "position": [2, 0.2, 0], "color": "0x222222",
         "label": "wheel_x", "material": "tire"},
    ]
    errs, _ = check_plan(spec, plan)
    assert any("car part" in e for e in errs)


def test_table_as_solid_block_fails():
    spec = next(s for s in SPECS if s["name"] == "table")
    plan = [{"action": "add_box", "width": 2.0, "height": 1.2, "depth": 1.2,
             "position": [0, 0.6, 0], "color": "0x9C6B3F",
             "label": "tabletop", "material": "wood"}]
    errs, _ = check_plan(spec, plan)
    assert any("leg" in e for e in errs)


def test_car_missing_wheels_fails():
    spec = next(s for s in SPECS if s["name"] == "car")
    plan = [a for a in get_fixture("car")["plan"]
            if "wheel" not in a["label"]]
    errs, _ = check_plan(spec, plan)
    assert any("wheel" in e for e in errs)


def test_duplicate_wheel_positions_rejected():
    from morphix_specs import repair_car_plan
    plan = get_fixture("car")["plan"]
    dup = [dict(a) for a in plan]
    dup.append(dict([a for a in plan if a["label"] == "wheel_front_left"][0]))
    out, repairs = repair_car_plan(dup)
    wheels = [a for a in out if a["action"] == "add_cylinder"]
    assert len(wheels) == 4
    pts = [(w["position"][0], w["position"][2]) for w in wheels]
    assert len(set(pts)) == 4
    assert any("duplicate" in r["reason"] for r in repairs)


def test_sphere_ring_fails_fill_check():
    # a wheel plan with its infill scanlines stripped: perimeter rings
    # remain, so layers that should be filled discs are empty.
    from morphix_path import compile_plan, merge_layers, optimize_toolpath
    plan = [a for a in get_fixture("car")["plan"]
            if a["label"] == "wheel_front_left"]
    compiled, order, errs, keep, _ = compile_plan(plan, {})
    assert errs == {}
    script, layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    opt, _ = optimize_toolpath(script)
    st = PathState(total_layers=layers)
    recs = []
    for a in opt:
        r = st.apply(a)
        if r.get("action") == "deposit_segment" \
                and "fill" not in a.get("label", ""):
            recs.append(r)
    assert check_round_fill(recs, "wheel_front_left",
                            0.35, 0.35, 0.0, 0.5) != []


def test_box_strips_fail_fill_check():
    fx = get_fixture("box_sphere")
    plan = [a for a in fx["plan"] if a["label"] == "box"]
    compiled, order, errs, keep, _ = compile_plan(plan, {})
    assert errs == {}
    # keep only the perimeter contour of layer 0: strips, not solid
    st = None
    from morphix_path import PathState as PS
    script, layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    st = PS(total_layers=layers)
    recs = []
    for a in script:
        r = st.apply(a)
        if r.get("action") == "deposit_segment" and "fill" not in a.get(
                "label", ""):
            recs.append(r)
    assert check_box_fill(recs, "box", 1.4, 1.4) != []


def test_missing_segments_break_layer_coverage():
    from morphix_specs import verify_component
    recs = _execute(get_fixture("cube")["plan"])
    full_layers = sorted({r["layer"] for r in recs})
    assert full_layers == list(range(5)), full_layers
    pruned = [r for r in recs if r.get("layer") != 2]
    comp = {"component": "cube", "planned_deposits": len(recs),
            "committed_deposits": len(pruned),
            "layers": full_layers, "expect_bbox": (1.0, 1.0, 1.0),
            "expect_center": (0, 0.5, 0)}
    ver = verify_component(None, comp, pruned)
    assert ver["verified"] is False


def test_open_contour_rejected():
    bad = [
        {"action": "move_to", "position": [0, 0.0, 0], "layer": 0,
         "label": "c_start"},
        {"action": "deposit_segment", "from": [0, 0.0, 0], "to": [1, 0.0, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "e1"},
        {"action": "complete_path", "layer": 0, "label": "c_done"},
    ]
    assert validate_geometry(bad) != []
    opt, stats = optimize_toolpath(bad)
    assert stats["fallback"] is True
    assert validate_geometry(opt) != []


def test_gap_between_segments_rejected():
    bad = [
        {"action": "move_to", "position": [0, 0.0, 0], "layer": 0,
         "label": "c_start"},
        {"action": "deposit_segment", "from": [0, 0.0, 0], "to": [1, 0.0, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "e1"},
        {"action": "deposit_segment", "from": [1.5, 0.0, 0],
         "to": [2.0, 0.0, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "e2"},
        {"action": "complete_path", "layer": 0, "label": "c_done"},
    ]
    assert any("continuity" in e or "toolhead" in e
               for e in validate_geometry(bad))


def test_wrong_material_rejected():
    spec = next(s for s in SPECS if s["name"] == "car")
    plan = [dict(a, material="neon") if a["label"] == "car_body" else a
            for a in get_fixture("car")["plan"]]
    errs, _ = check_plan(spec, plan)
    assert any("neon" in e for e in errs)


def test_wrong_proportions_rejected():
    spec = next(s for s in SPECS if s["name"] == "car")
    plan = [dict(a, width=4.0, height=0.2) if a["label"] == "car_body" else a
            for a in get_fixture("car")["plan"]]
    errs, warns = check_plan(spec, plan)
    assert errs == []  # dims parse; proportions are quality warnings...
    assert any("body" in w for w in warns)
    # ...which veto acceptance at the gate, never pass silently
    gate = quality_gate({"spec": [], "category": [],
                         "mesh": [], "continuity": [],
                         "completeness": [],
                         "fidelity": warns, "visual": []})
    assert gate["success"] is False
    assert gate["classification"] == "specification"


def test_incomplete_commit_fails_success():
    gate = quality_gate({"spec": [], "category": [], "mesh": [],
                         "continuity": [],
                         "completeness": ["car_body missing 2/3 layers"],
                         "fidelity": [], "visual": []})
    assert gate["success"] is False
    assert gate["stage"] == "completeness"
    assert gate["classification"] == "executor"


def test_valid_actions_but_wrong_object_fails_gate():
    # syntactically perfect chair plan with car parts removed leaves only a
    # seat: every action validates, the OBJECT is still wrong.
    spec = next(s for s in SPECS if s["name"] == "chair")
    plan = [a for a in get_fixture("chair")["plan"]
            if a["label"] == "seat"]
    errs, _ = check_plan(spec, plan)
    assert errs != []
    gate = quality_gate({"spec": errs, "category": [], "mesh": [],
                         "continuity": [], "completeness": [],
                         "fidelity": [], "visual": []})
    assert gate["success"] is False
