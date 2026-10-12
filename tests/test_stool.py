# tests/test_stool.py
# Stool category + chair armrest optionals + few-shot example completeness.
# No live model inference: deterministic fixtures and frozen plans.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphix_specs import (STOOL_SPEC, CHAIR_SPEC, spec_for_prompt,
                           check_plan, repair_plan_for_spec)
from morphix_ir import (ir_for_category, ir_to_actions, validate_ir,
                        _add_shape_checks)
from morphix_pipeline import print_suffix_for


def _stool_thinker_plan():
    return [
        {"action": "add_box", "width": 1.0, "height": 0.12, "depth": 1.0,
         "position": [0, 0.51, 0], "color": "0x9C6B3F", "label": "seat_a1",
         "material": "wood"},
        {"action": "add_box", "width": 0.12, "height": 0.9, "depth": 1.0,
         "position": [-0.44, 1.02, 0], "color": "0x9C6B3F",
         "label": "backrest_a1", "material": "wood"},
    ]


def test_stool_prompt_routes_to_stool_spec():
    assert spec_for_prompt("a wooden stool")["name"] == "stool"
    assert spec_for_prompt("footstool")["name"] == "stool"


def test_stool_template_is_seat_plus_four_grounded_legs():
    ir = ir_for_category("stool")
    assert validate_ir(ir) == []
    roles = [c.get("role") for c in ir["components"]]
    assert roles.count("seat") == 1 and roles.count("leg") == 4
    assert "backrest" not in roles
    fails = []
    _add_shape_checks("stool", ir["components"], ir["bbox"],
                      lambda n, ok, d: None if ok
                      else fails.append(f"{n}: {d}"))
    assert fails == [], fails


def test_stool_repair_drops_backrest_and_fills_legs():
    out, repairs = repair_plan_for_spec("stool", _stool_thinker_plan())
    assert not any("backrest" in a.get("label", "") for a in out)
    assert any("no backrest" in r["reason"] for r in repairs)
    legs = [a for a in out if "leg" in a.get("label", "")]
    assert len(legs) == 4
    errs, warns = check_plan(STOOL_SPEC, out)
    assert errs == [], errs
    assert warns == [], warns


def test_chair_accepts_valid_armrests():
    plan = [
        {"action": "add_box", "width": 1.1, "height": 0.12, "depth": 1.1,
         "position": [0, 0.51, 0], "color": "0x9C6B3F", "label": "seat_a1",
         "material": "wood"},
        {"action": "add_box", "width": 0.12, "height": 0.9, "depth": 1.0,
         "position": [-0.49, 1.02, 0], "color": "0x9C6B3F",
         "label": "backrest_a1", "material": "wood"},
        {"action": "add_box", "width": 0.1, "height": 0.4, "depth": 0.9,
         "position": [0.55, 0.77, 0], "color": "0x9C6B3F",
         "label": "armrest_left", "material": "wood"},
    ] + [{"action": "add_cylinder", "radiusTop": 0.07,
           "radiusBottom": 0.07, "height": 0.45,
           "position": [x, 0.225, z], "color": "0x9C6B3F",
           "label": f"leg_{i}", "material": "wood"}
          for i, (x, z) in enumerate([(0.45, 0.45), (0.45, -0.45),
                                      (-0.45, 0.45), (-0.45, -0.45)])]
    out, _ = repair_plan_for_spec("chair", plan)
    assert any(a.get("label") == "armrest_left" for a in out)
    errs, warns = check_plan(CHAIR_SPEC, out)
    assert errs == [], errs
    assert warns == [], warns


def test_chair_example_shows_all_four_legs():
    suffix = print_suffix_for("chair")
    for i in range(4):
        assert f'"label":"leg_{i}"' in suffix
    # no car-part JSON for the thinker to copy verbatim
    assert '"label":"car_body"' not in suffix
    assert '"label":"cabin_top"' not in suffix
    assert '"label":"wheel_0"' not in suffix
    stool = print_suffix_for("stool")
    assert '"label":"backrest' not in stool  # no backrest geometry offered
    assert "NO backrest" in stool  # ...and the ban is explicit
    neutral = print_suffix_for("table")
    assert "car_body" not in neutral and "seat_a1" not in neutral


def test_unknown_category_gets_no_repair_no_invention():
    plan = [{"action": "add_box", "width": 1.0, "height": 0.5, "depth": 1.0,
             "position": [0, 0.25, 0], "color": "0xFF0000",
             "label": "thing_a1"}]
    out, repairs = repair_plan_for_spec("object", plan)
    assert out == plan and repairs == []


def test_stool_prints_end_to_end(tmp_path):
    import morphix_pipeline as P
    stool = ir_to_actions(ir_for_category("stool"))
    rep = P.run_path_print("stool", out_dir=str(tmp_path),
                           plan_actions=stool, spec_name="stool",
                           stream_scene=False)
    assert rep["success"] is True
    assert rep["deposits"] > 10
    assert rep["spec"] == "stool"
