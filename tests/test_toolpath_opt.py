# tests/test_toolpath_opt.py
# Regression suite for toolpath optimization + car geometry fidelity.
# No live model inference: canonical templates, repair and the optimizer
# are deterministic.
import math
import os
import sys
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphix_ir import ir_for_category, ir_to_actions, _add_shape_checks
from morphix_path import (compile_plan, merge_layers, optimize_toolpath,
                          validate_geometry, BEAD_WIDTH, SEG_MAX_LEN)


def _car_script():
    plan = ir_to_actions(ir_for_category("car"))
    compiled, order, errs, keep, warns = compile_plan(plan, {})
    assert errs == {}, errs
    return merge_layers([compiled[lab] for lab in order if lab in compiled])


def _deposit_length(script):
    return sum(math.dist(a["from"], a["to"])
               for a in script if a.get("action") == "deposit_segment")


def test_optimizer_shrinks_car_keeps_deposited_length():
    script, layers = _car_script()
    opt, stats = optimize_toolpath(script)
    assert stats["fallback"] is False
    assert stats["after"] < stats["before"] * 0.5
    assert stats["merged_segments"] > 100
    # same bead laid down: total deposition length preserved
    assert abs(_deposit_length(opt) - _deposit_length(script)) < 1e-6
    assert validate_geometry(opt) == []


def test_optimizer_preserves_closure_layers_and_components():
    script, layers = _car_script()
    opt, stats = optimize_toolpath(script)
    # every layer still ends with its complete_layer marker, in order
    got_layers = [a["layer"] for a in opt
                  if a.get("action") == "complete_layer"]
    assert got_layers == sorted(got_layers) == list(range(layers))
    # every compiled component still deposits on each of its layers
    have = {(a.get("component"), a.get("layer")) for a in opt
            if a.get("action") in ("deposit_segment", "deposit_point")}
    for a in script:
        if a.get("action") in ("deposit_segment", "deposit_point"):
            assert (a.get("component"), a.get("layer")) in have


def test_optimizer_drops_zero_length_and_redundant_travel():
    y = 0.0
    script = [
        {"action": "move_to", "position": [0, y, 0], "layer": 0,
         "label": "c_start"},
        {"action": "deposit_segment", "from": [0, y, 0], "to": [0, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "zero"},
        {"action": "deposit_segment", "from": [0, y, 0], "to": [1, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "run"},
        {"action": "move_to", "position": [1, y, 0], "layer": 0,
         "label": "dup"},
        {"action": "deposit_segment", "from": [1, y, 0], "to": [2, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "run2"},
        {"action": "deposit_segment", "from": [2, y, 0], "to": [0, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "close"},
        {"action": "complete_path", "layer": 0, "label": "c_done"},
    ]
    # NOTE: the input is deliberately defective (zero-length bead); the
    # optimizer must remove the defect, keep the contour closed, and the
    # result must validate.
    opt, stats = optimize_toolpath(script)
    assert stats["fallback"] is False
    assert stats["removed_zero"] == 1
    assert stats["removed_travel"] == 1
    kinds = [a["action"] for a in opt]
    assert kinds == ["move_to", "deposit_segment", "deposit_segment",
                     "complete_path"]
    assert opt[1]["from"] == [0, y, 0] and opt[1]["to"] == [2, y, 0]
    assert validate_geometry(opt) == []


def test_optimizer_splits_oversize_segments():
    y = 0.0
    script = [
        {"action": "move_to", "position": [0, y, 0], "layer": 0,
         "label": "c_start"},
        {"action": "deposit_segment", "from": [0, y, 0], "to": [8, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "long"},
        {"action": "deposit_segment", "from": [8, y, 0], "to": [0, y, 0],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "back"},
        {"action": "complete_path", "layer": 0, "label": "c_done"},
    ]
    assert validate_geometry(script) != []  # oversize is a defect
    opt, stats = optimize_toolpath(script)
    segs = [a for a in opt if a.get("action") == "deposit_segment"]
    assert all(math.dist(s["from"], s["to"]) <= SEG_MAX_LEN + 1e-9
               for s in segs)
    assert abs(sum(math.dist(s["from"], s["to"]) for s in segs) - 16.0) < 1e-6
    assert validate_geometry(opt) == []


def test_optimizer_falls_back_on_unfixable_defect():
    y = 0.0
    script = [
        {"action": "move_to", "position": [0, y, 0], "layer": 0,
         "label": "c_start"},
        {"action": "deposit_segment", "from": [5, y, 5], "to": [6, y, 5],
         "bead_width": 0.2, "bead_height": 0.2, "material": "body",
         "layer": 0, "label": "gap"},  # starts away from the toolhead
        {"action": "complete_path", "layer": 0, "label": "c_done"},
    ]
    opt, stats = optimize_toolpath(script)
    assert stats["fallback"] is True
    assert opt == script  # returned untouched, loudly flagged


def test_canonical_wheels_grounded_filled_and_distinct():
    from morphix_specs import repair_car_plan
    plan = [
        {"action": "add_box", "width": 3.0, "height": 0.6, "depth": 1.6,
         "position": [0, 0.3, 0], "color": "0xFF0000", "label": "car_body"},
        {"action": "add_cylinder", "radiusTop": 0.3,
         "radiusBottom": 0.3, "height": 0.4,
         "position": [0.9, 0.2, 0.6], "color": "0x222222",
         "label": "wheel_x", "material": "tire"},
    ]
    out, _ = repair_car_plan(plan)
    wheels = [a for a in out if a.get("action") == "add_cylinder"]
    assert len(wheels) == 4
    for w in wheels:
        r = w["radiusTop"]
        assert r >= 0.3, "wheel must be wide enough for a filled disc layer"
        assert abs(w["position"][1] - w["height"] / 2) < 1e-9, \
            "wheel must rest exactly on the ground"
    quads = {(w["position"][0] > 0, w["position"][2] > 0) for w in wheels}
    assert len(quads) == 4


def test_template_glass_sits_proud_of_cabin_faces():
    ir = ir_for_category("car")
    by_id = {c["id"]: c for c in ir["components"]}
    cabin = by_id["cabin"]
    cx, _, cz = cabin["position"]
    cw, _, cd = cabin["dims"]
    shield = by_id["windshield"]
    assert shield["position"][0] >= cx + cw / 2 - 1e-9, \
        "windshield must reach the cabin front face (inset glass is invisible)"
    for side in ("side_window_left", "side_window_right"):
        w = by_id[side]
        assert abs(w["position"][2]) >= cz + cd / 2 - 1e-9, \
            f"{side} must reach a cabin side face"


def test_shape_gate_accepts_canonical_rejects_broken_car():
    from copy import deepcopy
    ir = ir_for_category("car")
    fails = []
    _add_shape_checks("car", ir["components"], ir["bbox"],
                      lambda n, ok, d: None if ok
                      else fails.append(f"{n}: {d}"))
    assert fails == [], fails
    bad = deepcopy(ir)
    by_id = {c["id"]: c for c in bad["components"]}
    by_id["car_body"]["dims"] = [4.0, 0.2, 2.0]  # flat slab
    by_id["wheel_rear_right"]["position"] = [1.3, 0.25, 0.75]  # dup corner
    by_id["cabin"]["position"][1] += 0.5  # floating
    fails2 = []
    _add_shape_checks("car", bad["components"], bad["bbox"],
                      lambda n, ok, d: None if ok
                      else fails2.append(f"{n}: {d}"))
    names = {f.split(":")[0] for f in fails2}
    assert {"car_body_substantial", "car_wheels_four_corners",
            "car_cabin_seated"} <= names


def test_sphere_keeps_curved_profile_per_layer():
    from morphix_path import _radius_at
    R, c = 0.6, 1.6
    assert _radius_at({"r": R}, "add_sphere", c, c - R, c + R) == R
    assert _radius_at({"r": R}, "add_sphere", c + R, c - R, c + R) == 0.0
    mid = _radius_at({"r": R}, "add_sphere", c + 0.3, c - R, c + R)
    assert abs(mid - (R ** 2 - 0.09) ** 0.5) < 1e-9
    # taper is smooth and symmetric, widest at the equator
    radii = [_radius_at({"r": R}, "add_sphere", c + dy, c - R, c + R)
             for dy in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    assert radii[2] == R and radii[0] == radii[4] and radii[0] < R
    assert radii[0] < radii[1] < radii[2] > radii[3] > radii[4]
    assert radii[1] == radii[3]
