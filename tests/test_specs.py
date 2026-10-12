# tests/test_specs.py
# Geometric fidelity: spec matching, plan gates (errors block, warnings
# print loudly), post-print verification.
# A "car" must look like a car — action counts alone prove nothing.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _car(body=None, cabin=None, wheels=4, windows=True, doors=True,
         material="body"):
    acts = []
    b = dict(action="add_box", width=3.0, height=0.6, depth=1.6,
             position=[0, 0.3, 0], color="0xFF0000", label="car_body")
    if body:
        b.update(body)
    acts.append(b)
    c = dict(action="add_box", width=1.6, height=0.5, depth=1.4,
             position=[0, 0.85, 0], color="0x44CCFF", label="cabin_top",
             material="accent")
    if cabin:
        c.update(cabin)
    acts.append(c)
    for i, (x, z) in enumerate([(-1.0, 0.7), (1.0, 0.7),
                                (-1.0, -0.7), (1.0, -0.7)][:wheels]):
        acts.append({"action": "add_cylinder", "radiusTop": 0.3,
                     "radiusBottom": 0.3, "height": 0.4,
                     "position": [x, 0.2, z], "color": "0x222222",
                     "label": f"wheel_{i}", "material": "tire"})
    if windows:
        for side, lab in ((0.71, "window_l"), (-0.71, "window_r")):
            acts.append({"action": "add_box", "width": 0.8, "height": 0.3,
                         "depth": 0.1, "position": [0, 0.85, side],
                         "color": "0x99DDFF", "label": lab,
                         "material": "glass"})
    if doors:
        for side, lab in ((0.81, "door_l"), (-0.81, "door_r")):
            acts.append({"action": "add_box", "width": 0.9, "height": 0.5,
                         "depth": 0.1, "position": [0, 0.55, side],
                         "color": "0xFF0000", "label": lab,
                         "material": material})
    return acts


def test_spec_matching():
    from morphix_specs import spec_for_prompt
    assert spec_for_prompt("a red car")["name"] == "car"
    assert spec_for_prompt("a small house")["name"] == "house"
    assert spec_for_prompt("an abstract sculpture") is None


def test_good_car_plan_passes():
    from morphix_specs import check_plan, spec_for_prompt
    errs, warns = check_plan(spec_for_prompt("car"), _car())
    assert errs == [] and warns == [], (errs, warns)


def test_thin_strip_body_warns_not_blocks():
    # Ugly but printable: warns loudly, still compiles (seating + support
    # turn it into real layers instead of refusing the whole run).
    from morphix_specs import check_plan, spec_for_prompt
    acts = _car(body={"width": 4.0, "height": 0.2, "depth": 0.5,
                      "position": [0, 0.1, 0]})
    errs, warns = check_plan(spec_for_prompt("a car"), acts)
    assert errs == [], errs
    assert any("body" in w and "dimensions" in w for w in warns), warns


def test_missing_wheels_is_an_error():
    from morphix_specs import check_plan, spec_for_prompt
    errs, _warns = check_plan(spec_for_prompt("car"), _car(wheels=2))
    assert any("wheel" in e for e in errs)


def test_missing_cabin_warns_but_prints():
    from morphix_specs import check_plan, spec_for_prompt
    acts = [a for a in _car() if "cabin" not in a.get("label", "")]
    errs, warns = check_plan(spec_for_prompt("car"), acts)
    assert errs == [], errs
    assert any("cabin" in w for w in warns)


def test_overlapping_wheels_warn():
    from morphix_specs import check_plan, spec_for_prompt
    spec = spec_for_prompt("car")
    acts = _car(wheels=0)
    for i in range(4):
        acts.append({"action": "add_cylinder", "radiusTop": 0.3,
                     "radiusBottom": 0.3, "height": 0.4,
                     "position": [1.0, 0.2, 0.7], "color": "0x222222",
                     "label": f"wheel_{i}", "material": "tire"})
    errs, warns = check_plan(spec, acts)
    assert any("share" in w or "spread" in w for w in warns)


def test_box_wheel_and_bad_material():
    from morphix_specs import check_plan, spec_for_prompt
    spec = spec_for_prompt("car")
    acts = _car(wheels=0)
    for i, (x, z) in enumerate([(-1.0, 0.7), (1.0, 0.7),
                                (-1.0, -0.7), (1.0, -0.7)]):
        acts.append({"action": "add_box", "width": 0.5, "height": 0.5,
                     "depth": 0.5, "position": [x, 0.25, z],
                     "color": "0x222222", "label": f"wheel_{i}",
                     "material": "tire"})
    errs, warns = check_plan(spec, acts)
    assert any("wheel" in w for w in warns) and errs == []
    acts = _car(material="lava")
    errs, _warns = check_plan(spec, acts)
    assert any("material" in e for e in errs)


def test_misplaced_window_warns():
    from morphix_specs import check_plan, spec_for_prompt
    acts = _car()
    for a in acts:
        if a["label"] == "window_l":
            a["position"] = [9, 9, 9]
    errs, warns = check_plan(spec_for_prompt("car"), acts)
    assert any("window" in w for w in warns)


def test_house_plan_gate():
    from morphix_specs import check_plan, spec_for_prompt
    spec = spec_for_prompt("house")
    ok = [{"action": "add_box", "width": 3, "height": 0.4, "depth": 2,
           "position": [0, 0.2, 0], "color": "0x777777",
           "label": "foundation"},
          {"action": "add_box", "width": 2.6, "height": 1.2, "depth": 1.8,
           "position": [0, 1.0, 0], "color": "0xF5DEB3", "label": "walls"},
          {"action": "add_box", "width": 3, "height": 0.3, "depth": 2.2,
           "position": [0, 1.75, 0], "color": "0x8B0000", "label": "roof"}]
    errs, warns = check_plan(spec, ok)
    assert errs == [] and warns == [], (errs, warns)
    floating = [dict(ok[0]),
                dict(ok[1], position=[0, 5, 0], label="walls")]
    errs, warns = check_plan(spec, floating)
    assert any("wall" in w for w in warns)


def test_verify_good_and_bad_records():
    from morphix_specs import verify_component
    comp = {"component": "car_body", "planned_deposits": 4,
            "committed_deposits": 4, "layers": [0, 1],
            "expect_bbox": (3.0, 0.6, 1.6),
            "expect_center": (0, 0.3, 0)}
    recs = [{"action": "deposit_segment", "component": "car_body",
             "from": [x, 0.0, -0.8], "to": [x + 0.5, 0.0, -0.8],
             "layer": 0, "label": f"s{i}"}
            for i, x in enumerate([-1.5, -1.0, 1.0, 0.5])]
    recs += [{"action": "deposit_segment", "component": "car_body",
              "from": [-1.5, 0.2, -0.8], "to": [-1.0, 0.2, -0.8],
              "layer": 1, "label": "s4"}]
    # z-span is 0 of 1.6: footprint coverage fails
    ver = verify_component(None, comp, recs)
    assert ver["verified"] is False
    assert any(c["name"] == "bbox" and not c["ok"] for c in ver["checks"])
    full = [{"action": "deposit_segment", "component": "car_body",
             "from": [-1.5, y, -0.8], "to": [1.5, y, 0.8],
             "layer": ly, "label": f"f{ly}"}
            for ly, y in ((0, 0.0), (1, 0.2))]
    full += [{"action": "deposit_segment", "component": "car_body",
              "from": [1.5, 0.3, 0.8], "to": [-1.5, 0.3, 0.8],
              "layer": 1, "label": "f2"}]
    comp2 = dict(comp, layers=[0, 1])
    ver2 = verify_component(None, comp2, full)
    assert ver2["verified"] is True, ver2["checks"]


def test_verify_missing_layer_fails():
    from morphix_specs import verify_component
    comp = {"component": "x", "planned_deposits": 4, "committed_deposits": 2,
            "layers": [0, 1],
            "expect_bbox": (0.4, 0.2, 0.4), "expect_center": (0, 0.1, 0)}
    recs = [{"action": "deposit_segment", "component": "x",
             "from": [-0.2, 0, -0.2], "to": [0.2, 0, -0.2],
             "layer": 0, "label": "s0"},
            {"action": "deposit_segment", "component": "x",
             "from": [0.2, 0, -0.2], "to": [0.2, 0, 0.2],
             "layer": 0, "label": "s1"}]
    ver = verify_component(None, comp, recs)
    assert ver["verified"] is False
    assert any(c["name"] == "layer_coverage" and not c["ok"]
               for c in ver["checks"])


def _one_wheel_plan():
    return [
        {"action": "add_box", "width": 3.0, "height": 0.6, "depth": 1.6,
         "position": [0, 0.3, 0], "color": "0xFF0000", "label": "car_body"},
        {"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.3,
         "height": 0.4, "position": [0.9, 0.2, 0.6], "color": "0x222222",
         "label": "wheel_x", "material": "tire"}]


def test_four_valid_wheels_get_stable_ids():
    from morphix_specs import repair_car_plan
    acts = _car()
    out, repairs = repair_car_plan([dict(a) for a in acts])
    wheels = [a for a in out if a.get("action") == "add_cylinder"]
    assert sorted(a["label"] for a in wheels) == [
        "wheel_front_left", "wheel_front_right",
        "wheel_rear_left", "wheel_rear_right"]
    # geometry preserved, only IDs normalized (logged)
    got = sorted((a["position"][0], a["position"][2]) for a in wheels)
    assert got == [(-1.0, -0.7), (-1.0, 0.7), (1.0, -0.7), (1.0, 0.7)]
    assert any("stable ID" in r["reason"] for r in repairs)


def test_one_wheel_repaired_to_four():
    from morphix_specs import repair_car_plan, check_plan, spec_for_prompt
    out, repairs = repair_car_plan(_one_wheel_plan())
    wheels = [a for a in out if a.get("action") == "add_cylinder"]
    assert sorted(a["label"] for a in wheels) == [
        "wheel_front_left", "wheel_front_right",
        "wheel_rear_left", "wheel_rear_right"]
    kept = [a for a in wheels if a["label"] == "wheel_front_left"]
    assert kept and kept[0]["position"] == [0.9, 0.2, 0.6]  # original kept
    for a in wheels:
        if a["label"] != "wheel_front_left":
            assert a["position"][1] == 0.3  # synthesized rest on ground
    assert any(r["component"] == "cabin" for r in repairs)
    errs, _warns = check_plan(spec_for_prompt("car"), out)
    assert errs == [], errs


def test_three_wheels_keep_existing():
    from morphix_specs import repair_car_plan
    acts = _one_wheel_plan()
    acts += [{"action": "add_cylinder", "radiusTop": 0.3,
              "radiusBottom": 0.3, "height": 0.4,
              "position": [-0.9, 0.2, -0.6], "color": "0x222222",
              "label": "wheel_old_rear", "material": "tire"},
             {"action": "add_cylinder", "radiusTop": 0.3,
              "radiusBottom": 0.3, "height": 0.4,
              "position": [0.9, 0.2, -0.6], "color": "0x222222",
              "label": "wheel_old_front", "material": "tire"}]
    out, _repairs = repair_car_plan(acts)
    labels = [a["label"] for a in out if a.get("action") == "add_cylinder"]
    assert sorted(labels) == ["wheel_front_left", "wheel_front_right",
                              "wheel_rear_left", "wheel_rear_right"]
    # original wheel geometry preserved under stable IDs
    by_label = {a["label"]: a for a in out
                if a.get("action") == "add_cylinder"}
    assert by_label["wheel_front_left"]["position"] == [0.9, 0.2, 0.6]


def test_duplicate_wheel_positions_detected():
    from morphix_specs import repair_car_plan
    acts = _one_wheel_plan()
    acts.append({"action": "add_cylinder", "radiusTop": 0.3,
                 "radiusBottom": 0.3, "height": 0.4,
                 "position": [0.9, 0.2, 0.6], "color": "0x222222",
                 "label": "wheel_copy", "material": "tire"})
    out, repairs = repair_car_plan(acts)
    labels = [a["label"] for a in out if a.get("action") == "add_cylinder"]
    assert "wheel_copy" not in labels and len(labels) == 4
    assert any("duplicate" in r["reason"] for r in repairs)
    pts = [(a["position"][0], a["position"][2]) for a in out
           if a.get("action") == "add_cylinder"]
    import math
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert math.dist(pts[i], pts[j]) >= 0.4


def test_missing_body_dims_use_defaults():
    from morphix_specs import repair_car_plan
    acts = [{"action": "add_box", "width": 3.0,
             "position": [0, 0.3, 0], "color": "0xFF0000",
             "label": "car_body"}]
    out1, r1 = repair_car_plan([dict(a) for a in acts])
    out2, r2 = repair_car_plan([dict(a) for a in acts])
    assert out1 == out2  # deterministic
    body = next(a for a in out1 if a.get("label") == "car_body")
    assert (body["height"], body["depth"]) == (0.6, 1.6)
    assert any("height" in r["reason"] for r in r1)


def test_invalid_body_fails_explicitly():
    import pytest
    from morphix_specs import repair_car_plan
    bad = {"action": "add_box", "width": -2, "height": 0.6, "depth": 1.6,
           "position": [0, 0.3, 0], "color": "0xFF0000", "label": "car_body"}
    with pytest.raises(ValueError) as e:
        repair_car_plan([bad])
    assert "width" in str(e.value)
    with pytest.raises(ValueError) as e:
        repair_car_plan([{"action": "add_box", "width": 1, "height": 1,
                           "depth": 1, "position": [0, 0.5, 0],
                           "color": "0xFF0000", "label": "table_top"}])
    assert "no body" in str(e.value)
    cyl_body = {"action": "add_cylinder", "radiusTop": 1,
                "radiusBottom": 1, "height": 1, "position": [0, 0.5, 0],
                "color": "0xFF0000", "label": "car_body"}
    with pytest.raises(ValueError) as e:
        repair_car_plan([cyl_body])
    assert "add_box" in str(e.value)


def test_stable_ids_and_positions():
    from morphix_specs import repair_car_plan
    out, _repairs = repair_car_plan(_one_wheel_plan())
    by_label = {a["label"]: a for a in out}
    assert by_label["wheel_front_left"]["position"] == [0.9, 0.2, 0.6]
    # synthesized slots follow the template convention (hx = bw/2 - 0.5,
    # hz = bd/2 - 0.15) and rest on the ground (y = wh/2 with wh = 2*wr):
    # body is 3.0 x 0.6 x 1.6 so wr = 0.3, slots land on (+-1.0, +-0.65).
    assert by_label["wheel_front_right"]["position"] == [1.0, 0.3, -0.65]
    assert by_label["wheel_rear_left"]["position"] == [-1.0, 0.3, 0.65]
    assert by_label["wheel_rear_right"]["position"] == [-1.0, 0.3, -0.65]
    cabin = by_label["cabin"]
    assert cabin["position"][1] == 0.6 + cabin["height"] / 2  # on body top
    shield = by_label["windshield"]
    assert shield["material"] == "glass"
    assert {a["label"] for a in out if a.get("material") == "tire"} == {
        "wheel_front_left", "wheel_front_right",
        "wheel_rear_left", "wheel_rear_right"}
