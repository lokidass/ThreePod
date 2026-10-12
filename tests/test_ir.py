# tests/test_ir.py
# Canonical Object IR: strict schema, deterministic templates, exact primitive
# conversion, and the geometric guarantees a "car" must satisfy.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_car_template_is_deterministic_and_valid():
    import morphix_ir as IR
    from morphix_specs import check_plan, spec_for_prompt
    a = IR.ir_for_category("car")
    b = IR.ir_for_category("car")
    assert a == b                                    # deterministic
    assert IR.validate_ir(a) == []                   # strict schema + sanity
    acts = IR.ir_to_actions(a)
    errs, warns = check_plan(spec_for_prompt("car"), acts)
    assert errs == []
    assert a["bbox"][0] > a["bbox"][2]               # longer than it is wide


def test_car_has_body_cabin_windshield_windows_and_four_wheels():
    import morphix_ir as IR
    ir = IR.ir_for_category("car")
    by_id = {c["id"]: c for c in ir["components"]}
    wheels = [c for c in ir["components"] if c["role"] == "wheel"]
    assert len(wheels) == 4
    assert len({tuple(c["position"]) for c in wheels}) == 4   # 4 distinct spots
    assert all(c["primitive"] == "cylinder" for c in wheels)
    assert by_id["car_body"]["primitive"] == "box"
    # exactly one windshield and two side windows
    assert sum(c["role"] == "windshield" for c in ir["components"]) == 1
    assert sum(c["role"] == "window" for c in ir["components"]) == 2
    # explicit material assignments: body / glass / rubber / lights / trim
    mats = {c["material"] for c in ir["components"]}
    assert {"body", "accent", "glass", "tire"} <= mats


def test_cabin_is_raised_above_the_body():
    import morphix_ir as IR
    ir = IR.ir_for_category("car")
    by_id = {c["id"]: c for c in ir["components"]}
    body = by_id["car_body"]
    cabin = by_id["cabin"]
    assert body["parent"] is None
    assert cabin["parent"] == "car_body"
    body_top = body["position"][1] + body["dims"][1] / 2
    cabin_base = cabin["position"][1] - cabin["dims"][1] / 2
    assert abs(cabin_base - body_top) < 1e-6         # seated, exactly raised
    # windows live on the cabin
    for cid in ("windshield", "side_window_left", "side_window_right"):
        assert by_id[cid]["parent"] == "cabin"
        assert by_id[cid]["material"] == "glass"


def test_wheels_touch_the_ground_on_one_consistent_axis():
    import morphix_ir as IR
    ir = IR.ir_for_category("car")
    wheels = [c for c in ir["components"] if c["role"] == "wheel"]
    for w in wheels:
        base = w["position"][1] - w["dims"][1] / 2
        assert abs(base) < 1e-6                      # grounded
        assert "axis:+y" in w["constraints"]         # one consistent axis
    # spread along both axes: two on each side
    xs = {round(w["position"][0], 3) for w in wheels}
    zs = {round(w["position"][2], 3) for w in wheels}
    assert len(xs) == 2 and len(zs) == 2


def test_ir_to_actions_and_back_round_trip():
    import morphix_ir as IR
    ir = IR.ir_for_category("car")
    actions = IR.ir_to_actions(ir)
    assert {a["action"] for a in actions} <= {"add_box", "add_cylinder"}
    assert [a["label"] for a in actions] == [c["id"] for c in ir["components"]]
    back = IR.actions_to_ir(actions, "car", source="text")
    assert back["category"] == "car"
    assert [c["id"] for c in back["components"]] == [c["id"] for c in ir["components"]]
    assert IR.validate_ir(back) == []


def test_validate_ir_rejects_bad_specs():
    import morphix_ir as IR
    good = IR.ir_for_category("object")
    assert IR.validate_ir(IR.ir_for_category("house")) == []

    bad = IR.ir_for_category("object")
    bad["components"][0]["dims"] = [0.0, 0.5, 0.5]           # zero width
    assert any("dims.w" in e for e in IR.validate_ir(bad))

    big = IR.ir_for_category("object")
    big["components"][0]["dims"] = [99.0, 0.5, 1.0]
    assert any("exceeds" in e for e in IR.validate_ir(big))

    dup = IR.ir_for_category("object")
    dup["components"][1]["id"] = "object_base"       # duplicate
    assert any("duplicate" in e for e in IR.validate_ir(dict(dup, bbox=[0, 0, 0])))


def test_validate_ir_rejects_bad_material_and_nonround_cylinder():
    import morphix_ir as IR
    bad_mat = IR.ir_for_category("car")
    bad_mat["components"][0]["material"] = "plutonium"
    assert any("material" in e for e in IR.validate_ir(bad_mat))

    nonround = IR.ir_for_category("car")
    nonround["components"][0]["dims"] = [0.7, 0.5, 0.9]
    assert any("circular" in e for e in IR.validate_ir(nonround))


def test_validate_ir_rejects_unknown_source_and_units():
    import morphix_ir as IR
    ir = IR.ir_for_category("car")
    ir["source"] = "guesswork"
    assert any("source" in e for e in IR.validate_ir(ir))
    ir2 = IR.ir_for_category("car")
    ir2["up"] = "z"
    assert any("up" in e for e in IR.validate_ir(ir2))


def test_unknown_category_uses_generic_template_with_documented_assumption():
    import morphix_ir as IR
    ir = IR.ir_for_category("bicycle")
    assert ir["template"] == "object"
    assert ir["category"] == "bicycle"
    assert IR.validate_ir(ir) == []
    assert any("no template" in a for a in ir["assumptions"])


def test_component_hierarchy_nests_parts():
    import morphix_ir as IR
    rows = {r["id"]: r for r in IR.component_hierarchy(IR.ir_for_category("car"))}
    assert rows["car_body"]["depth"] == 0
    assert rows["cabin"]["depth"] == 1
    assert rows["windshield"]["depth"] == 2
    assert rows["side_window_left"]["depth"] == 2


def test_car_template_compiles_into_distinct_components():
    from morphix_path import compile_plan
    import morphix_ir as IR
    compiled, order, errors, _keep, _warns = compile_plan(
        IR.ir_to_actions(IR.ir_for_category("car")), {})
    assert errors == {}
    wheels = [lab for lab in order if lab.startswith("wheel_")]
    assert len(wheels) == 4
    # each wheel realises its own multi-layer solid, and the body is bigger
    wheel_layers = [sorted(compiled[w].keys()) for w in wheels]
    assert all(len(ls) >= 2 for ls in wheel_layers)
    for i in range(len(wheels)):
        for j in range(i + 1, len(wheels)):
            assert wheel_layers[i] != wheel_layers[j] or \
                compiled[wheels[i]] != compiled[wheels[j]]
