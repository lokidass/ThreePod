# tests/test_path.py
# Point-by-point printer core: toolhead continuity, y-locked layers,
# deterministic generators, engine rendering, checkpoints, components, demo.
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEG = {"action": "deposit_segment", "from": [0.0, 0.0, 0.0],
       "to": [0.2, 0.0, 0.0], "bead_width": 0.08, "bead_height": 0.08,
       "material": "body", "layer": 0, "label": "s_000"}


def test_primitives_rejected_in_printer_mode():
    # §11.1-2: whole-object actions never construct geometry here, and are
    # never translated into deposition steps.
    from morphix_path import PathState
    st = PathState(total_layers=1)
    box = {"action": "add_box", "width": 1, "height": 1, "depth": 1,
           "position": [0, 0.5, 0], "color": "0xFF0000", "label": "b"}
    sphere = {"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0],
              "color": "0xFF0000", "label": "s"}
    assert any("forbidden" in e for e in st.check(box))
    assert any("forbidden" in e for e in st.check(sphere))
    with pytest.raises(ValueError):
        st.apply(box)
    assert st.pos == [0.0, 0.0, 0.0] and st.seq == 0


def test_one_valid_deposit_creates_one_local_segment():
    # §11.3: exactly one local segment lands in the registry + renderer.
    from morphix_path import PathState, render_deposit
    st = PathState(total_layers=1)
    rec = st.apply(dict(SEG))
    assert len(st.layers[0]["segments"]) == 1
    js = render_deposit(rec)
    assert "scene.add" in js and "0.2" in js and "lookAt" in js
    # segment spans exactly from->to on x, bead cross-section otherwise
    assert "0.1" in js  # midpoint x of [0, 0.2]


def test_segment_must_start_at_toolhead():
    # §11.4-6: commits advance; rejects preserve position and geometry.
    from morphix_path import PathState
    st = PathState(total_layers=1)
    bad = dict(SEG, **{"from": [5.0, 0.0, 0.0]})
    assert any("toolhead" in e for e in st.check(bad))
    with pytest.raises(ValueError):
        st.apply(bad)
    assert st.pos == [0.0, 0.0, 0.0]  # failed step never advances
    assert st.layers[0]["segments"] == []
    rec = st.apply(dict(SEG))
    assert st.pos == [0.2, 0.0, 0.0]
    assert rec["seq"] == 1


def test_deposition_y_locked_to_active_layer():
    from morphix_path import PathState
    st = PathState(total_layers=2)
    high = dict(SEG, **{"from": [0.0, 0.0, 0.0],
                        "to": [0.2, 0.5, 0.0]})
    assert any("active layer 0" in e for e in st.check(high))
    point = {"action": "deposit_point", "position": [0.0, 0.0, 0.0],
             "size": 0.08, "material": "accent", "layer": 0,
             "label": "p_000"}
    assert st.check(point) == []
    rec = st.apply(point)
    assert st.pos == [0.0, 0.0, 0.0]
    assert rec["material"] == "accent"


def test_bad_dimensions_bounds_and_layer_rejected():
    # §11.7: non-finite coordinates and invalid dimensions rejected.
    from morphix_path import PathState
    st = PathState(total_layers=2)
    assert any("zero length" in e
               for e in st.check(dict(SEG, **{"to": [0.0, 0.0, 0.0]})))
    assert any("too long" in e
               for e in st.check(dict(SEG, **{"to": [9.0, 0.0, 0.0]})))
    assert any("bead_width" in e for e in st.check(dict(SEG, bead_width=0)))
    assert any("bead_height" in e
               for e in st.check(dict(SEG, bead_height=99)))
    assert any("material" in e
               for e in st.check(dict(SEG, material="unobtainium")))
    assert any("finite" in e for e in st.check(
        dict(SEG, **{"to": [float("nan"), 0.0, 0.0]})))
    assert any("finite" in e for e in st.check(
        dict(SEG, **{"from": [float("inf"), 0.0, 0.0]})))
    assert any("out of bounds" in e for e in st.check(
        dict(SEG, **{"to": [99.0, 0.0, 0.0]})))
    assert any("active layer 0" in e for e in st.check(
        dict(SEG, layer=1)))
    assert any("bounds" in e for e in st.check(
        {"action": "move_to", "position": [0, 0, 99.0],
         "layer": 0, "label": "m"}))
    st.apply({"action": "move_to", "position": [1.0, 0.0, 2.0],
              "layer": 0, "label": "m"})
    assert st.pos == [1.0, 0.0, 2.0]


def test_layer_completion_rules():
    # §11.8: layer transitions require authorization.
    from morphix_path import PathState
    st = PathState(total_layers=2)
    assert any("no segments" in e for e in st.check(
        {"action": "complete_layer", "layer": 0, "label": "d"}))
    assert any("no open path" in e for e in st.check(
        {"action": "complete_path", "layer": 0, "label": "d"}))
    st.apply(dict(SEG))
    st.apply({"action": "complete_path", "layer": 0, "label": "d"})
    assert any("complete_object" in e for e in st.check(
        {"action": "complete_object", "label": "done"}))
    st.apply({"action": "complete_layer", "layer": 0, "label": "d"})
    assert st.active_layer == 1
    assert any("active layer 1" in e for e in st.check(dict(SEG)))


def test_checkpoint_round_trip():
    from morphix_path import PathState
    st = PathState(total_layers=2)
    st.apply(dict(SEG))
    st.apply({"action": "move_to", "position": [0.2, 0.0, 0.0],
              "layer": 0, "label": "m"})
    data = json.loads(json.dumps(st.checkpoint()))
    st2 = PathState.restore(data)
    assert st2.checkpoint() == data
    assert st2.pos == st.pos
    assert st2.seq == 1


def test_component_lifecycle():
    # §11.11: planned/started/partial/completed/missing are explicit,
    # matched by slicer-stamped tags, never label guessing.
    from morphix_path import component_status
    thought = [{"action": "add_box", "label": "walls"},
               {"action": "add_box", "label": "roof"}]
    script = [
        {"action": "deposit_segment", "component": "walls", "layer": 0,
         "from": [0, 0, 0], "to": [0.2, 0, 0], "label": "a"},
        {"action": "deposit_segment", "component": "walls", "layer": 0,
         "from": [0.2, 0, 0], "to": [0.4, 0, 0], "label": "b"},
        {"action": "deposit_segment", "component": "roof", "layer": 1,
         "from": [0, 0.2, 0], "to": [0.2, 0.2, 0], "label": "c"}]
    assert [c["status"] for c in component_status(thought, script, [])] == [
        "missing", "missing"]
    half = component_status(thought, script, script[:1])
    assert [(c["status"], c["started"]) for c in half] == [
        ("partial", True), ("missing", False)]
    full = component_status(thought, script, script)
    assert [c["status"] for c in full] == ["completed", "completed"]
    assert full[0]["planned_deposits"] == 2
    assert full[0]["committed_deposits"] == 2
    assert full[1]["layers"] == [1]


def test_connected_segments_bounding_box():
    # §11.12: two connected deposits span the expected extents.
    from morphix_path import PathState
    st = PathState(total_layers=1)
    st.apply(dict(SEG, label="s1"))
    st.apply({"action": "deposit_segment", "from": [0.2, 0.0, 0.0],
              "to": [0.2, 0.0, 0.3], "bead_width": 0.08,
              "bead_height": 0.08, "material": "body", "layer": 0,
              "label": "s2"})
    xs = [c for r in st.layers[0]["segments"]
          for c in (r["from"][0], r["to"][0])]
    zs = [c for r in st.layers[0]["segments"]
          for c in (r["from"][2], r["to"][2])]
    assert (min(xs), max(xs)) == (0.0, 0.2)
    assert (min(zs), max(zs)) == (0.0, 0.3)
    assert st.pos == [0.2, 0.0, 0.3]


def test_closed_square_segment_by_segment():
    # §11.13: deterministic closed square, one action per commit.
    from morphix_path import PathState, trace_rect
    acts = trace_rect(0, 0.0, 0.0, 0.4, 0.4, step=0.2, label="sq")
    st = PathState(total_layers=1)
    for a in acts:
        assert st.check(a) == [], (a, st.check(a))
        st.apply(a)
    assert st.pos == [-0.2, 0.0, -0.2]  # back at the start corner
    assert st.layers[0]["paths_done"] == 1
    # 4 sides x (0.4/0.2) segments + one join bead per right-angle corner
    assert st.seq == 12


def test_schema_covers_path_protocol():
    from morphix_validator import validate_actions
    acts = [{"action": "move_to", "position": [0, 0, 0],
             "layer": 0, "label": "m_0"},
            dict(SEG),
            {"action": "deposit_point", "position": [0.2, 0, 0],
             "size": 0.08, "material": "support", "layer": 0,
             "label": "p_0"},
            {"action": "complete_path", "layer": 0, "label": "c_0"},
            {"action": "complete_layer", "layer": 0, "label": "c_1"},
            {"action": "complete_object", "label": "c_2"}]
    assert validate_actions(acts) == []
    assert any("layer" in e for e in validate_actions(
        [dict(SEG, layer=-1)]))
    assert any("layer" in e for e in validate_actions(
        [dict(SEG, layer=True)]))
    assert any("[x,y,z]" in e for e in validate_actions(
        [dict(SEG, **{"from": [0, 0]})]))
    assert any("material" in e for e in validate_actions(
        [dict(SEG, material="lava")]))
    # SCREAMING_CASE is not the path protocol (mode gate for primitives
    # lives in PathState.check, covered by test_primitives_rejected_*).
    assert any("unknown action" in e for e in validate_actions(
        [{"action": "MOVE_TO", "position": [0, 0, 0],
          "layer": 0, "label": "m"}]))


def test_engine_rendering_is_node_clean():
    from morphix_path import render_deposit
    from morphix_validator import validate_js_syntax
    seg_js = render_deposit({"seq": 1, "action": "deposit_segment",
                             "label": "s_000", "from": [0, 0, 0],
                             "to": [0.2, 0, 0], "bead_width": 0.08,
                             "bead_height": 0.08, "material": "accent"})
    pt_js = render_deposit({"seq": 2, "action": "deposit_point",
                            "label": "p_000", "at": [0.2, 0, 0],
                            "size": 0.08, "material": "body"})
    assert "scene.add" in seg_js and "lookAt" in seg_js
    assert "scene.add" in pt_js
    assert "0x44CCFF" in seg_js  # accent palette, engine-owned
    assert validate_js_syntax(seg_js + "\n" + pt_js) == []


def test_demo_deterministic_and_valid(tmp_path):
    import demo_print
    a1, js1, rows1 = demo_print.build()
    a2, js2, rows2 = demo_print.build()
    assert a1 == a2 and js1 == js2
    assert len(rows1) > 50
    for r in rows1:
        assert set(("state", "plan", "layer", "action")) <= set(r)
        assert r["layer"]["y"] == r["layer"]["index"] * 0.2
    # every committed deposit sits on its layer plane
    for r in rows1:
        c = r["committed"]
        if c.get("action") == "deposit_segment":
            assert abs(c["to"][1] - r["layer"]["y"]) <= 0.05


def _box(label, w=2.0, h=0.4, d=1.0, pos=(0, 0.2, 0)):
    return {"action": "add_box", "width": w, "height": h, "depth": d,
            "position": list(pos), "color": "0xFF0000", "label": label}


def test_compile_grounded_box_and_tapered_cylinder():
    from morphix_path import compile_plan, merge_layers
    import math
    cyl = {"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.5,
           "height": 0.6, "position": [3, 0.3, 0], "color": "0x00FF00",
           "label": "post"}
    comp, order, errors, _keep, _warns = compile_plan([_box("base"), cyl])
    assert errors == {}, errors
    assert order == ["base", "post"]
    script, total = merge_layers([comp[lab] for lab in order])
    assert total == 3  # box 0.4 -> layers 0-1; cylinder 0.6 -> layers 0-2
    assert len(script) > 0 and all(
        a.get("component") in ("base", "post") or
        a["action"] == "complete_layer" for a in script)

    def ring_r(layer_actions):
        ext = [max(abs(p[0] - 3), abs(p[2]))
               for a in layer_actions
               if a.get("action") == "deposit_segment" for p in (a["to"],)]
        return max(ext)

    l0 = comp["post"][0]
    l2 = comp["post"][2]
    assert ring_r(l2) < ring_r(l0)  # real taper cross-sections


def test_compile_elevated_supported_and_airborne_seated():
    from morphix_path import compile_plan
    base = _box("base")
    door = _box("door_a1", w=1.0, h=1.6, d=0.2, pos=(0, 1.2, 0.4))
    comp, _order, errors, _keep, warns = compile_plan([base, door])
    assert errors == {}, errors
    assert warns.get("door_a1") in (None, [])
    assert sorted(comp["door_a1"].keys()) == list(range(2, 10))
    # genuinely airborne parts are seated onto support/ground with a warning,
    # never silently and never dropped
    sky = _box("sky", pos=(9, 5, 9))
    comp2, _o2, errors2, _k2, warns2 = compile_plan([base, sky])
    assert errors2 == {}, errors2
    assert "base" in comp2
    assert any("seated" in w for w in warns2.get("sky", []))
    assert sorted(comp2["sky"].keys()) == [0, 1]


def test_compile_car_like_multicomponent():
    from morphix_path import compile_plan, merge_layers

    def wheel(x, z, label):
        return {"action": "add_cylinder", "radiusTop": 0.3,
                "radiusBottom": 0.3, "height": 0.6,
                "position": [x, 0.3, z], "color": "0x222222", "label": label}

    wheels = [wheel(-1, 0.8, "wheel_fl"), wheel(1, 0.8, "wheel_fr"),
              wheel(-1, -0.8, "wheel_rl"), wheel(1, -0.8, "wheel_rr")]
    body = {"action": "add_box", "width": 4.0, "height": 0.8, "depth": 1.8,
            "position": [0, 1.0, 0], "color": "0xFF0000", "label": "car_body"}
    comp, order, errors, _keep, _warns = compile_plan(wheels + [body])
    assert errors == {}, errors
    assert sorted(comp["car_body"].keys()) == [3, 4, 5, 6]
    script, total = merge_layers([comp[lab] for lab in order])
    assert total == 7
    # bottom layer of the body carries infill as well as perimeter
    assert any("_fill" in a.get("label", "")
               for a in comp["car_body"][3])


def test_compile_invalid_dims_and_refs_are_precise():
    from morphix_path import compile_plan
    bad_w = _box("badw")
    bad_w["width"] = -1
    _c, _o, errors, _k, _warns = compile_plan([bad_w])
    assert "badw" in errors and "width" in errors["badw"]
    huge = _box("huge")
    huge["width"] = 100
    _c, _o, errors, _k, _warns = compile_plan([huge])
    assert "huge" in errors and "exceeds" in errors["huge"]
    no_h = _box("noh")
    del no_h["height"]
    _c, _o, errors, _k, _warns = compile_plan([no_h])
    assert "noh" in errors and "missing height" in errors["noh"]
    bad_r = {"action": "add_cylinder", "radiusTop": "x", "height": 0.5,
             "position": [0, 0.25, 0], "color": "0xFF0000", "label": "badr"}
    _c, _o, errors, _k, _warns = compile_plan([bad_r])
    assert "badr" in errors and "number" in errors["badr"]
    dangling = _box("dangle")
    dangling["on_top_of"] = "ghost"
    _c, _o, errors, _k, _warns = compile_plan([dangling])
    assert "dangle" in errors and "ghost" in errors["dangle"]
    # resolved stacking: roof sits exactly on the walls' top
    walls = _box("walls", w=2.0, h=1.0, d=2.0, pos=(0, 0.5, 0))
    roof = _box("roof", w=1.0, h=0.4, d=1.0, pos=(9, 9, 9))
    roof["on_top_of"] = "walls"
    comp, _o, errors, _k, _warns = compile_plan([walls, roof])
    assert errors == {}, errors
    assert sorted(comp["roof"].keys()) == [5, 6]  # y 1.0..1.4


def test_keep_reuses_identical_components():
    from morphix_path import compile_plan
    base = _box("base")
    sky = _box("sky", pos=(9, 5, 9))
    comp1, _o1, e1, keep, w1 = compile_plan([base, sky])
    assert e1 == {}, e1  # sky seated, not dropped
    assert any("seated" in w for w in w1.get("sky", []))
    fixed = _box("sky", pos=(0, 0.2, 0))
    comp2, _o2, e2, _k2, w2 = compile_plan([dict(base), fixed], keep=keep)
    assert e2 == {}, e2
    assert comp2["base"] == comp1["base"]  # identical work reused
    assert w2.get("sky", []) == []  # grounded needs no seating
