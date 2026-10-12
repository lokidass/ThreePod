# tests/test_geometry_pipeline.py
# Regression suite for the printer geometry pipeline: connected, validated
# segments and successive solid layers — and the canonical
# "a blue box with a sphere on top" prompt compiled deterministically with no
# model in the loop.
#
# These tests check GEOMETRY (continuity, closure, coverage, dimensions,
# roundness, contact, material), never merely that the action JSON parsed.
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphix_path import (BEAD_HEIGHT, BEAD_WIDTH, LAYER_HEIGHT,
                          PathState, compile_plan, layer_y, merge_layers,
                          trace_rect, validate_geometry)

PROMPT = "a blue box with a sphere on top"


def _compile(actions):
    """Compile + merge a plan; assert no per-component compile errors."""
    compiled, order, errors, _keep, warns = compile_plan(actions, {})
    assert errors == {}, errors
    script, total = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    return compiled, order, script, total


def _extent(acts, axis):
    vals = [p for a in acts if a["action"] == "deposit_segment"
            for p in (a["from"][axis], a["to"][axis])]
    return round(min(vals), 3), round(max(vals), 3)


# ---------------------------------------------------------------- generators
def test_straight_path_is_connected_and_valid():
    # every deposit starts exactly at the live toolhead; no drift, no gap
    acts = trace_rect(0, 0.0, 0.0, 0.8, 0.4, label="r")
    head = None
    segs = 0
    for a in acts:
        if a["action"] == "move_to":
            head = a["position"]
        elif a["action"] == "deposit_segment":
            assert a["from"] == head, (a, head)
            head = a["to"]
            segs += 1
    assert segs > 0
    assert validate_geometry(acts) == []


def test_closed_rectangle_has_all_edges_and_closes():
    acts = trace_rect(0, 0.0, 0.0, 0.6, 0.4, label="r")
    segs = [a for a in acts if a["action"] == "deposit_segment"]
    # all four edges present: full X and Z extent reached
    assert (min(_extent(segs, 0)), max(_extent(segs, 0))) == (-0.3, 0.3)
    assert (min(_extent(segs, 2)), max(_extent(segs, 2))) == (-0.2, 0.2)
    # the contour returns exactly to where it started: no missing edge
    assert segs[0]["from"] == segs[-1]["to"]
    assert validate_geometry(acts) == []


def test_right_angle_corners_are_joined():
    # a 90-degree turn between two flat-capped beads leaves a bead-half notch;
    # one join bead is dropped on each of the four vertices to fill it.
    acts = trace_rect(0, 0.0, 0.0, 0.4, 0.4, label="c")
    joins = [a for a in acts if a["action"] == "deposit_point"]
    assert len(joins) == 4
    corners = {(round(a["position"][0], 3), round(a["position"][2], 3))
               for a in joins}
    assert corners == {(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)}
    assert all(a["size"] == BEAD_WIDTH and a["material"] == "body"
               for a in joins)
    # joins sit exactly on the toolhead: continuity is preserved
    assert validate_geometry(acts) == []


def test_adjacent_layers_stack_without_vertical_gap():
    # the bead cross-section is one layer tall, so consecutive courses share a
    # face; a box is solid on every layer (perimeter + infill), not a shell.
    assert BEAD_HEIGHT == LAYER_HEIGHT
    box = {"action": "add_box", "width": 1.0, "height": 0.6, "depth": 1.0,
           "position": [0, 0.3, 0], "color": "0xFF0000", "label": "b",
           "material": "body"}
    compiled, _order, script, total = _compile([box])
    assert sorted(compiled["b"].keys()) == [0, 1, 2]
    assert total == 3
    for k in range(3):
        layer = compiled["b"][k]
        # every layer has fill as well as a perimeter
        assert any(a["action"] == "deposit_point" for a in layer)  # corners
        assert any("fill" in a.get("label", "")
                   for a in layer if a["action"] == "deposit_segment")
    # top of each layer's bead == bottom of the next layer's bead
    for k in range(total - 1):
        top = layer_y(k) + BEAD_HEIGHT / 2.0
        bottom = layer_y(k + 1) - BEAD_HEIGHT / 2.0
        assert math.isclose(top, bottom, abs_tol=1e-9)
    assert validate_geometry(script) == []


def test_validate_geometry_rejects_gaps_and_unclosed_paths():
    good = {"bead_width": BEAD_WIDTH, "bead_height": BEAD_HEIGHT,
            "material": "body", "layer": 0}
    gap = [{"action": "move_to", "position": [0, 0, 0], "layer": 0,
            "label": "m"},
           dict(good, action="deposit_segment", **{"from": [0, 0, 0],
                "to": [0.2, 0, 0]}, label="s0"),
           dict(good, action="deposit_segment", **{"from": [0.4, 0, 0],
                "to": [0.6, 0, 0]}, label="s1"),
           {"action": "complete_path", "layer": 0, "label": "d"}]
    assert any("continuity" in e for e in validate_geometry(gap))
    unclosed = [{"action": "move_to", "position": [0, 0, 0], "layer": 0,
                 "label": "m"},
                dict(good, action="deposit_segment", **{"from": [0, 0, 0],
                     "to": [0.2, 0, 0]}, label="s0"),
                dict(good, action="deposit_segment", **{"from": [0.2, 0, 0],
                     "to": [0.2, 0, 0.2]}, label="s1"),
                {"action": "complete_path", "layer": 0, "label": "d"}]
    assert any("not closed" in e for e in validate_geometry(unclosed))


def test_rejected_segment_does_not_modify_scene():
    st = PathState(total_layers=1)
    bad = {"action": "deposit_segment", "from": [5, 0, 5], "to": [5.2, 0, 5],
           "bead_width": BEAD_WIDTH, "bead_height": BEAD_HEIGHT,
           "material": "body", "layer": 0, "label": "bad"}
    assert st.check(bad)  # rejected before it can commit
    with pytest.raises(ValueError):
        st.apply(bad)
    assert st.pos == [0.0, 0.0, 0.0]
    assert st.seq == 0
    assert st.layers[0]["segments"] == []


# ---------------------------------------------------------------- canonical
def test_box_sphere_spec_is_canonical():
    from morphix_specs import spec_for_prompt
    spec = spec_for_prompt(PROMPT)
    assert spec is not None and spec["name"] == "box_sphere"
    # a lone round word is NOT gated by this spec (no false positives)
    assert spec_for_prompt("a ball") is None


def test_canonical_ir_describes_blue_box_with_sphere_on_top():
    import morphix_ir as IR
    spec = IR.ir_for_prompt(PROMPT)
    assert IR.validate_ir(spec) == []
    box = next(c for c in spec["components"] if c["id"] == "box")
    sph = next(c for c in spec["components"] if c["id"] == "sphere")
    assert box["primitive"] == "box" and box["material"] == "blue"
    assert all(v > 0 for v in box["dims"])
    assert sph["primitive"] == "sphere" and sph["parent"] == "box"
    # sphere rests concentrically on the box top face
    assert math.isclose(sph["position"][1] - sph["dims"][1] / 2,
                        box["position"][1] + box["dims"][1] / 2)
    assert sph["position"][0] == box["position"][0]
    assert sph["position"][2] == box["position"][2]
    # both poles land on layer planes so the sphere slices to real circles
    base = sph["position"][1] - sph["dims"][1] / 2
    apex = sph["position"][1] + sph["dims"][1] / 2
    assert math.isclose(base / LAYER_HEIGHT, round(base / LAYER_HEIGHT),
                        abs_tol=1e-9)
    assert math.isclose(apex / LAYER_HEIGHT, round(apex / LAYER_HEIGHT),
                        abs_tol=1e-9)


def test_sphere_slices_to_round_per_layer_cross_sections():
    import morphix_ir as IR
    compiled, _order, script, total = _compile(
        IR.ir_to_actions(IR.ir_for_prompt(PROMPT)))
    # a full round solid: many layers, the middle one is the widest
    assert sorted(compiled["sphere"].keys()) == [5, 6, 7, 8, 9, 10, 11]
    widths = []
    for k in sorted(compiled["sphere"].keys()):
        xs = []
        for a in compiled["sphere"][k]:
            if a["action"] == "deposit_segment":
                xs += [a["from"][0], a["to"][0]]
            elif a["action"] == "deposit_point":
                xs.append(a["position"][0])
        widths.append(round(max(xs) - min(xs), 3))
    assert widths[0] < widths[3] > widths[-1]  # narrow, widest, narrow
    assert widths[3] == pytest.approx(1.2, abs=0.05)  # diameter == 2R
    assert validate_geometry(script) == []


def test_canonical_blue_box_sphere_end_to_end(tmp_path, monkeypatch):
    import morphix_pipeline as P

    def boom(*a, **k):
        raise AssertionError("canonical prompt must not consult the model")

    monkeypatch.setattr(P, "think", boom)
    rep = P.run_path_print(PROMPT, out_dir=str(tmp_path),
                           stream_scene=False)
    assert rep["ok"] and rep["complete"] and rep["quality_accepted"]
    assert rep["spec"] == "box_sphere"
    comps = {c["component"]: c for c in rep["components"]}
    assert comps["box"]["verified"] and comps["sphere"]["verified"]
    # the final result passes geometry-level checks, not just syntax
    assert rep["geometry_checks"]
    assert all(g["ok"] for g in rep["geometry_checks"])
    names = {(g["component"], g["name"]) for g in rep["geometry_checks"]}
    assert ("box", "box_extent") in names
    assert ("sphere", "sphere_round") in names
    assert ("sphere", "contact_box") in names
    # the requested blue actually reaches the rendered scene
    scene = open(os.path.join(str(tmp_path), "scene.js"),
                 encoding="utf-8").read()
    assert "0x2A6FDB" in scene


def test_canonical_toolpath_is_gapless():
    import morphix_ir as IR
    _compiled, _order, script, total = _compile(
        IR.ir_to_actions(IR.ir_for_prompt(PROMPT)))
    assert total == 12  # box layers 0-4, sphere layers 5-11
    assert validate_geometry(script) == []
    # every deposit carries a component tag and a declared material
    for a in script:
        if a["action"] in ("deposit_segment", "deposit_point"):
            assert a.get("component") in ("box", "sphere")
            assert a.get("material") in ("blue", "accent")
