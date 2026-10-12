# tests/test_print.py
# Path-mode integration: deterministic slicing runs, guided loop with fake
# builders (echo / adversarial), resume, loud failures, single-action parse.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOX = {"action": "add_box", "width": 2.0, "height": 0.4, "depth": 1.0,
       "position": [0, 0.2, 0], "color": "0xFF0000", "label": "base_a1"}


def _mock_think(monkeypatch, actions):
    import morphix_pipeline as P

    def fake_think(prompt, model=None, options=None):
        return ([dict(a) for a in actions], "MODEL x")

    monkeypatch.setattr(P, "think", fake_think)
    return P


def test_deterministic_print_completes(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])
    rep = P.run_path_print("a flat base", out_dir=str(tmp_path))
    assert rep["ok"] is True
    assert rep["mode"] == "path" and rep["guided"] is False
    assert rep["layers"] == 2 and rep["deposits"] > 20
    assert rep["syntax_scene_js"] == []
    assert os.path.isfile(os.path.join(str(tmp_path), "path.json"))
    assert os.path.isfile(os.path.join(str(tmp_path), "scene.html"))
    acts = json.load(open(os.path.join(str(tmp_path), "actions.json")))
    assert acts[-1] == {"action": "complete_object", "label": "obj_done"}
    # every deposit sits on its own layer plane, layers complete in order
    from morphix_path import layer_y
    done_layers = 0
    for a in acts:
        if a["action"] == "complete_layer":
            assert a["layer"] == done_layers
            done_layers += 1
        elif a["action"] == "deposit_segment":
            assert a["layer"] == done_layers
            assert abs(a["to"][1] - layer_y(a["layer"])) <= 0.05
    assert done_layers == rep["layers"]
    # components tracked explicitly per planned primitive
    comp = rep["components"][0]
    assert (comp["component"], comp["action"], comp["status"]) == (
        "base_a1", "add_box", "completed")
    assert comp["layers"] == [0, 1]
    assert comp["verified"] is True
    assert all(c["ok"] for c in comp["checks"]), comp["checks"]
    assert rep["complete"] is True and rep["quality_accepted"] is True
    assert rep["budgets"]["committed_actions"] == len(acts)
    assert rep["budgets"]["planner_attempts"] == 1


def test_guided_echo_completes(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])

    def echo(ctx):
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    rep = P.run_path_print("a flat base", out_dir=str(tmp_path),
                           guided=True, build_one=echo)
    assert rep["ok"] is True and rep["guided"] is True
    assert rep["deposits"] > 20


def test_guided_adversarial_fails_loudly(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])
    calls = {"n": 0}

    def bad(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return '[{"action": "move_to"}]'  # array smuggling refused
        return json.dumps({"action": "deposit_segment",
                           "from": [9, 9, 9], "to": [9.2, 9, 9],
                           "bead_width": 0.08, "bead_height": 0.08,
                           "material": "body",
                           "layer": 0, "label": "evil_01"})

    with pytest.raises(Exception):
        P.run_path_print("a flat base", out_dir=str(tmp_path),
                         guided=True, build_one=bad, retries=1)
    assert calls["n"] >= 2


def test_resume_replays_without_duplication(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])
    r1 = P.run_path_print("a flat base", out_dir=str(tmp_path))
    r2 = P.run_path_print("a flat base", out_dir=str(tmp_path), resume=True)
    assert r1["deposits"] == r2["deposits"] > 0
    assert r1["path_actions"] == r2["path_actions"]
    acts = json.load(open(os.path.join(str(tmp_path), "actions.json")))
    assert sum(1 for a in acts
               if a.get("action") == "complete_object") == 1


def test_unprintable_plans_fail_loudly(tmp_path, monkeypatch):
    import morphix_pipeline as P
    from morphix_config import PipelineError
    cone = {"action": "add_cone", "radius": 0.5, "height": 1.0,
            "position": [0, 1, 0], "color": "0xFF0000", "label": "cone_a1"}
    _mock_think(monkeypatch, [cone])
    with pytest.raises(PipelineError):
        P.run_path_print("a cone", out_dir=str(tmp_path))
    # airborne parts are seated (not dropped): prints fine with a warning
    floater = dict(BOX, position=[0, 5, 0], label="float_a1")
    _mock_think(monkeypatch, [floater])
    rep = P.run_path_print("a floating base", out_dir=str(tmp_path))
    assert rep["ok"] is True
    assert any("seated" in w for ws in rep["seat_warnings"].values()
               for w in ws)


def test_plan_retried_once_with_feedback(tmp_path, monkeypatch):
    import morphix_pipeline as P
    calls = {"n": 0, "prompts": []}

    def flaky(prompt, model=None, options=None):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        if calls["n"] == 1:
            return ([{"action": "add_cone", "radius": 0.5, "height": 1.0,
                      "position": [0, 1, 0], "color": "0xFF0000",
                      "label": "cone_a1"}], "MODEL x")  # unsliceable
        return ([dict(BOX)], "MODEL x")

    monkeypatch.setattr(P, "think", flaky)
    rep = P.run_path_print("a base", out_dir=str(tmp_path))
    assert rep["ok"] is True and calls["n"] == 2
    assert "PRINT MODE" in calls["prompts"][0]


def test_sunk_plan_seated_without_retry(tmp_path, monkeypatch):
    import morphix_pipeline as P

    def sunk_once(prompt, model=None, options=None):
        return ([dict(BOX, position=[0, 0, 0])], "MODEL x")

    monkeypatch.setattr(P, "think", sunk_once)
    rep = P.run_path_print("a base", out_dir=str(tmp_path))
    # sunk plan is seated onto support on attempt 1: no retry needed
    assert rep["ok"] is True
    assert any("seated" in w for ws in rep["seat_warnings"].values()
               for w in ws)


def test_plan_retry_exhausted_still_fails(tmp_path, monkeypatch):
    import morphix_pipeline as P
    from morphix_config import PipelineError
    calls = {"n": 0}

    def sunk(prompt, model=None, options=None):
        calls["n"] += 1
        return ([{"action": "add_cone", "radius": 0.5, "height": 1.0,
                  "position": [0, 1, 0], "color": "0xFF0000",
                  "label": "cone_a1"}], "MODEL x")

    monkeypatch.setattr(P, "think", sunk)
    with pytest.raises(PipelineError):
        P.run_path_print("a base", out_dir=str(tmp_path))
    assert calls["n"] == 2


def test_partial_subset_prints_loudly(tmp_path, monkeypatch):
    # One unprintable component does not kill valid work: the box prints,
    # the cone is omitted with its precise error, complete stays False.
    import morphix_pipeline as P
    cone = {"action": "add_cone", "radius": 0.5, "height": 1.0,
            "position": [0, 1, 0], "color": "0xFF0000", "label": "cone_a1"}
    events = []
    P2 = _mock_think(monkeypatch, [dict(BOX), cone])
    rep = P2.run_path_print("a base and a cone", out_dir=str(tmp_path),
                            emit=events.append)
    assert rep["ok"] is True and rep["deposits"] > 0
    assert rep["complete"] is False
    assert [o["component"] for o in rep["omitted"]] == ["cone_a1"]
    assert "add_cone" in rep["omitted"][0]["error"]
    by_status = {c["component"]: c["status"] for c in rep["components"]}
    assert by_status == {"base_a1": "completed", "cone_a1": "missing"}
    fails = [e for e in events if e.get("type") == "compile"
             and e.get("status") == "failed"]
    assert len(fails) == 2  # one per plan attempt, same component


def test_guided_finish_directive(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])
    ctxs = []

    def echo(ctx):
        ctxs.append(dict(ctx))
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    rep = P.run_path_print("a base", out_dir=str(tmp_path),
                           guided=True, build_one=echo)
    assert rep["ok"] is True
    finals = [c for c in ctxs if "suggestion" not in c]
    assert finals
    assert all("complete_object" in c.get("next", "") for c in finals)


def test_engine_owns_travel_moves(tmp_path, monkeypatch):
    P = _mock_think(monkeypatch, [BOX])
    builder_calls = {"n": 0}

    def no_travel(ctx):
        builder_calls["n"] += 1
        sug = ctx.get("suggestion")
        assert sug is None or sug["action"] != "move_to", \
            "engine must own travel, not the builder"
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    rep = P.run_path_print("a base", out_dir=str(tmp_path),
                           guided=True, build_one=no_travel)
    assert rep["ok"] is True
    acts = json.load(open(os.path.join(str(tmp_path), "actions.json")))
    moves = [a for a in acts if a.get("action") == "move_to"]
    assert len(moves) >= 2  # engine-inserted, not builder-proposed
    assert builder_calls["n"] == len(acts) - len(moves)


def test_single_action_parse():
    import morphix_pipeline as P
    assert P._parse_single_action(
        '```json\n{"action": "move_to", "position": [0,0,0],'
        ' "layer": 0, "label": "m"}\n```')["action"] == "move_to"
    with pytest.raises(ValueError):
        P._parse_single_action('[{"action": "move_to"}]')
    with pytest.raises(ValueError):
        P._parse_single_action(
            '{"action": "move_to"} {"action": "move_to"}')
    with pytest.raises(ValueError):
        P._parse_single_action('just some words')


def test_attempt_budget_is_hard(tmp_path, monkeypatch):    # §11.9-10: attempts are bounded, rejects never commit, failures make
    # no geometry. Tiny 1-layer plan (11 script actions: 4 perimeter
    # segments + 4 corner joins + move_to + complete_path + complete_layer),
    # segment_cap=11 -> ceiling max(44,19)=44, checked before each builder
    # call, so exactly 44 calls happen and the 45th attempt dies with zero
    # deposits committed.
    tiny = {"action": "add_box", "width": 0.2, "height": 0.2, "depth": 0.2,
            "position": [0, 0.1, 0], "color": "0xFF0000", "label": "tiny_a1"}
    P = _mock_think(monkeypatch, [tiny])
    calls = {"n": 0}

    def bad(ctx):
        calls["n"] += 1
        return json.dumps({"action": "deposit_segment",
                           "from": [9, 9, 9], "to": [9.2, 9, 9],
                           "bead_width": 0.08, "bead_height": 0.08,
                           "material": "body", "layer": 0,
                           "label": f"evil_{calls['n']:03d}"})

    with pytest.raises(Exception):
        P.run_path_print("a tiny base", out_dir=str(tmp_path),
                         guided=True, build_one=bad, retries=100,
                         segment_cap=11)
    assert calls["n"] == 44
    acts = json.load(open(os.path.join(str(tmp_path), "actions.json")))
    assert all(a.get("action") == "move_to" for a in acts)  # engine travel
    assert "seg_" not in open(
        os.path.join(str(tmp_path), "scene.js"), encoding="utf-8").read()


def _car_plan():
    acts = [
        {"action": "add_box", "width": 3.0, "height": 0.6, "depth": 1.6,
         "position": [0, 0.3, 0], "color": "0xFF0000", "label": "car_body",
         "material": "body"},
        {"action": "add_box", "width": 1.6, "height": 0.5, "depth": 1.4,
         "position": [0, 0.85, 0], "color": "0x44CCFF", "label": "cabin_top",
         "material": "accent"}]
    for i, (x, z) in enumerate([(-1.0, 0.7), (1.0, 0.7),
                                (-1.0, -0.7), (1.0, -0.7)]):
        acts.append({"action": "add_cylinder", "radiusTop": 0.3,
                     "radiusBottom": 0.3, "height": 0.4,
                     "position": [x, 0.2, z], "color": "0x222222",
                     "label": f"wheel_{i}", "material": "tire"})
    for side, lab in ((0.71, "window_l"), (-0.71, "window_r")):
        acts.append({"action": "add_box", "width": 0.8, "height": 0.3,
                     "depth": 0.1, "position": [0, 0.85, side],
                     "color": "0x99DDFF", "label": lab, "material": "glass"})
    for side, lab in ((0.81, "door_l"), (-0.81, "door_r")):
        acts.append({"action": "add_box", "width": 0.9, "height": 0.5,
                     "depth": 0.1, "position": [0, 0.55, side],
                     "color": "0xFF0000", "label": lab, "material": "body"})
    return acts


def test_car_end_to_end_verified(tmp_path, monkeypatch):
    # Acceptance: a recognizable car from validated segments with truthful
    # quality reporting — every component geometrically verified.
    P = _mock_think(monkeypatch, _car_plan())
    rep = P.run_path_print("a red car", out_dir=str(tmp_path))
    assert rep["spec"] == "car"
    assert rep["ok"] is True
    assert rep["complete"] is True, rep["components"]
    assert rep["quality_accepted"] is True
    by_comp = {c["component"]: c for c in rep["components"]}
    assert set(by_comp) == {"car_body", "cabin_top", "windshield",
                            "wheel_front_left", "wheel_front_right",
                            "wheel_rear_left", "wheel_rear_right",
                            "window_l", "window_r", "door_l", "door_r"}
    for label, comp in by_comp.items():
        assert comp["status"] == "completed", (label, comp)
        assert comp["verified"] is True, (label, comp["checks"])
    assert by_comp["car_body"]["layers"] == [0, 1, 2]
    scene = open(os.path.join(str(tmp_path), "scene.js"),
                 encoding="utf-8").read()
    for color in ("0xFFAA33", "0x44CCFF", "0x99DDFF", "0x222222"):
        assert color in scene, color  # body/accent/glass/tire all rendered


def test_segment_cap_refuses_oversize_prints(tmp_path, monkeypatch):
    import morphix_pipeline as P
    from morphix_config import PipelineError
    P2 = _mock_think(monkeypatch, [BOX])
    with pytest.raises(PipelineError) as exc:
        P2.run_path_print("a base", out_dir=str(tmp_path), segment_cap=5)
    assert "segment_cap=5" in str(exc.value)


def test_complete_object_gated_on_components(tmp_path, monkeypatch):
    # §11.8/11: finishing early is refused as a goal rejection, retried,
    # then fails loudly — nothing half-built is ever marked complete.
    tiny = {"action": "add_box", "width": 0.2, "height": 0.2, "depth": 0.2,
            "position": [0, 0.1, 0], "color": "0xFF0000", "label": "tiny_a1"}
    P = _mock_think(monkeypatch, [tiny])
    events = []
    plan = {"n": 0}

    def forge(ctx):
        # valid but untagged deposits fill the layer, then an early finish
        plan["n"] += 1
        pos = ctx["toolhead"]
        if plan["n"] == 1:
            return json.dumps({"action": "deposit_segment",
                               "from": pos, "to": [pos[0] + 0.2, pos[1],
                                                   pos[2]],
                               "bead_width": 0.08, "bead_height": 0.08,
                               "material": "body", "layer": 0,
                               "label": "forged_01"})
        if plan["n"] == 2:
            return json.dumps({"action": "complete_path", "layer": 0,
                               "label": "forged_p"})
        if plan["n"] == 3:
            return json.dumps({"action": "complete_layer", "layer": 0,
                               "label": "forged_l"})
        return json.dumps({"action": "complete_object", "label": "done"})

    with pytest.raises(Exception):
        P.run_path_print("a tiny base", out_dir=str(tmp_path),
                         guided=True, build_one=forge, retries=0,
                         emit=events.append)
    phases = [e.get("phase") for e in events
              if e.get("type") == "reject"]
    assert phases == ["goal"]


def test_builder_receives_updated_toolhead(tmp_path, monkeypatch):    # §11.14: every builder call sees the state the previous commit left.
    P = _mock_think(monkeypatch, [BOX])
    seen = []

    def echo(ctx):
        seen.append((list(ctx["toolhead"]), dict(ctx.get("suggestion") or {})))
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    P.run_path_print("a base", out_dir=str(tmp_path),
                     guided=True, build_one=echo)
    assert seen, "builder was never asked"
    assert seen[0][0] == [-1.0, 0.0, -0.5], (
        "first call must see post-travel toolhead, not the origin")
    for toolhead, sug in seen:
        # every call's state is self-consistent: a suggested deposit starts
        # exactly at the toolhead handed to that same call (engine travel
        # between calls is invisible here, so chain across it is not
        # asserted — the per-call consistency is the §11.14 invariant).
        if sug.get("action") == "deposit_segment":
            assert sug["from"] == toolhead, (sug, toolhead)


def test_noncar_plans_skip_repair(tmp_path, monkeypatch):
    # Non-car prompts never receive car components: a house plan passes
    # through byte-identical with no repair events.
    P = _mock_think(monkeypatch, [
        {"action": "add_box", "width": 3, "height": 0.4, "depth": 2,
         "position": [0, 0.2, 0], "color": "0x777777", "label": "base_h1"}])
    events = []
    rep = P.run_path_print("a small house", out_dir=str(tmp_path),
                           emit=events.append)
    assert rep["ok"] is True
    assert not [e for e in events if e.get("type") == "repair"]
    assert all("wheel" not in a.get("label", "") for a in
               json.load(open(os.path.join(str(tmp_path), "actions.json"))))
    assert rep.get("repairs", []) == []


def test_printer_rejects_primitive_then_prints(tmp_path, monkeypatch):
    # Printer mode end-to-end: a schema-valid add_box proposal is refused
    # pre-execution (never converted), the toolhead and scene files are
    # untouched by the rejection, then a valid segment commits and renders.
    P = _mock_think(monkeypatch, [BOX])
    events = []
    toolheads = []
    calls = {"n": 0}

    def fake(ctx):
        calls["n"] += 1
        toolheads.append(list(ctx["toolhead"]))
        if calls["n"] == 1:
            return json.dumps({"action": "add_box", "width": 1, "height": 1,
                               "depth": 1, "position": [0, 0.5, 0],
                               "color": "0xFF0000", "label": "primitive_box"})
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    rep = P.run_path_print("a base", out_dir=str(tmp_path),
                           guided=True, build_one=fake, emit=events.append)
    assert rep["ok"] is True and rep["deposits"] > 0
    forb = [e for e in events if e.get("type") == "reject"
            and "forbidden action 'add_box'" in "; ".join(e.get("errors", []))]
    assert len(forb) == 1
    # rejection preserved toolhead: next call sees the same position
    assert toolheads[0] == toolheads[1]
    # nothing primitive was committed or rendered
    acts = json.load(open(os.path.join(str(tmp_path), "actions.json")))
    assert all(a.get("action") != "add_box" for a in acts)
    scene = open(os.path.join(str(tmp_path), "scene.js"),
                 encoding="utf-8").read()
    assert "primitive_box" not in scene
    assert "seg_" in scene
