# tests/test_core.py
# Offline deterministic tests for the Morphix construction core:
# action schema, JS syntax check, anti-hallucination guard, geometry state,
# job cancellation plumbing, and the HTTP API surface (no models needed).
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOOD_BOX = {"action": "add_box", "width": 2, "height": 1, "depth": 2,
            "position": [0, 0.5, 0], "color": "0xFF0000", "label": "box_t1"}

GOOD_JS = (
    "const geo_box_t1 = new THREE.BoxGeometry(2,1,2);\n"
    "const mat_box_t1 = new THREE.MeshStandardMaterial({color:0xFF0000});\n"
    "const mesh_box_t1 = new THREE.Mesh(geo_box_t1,mat_box_t1);\n"
    "mesh_box_t1.position.set(0,0.5,0);\n"
    "mesh_box_t1.castShadow = true;\n"
    "mesh_box_t1.receiveShadow = true;\n"
    "scene.add(mesh_box_t1);"
)


def test_schema_accepts_valid_action():
    from morphix_validator import validate_actions
    assert validate_actions([GOOD_BOX]) == []


def test_schema_rejects_missing_fields():
    from morphix_validator import validate_actions
    errs = validate_actions([{"action": "add_box", "label": "x"}])
    assert any("missing width" in e for e in errs)


def test_schema_rejects_bad_color_and_unknown_action():
    from morphix_validator import validate_actions
    errs = validate_actions([dict(GOOD_BOX, color="#FF0000")])
    assert any("0xRRGGBB" in e for e in errs)
    errs = validate_actions([{"action": "add_tree", "label": "t"}])
    assert any("unknown action" in e for e in errs)


def test_schema_rejects_duplicate_labels():
    from morphix_validator import validate_actions
    errs = validate_actions([GOOD_BOX, dict(GOOD_BOX)])
    assert any("duplicate label" in e for e in errs)


def test_schema_rejects_invalid_labels():
    from morphix_validator import validate_actions
    for bad in ("Stone Foundation", "a b", "9lives", "x" * 41, "dash-ed"):
        errs = validate_actions([dict(GOOD_BOX, label=bad)])
        assert any("invalid label" in e for e in errs), bad
    assert validate_actions([dict(GOOD_BOX, label="ok_label_01")]) == []


def test_js_syntax_clean_and_broken():
    from morphix_validator import validate_js_syntax
    assert validate_js_syntax(GOOD_JS) == []
    # unbalanced + unparseable: node --check or the fallback must flag it
    assert validate_js_syntax("function broken( { scene.add(") != []


def test_guard_accepts_grounded_block():
    from morphix_guard import enforce
    final, status, errors = enforce(GOOD_BOX, GOOD_JS)
    assert status == "OK", errors
    assert errors == []
    assert "mesh_box_t1" in final


def test_guard_rejects_ungrounded_label():
    from morphix_guard import enforce
    final, status, errors = enforce(
        GOOD_BOX, "const x = 1; scene.add(new THREE.Mesh());")
    assert status != "OK" or errors


def test_state_tracks_and_rejects_duplicates_and_collisions():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    assert validate_against_state(GOOD_BOX, s) == []
    s.add(GOOD_BOX)
    assert "box_t1" in s.labels()
    dup = validate_against_state(dict(GOOD_BOX), s)
    assert any("duplicate label" in e for e in dup)
    assert any("collides" in e for e in dup)


def test_state_rejects_out_of_bounds():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    far = dict(GOOD_BOX, label="box_far", position=[999, 0.5, 0])
    assert any("out of bounds" in e for e in validate_against_state(far, s))


def test_suggest_spot_grows_from_origin():
    import math
    from morphix_state import SceneState
    s = SceneState()
    assert s.suggest_spot(dict(GOOD_BOX)) == [0, 0.5, 0]
    s.add(dict(GOOD_BOX, position=[0, 0.5, 0]))
    spot = s.suggest_spot({"action": "add_box", "width": 1, "height": 1,
                           "depth": 1, "label": "next"})
    assert spot is not None
    assert math.dist(spot, [0, 0.5, 0]) <= 8.0


def test_cancel_stops_incremental_before_any_model_call(tmp_path):
    from morphix_config import Cancelled
    from morphix_pipeline import run_incremental
    calls = []

    def fake_emit(ev):
        calls.append(ev.get("type"))

    with pytest.raises(Cancelled):
        run_incremental("a box", builder="ollama", out_dir=str(tmp_path),
                        max_steps=4, emit=fake_emit,
                        should_stop=lambda: True)
    # cancellation is announced on the event stream; no model was reached
    assert "cancelled" in calls


def test_job_cancel_flag_and_snapshot():
    from morphix_jobs import Job
    job = Job("test123")
    assert job.cancel_requested() is False
    assert job.snapshot()["cancelled"] is False
    job.request_cancel()
    assert job.cancel_requested() is True
    assert job.snapshot()["cancelled"] is True


def test_first_piece_must_start_at_origin():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    far = dict(GOOD_BOX, label="far", position=[10, 0.5, 0])
    assert any("origin" in e for e in validate_against_state(far, s))
    high = dict(GOOD_BOX, label="high", position=[0, 5, 0])
    assert any("float" in e for e in validate_against_state(high, s))
    ok = dict(GOOD_BOX, position=[0, 0.5, 0])
    assert validate_against_state(ok, s) == []


def test_floating_and_sinking_rejected_stacking_allowed():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    base = dict(GOOD_BOX, position=[0, 0.5, 0])
    assert validate_against_state(base, s) == []
    s.add(base)
    floater = {"action": "add_box", "width": 1, "height": 1, "depth": 1,
               "position": [0, 5, 0], "color": "0x00FF00", "label": "float"}
    assert any("floats" in e for e in validate_against_state(floater, s))
    sunk = {"action": "add_box", "width": 2, "height": 4, "depth": 2,
            "position": [5, 0.5, 0], "color": "0x00FF00", "label": "sunk"}
    assert any("sinks" in e for e in validate_against_state(sunk, s))
    stacked = s.resolve({"action": "add_box", "width": 1, "height": 1,
                         "depth": 1, "color": "0x00FF00", "label": "top",
                         "on_top_of": "box_t1"})
    assert validate_against_state(stacked, s) == []
    s.add(stacked)
    assert abs(s.top_y(0, 0) - 2.0) < 1e-9


def test_terrain_exempt_from_placement_rules():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    hill = {"action": "add_terrain", "width": 20, "segments": 32,
            "amplitude": 2.0, "frequency": 0.5, "color": "0x70B6C5",
            "label": "hill", "position": [10, 0, 10]}
    assert validate_against_state(hill, s) == []


def test_origin_rule_survives_nonphysical_commits():
    from morphix_state import SceneState
    from morphix_validator import validate_against_state
    s = SceneState()
    s.add({"action": "add_light", "type": "directional", "intensity": 2.0,
           "label": "sun", "position": [10, 15, 8]})
    assert s.has_physical() is False
    far = dict(GOOD_BOX, label="far", position=[10, 0.5, 0])
    assert any("origin" in e for e in validate_against_state(far, s))
    ok = dict(GOOD_BOX, position=[0, 0.5, 0])
    assert validate_against_state(ok, s) == []


def test_done_refused_while_goal_parts_missing(tmp_path, monkeypatch):
    import morphix_pipeline as P
    seen = []

    def fake_think_next(goal, summary, step, feedback="", model=None,
                        options=None, _retried=False):
        seen.append(summary)
        return ({"action": "done", "reason": "all good"}, "MODEL x")

    monkeypatch.setattr(P, "think_next", fake_think_next)
    report = P.run_incremental("a house with a door", builder="ollama",
                               out_dir=str(tmp_path), max_steps=2,
                               emit=lambda e: None)
    # done refused on every attempt (bounded by the attempt ceiling)
    assert len(seen) == max(2 * 4, 2 + 4)
    assert any("missing" in s for s in seen)
    assert report["complete"] is False
    assert "door" in report["goal_missing"]


def test_goal_parts_and_coverage():
    from morphix_thinker import goal_parts, goal_coverage
    parts = goal_parts("a small house with a tree and warm sun light")
    for tok in ("foundation", "wall", "roof", "door", "window",
                "trunk", "leaves", "light"):
        assert tok in parts, parts
    assert goal_parts("something utterly abstract xyzqq") == []
    acts = [{"action": "add_box", "label": "foundation_a1"},
            {"action": "add_light", "label": "sun_a1"}]
    covered, missing = goal_coverage(parts, acts)
    assert "foundation" in covered and "light" in covered
    assert "roof" in missing and "trunk" in missing


def test_build_block_failure_shows_raw_output(monkeypatch):
    import morphix_pipeline as P
    monkeypatch.setattr(P, "_ollama_build", lambda *a, **k: "oops {{{(")
    final, status, errors = P._build_block(
        dict(GOOD_BOX), "ollama", "qwen2.5-coder:7b", None)
    assert final is None
    assert status.endswith("/FAIL")
    assert any("builder wrote:" in e for e in errors)


def test_api_health_shape_and_cancel_unknown_job():
    import server
    client = server.app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    body = json.loads(r.data)
    for key in ("thinker_model", "builder_model", "ready", "strict"):
        assert key in body
    r = client.post("/api/jobs/does-not-exist/cancel")
    assert r.status_code == 404
