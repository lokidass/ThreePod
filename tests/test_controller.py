# tests/test_controller.py
# Regression suite for the run controller: deadlines, reject budgets,
# checkpoints/resume, rejection purity, and the success gate.
# No live model inference: frozen plans (plan_actions) and scripted
# build_one fakes drive every test.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import morphix_pipeline as P

BOX = {"action": "add_box", "width": 2.0, "height": 0.4, "depth": 1.0,
       "position": [0, 0.2, 0], "color": "0xFF0000", "label": "base_a1"}


def _box_actions():
    return [dict(BOX)]


def test_timeout_writes_partial_report_and_resumes_without_dupes(tmp_path):
    # deadline_s=-1 expires on the first check: nothing commits, but the
    # checkpoint and an explicit INCOMPLETE report must exist.
    with pytest.raises(Exception, match="deadline exceeded"):
        P.run_path_print("a flat base", out_dir=str(tmp_path),
                         plan_actions=_box_actions(), spec_name=None,
                         stream_scene=False, deadline_s=-1)
    rep = json.load(open(os.path.join(str(tmp_path), "report.json")))
    assert rep["ok"] is False and rep["complete"] is False
    assert rep["success"] is False and rep["status"] == "incomplete"
    assert rep["remaining_actions"] == rep["total_actions"] > 0
    assert os.path.isfile(os.path.join(str(tmp_path), "path.json"))
    # resume finishes the same script with no duplicated deposits
    rep2 = P.run_path_print("a flat base", out_dir=str(tmp_path),
                            plan_actions=_box_actions(), spec_name=None,
                            stream_scene=False, deadline_s=1800,
                            resume=True)
    assert rep2["success"] is True
    fresh = os.path.join(str(tmp_path), "fresh.json")
    rep3 = P.run_path_print("a flat base", out_dir=str(tmp_path) + "_f",
                            plan_actions=_box_actions(), spec_name=None,
                            stream_scene=False, deadline_s=1800)
    assert rep2["deposits"] == rep3["deposits"] > 0
    assert rep2["path_actions"] == rep3["path_actions"]


def test_guided_reject_budget_terminates(tmp_path):
    calls = {"n": 0}

    def bad(ctx):
        calls["n"] += 1
        return json.dumps({"action": "deposit_segment",
                           "from": [9, 9, 9], "to": [9.2, 9, 9],
                           "bead_width": 0.2, "bead_height": 0.2,
                           "material": "body", "layer": 0,
                           "label": f"evil_{calls['n']:02d}"})

    with pytest.raises(Exception, match="without converging|attempt ceiling"):
        P.run_path_print("a flat base", out_dir=str(tmp_path),
                         plan_actions=_box_actions(), spec_name=None,
                         stream_scene=False, guided=True, build_one=bad,
                         retries=10, max_rejects=5)
    # bounded: a handful of builder calls, not an endless loop
    assert calls["n"] <= 8
    rep = json.load(open(os.path.join(str(tmp_path), "report.json")))
    assert rep["success"] is False and rep["status"] == "incomplete"


def test_flaky_builder_recovers_and_matches_clean_run(tmp_path):
    def echo(ctx):
        sug = ctx.get("suggestion")
        if sug is not None:
            return json.dumps(sug)
        return json.dumps({"action": "complete_object", "label": "obj_done"})

    calls = {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"action": "deposit_segment",
                               "from": [9, 9, 9], "to": [9.2, 9, 9],
                               "bead_width": 0.2, "bead_height": 0.2,
                               "material": "body", "layer": 0,
                               "label": "evil_01"})
        return echo(ctx)

    rep_flaky = P.run_path_print("a flat base", out_dir=str(tmp_path) + "_f",
                                 plan_actions=_box_actions(), spec_name=None,
                                 stream_scene=False, guided=True,
                                 build_one=flaky)
    rep_clean = P.run_path_print("a flat base", out_dir=str(tmp_path) + "_c",
                                 plan_actions=_box_actions(), spec_name=None,
                                 stream_scene=False, guided=True,
                                 build_one=echo)
    # the rejected proposal left no trace: identical geometry
    assert rep_flaky["success"] is True
    js_f = open(os.path.join(str(tmp_path) + "_f", "scene.js")).read()
    js_c = open(os.path.join(str(tmp_path) + "_c", "scene.js")).read()
    assert js_f == js_c


def test_success_impossible_when_quality_fails_but_execution_passes(tmp_path):
    # floating sphere: schema-valid, slices fine, but the box_sphere spec
    # warns -> quality_accepted False while execution ok stays True.
    box = {"action": "add_box", "width": 2.0, "height": 0.4, "depth": 1.0,
           "position": [0, 0.2, 0], "color": "0xFF0000", "label": "box_a1"}
    ball = {"action": "add_sphere", "radius": 0.5,
            "position": [0, 3.0, 0], "color": "0xFF0000", "label": "ball_a1"}
    rep = P.run_path_print("box and ball", out_dir=str(tmp_path),
                           plan_actions=[box, ball], spec_name="box_sphere",
                           stream_scene=False, deadline_s=1800)
    assert rep["ok"] is True, "execution (syntax/geometry) is valid"
    assert rep["quality_accepted"] is False, "fidelity must veto acceptance"
    assert rep["success"] is False, "success requires quality acceptance"
