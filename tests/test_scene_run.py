# tests/test_scene_run.py
# Completion state machine (Phase 4.1): explicit PLANNING -> ... ->
# COMPLETE | FAILED transitions, no false COMPLETE, bounded repairs,
# preserved partials. Stub renderer for determinism + one live run.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _design_graph(ir):
    """Canned live-style graph with geometrically coherent bboxes derived
    from the IR design geometry (axis-aligned parts; rotations ignored —
    sufficient for contact/bounds reasoning in stub renders)."""
    graph = {}
    for n in ir["nodes"]:
        p = n["transform"]["position"]
        geo = n.get("geometry") or {}
        prm = geo.get("params", {})
        if geo.get("generator"):
            box = [[-45, -3, -45], [45, 6, 45]]
        elif geo.get("ctor") == "BoxGeometry":
            w, h, d = prm["w"], prm["h"], prm["d"]
            box = [[p[0] - w / 2, p[1] - h / 2, p[2] - d / 2],
                   [p[0] + w / 2, p[1] + h / 2, p[2] + d / 2]]
        elif geo.get("ctor") in ("CylinderGeometry", "ConeGeometry"):
            r = max(prm.get("rt", prm.get("r", 0.5)),
                    prm.get("rb", prm.get("r", 0.5)))
            h = prm["h"]
            box = [[p[0] - r, p[1] - h / 2, p[2] - r],
                   [p[0] + r, p[1] + h / 2, p[2] + r]]
        elif geo.get("ctor") == "SphereGeometry":
            r = prm["r"]
            box = [[p[0] - r, p[1] - r, p[2] - r],
                   [p[0] + r, p[1] + r, p[2] + r]]
        else:
            box = [[p[0] - 1, p[1] - 1, p[2] - 1],
                   [p[0] + 1, p[1] + 1, p[2] + 1]]
        graph[n["id"]] = {"present": True, "empty": False, "visible": True,
                          "bbox": box}
    return graph


def _stub_result(ir, png=None):
    import morphix_scene_ir as S
    return {"ready": True, "scene_error": None, "console_errors": [],
            "stats": {"drawCalls": 4, "triangles": 2000,
                      "instances": S.instance_total(ir)},
            "graph": _design_graph(ir), "png": png, "elapsed_s": 0.1}


def test_stub_run_completes_with_evidence(tmp_path):
    import morphix_scene_plan as P
    import morphix_scene_run as R
    png = os.path.join(str(tmp_path), "r.png")
    # stub screenshot: bright frame so subject_lit passes if framed
    from PIL import Image
    Image.new("RGB", (640, 480), (200, 200, 200)).save(png)
    routed = P.route_systems("a wooden table in a room")
    ir, _ = P.compose_routed_scene(routed, {"objective": "t"}, seed=11)

    rep = R.run_scene(
        {"prompt": "a wooden table in a room"}, seed=11,
        render_fn=lambda js: _stub_result(ir, png=png),
        out_dir=str(tmp_path))
    assert rep["state"] == "COMPLETE" and rep["ok"] is True
    assert [t["state"] for t in rep["transitions"]][:4] == \
        ["PLANNING", "GENERATING", "COMPILING", "RENDERING"]
    assert rep["transitions"][-1]["state"] == "COMPLETE"
    assert os.path.isfile(os.path.join(str(tmp_path), "scene_ir.json"))
    assert os.path.isfile(os.path.join(str(tmp_path), "scene.js"))


def test_planning_retry_recovers_coverage_then_completes(tmp_path,
                                                            monkeypatch):
    """Thinker invents an unresolvable requirement first: the state
    machine re-plans once with the failure as feedback instead of dying."""
    import morphix_scene_plan as P
    import morphix_scene_run as R
    calls = {"n": 0}

    def fake_plan(prompt, model, systems=()):
        calls["n"] += 1
        req = ["dragon_castle_xyz"] if calls["n"] == 1 else []
        return {"version": 2, "objective": "o", "style": "s",
                "subject": "s", "scene_type": "other",
                "backend": "webgl", "templates": [],
                "required": req, "freestyle": [],
                "seed": 11, "animations": False,
                "substitutions": {}}

    monkeypatch.setattr(P, "plan_scene", fake_plan)
    rep = R.run_scene({"prompt": "a wooden table in a room"},
                      thinker_model="mock", seed=11,
                      render_fn=None, out_dir=str(tmp_path))
    assert calls["n"] == 2
    assert rep["state"] == "NEEDS_RENDER"
    assert any("re-plan" in r for r in rep.get("plan_repairs", []))


def test_unresolvable_thinker_requirements_fall_back_to_router(tmp_path):
    """A thinker that insists on unresolvable requirements must not kill
    the run: one re-plan, then the router-only deterministic fallback —
    everything logged, nothing silently substituted."""
    import morphix_scene_plan as P
    import morphix_scene_run as R

    def stubborn(prompt, model, systems=()):
        return {"version": 2, "objective": "o", "style": "s",
                "subject": "s", "scene_type": "other",
                "backend": "webgl", "templates": [],
                "required": ["dragon_castle_xyz"], "freestyle": [],
                "seed": 11, "animations": False, "substitutions": {}}

    orig = P.plan_scene
    P.plan_scene = stubborn
    try:
        rep = R.run_scene({"prompt": "a wooden table in a room"},
                          thinker_model="mock", seed=11,
                          render_fn=None, out_dir=str(tmp_path))
    finally:
        P.plan_scene = orig
    assert rep["state"] == "NEEDS_RENDER"
    assert any("router-only" in r for r in rep.get("plan_repairs", []))
    assert rep["required_ids"], "router requirements still tracked"


def test_empty_prompt_fails_at_planning(tmp_path):
    import morphix_scene_run as R
    rep = R.run_scene({"prompt": "   "}, out_dir=str(tmp_path))
    assert rep["state"] == "FAILED" and rep["stage"] == "planning"


def test_gibberish_prompt_fails_loudly_never_silently(tmp_path):
    import morphix_scene_run as R
    rep = R.run_scene({"prompt": "some random xyzzy bildings"},
                      out_dir=str(tmp_path))
    assert rep["state"] == "FAILED"
    assert rep["ok"] is False


def test_render_failure_is_failed_not_complete(tmp_path):
    import morphix_scene_run as R

    def bad_render(js):
        return {"ready": False,
                "scene_error": "Uncaught ReferenceError: nope",
                "console_errors": [], "stats": {}, "png": None,
                "elapsed_s": 0.1, "graph": {}}

    rep = R.run_scene({"prompt": "a wooden table in a room"},
                      render_fn=bad_render, out_dir=str(tmp_path))
    assert rep["state"] == "FAILED"
    assert rep["ok"] is False
    assert os.path.isfile(os.path.join(str(tmp_path), "scene_ir.json"))


def test_unlit_subject_triggers_light_repair(tmp_path):
    import morphix_scene_run as R
    from PIL import Image
    black = os.path.join(str(tmp_path), "black.png")
    Image.new("RGB", (640, 480), (2, 2, 3)).save(black)
    import morphix_scene_plan as P
    routed = P.route_systems("a wooden table in a room")
    ir, _ = P.compose_routed_scene(routed, {"objective": "t"}, seed=11)

    def dark_render(js):
        import morphix_scene_ir as S
        return {"ready": True, "scene_error": None, "console_errors": [],
                "stats": {"drawCalls": 4, "triangles": 2000,
                          "instances": S.instance_total(ir)},
                "graph": _design_graph(ir), "png": black, "elapsed_s": 0.1}

    rep = R.run_scene({"prompt": "a wooden table in a room"},
                      render_fn=dark_render, out_dir=str(tmp_path),
                      max_repairs=1)
    # the stub stays dark, so the honest terminal state is FAILED — but
    # only AFTER the light repair was attempted and re-rendered
    assert rep["state"] == "FAILED", rep
    assert any("light" in r for r in rep.get("repairs", [])), rep


def test_live_run_completes_table_room(tmp_path):
    import morphix_scene_run as R
    from scene_browser import render_scenes
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "shots"), exist_ok=True)

    def live(js):
        png = os.path.join(out, "shots", "live.png")
        r = render_scenes([("live", js, png)])["live"]
        return r

    rep = R.run_scene({"prompt": "a wooden table in a room"}, seed=11,
                      render_fn=live, out_dir=out)
    assert rep["state"] == "COMPLETE", rep
    assert rep["ok"] is True
    assert len(rep["required_ids"]) >= 8
