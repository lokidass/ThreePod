# tests/test_scene_verify.py
# Render-and-repair loop: evaluation branches, repair mapping, honest
# verdicts (never success-on-compile-alone), plus one real-browser loop.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ir():
    import morphix_scene_templates as T
    return T.build("fog_benchmark")


def test_evaluate_accepts_clean_render():
    import morphix_scene_verify as V
    ok, findings = V.evaluate_render(_ir(), {
        "ready": True, "scene_error": None, "console_errors": [],
        "stats": {"drawCalls": 6, "triangles": 7514, "instances": 0},
        "elapsed_s": 0.3})
    assert ok and findings == []


def test_evaluate_rejects_each_failure_class():
    import morphix_scene_verify as V
    base = {"ready": True, "scene_error": None, "console_errors": [],
            "stats": {"drawCalls": 6, "triangles": 7514, "instances": 0},
            "elapsed_s": 0.3}
    import copy
    cases = [
        dict(base, ready=False),
        dict(base, scene_error="boom"),
        dict(base, console_errors=["THREE.WebGLProgram: Shader Error 0"]),
        dict(base, stats={"drawCalls": 0, "triangles": 0, "instances": 0}),
        dict(base, stats={"drawCalls": 6, "triangles": 0, "instances": 0}),
    ]
    for res in cases:
        ok, findings = V.evaluate_render(_ir(), res)
        assert not ok and findings, res


def test_evaluate_warns_on_instance_shortfall():
    import morphix_scene_templates as T
    import morphix_scene_verify as V
    ir = T.build("forest")                        # plans 4000 instances
    ok, findings = V.evaluate_render(ir, {
        "ready": True, "scene_error": None, "console_errors": [],
        "stats": {"drawCalls": 4, "triangles": 90000, "instances": 3600},
        "elapsed_s": 0.6})
    assert ok                                     # render itself is fine ...
    assert any(f.get("warning") and "3600 of 4000" in f["detail"]
               for f in findings)                 # ... but reported, not hidden


def test_repair_only_handles_init_failures():
    import morphix_scene_verify as V
    ir = _ir()
    fixed, note = V.repair_for_findings(
        ir, [{"stage": "render", "detail": "webgl context lost"}])
    assert fixed is not None and "safe mode" in note
    fixed, _ = V.repair_for_findings(
        ir, [{"stage": "render", "detail": "webgl context lost"}], safe=True)
    assert fixed is None                           # budget: nothing left
    fixed, _ = V.repair_for_findings(
        ir, [{"stage": "render", "detail": "Shader Error 0"}])
    assert fixed is None                           # shader bugs are reported


def test_loop_succeeds_on_real_browser():
    import morphix_scene_verify as V
    from scene_browser import render_scenes
    import tempfile
    out = os.path.join(tempfile.mkdtemp(), "v.png")

    def render_one(js, pixel_ratio, antialias):
        r = render_scenes([("v", js, out)])
        return r["v"]

    verdict = V.render_and_repair(_ir(), render_one)
    assert verdict["ok"] is True
    assert verdict["repairs"] == []
    assert verdict["attempts"] == 1
    assert verdict["stats"]["drawCalls"] >= 1


def test_loop_never_claims_success_on_errors():
    import morphix_scene_verify as V

    def bad_render(js, pixel_ratio, antialias):
        return {"ready": False,
                "scene_error": "Uncaught ReferenceError: x is not defined",
                "console_errors": [], "stats": {}, "elapsed_s": 0.1}

    verdict = V.render_and_repair(_ir(), bad_render, max_repairs=2)
    assert verdict["ok"] is False
    # nothing repairable about a reference error: no budget burned
    assert verdict["attempts"] == 1
    assert "ReferenceError" in verdict["error"]


def test_loop_exhausts_budget_on_persistent_init_failure():
    import morphix_scene_verify as V

    def dead(js, pixel_ratio, antialias):
        return {"ready": False,
                "scene_error": "failed to create webgl context",
                "console_errors": [], "stats": {}, "elapsed_s": 0.1}

    verdict = V.render_and_repair(_ir(), dead, max_repairs=2,
                                  pixel_ratio=2, antialias=True)
    assert verdict["ok"] is False
    assert verdict["attempts"] == 2                # safe retry, then give up
    assert len(verdict["repairs"]) == 1


def test_loop_repairs_init_failure_then_succeeds():
    import morphix_scene_verify as V
    calls = {"n": 0, "modes": []}

    def flaky(js, pixel_ratio, antialias):
        calls["n"] += 1
        calls["modes"].append((pixel_ratio, antialias))
        if calls["n"] == 1:
            return {"ready": False,
                    "scene_error": "failed to create webgl context",
                    "console_errors": [], "stats": {}, "elapsed_s": 0.1}
        return {"ready": True, "scene_error": None, "console_errors": [],
                "stats": {"drawCalls": 6, "triangles": 7514, "instances": 0},
                "elapsed_s": 0.2}

    verdict = V.render_and_repair(_ir(), flaky, max_repairs=2,
                                  pixel_ratio=2, antialias=True)
    assert verdict["ok"] is True
    assert len(verdict["repairs"]) == 1
    assert calls["n"] == 2
    assert calls["modes"][1] == (1, False)          # retried in safe mode


def test_loop_rejects_invalid_ir_without_rendering():
    import morphix_scene_verify as V
    calls = {"n": 0}

    def never(js, pixel_ratio, antialias):
        calls["n"] += 1
        raise AssertionError("must not render invalid IR")

    verdict = V.render_and_repair({"kind": "scene"}, never)
    assert verdict["ok"] is False and verdict["stage"] == "ir"
    assert calls["n"] == 0
