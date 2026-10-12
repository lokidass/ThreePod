# tests/test_vision.py
# Vision inspector: verdict parsing, streak warnings, advisory-only wiring
# (vision never fails a run), screenshot capture.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOX = {"action": "add_box", "width": 2.0, "height": 0.4, "depth": 1.0,
       "position": [0, 0.2, 0], "color": "0xFF0000", "label": "base_a1"}


def test_track_verdict_streak():
    from morphix_vision import track_verdict
    s, w = track_verdict(0, "GOOD")
    assert (s, w) == (0, "")
    s, w = track_verdict(0, "BROKEN")
    assert s == 1 and w == ""
    s, w = track_verdict(1, "BROKEN")
    assert s == 2 and "Stop" in w
    s, w = track_verdict(2, "DRIFTING")
    assert (s, w) == (0, "")


def test_judge_parses_verdict_and_note(monkeypatch):
    import morphix_vision as V
    seen = {}

    def fake_generate(model, system, prompt, images=None, timeout=None):
        seen["images"] = images
        return "DRIFTING\nwest wall leans outward"
    monkeypatch.setattr(V, "ollama_generate", fake_generate)
    monkeypatch.setattr(V, "available_vision_model", lambda: "moondream:latest")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"fakepng")
        png = f.name
    try:
        res = V.judge_print(png, "a house", 12, 2)
    finally:
        os.unlink(png)
    assert res["available"] is True
    assert res["verdict"] == "DRIFTING"
    assert "leans" in res["note"]
    assert seen["images"] and len(seen["images"]) == 1


def test_verdict_parser_shapes():
    from morphix_vision import _parse_verdict, _parse_note
    assert _parse_verdict("GOOD on track, beads visible") == "GOOD"
    assert _parse_verdict("broken beyond repair, empty view") == "BROKEN"
    assert _parse_verdict("nothing to say here") == "UNKNOWN"
    assert _parse_verdict("VERDICT: DRIFTING\nNOTE: leaning") == "DRIFTING"
    assert "leaning" in _parse_note("VERDICT: DRIFTING\nNOTE: leaning")
    assert _parse_verdict("A\ngeometry matches the goal") == "GOOD"
    assert _parse_verdict("(b) placement looks off") == "DRIFTING"
    assert _parse_verdict("C. viewport is empty") == "BROKEN"
    # article 'a' must never count as a verdict letter (cue fallback may
    # still judge the sentence on its merits, tested below)
    assert _parse_verdict("A yellow box sits on the grid") == "UNKNOWN"


def test_cue_fallback_on_real_moondream_rambles():
    from morphix_vision import _parse_verdict
    # verbatim shapes observed live (moondream 1.7b, clean box print)
    assert _parse_verdict("This is a print going well.") == "GOOD"
    assert _parse_verdict(
        "This is a print going well. The image features a 3D rendering "
        "of a yellow brick box on a grid-like surface.") == "GOOD"
    # render-quality complaint: negative cues win (advisory-only anyway)
    assert _parse_verdict(
        "The print quality is not very good, a bad print, "
        "not very well printed.") == "BROKEN"
    # balanced/empty text stays honest
    assert _parse_verdict("A box.") == "UNKNOWN"
    # negated verdict words flip instead of misfiring
    assert _parse_verdict("The result is not very good.") == "BROKEN"
    assert _parse_verdict("It is not broken at all.") == "GOOD"


def test_judge_handles_garbage_and_outages(monkeypatch):
    import morphix_vision as V
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"x")
        png = f.name
    try:
        monkeypatch.setattr(V, "available_vision_model", lambda: "")
        assert V.judge_print(png, "g", 1, 1)["available"] is False
        monkeypatch.setattr(V, "available_vision_model",
                            lambda: "moondream:latest")
        monkeypatch.setattr(V, "ollama_generate",
                            lambda *a, **k: "some rambling without format")
        assert V.judge_print(png, "g", 1, 1)["verdict"] == "UNKNOWN"

        def boom(*a, **k):
            raise RuntimeError("ollama down")
        monkeypatch.setattr(V, "ollama_generate", boom)
        assert V.judge_print(png, "g", 1, 1)["verdict"] == "UNKNOWN"
    finally:
        os.unlink(png)


def test_screenshot_captures_page(tmp_path):
    from morphix_vision import screenshot_scene
    out = str(tmp_path / "shot.png")
    screenshot_scene("data:text/html,<html><body><h1>print</h1></body></html>",
                     out, wait_ms=500)
    assert os.path.isfile(out) and os.path.getsize(out) > 1000


def _run_print(monkeypatch, tmp_path, vision_every, judge=None,
               shot_fail=False):
    import morphix_pipeline as P
    import morphix_vision as V
    monkeypatch.setattr(
        P, "think", lambda *a, **k: ([dict(BOX)], "MODEL x"))

    def fake_shot(url, out_png, **kw):
        if shot_fail:
            raise RuntimeError("no browser")
        with open(out_png, "wb") as f:
            f.write(b"png")
        return out_png

    monkeypatch.setattr(V, "screenshot_scene", fake_shot)
    if judge is not None:
        monkeypatch.setattr(V, "judge_print", judge)
    events = []
    rep = P.run_path_print("a base", out_dir=str(tmp_path),
                           vision_every=vision_every,
                           emit=events.append)
    return rep, events


def test_vision_events_stream_and_land_in_report(tmp_path, monkeypatch):
    rep, events = _run_print(
        monkeypatch, tmp_path, 1000,
        judge=lambda *a, **k: {"available": True, "model": "m",
                               "verdict": "GOOD", "note": "on track"})
    assert rep["ok"] is True
    visions = [e for e in events if e["type"] == "vision"]
    assert len(visions) == 1  # final forced inspection only
    assert visions[0]["verdict"] == "GOOD"
    assert rep["vision"] and rep["vision"][0]["verdict"] == "GOOD"
    shots = [f for f in os.listdir(str(tmp_path)) if f.endswith(".png")]
    assert len(shots) == 1
    inspect = os.path.join(str(tmp_path), "inspect.html")
    assert os.path.isfile(inspect)
    html = open(inspect, encoding="utf-8").read()
    assert "lookAt(0, 0.25, 0)" in html and "GridHelper" in html


def test_broken_twice_warns_but_never_fails(tmp_path, monkeypatch):
    rep, events = _run_print(
        monkeypatch, tmp_path, 1,
        judge=lambda *a, **k: {"available": True, "model": "m",
                               "verdict": "BROKEN", "note": "looks wrong"})
    assert rep["ok"] is True  # advisory only
    warns = [e for e in events if e["type"] == "warn"
             and e.get("phase") == "vision" and "Stop" in e.get("message", "")]
    assert warns
    assert all(v["verdict"] == "BROKEN" for v in rep["vision"])


def test_vision_infra_failure_disables_quietly(tmp_path, monkeypatch):
    rep, events = _run_print(monkeypatch, tmp_path, 1, shot_fail=True)
    assert rep["ok"] is True
    assert rep["vision"] == []
    assert any(e["type"] == "warn" and "disabled" in e.get("message", "")
               for e in events)
