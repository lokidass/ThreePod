# tests/test_edit.py
# Prompt-based object editing (morphix_edit): rule parsing, strict
# validation of operations against the object's real parts, geometric
# application, and an end-to-end edit that re-prints a new run.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "image3d"))

import morphix_edit as E  # noqa: E402
from morphix_config import PipelineError  # noqa: E402


@pytest.fixture(scope="module")
def car_ir():
    import morphix_image_seg as S
    import morphix_image_shapes as SH
    import morphix_image3d as I3
    import image3d_fixtures as F
    path = F.make_all()["car_side_platform"]
    seg = S.segment(path)
    paint = SH.paint_from_clusters(seg["subject"]["colors"], "vehicle")
    fit = SH.fit("vehicle", seg["_mask"], paint, budget=150)
    ir = I3.build_ir(I3._manual_observation("car"), seg, "family",
                     "vehicle", fit, "object-only")
    ir["appearance"]["palette"] = I3.palette_for(ir["components"])
    return ir


@pytest.mark.parametrize("prompt,expect", [
    ("paint it red", [("color", "all")]),
    ("make the wheels bigger", [("scale", "wheel")]),
    ("make it 30% taller", [("scale", "all")]),
    ("remove the headlights", [("remove", "light")]),
    ("make the windows black", [("color", "material:glass")]),
    ("paint the body light blue", [("color", "body")]),
    ("make the roof white, wheels twice as big",
     [("color", "roof"), ("scale", "wheel")]),
])
def test_rule_parsing(car_ir, prompt, expect):
    ops = E.parse_rules(prompt, car_ir)
    assert [(o["op"], o["target"]) for o in ops] == expect


def test_factors(car_ir):
    assert E.parse_rules("make it 30% taller", car_ir)[0]["factor"] == 1.3
    assert E.parse_rules("make the wheels twice as big", car_ir)[0]["factor"] == 2.0
    assert E.parse_rules("make it half the size", car_ir)[0]["factor"] == 0.5


@pytest.mark.parametrize("ops,msg", [
    ([{"op": "remove", "target": "spaceship"}], "no part"),
    ([{"op": "color", "target": "all", "color": "red"}], "bad colour"),
    ([{"op": "scale", "target": "all", "axis": "all", "factor": 40}], "outside"),
    ([{"op": "remove", "target": "all"}], "whole object"),
    ([{"op": "teleport", "target": "all"}], "unknown edit"),
])
def test_invalid_ops_rejected(car_ir, ops, msg):
    with pytest.raises(PipelineError) as e:
        E.validate_ops(ops, car_ir)
    assert msg in str(e.value)


def test_apply_paint_scale_remove(car_ir):
    before = json.dumps(car_ir, sort_keys=True)
    ops = E.validate_ops(E.parse_rules(
        "make the wheels bigger and paint it red and remove the headlights",
        car_ir), car_ir)
    new, log = E.apply_ops(car_ir, ops)
    by = {c["id"]: c for c in new["components"]}
    assert "headlight_left" not in by
    assert by["car_body"]["color"] == "0xC83232"
    assert by["wheel_front_left"]["material"] == "tire"
    assert by["wheel_front_left"]["color"] != "0xC83232"   # tires not painted
    old_w = next(c for c in car_ir["components"] if c["id"] == "wheel_front_left")
    assert by["wheel_front_left"]["dims"][1] == pytest.approx(old_w["dims"][1] * 1.25, rel=1e-3)
    import morphix_image_shapes as SH
    lo, _ = SH.scene_bounds(new["components"])
    assert abs(lo[1]) < 1e-3                              # still on the ground
    from morphix_ir import validate_ir
    assert validate_ir(new) == []
    assert json.dumps(car_ir, sort_keys=True) == before   # original untouched


def test_paint_tints_photo_texture():
    ir = {"components": [{"id": "silhouette_body", "role": "body",
                          "primitive": "box", "dims": [1, 1, 1],
                          "position": [0, 0.5, 0], "material": "body",
                          "color": "0xEEEEEE", "texture": "photo"}]}
    new, _ = E.apply_ops(ir, [{"op": "color", "target": "all", "color": "0xC83232"}])
    c = new["components"][0]
    assert c["texture"] == "photo" and c["tint"] == "0xC83232"


def test_edit_run_creates_new_verified_version(car_ir, tmp_path):
    base = tmp_path / "20260101_000000"
    base.mkdir()
    (base / "object.json").write_text(json.dumps(car_ir), encoding="utf-8")
    rep = E.edit_run(str(tmp_path), base.name, "make it 20% taller and paint it blue")
    new = tmp_path / rep["run_id"]
    assert rep["mode"] == "edit" and rep["base_run"] == base.name
    assert rep["success"] is True and rep["complete"] is True
    for f in ("object.json", "report.json", "scene.html", "model.html"):
        assert (new / f).exists(), f
    obj = json.loads((new / "object.json").read_text(encoding="utf-8"))
    assert obj["edit_history"][-1]["prompt"].startswith("make it 20%")
    assert json.loads((base / "object.json").read_text(encoding="utf-8")) == car_ir


def test_edit_run_requires_an_object(tmp_path):
    (tmp_path / "scene_run").mkdir()
    with pytest.raises(PipelineError) as e:
        E.edit_run(str(tmp_path), "scene_run", "paint it red")
    assert "no editable object" in str(e.value)
