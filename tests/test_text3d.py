# tests/test_text3d.py
# Text -> realistic 3D (morphix_text3d): subject/family routing, modifier
# words bound to the right parts, honest refusal for subjects without a
# builder, and an end-to-end build that prints and verifies.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import morphix_text3d as T  # noqa: E402
from morphix_config import PipelineError  # noqa: E402


@pytest.mark.parametrize("prompt,cat,fam,preset", [
    ("a red sports car with big wheels", "car", "vehicle", "sports"),
    ("a black SUV with white roof", "car", "vehicle", "suv"),
    ("a tall wooden chair", "chair", "chair", "dining"),
    ("a coffee table", "table", "table", "coffee"),
    ("a small house with a blue roof", "house", "house", "cottage"),
    ("an orange cat", "cat", "quadruped", "cat"),
    ("human skull", "skull", None, None),
])
def test_subject_routing(prompt, cat, fam, preset):
    c, f, _label = T.understand(prompt)
    assert c == cat and f == fam
    if fam is None:
        with pytest.raises(PipelineError) as e:
            T.build(prompt)
        assert "Image mode" in str(e.value)
    else:
        ir, _ops, _log = T.build(prompt)
        assert ir["method"]["preset"] == preset


def test_modifiers_bind_to_parts():
    ir, ops, _ = T.build("a black SUV with white roof")
    pairs = [(o["op"], o["target"], o.get("color")) for o in ops]
    assert ("color", "all", "0x141414") in pairs
    assert ("color", "roof", "0xF0F0F0") in pairs
    by = {c["id"]: c for c in ir["components"]}
    assert by["car_body"]["color"] == "0x141414"
    assert by["cabin_roof"]["color"] == "0xF0F0F0"
    assert by["wheel_front_left"]["material"] == "tire"          # tires not repainted


def test_big_wheels_grow_arches_not_overlap():
    base, _, _ = T.build("a red sports car")
    big, ops, _ = T.build("a red sports car with big wheels")
    assert not any(o["op"] == "scale" for o in ops)     # built in, not scaled after
    assert big["method"]["params"]["wr"] > base["method"]["params"]["wr"]
    body = next(c for c in big["components"] if c["id"] == "car_body")
    wheel = next(c for c in big["components"] if c["role"] == "wheel")
    top_wheel = wheel["position"][1] + wheel["dims"][1] / 2
    top_body = body["position"][1] + body["dims"][1] / 2
    assert top_wheel < top_body                          # wheel stays in the arch


@pytest.mark.visual
def test_end_to_end_text_build(tmp_path):
    rep = T.run("an orange cat", str(tmp_path))
    run = tmp_path / rep["run_id"]
    assert rep["mode"] == "text-3d" and rep["success"] is True
    checks = {c["name"]: c for c in rep["visual_verification"]["checks"]}
    assert checks["print_matches_model"]["ok"]
    for f in ("object.json", "model.html", "scene.html", "trace/compare.png"):
        assert (run / f).exists(), f
    ir = json.loads((run / "object.json").read_text(encoding="utf-8"))
    torso = next(c for c in ir["components"] if c["id"] == "torso")
    assert torso["primitive"] == "inflate"               # rounded, not a disc
