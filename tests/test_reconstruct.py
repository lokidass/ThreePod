# tests/test_reconstruct.py
# Semantic image -> 3D: a photo becomes a small set of named components via the
# canonical IR — NOT a grid of px_ columns — and image relief stays a separate
# pipeline. All model access is mocked; the geometry is deterministic.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_car_image(path):
    """A wide body with two dark wheels on a light background."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (400, 200), (240, 240, 240))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 80, 360, 160], fill=(200, 40, 40))
    dr.ellipse([80, 140, 140, 200], fill=(15, 15, 15))
    dr.ellipse([260, 140, 320, 200], fill=(15, 15, 15))
    im.save(path)


@pytest.fixture(scope="module")
def car_image(tmp_path_factory):
    d = tmp_path_factory.mktemp("img")
    p = d / "car.png"
    _make_car_image(str(p))
    return str(p)


@pytest.fixture(scope="module")
def demo_report(car_image, tmp_path_factory):
    """Semantic reconstruction with NO vision model (documented demo path)."""
    import morphix_reconstruct as R
    old = (R.available_vision_model, R.VISION_MODEL, R.ollama_generate)
    R.available_vision_model = lambda: ""
    R.VISION_MODEL = ""
    try:
        out = str(tmp_path_factory.mktemp("demo"))
        rep = R.run_reconstruction(car_image, out_dir=out)
    finally:
        R.available_vision_model, R.VISION_MODEL, R.ollama_generate = old
    rep["_out_dir"] = out
    return rep


@pytest.fixture(scope="module")
def vision_report(car_image, tmp_path_factory):
    """Semantic reconstruction with a mocked vision model."""
    import morphix_reconstruct as R
    old = (R.available_vision_model, R.VISION_MODEL, R.ollama_generate)
    R.available_vision_model = lambda: "moondream"
    R.VISION_MODEL = ""

    def fake(model, system, prompt, images=None, timeout=None, options=None):
        return json.dumps({"category": "car", "confidence": 0.82,
                           "parts": ["body", "wheel", "window"],
                           "hidden": ["engine"]})
    R.ollama_generate = fake
    try:
        out = tmp_path_factory.mktemp("vision")
        rep = R.run_reconstruction(car_image, out_dir=str(out))
        rep["_out_dir"] = str(out)
        return rep
    finally:
        R.available_vision_model, R.VISION_MODEL, R.ollama_generate = old


# ---------------------------------------------------------------- vision path
def test_vision_reconstruction_identifies_and_verifies(vision_report):
    r = vision_report
    assert r["mode"] == "image-reconstruct"
    assert r["image"]["source"] == "image-vision"
    assert r["image"]["category"] == "car"
    assert r["image"]["confidence"] == pytest.approx(0.82)
    assert r["ok"] and r["complete"] and r["quality_accepted"]
    comps = r["components"]
    assert comps and all(c.get("verified") for c in comps)


def test_car_photo_is_not_a_grid_of_columns(vision_report):
    r = vision_report
    labels = [c["component"] for c in r["components"]]
    assert labels and not any(l.startswith("px_") for l in labels)
    assert len(labels) < 20                       # semantic parts, not per-pixel
    wheels = [l for l in labels if "wheel" in l]
    assert len(wheels) == 4
    assert len(set(wheels)) == 4
    assert any("body" in l for l in labels)
    assert any("window" in l for l in labels)


def test_ir_records_observed_parts_and_assumptions(vision_report):
    ir = vision_report["ir"]
    assert ir["category"] == "car"
    assert "body" in ir["observed"]["visible_parts"]
    assert ir["assumptions"]                     # hidden geometry is documented
    assert any("not visible" in a for a in ir["assumptions"])


# ---------------------------------------------------------------- demo path
def test_no_vision_model_is_reported_clearly(demo_report):
    r = demo_report
    assert r["image"]["source"] == "image-demo"
    assert r["image"]["vision_available"] is False
    assert any("no vision model" in w for w in r["warnings"])
    # still a real volumetric object, and still deterministic
    labels = [c["component"] for c in r["components"]]
    assert labels and not any(l.startswith("px_") for l in labels)


def test_demo_path_still_fits_a_car_silhouette(demo_report):
    # the deterministic demo classifier notices two wheels on a wide body
    assert demo_report["image"]["category"] == "car"
    fid = demo_report["image_fidelity"]
    assert fid["verified"]
    names = {c["name"] for c in fid["checks"]}
    assert "not_a_voxel_grid" in names
    # a single photo cannot determine occluded surfaces: reported, not guessed
    assert fid["unsupported"]


# ---------------------------------------------------------------- pipelines
def test_relief_and_semantic_use_different_pipelines(car_image, tmp_path):
    from morphix_image import image_to_actions, run_image
    import morphix_reconstruct as R

    relief_actions, meta = image_to_actions(car_image, grid=8, mode="relief")
    assert relief_actions
    assert all(a["label"].startswith("px_") for a in relief_actions)

    relief_report = run_image(car_image, os.path.join(str(tmp_path), "relief"),
                              8, "relief")
    assert relief_report["mode"] == "image-relief"

    old = (R.available_vision_model, R.VISION_MODEL)
    R.available_vision_model, R.VISION_MODEL = (lambda: "", "")
    try:
        sem = R.run_reconstruction(car_image,
                                   out_dir=os.path.join(str(tmp_path), "sem"))
    finally:
        R.available_vision_model, R.VISION_MODEL = old
    assert sem["mode"] == "image-reconstruct"
    assert not any(c["component"].startswith("px_") for c in sem["components"])
    # the two pipelines produce structurally different scenes
    assert len(sem["components"]) < len(relief_actions)


# ---------------------------------------------------------------- honesty
def test_invalid_geometry_cannot_pass_final_quality(tmp_path):
    import morphix_ir as IR
    import morphix_pipeline as P
    from morphix_config import PipelineError

    actions = IR.ir_to_actions(IR.ir_for_category("car"))
    # a "car" whose body is a 5-unit-tall slab is reported and REJECTED by the
    # final quality gate (it is never silently accepted as a good car)
    actions[4]["height"] = 5.0
    rep = P.run_path_print("broken car", out_dir=str(tmp_path),
                           plan_actions=actions, spec_name="car",
                           segment_cap=10000, stream_scene=False)
    assert rep["fidelity_warnings"]                 # reported loudly
    assert rep["quality_accepted"] is False         # never accepted
    assert any(not c.get("verified") for c in rep["components"])


def test_missing_required_component_is_a_hard_error(tmp_path):
    import morphix_ir as IR
    import morphix_pipeline as P
    from morphix_config import PipelineError

    actions = [a for a in IR.ir_to_actions(IR.ir_for_category("car"))
               if "wheel" not in a["label"]]
    with pytest.raises(PipelineError):
        P.run_path_print("car with no wheels", out_dir=str(tmp_path),
                         plan_actions=actions, spec_name="car",
                         stream_scene=False)


def test_scene_reproduces_from_committed_state(demo_report):
    from morphix_path import PathState
    r = demo_report
    out_dir = r["_out_dir"]
    with open(os.path.join(out_dir, "actions.json"), encoding="utf-8") as f:
        actions = json.load(f)
    st = PathState(total_layers=r["layers"])
    deposits = 0
    for act in actions:
        rec = st.apply(act)
        if rec.get("action") in ("deposit_segment", "deposit_point"):
            deposits += 1
    assert deposits == r["deposits"]
    assert st.object_done


# ---------------------------------------------------------------- fast units
def test_describe_object_parses_json(monkeypatch):
    import morphix_reconstruct as R
    monkeypatch.setattr(R, "ollama_generate", lambda *a, **k: (
        "Sure! {\"category\": \"Sedan!\", \"confidence\": 0.7, "
        "\"parts\": [\"body\"], \"hidden\": [\"engine\"]}"))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        _make_car_image(fh.name)
        det = R.describe_object(fh.name, "fake")
    os.unlink(fh.name)
    assert det["category"] == "sedan"                # cleaned + lowercased
    assert det["parts"] == ["body"]
    assert det["confidence"] == pytest.approx(0.7)


def test_describe_object_rejects_unusable_output(monkeypatch):
    import morphix_reconstruct as R
    monkeypatch.setattr(R, "ollama_generate", lambda *a, **k: "???")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        _make_car_image(fh.name)
        det = R.describe_object(fh.name, "fake")
    os.unlink(fh.name)
    assert det is None


def test_demo_category_notices_wheels_on_a_wide_body():
    import morphix_reconstruct as R
    assert R.demo_category({"aspect": 2.4, "dark_blobs": 2}) == "car"
    assert R.demo_category({"aspect": 1.0, "dark_blobs": 0}) == "object"
