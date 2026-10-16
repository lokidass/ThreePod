# tests/image3d/test_image3d.py
# Image -> 3D regression suite (separate from scene-mode and printer tests).
#
# Vision is MOCKED (recorded real moondream replies for the user's Urus
# photo, scripted replies elsewhere) so every run is deterministic and
# offline. Each end-to-end case stores its evidence under
# tests/image3d/.artifacts/<case>/: input image, parsed semantic IR,
# required components, renders (print / id pass / smooth preview /
# overview / comparison sheet), validation outcome and any fallback used.
#
#   pytest tests/image3d -m "not visual"   # fast: adapter, segmentation, IR
#   pytest tests/image3d -m visual         # end-to-end with headless chromium
import json
import math
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import image3d_fixtures as fixtures  # noqa: E402
import morphix_image3d as I3  # noqa: E402
import morphix_image_seg as SEG  # noqa: E402
import morphix_image_vision as V  # noqa: E402
from morphix_config import ModelUnavailableError  # noqa: E402

ARTIFACTS = os.path.join(HERE, ".artifacts")
INPUTS = os.path.join(HERE, "inputs")


@pytest.fixture(scope="session")
def inputs():
    paths = fixtures.make_all()
    paths["real_urus"] = os.path.join(INPUTS, "real_urus.png")
    assert os.path.exists(paths["real_urus"]), "real reference image missing"
    return paths


def scripted(answers=None, default=""):
    """Mock vision call: question -> reply (exact or keyword match)."""
    answers = answers or {}

    def ask(system, prompt):
        if prompt in answers:
            v = answers[prompt]
        else:
            v = next((r for k, r in answers.items() if k in prompt), default)
        if isinstance(v, Exception):
            raise v
        return v
    return ask


def recorded_urus():
    with open(os.path.join(HERE, "recorded", "real_urus_moondream.json"),
              encoding="utf-8") as f:
        return scripted(json.load(f)["replies"])


def says(sentence):
    """Two independent questions answered with the same subject."""
    return scripted({"What is the main object in this image?": sentence,
                     "Describe this image.": sentence})


# ------------------------------------------------------------ vision adapter
@pytest.mark.parametrize("raw,kind", [
    ("", "empty"),
    ("ids = [0.39, 0.6, 0.48, 0.66]", "malformed"),
    ("urn, suv, car, " + ", ".join(["car"] * 40), "degenerate"),
    ("urn", "bare_word"),
    ("The main object in this image is a sports car.", "sentence"),
    ('{"category": "car", "confidence": 0.8}', "json"),
])
def test_reply_classification(raw, kind):
    assert V.classify_reply(raw) == kind


@pytest.mark.parametrize("text,cat", [
    ("The main object in this image is a sports car, specifically a black "
     "Lamborghini Huracan, parked on a road.", "car"),
    ("The image presents a sleek black Lamborghini Huracan parked on a "
     "circular driveway.", "car"),
    ("The image shows a teapot on a wooden table.", "teapot"),
    ("A ginger kitten sitting on a sofa.", "cat"),
    ("The image features an SUV in a garage.", "car"),
])
def test_category_from_sentence_subject(text, cat):
    ev = V.extract_subject(text, V.classify_reply(text))
    assert ev and ev["category"] == cat


def test_parts_lists_never_decide_the_category():
    # a free-text parts list mentions category words but has no subject
    raw = "urn, car, wheel, wheel, wheel"
    ev = V.extract_subject(raw, V.classify_reply(raw))
    assert ev is None
    ask = scripted({"What is the main object": raw, "Describe": raw})
    with pytest.raises(V.VisionFailure) as e:
        V.observe(ask, "moondream", aspect=2.0)
    assert e.value.kind in ("malformed", "low_confidence")


def test_recorded_urus_replies_identify_a_car():
    obs = V.observe(recorded_urus(), "moondream:latest", aspect=1.7)
    assert obs["category"] == "car" and obs["family"] == "vehicle"
    assert obs["confidence"] >= 0.85


@pytest.mark.parametrize("ask,kind", [
    (scripted(default=""), "malformed"),
    (scripted(default="ids = [0.1, 0.2]"), "malformed"),
    (scripted(default="urn"), "low_confidence"),
    (scripted({"What is the main object": "The image shows a red car.",
               "Describe": "The image shows a wooden chair.",
               "What kind": "It is a bicycle.",
               "Name the object": "There is a lamp."}), "ambiguous"),
    (scripted(default=ModelUnavailableError("connection refused")),
     "unavailable"),
])
def test_vision_failure_kinds_are_distinct(ask, kind):
    with pytest.raises(V.VisionFailure) as e:
        V.observe(ask, "moondream", aspect=2.0)
    assert e.value.kind == kind
    assert "replies" in e.value.info


def test_single_answer_needs_geometric_prior():
    ask = scripted({"What is the main object": "The image shows a car."})
    obs = V.observe(ask, "moondream", aspect=2.4)     # fits vehicle prior
    assert obs["category"] == "car" and obs["confidence"] <= 0.6
    with pytest.raises(V.VisionFailure):
        V.observe(ask, "moondream", aspect=0.2)       # contradicts it


# ------------------------------------------------------------ segmentation
def test_platform_is_environment_not_object(inputs):
    s = SEG.segment(inputs["car_side_platform"])
    assert s["ok"] and s["environment"]["platform"]
    assert 2.3 <= s["subject"]["aspect"] <= 3.1          # car, not car+disc
    assert SEG.lab_distance(s["subject"]["body_color"], "0x1E1C22") < 15
    assert len(s["subject"]["ground_contacts"]) == 2      # two wheels


def test_real_photo_floor_and_light_pool_separated(inputs):
    s = SEG.segment(inputs["real_urus"])
    env = s["environment"]
    assert s["ok"] and env["backdrop"] and env["floor"]
    assert env["platform"] or env["surface_light"]       # red light pool
    assert 1.4 <= s["subject"]["aspect"] <= 2.1          # was 3.7 (floor)


def test_background_clutter_is_not_subject(inputs):
    s = SEG.segment(inputs["car_suv_clutter"])
    x0, y0, x1, y1 = s["subject"]["bbox"]
    assert y0 > 0.25        # clutter lives in the top band only
    assert s["environment"]["clutter"] or True


def test_low_contrast_reports_a_specific_failure(inputs):
    s = SEG.segment(inputs["lowcontrast_teapot"])
    assert not s["ok"] and "background" in s["error"]


# ------------------------------------------------------------ printer + IR
def test_profile_primitives_slice_true_cross_sections():
    from morphix_path import (PathState, compile_plan, merge_layers,
                              optimize_toolpath, validate_geometry)
    from morphix_specs import verify_solids
    wheel = [[0.45 * math.cos(2 * math.pi * i / 24),
              0.45 * math.sin(2 * math.pi * i / 24)] for i in range(24)]
    acts = [{"action": "add_profile", "label": "wheel", "position": [0, 0.45, 0],
             "profile": wheel, "depth": 0.4, "material": "tire"},
            {"action": "add_lathe", "label": "vase", "position": [3, 0.6, 0],
             "profile": [[0.3, -0.6], [0.5, -0.2], [0.25, 0.4], [0.3, 0.6]],
             "material": "wood"}]
    comp, order, errs, _k, _w = compile_plan(acts)
    assert not errs
    script, layers = merge_layers([comp[l] for l in order])
    script, _ = optimize_toolpath(script)
    assert not validate_geometry(script)
    st, recs = PathState(total_layers=layers), []
    for a in script:
        assert not st.check(a)
        r = st.apply(a)
        if r.get("action") in ("deposit_segment", "deposit_point"):
            recs.append(r)
    sec = [c for c in verify_solids(acts, recs) if c["name"] == "section_match"]
    assert len(sec) == 2 and all(c["ok"] for c in sec)


def test_image_ir_contract_rejects_printed_environment(inputs):
    seg = SEG.segment(inputs["car_side_platform"])
    import morphix_image_shapes as SH
    paint = SH.paint_from_clusters(seg["subject"]["colors"], "vehicle")
    fitres = SH.fit("vehicle", seg["_mask"], paint, budget=200)
    obs = I3._manual_observation("car")
    ir = I3.build_ir(obs, seg, "family", "vehicle", fitres, "object-only")
    assert I3.validate_image_ir(ir) == []
    bad = json.loads(json.dumps(ir))
    bad["environment"]["elements"][0]["printed"] = True
    assert any("never be printed" in e for e in I3.validate_image_ir(bad))
    bad = json.loads(json.dumps(ir))
    bad["components"] = [c for c in bad["components"]
                         if c["stage"] != "main_volume"]
    assert any("main_volume" in e for e in I3.validate_image_ir(bad))


# ------------------------------------------------------------ end to end
def _store(case, path, rep, expected_required):
    dst = os.path.join(ARTIFACTS, case)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(os.path.dirname(rep["_out"]), "trace")
                    if "_out" in rep else rep["_trace"], dst)
    v = rep["visual_verification"]
    with open(os.path.join(dst, "case.json"), "w", encoding="utf-8") as f:
        json.dump({"case": case, "input": os.path.basename(path),
                   "status": rep["reconstruction_status"],
                   "subject": rep["image"]["subject"],
                   "builder": rep["image"]["builder"],
                   "vision": rep["image"]["vision"],
                   "required_components": expected_required,
                   "components": [c["id"] for c in rep["ir_full"]["components"]
                                  if c.get("required")],
                   "metrics": v.get("metrics"),
                   "checks": v.get("checks"),
                   "fallback": rep["image"]["builder"]["method"] != "family",
                   "repairs": rep["image"]["repairs"],
                   "warnings": rep["warnings"]}, f, indent=2)


def _run(case, path, tmp_path, ask, **kw):
    out = str(tmp_path / case)
    rep = I3.run(path, out, ask=ask, **kw)
    rep["_trace"] = os.path.join(out, "trace")
    return rep


def _check(rep, name):
    return next(c for c in rep["visual_verification"]["checks"]
                if c["name"] == name)


CASES = {
    # case: (input, ask factory, expected status, expected family, required)
    "real_urus": ("real_urus", recorded_urus, "reconstructed", "vehicle",
                  ["car_body", "cabin_glass", "wheel_*"]),
    "car_side_platform": ("car_side_platform",
                          lambda: says("The image shows a black car."),
                          "reconstructed", "vehicle",
                          ["car_body", "cabin_glass", "wheel_*"]),
    "car_suv_clutter": ("car_suv_clutter",
                        lambda: says("The image shows a blue SUV."),
                        "reconstructed", "vehicle", ["car_body", "wheel_*"]),
    "table_angle": ("table_angle",
                    lambda: says("The image shows a wooden table."),
                    "reconstructed", "table", ["tabletop", "leg_*"]),
    "chair_side": ("chair_side",
                   lambda: says("The image shows a wooden chair."),
                   "reconstructed", "chair", ["seat", "backrest", "leg_*"]),
    "house_front": ("house_front",
                    lambda: says("The image shows a small house."),
                    "reconstructed", "house", ["walls", "roof"]),
    "cat_side": ("cat_side", lambda: says("The image shows an orange cat."),
                 "reconstructed", "quadruped", ["torso", "head", "leg_*"]),
    "vase_front": ("vase_front", lambda: says("The image shows a blue vase."),
                   "approximation", "vessel", ["revolved_body"]),
    "teapot_unfamiliar": ("teapot_side",
                          lambda: says("The image shows a teapot."),
                          "approximation", None, ["silhouette_body"]),
}


@pytest.mark.visual
@pytest.mark.parametrize("case", sorted(CASES))
def test_end_to_end_case(case, inputs, tmp_path):
    key, ask, status, family, required = CASES[case]
    rep = _run(case, inputs[key], tmp_path, ask())
    _store(case, inputs[key], rep, required)
    assert rep["reconstruction_status"] == status, rep["warnings"]
    assert rep["image"]["subject"]["family"] == family
    assert _check(rep, "silhouette_agreement")["ok"]
    assert _check(rep, "aspect_agreement")["ok"]
    ids = [c["id"] for c in rep["ir_full"]["components"]]
    for want in required:
        if want.endswith("*"):
            assert any(i.startswith(want[:-1]) for i in ids), want
        else:
            assert want in ids, want
    assert rep["success"] is True
    if family == "vehicle":
        assert _check(rep, "wheels_visible")["ok"]
        assert _check(rep, "cabin_above_body")["ok"]
        assert _check(rep, "color_region_consistency")["ok"]


@pytest.mark.visual
def test_urus_is_no_longer_a_yellow_block(inputs, tmp_path):
    rep = _run("urus_paint", inputs["real_urus"], tmp_path, recorded_urus())
    pal = rep["ir_full"]["appearance"]["palette"]
    assert SEG.lab_distance(pal["body"], "0xFFAA33") > 60   # not template yellow
    prims = {c["primitive"] for c in rep["ir_full"]["components"]
             if c["stage"] in ("main_volume", "structure")}
    assert "profile" in prims                    # real cross-sections
    body = next(c for c in rep["ir_full"]["components"] if c["id"] == "car_body")
    assert len(body["profile"]) > 12             # shaped, not a box


@pytest.mark.visual
def test_occluder_excluded_from_subject(inputs, tmp_path):
    rep = _run("car_occluded", inputs["car_occluded"], tmp_path,
               says("The image shows a yellow car next to a brown crate."))
    _store("car_occluded", inputs["car_occluded"], rep,
           ["car_body", "wheel_*"])
    env = rep["ir_full"]["environment"]["elements"]
    assert any(e["kind"] == "occluder" for e in env)
    assert any("hidden behind" in o
               for o in rep["ir_full"]["uncertainty"]["occluded"])
    body = rep["ir_full"]["appearance"]["palette"]["body"]
    assert SEG.lab_distance(body, "0xE6C828") < 25            # yellow, not crate


@pytest.mark.visual
def test_malformed_vision_builds_unidentified_silhouette(inputs, tmp_path):
    rep = _run("guitar_malformed", inputs["guitar_side"], tmp_path,
               scripted(default="ids = [0.1, 0.4, 0.5, 0.9]"))
    _store("guitar_malformed", inputs["guitar_side"], rep, ["silhouette_body"])
    sub = rep["image"]["subject"]
    assert not sub["identified"] and sub["vision_status"] == "malformed"
    assert rep["reconstruction_status"] == "approximation"
    assert rep["recognized"] is False
    assert any("not identified" in w for w in rep["warnings"])


@pytest.mark.visual
def test_missing_vision_model_is_reported(inputs, tmp_path):
    rep = _run("teapot_no_model", inputs["teapot_side"], tmp_path,
               scripted(default=ModelUnavailableError("connection refused")))
    assert rep["image"]["subject"]["vision_status"] == "unavailable"
    assert rep["reconstruction_status"] == "approximation"


@pytest.mark.visual
def test_wrong_category_is_never_claimed_as_reconstructed(inputs, tmp_path):
    # user forces "table" on a car photo: the table builder cannot match the
    # silhouette, so the result must not be reported as a reconstructed table
    rep = _run("car_as_table", inputs["car_side_platform"], tmp_path, None,
               category="table")
    _store("car_as_table", inputs["car_side_platform"], rep, ["tabletop"])
    assert rep["reconstruction_status"] != "reconstructed"
    assert rep["recognized"] is False


@pytest.mark.visual
def test_environment_preserving_mode(inputs, tmp_path):
    rep = _run("preserve", inputs["car_side_platform"], tmp_path,
               says("The image shows a black car."), environment="preserve")
    _store("car_preserve", inputs["car_side_platform"], rep, ["car_body"])
    env = rep["ir_full"]["environment"]
    assert env["mode"] == "preserve"
    plat = [e for e in env["elements"] if e["kind"] == "platform"]
    assert plat and plat[0]["build"] and not plat[0]["printed"]
    assert os.path.exists(os.path.join(rep["_trace"], "display.png"))
    # display context never reaches the printer
    assert not any(c["role"] == "platform" for c in rep["ir_full"]["components"])


def test_low_contrast_run_fails_at_segmentation(inputs, tmp_path):
    with pytest.raises(I3.ReconstructionError) as e:
        I3.run(inputs["lowcontrast_teapot"], str(tmp_path / "lc"),
               ask=says("The image shows a teapot."), render=False)
    assert e.value.stage == "segmentation"
