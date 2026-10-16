# tests/test_image_reconstruction.py
# Regression suite for the image -> 3D semantic pipeline:
#   * a table photograph becomes ONE thin tabletop + FOUR separate corner legs
#     (never a block, never an urn);
#   * an urn becomes round urn geometry (never table parts);
#   * the vision answer is schema-validated, confidence-gated and cross-checked
#     against the measured silhouette; a low-confidence / conflicting answer is
#     re-analysed once and then asks the user — it is NEVER silently swapped;
#   * reconstruction-quality checks stay separate from technical action checks.
# All model access is mocked; the geometry is deterministic.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ images
def _make_table_image(path):
    """A thin wide top resting on four separated dark legs (light bg)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (420, 220), (245, 245, 245))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 44, 380, 64], fill=(150, 96, 48))     # tabletop
    for x in (58, 342):
        dr.rectangle([x, 64, x + 20, 192], fill=(120, 74, 34))   # legs
    im.save(path)


def _make_urn_image(path):
    """A round-ish vessel taller than it is wide (a plausible urn)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (240, 320), (245, 245, 245))
    dr = ImageDraw.Draw(im)
    dr.ellipse([55, 40, 185, 120], fill=(160, 90, 60))
    dr.rectangle([65, 110, 175, 240], fill=(160, 90, 60))
    dr.rectangle([90, 240, 150, 290], fill=(160, 90, 60))
    im.save(path)


@pytest.fixture(scope="module")
def table_image(tmp_path_factory):
    p = tmp_path_factory.mktemp("img") / "table.png"
    _make_table_image(str(p))
    return str(p)


@pytest.fixture(scope="module")
def urn_image(tmp_path_factory):
    p = tmp_path_factory.mktemp("img") / "urn.png"
    _make_urn_image(str(p))
    return str(p)


def _vision(monkeypatch, responses, model="moondream:latest"):
    """Install a deterministic vision model returning `responses` in order
    (the last response repeats). `responses` is one JSON-able dict/string or a
    list of them."""
    import morphix_reconstruct as R
    if isinstance(responses, (dict, str)):
        seq = [responses]
    else:
        seq = list(responses)

    def fake(*a, **k):
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        return r if isinstance(r, str) else json.dumps(r)

    monkeypatch.setattr(R, "ollama_generate", fake)
    monkeypatch.setattr(R, "available_vision_model", lambda: model)


# ------------------------------------------------------------ templates
def test_table_template_is_one_thin_top_and_four_grounded_legs():
    import morphix_ir as IR
    from morphix_path import compile_plan, merge_layers, validate_geometry

    ir = IR.ir_for_category("table")
    assert IR.validate_ir(ir) == []
    assert IR.template_key("table") == "table"
    comps = ir["components"]
    tops = [c for c in comps if c["role"] == "tabletop"]
    legs = [c for c in comps if c["role"] == "leg"]
    assert len(tops) == 1 and len(legs) == 4
    w, h, d = tops[0]["dims"]
    assert h <= 0.3 * min(w, d)                      # a thin slab, not a block
    heights = [c["dims"][1] for c in legs]
    assert max(heights) - min(heights) <= 0.05       # consistent leg lengths
    for c in legs:
        assert abs(c["position"][1] - c["dims"][1] / 2) <= 0.05   # grounded
    top_base = tops[0]["position"][1] - h / 2
    assert abs(top_base - max(c["position"][1] + c["dims"][1] / 2
                              for c in legs)) <= 0.1              # top on legs
    # +x is the length: the table is longer than it is deep
    assert ir["bbox"][0] > ir["bbox"][2]

    compiled, order, errors, _keep, _warns = compile_plan(
        IR.ir_to_actions(ir), {})
    assert errors == {}
    script, _layers = merge_layers([compiled[l] for l in order])
    assert validate_geometry(script) == []


def test_urn_template_is_urn_geometry_not_table_parts():
    import morphix_ir as IR
    from morphix_path import compile_plan, merge_layers, validate_geometry

    ir = IR.ir_for_category("urn")
    assert IR.validate_ir(ir) == []
    roles = {c["role"] for c in ir["components"]}
    assert "tabletop" not in roles and "leg" not in roles
    assert any(c["primitive"] == "cylinder" for c in ir["components"])
    assert ir["bbox"][1] > ir["bbox"][0]              # taller than wide

    compiled, order, errors, _keep, _warns = compile_plan(
        IR.ir_to_actions(ir), {})
    assert errors == {}
    script, _layers = merge_layers([compiled[l] for l in order])
    assert validate_geometry(script) == []


def test_every_supported_category_compiles_to_valid_layer_paths():
    import morphix_ir as IR
    from morphix_path import compile_plan, merge_layers, validate_geometry

    for cat in IR.SUPPORTED_CATEGORIES:
        ir = IR.ir_for_category(cat)
        assert IR.validate_ir(ir) == [], cat
        compiled, order, errors, _keep, _warns = compile_plan(
            IR.ir_to_actions(ir), {})
        assert errors == {}, cat
        script, _layers = merge_layers([compiled[l] for l in order])
        assert validate_geometry(script) == [], cat


# ------------------------------------------------------ table end-to-end
def test_table_photo_reconstructs_to_tabletop_and_four_legs(
        table_image, tmp_path, monkeypatch):
    import morphix_reconstruct as R
    _vision(monkeypatch, [{
        "category": "table", "confidence": 0.9,
        "proportions": {"width": 2.0, "height": 1.2},
        "components": [{"name": "top", "position": "top"},
                       {"name": "leg", "position": "bottom"}],
        "materials": ["wood"], "colors": ["brown"],
        "parts": ["top", "leg"], "hidden": [],
    }])
    rep = R.run_reconstruction_legacy(table_image, out_dir=str(tmp_path),
                               vision_model="moondream:latest")
    assert rep["image"]["source"] == "image-vision"
    assert rep["image"]["category"] == "table"
    assert rep["quality_accepted"] and rep["ok"] and rep["complete"]
    labels = [c["component"] for c in rep["components"]]
    assert not any(l.startswith("px_") for l in labels)
    assert sum(1 for l in labels if l.startswith("leg_")) == 4
    assert "tabletop" in labels
    fid = rep["reconstruction_quality"]
    assert fid["verified"]
    names = {c["name"] for c in fid["checks"]}
    assert {"tabletop_count", "leg_count", "tabletop_thin",
            "leg_lengths_consistent", "legs_grounded", "legs_at_corners",
            "tabletop_supported"} <= names


def test_table_photo_is_not_silently_accepted_as_an_urn(monkeypatch, table_image):
    import morphix_reconstruct as R

    # The model insists (confidently) that the table is an urn. The template
    # shape contradicts the wide photo silhouette, so after ONE bounded
    # re-analysis the pipeline must ask the user, not substitute.
    _vision(monkeypatch, {"category": "urn", "confidence": 0.95,
                          "parts": ["body"]})
    with pytest.raises(R.VisionError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=str(table_image) + "_o1",
                             vision_model="moondream:latest")
    info = ei.value.info
    assert info.get("needs_category") is True
    assert info.get("detected_category") == "urn"
    assert len(info.get("attempts", [])) == R.VISION_MAX_ATTEMPTS


def test_conflicting_vision_is_reanalysed_and_recovers(monkeypatch, table_image):
    import morphix_reconstruct as R

    _vision(monkeypatch, [{"category": "urn", "confidence": 0.9,
                           "parts": ["body"]},
                          {"category": "table", "confidence": 0.9,
                           "parts": ["top", "leg"]}])
    rep = R.run_reconstruction_legacy(table_image, out_dir=table_image + "_o2",
                               vision_model="moondream:latest")
    assert rep["image"]["category"] == "table"
    assert rep["quality_accepted"]
    assert len(rep["image"]["vision_analysis"]["attempts"]) == 2


def test_low_confidence_is_bounded_then_asks_the_user(monkeypatch, table_image):
    import morphix_reconstruct as R

    _vision(monkeypatch, {"category": "table", "confidence": 0.05,
                          "parts": ["top"]})
    with pytest.raises(R.VisionError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_o3",
                             vision_model="moondream:latest")
    assert ei.value.info.get("needs_category") is True
    assert len(ei.value.info["attempts"]) == R.VISION_MAX_ATTEMPTS


def test_unparseable_vision_response_is_a_clear_error(monkeypatch, table_image):
    import morphix_reconstruct as R

    _vision(monkeypatch, "I am sorry, I cannot see any photograph.")
    with pytest.raises(R.VisionError):
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_o4",
                             vision_model="moondream:latest")


def test_text_only_model_is_refused_for_images(monkeypatch, table_image):
    import morphix_reconstruct as R
    from morphix_config import PipelineError

    monkeypatch.setattr(R, "ollama_generate",
                        lambda *a, **k: pytest.fail("must not call the model"))
    with pytest.raises(PipelineError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_o5",
                             vision_model="deepseek-r1:1.5b")
    assert "not a known image-capable model" in str(ei.value)


def test_urn_photo_reconstructs_as_urn_geometry(monkeypatch, urn_image):
    import morphix_reconstruct as R

    _vision(monkeypatch, {"category": "urn", "confidence": 0.9,
                          "parts": ["body", "neck"]})
    rep = R.run_reconstruction_legacy(urn_image, out_dir=urn_image + "_o6",
                               vision_model="moondream:latest")
    assert rep["image"]["category"] == "urn"
    assert rep["ok"] and rep["complete"] and rep["quality_accepted"]
    ids = [c["id"] for c in rep["ir_full"]["components"]]
    assert not any("leg" in i or "tabletop" in i for i in ids)
    assert rep["ir_full"]["bbox"][1] > rep["ir_full"]["bbox"][0]


def test_manual_category_uses_template_without_any_vision(table_image):
    import morphix_reconstruct as R

    rep = R.run_reconstruction_legacy(table_image, out_dir=table_image + "_man",
                               category="table")
    assert rep["image"]["source"] == "image-manual"
    assert rep["image"]["category"] == "table"
    labels = [c["component"] for c in rep["components"]]
    assert sum(1 for l in labels if l.startswith("leg_")) == 4
    assert "tabletop" in labels


# ------------------------------------------------------------ quality split
def test_reconstruction_quality_rejects_a_thick_tabletop(table_image):
    import morphix_ir as IR
    import morphix_reconstruct as R
    from morphix_path import compile_plan, merge_layers, validate_geometry

    ir = IR.ir_for_category("table")
    top = [c for c in ir["components"] if c["role"] == "tabletop"][0]
    top["dims"][1] = 0.8                       # pretend the top is a thick block
    ir["bbox"] = IR._object_bbox(ir["components"])
    stats = R.image_stats(table_image)
    fid = IR.image_fidelity(ir, stats, ["top", "leg"])
    assert not fid["verified"]
    bad = {c["name"]: c for c in fid["checks"]}
    assert bad["tabletop_thin"]["ok"] is False
    # ...while the same (thick-top) plan still compiles as valid TECHNICAL
    # geometry: the two verdicts are independent.
    compiled, order, errors, _keep, _warns = compile_plan(
        IR.ir_to_actions(ir), {})
    assert errors == {}
    script, _layers = merge_layers([compiled[l] for l in order])
    assert script


# ------------------------------------------------------------ fast units
def test_vision_model_capability_helper():
    from morphix_image import vision_model_supports_images as sup
    assert sup("moondream:latest")
    assert sup("llava")
    assert sup("qwen2.5vl:7b")
    assert not sup("deepseek-r1:1.5b")
    assert not sup("qwen2.5-coder:7b")
    assert not sup("")


def test_parse_vision_response_schema():
    import morphix_reconstruct as R
    det, errors = R.parse_vision_response(
        '{"category": "Table!", "confidence": 0.8, "parts": ["top"]}')
    assert det["category"] == "table"
    assert det["confidence"] == pytest.approx(0.8)
    assert det["parts"] == ["top"]
    # missing category -> unusable, with a schema error recorded
    det2, errors2 = R.parse_vision_response('{"confidence": 0.5}')
    assert det2 is None and errors2
    # prose is never mined for a category
    det3, _ = R.parse_vision_response("the object is maybe a table")
    assert det3 is None


def test_bare_word_parses_weak_but_prose_does_not():
    import morphix_reconstruct as R
    det, _ = R.parse_vision_response("urn")
    assert det["category"] == "urn" and det["weak"] is True
    assert det["confidence"] == pytest.approx(R.VISION_WEAK_CONFIDENCE)
    det, _ = R.parse_vision_response("urn.")     # trailing punctuation
    assert det["category"] == "urn" and det["weak"] is True
    assert R.parse_vision_response("")[0] is None
    assert R.parse_vision_response("???")[0] is None


def test_weak_fitting_answer_accepted_after_retry(monkeypatch, table_image):
    import morphix_reconstruct as R
    # strong conflicting JSON, then a weak single word that fits
    _vision(monkeypatch, [
        {"category": "urn", "confidence": 0.9, "parts": ["body"]},
        "table",
    ])
    rep = R.run_reconstruction_legacy(table_image, out_dir=table_image + "_weak",
                               vision_model="moondream:latest")
    assert rep["image"]["category"] == "table"
    assert rep["quality_accepted"]
    attempts = rep["image"]["vision_analysis"]["attempts"]
    assert len(attempts) == 2
    assert attempts[-1].get("resolved_weak") is True


def test_description_with_parts_resolves_conflicting_label(
        monkeypatch, table_image):
    import morphix_reconstruct as R
    # the real-world transcript shape: confident mislabel, empty retry,
    # then a description naming the true subject WITH its parts. The
    # decision uses every reply jointly (votes + part corroboration +
    # silhouette) — never the word count of one reply alone.
    _vision(monkeypatch, [
        {"category": "urn", "confidence": 0.9, "parts": ["body"]},
        "",
        "a table of wood with a rectangular top and four legs",
    ])
    rep = R.run_reconstruction_legacy(table_image, out_dir=table_image + "_parts",
                               vision_model="moondream:latest")
    assert rep["image"]["category"] == "table"
    assert rep["quality_accepted"]
    attempts = rep["image"]["vision_analysis"]["attempts"]
    assert len(attempts) == 3
    assert attempts[-1].get("via") == "decide"
    assert "urn" in attempts[-1].get("overrode", [])


def test_parts_without_dominance_escalates(monkeypatch, table_image):
    import morphix_reconstruct as R
    _vision(monkeypatch, [
        {"category": "urn", "confidence": 0.9, "parts": ["body"]},
        "urn",
        "wood, brown, rectangular, smooth",
    ])
    with pytest.raises(R.VisionError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_nodom",
                             vision_model="moondream:latest")
    assert ei.value.info.get("needs_category") is True
    assert len(ei.value.info["attempts"]) == R.VISION_MAX_ATTEMPTS


def test_unavailable_model_is_a_distinct_error(monkeypatch, table_image):
    import morphix_reconstruct as R
    from morphix_config import ModelUnavailableError

    def dead(*a, **k):
        raise ModelUnavailableError("connection refused")
    monkeypatch.setattr(R, "ollama_generate", dead)
    with pytest.raises(R.VisionError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_dead",
                             vision_model="moondream:latest")
    assert ei.value.info.get("unavailable") is True
    assert ei.value.info.get("needs_category") is False


def test_unknown_category_escalates(monkeypatch, table_image):
    import morphix_reconstruct as R
    _vision(monkeypatch, "dragon")
    with pytest.raises(R.VisionError) as ei:
        R.run_reconstruction_legacy(table_image, out_dir=table_image + "_dragon",
                             vision_model="moondream:latest")
    assert ei.value.info.get("needs_category") is True


@pytest.mark.live
def test_real_wooden_table_classifies_as_table(tmp_path):
    """Live regression: the actual uploaded table photograph through the
    real moondream pipeline must resolve to a table (no manual category).
    Needs a warm, uncontended Ollama; excluded from normal runs."""
    import morphix_reconstruct as R
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "uploads",
        "20261013_043108_images_4.jpg")
    assert os.path.isfile(src)
    rep = R.run_reconstruction_legacy(src, out_dir=str(tmp_path),
                               vision_model="moondream:latest")
    assert rep["image"]["source"] == "image-vision"
    assert rep["image"]["category"] == "table"
    assert rep["quality_accepted"]
    labels = [c["component"] for c in rep["components"]]
    assert "tabletop" in labels
    assert sum(1 for l in labels if l.startswith("leg_")) == 4


def test_rejected_action_does_not_modify_committed_geometry():
    from morphix_path import PathState
    st = PathState(total_layers=1)
    good = {"action": "deposit_segment", "from": [0.0, 0.0, 0.0],
            "to": [0.2, 0.0, 0.0], "bead_width": 0.2, "bead_height": 0.2,
            "material": "wood", "layer": 0, "label": "good"}
    st.apply(dict(good))
    before = list(st.layers[0]["segments"])
    # starts away from the live toolhead -> rejected, never committed
    bad = {"action": "deposit_segment", "from": [5.0, 0.0, 0.0],
           "to": [5.2, 0.0, 0.0], "bead_width": 0.2, "bead_height": 0.2,
           "material": "wood", "layer": 0, "label": "bad"}
    assert st.check(bad)                          # the validator rejects it
    with pytest.raises(ValueError):
        st.apply(dict(bad))
    # the rejected step changed nothing: same committed segments, same toolhead
    assert st.layers[0]["segments"] == before
    assert st.pos == [0.2, 0.0, 0.0]
