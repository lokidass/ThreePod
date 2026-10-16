# tests/test_image_modes.py
# The image pipeline has two distinct modes that must never be conflated:
# relief (deterministic per-pixel voxel print, honest demo geometry) and
# semantic reconstruction (canonical object IR from a category, gated on
# the photo silhouette BEFORE any path compiles). No live vision needed:
# the manual-category path is fully deterministic.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _photo(path, w=120, h=120):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (245, 245, 245))
    dr = ImageDraw.Draw(im)
    dr.ellipse([30, 30, 90, 90], fill=(200, 60, 40))
    im.save(path)


def _car_photo(path):
    # wide car-proportioned silhouette (aspect ~3) so the silhouette gate
    # accepts the car template (band 0.5-2.0 around template aspect 2.18).
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (200, 100), (245, 245, 245))
    dr = ImageDraw.Draw(im)
    dr.ellipse([20, 30, 180, 80], fill=(200, 60, 40))
    im.save(path)


def _pano(path):
    from PIL import Image
    Image.new("RGB", (200, 20), (245, 245, 245)).save(path)


def test_relief_reports_relief_not_semantics(tmp_path):
    from morphix_image import run_image
    img = os.path.join(str(tmp_path), "ball.png")
    _photo(img)
    rep = run_image(img, os.path.join(str(tmp_path), "out"), grid=8,
                    mode="relief")
    assert rep["mode"] == "image-relief"
    assert "image_meta" in rep and "ir_full" not in rep
    assert rep["ok"] is True


def test_semantic_reconstruction_reports_template_and_quality(tmp_path):
    from morphix_reconstruct import run_reconstruction_legacy
    img = os.path.join(str(tmp_path), "car.png")
    _car_photo(img)
    rep = run_reconstruction_legacy(img, os.path.join(str(tmp_path), "out"),
                             category="car")
    assert rep["mode"] == "image-reconstruct"
    assert rep["ir"]["category"] == "car"
    assert "reconstruction_quality" in rep and "image_fidelity" in rep
    # semantic path has no voxel grid metadata
    assert "image_meta" not in rep


def test_manual_category_with_conflicting_silhouette_refuses_to_compile(
        tmp_path):
    from morphix_config import PipelineError
    from morphix_reconstruct import run_reconstruction_legacy
    img = os.path.join(str(tmp_path), "pano.png")
    _pano(img)  # aspect ~10 vs car template ~2.2: irreconcilable
    with pytest.raises(PipelineError, match="silhouette|aspect"):
        run_reconstruction_legacy(img, os.path.join(str(tmp_path), "out"),
                           category="car")
    # refused BEFORE compiling: no paths were ever generated
    assert not os.path.isfile(os.path.join(str(tmp_path), "out", "scene.js"))
