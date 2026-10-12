# tests/visual_geometry/test_live_adapters.py
# OPTIONAL integration test: exercises the real model adapters once
# (thinker + builder + vision). Excluded from normal runs; invoke
# explicitly with:  pytest tests/visual_geometry -m live
# Timings printed here feed the bottleneck report.
import json
import os
import sys
import time

import pytest

pytestmark = pytest.mark.live

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def test_live_thinker_builder_vision():
    from morphix_config import ollama_generate, THINKER_MODEL, BUILDER_MODEL
    from morphix_image import available_vision_model, describe_image

    t0 = time.perf_counter()
    raw = ollama_generate(
        THINKER_MODEL, "Output ONLY the JSON object.",
        "GOAL: a red sphere. Reply with the next action as ONE JSON object.",
        options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 1500})
    t_think = time.perf_counter() - t0
    print(f"\nlive think: {t_think:.1f}s ({len(raw)} bytes)")
    assert raw.strip()

    action = {"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0],
              "color": "0xFF0000", "label": "t1"}
    t0 = time.perf_counter()
    out = ollama_generate(
        BUILDER_MODEL, "Output ONLY raw JavaScript.",
        json.dumps(action),
        options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 1500})
    t_build = time.perf_counter() - t0
    print(f"live build: {t_build:.1f}s ({len(out)} bytes)")
    assert "THREE" in out or "Sphere" in out

    vm = available_vision_model()
    if vm:
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png",
                                         delete=False) as f:
            Image.new("RGB", (64, 64), (200, 60, 40)).save(f.name)
            path = f.name
        t0 = time.perf_counter()
        desc = describe_image(path, vm)
        t_vis = time.perf_counter() - t0
        print(f"live vision ({vm}): {t_vis:.1f}s: {desc[:80]!r}")
        os.unlink(path)
        assert isinstance(desc, str)
    else:
        pytest.skip("no vision model installed")
