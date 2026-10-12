# tests/test_knowledge.py
# Three.js knowledge index: the frozen export list matches the vendored
# build, curated entries are sane, and API gating accepts real APIs while
# rejecting fabrications and WebGPU-only APIs on the WebGL backend.
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_frozen_exports_match_vendor_build():
    import morphix_three_knowledge as K
    src = open(K.VENDOR_PATH, encoding="utf-8").read()
    m = re.search(r"export\s*\{([^}]*)\}\s*;?\s*$", src[-12000:])
    live = sorted(e.strip().split(" as ")[-1].strip()
                  for e in m.group(1).split(",") if e.strip())
    assert list(K.vendor_exports()) == live
    assert len(live) > 300


def test_curated_entries_are_wellformed():
    import morphix_three_knowledge as K
    assert len(K.CURATED) >= 40
    for name, e in K.CURATED.items():
        for key in ("description", "since", "imports", "dependencies",
                    "example", "performance", "modes", "docs"):
            assert key in e, (name, key)
        assert e["modes"], name
        if name not in K.UNSUPPORTED:
            assert name in K.vendor_exports(), name


def test_api_supported_gate():
    import morphix_three_knowledge as K
    ok, _ = K.api_supported("InstancedMesh")
    assert ok
    ok, why = K.api_supported("WebGPURenderer")
    assert not ok and "WebGLRenderer" in why
    ok, why = K.api_supported("rangeFogFactor")
    assert not ok and "Fog" in why
    ok, why = K.api_supported("BogusGeometry")
    assert not ok and "unknown" in why


def test_check_js_apis_accepts_real_code():
    import morphix_three_knowledge as K
    code = ("const scene = new THREE.Scene();\n"
            "scene.fog = new THREE.Fog(0x4080cc, 2.7, 4);\n"
            "const m = new THREE.InstancedMesh(geo, mat, 100);\n"
            "const c = new THREE.Color(0xffffff);")
    assert K.check_js_apis(code) == []


def test_check_js_apis_rejects_fabrication_and_tsl():
    import morphix_three_knowledge as K
    assert any("BogusGeometry" in e
               for e in K.check_js_apis("new BogusGeometry(1);"))
    errs = K.check_js_apis("import { pass } from 'three/tsl';")
    assert any("WebGL" in e for e in errs)
    assert K.check_js_apis("") != []
