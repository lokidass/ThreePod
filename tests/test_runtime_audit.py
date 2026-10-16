# tests/test_runtime_audit.py
# Priority 3: the capability registry is audited against the ACTUAL
# vendored runtime — revision pin, constructor signatures the compiler
# emits, and the UI CDN pin must all agree. A silent version drift fails
# loudly here instead of generating subtly wrong scenes.
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _vendor_source():
    import morphix_three_knowledge as K
    with open(K.VENDOR_PATH, encoding="utf-8") as f:
        return f.read()


def test_vendor_revision_is_r152():
    src = _vendor_source()
    m = re.search(r"const REVISION = ['\"]([^'\"]+)['\"]", src)
    assert m and m.group(1) == "152"


def test_emitted_constructor_signatures_match_vendor():
    src = _vendor_source()
    assert re.search(r"class Fog \{\s+constructor\(\s*color,\s*near",
                     src), "Fog(color, near, far)"
    assert re.search(r"class FogExp2 \{\s+constructor\(\s*color,\s*density",
                     src), "FogExp2(color, density)"
    assert re.search(r"class InstancedMesh extends Mesh \{\s+constructor\("
                     r"\s*geometry,\s*material,\s*count", src), \
        "InstancedMesh(geometry, material, count)"
    assert "fromEquirectangular(" in src
    assert "setMatrixAt(" in src


def test_ui_cdn_pin_matches_vendored_revision():
    import morphix_scene_compile as C
    html = C.scene_html("import * as THREE from 'three';\n"
                        "window.__scene_ready = true;")
    assert "three@0.152.2" in html
    # no other three version may sneak in through the wrapper
    assert len(re.findall(r"three@[\d.]+", html)) == 1


def test_no_package_json_dependency_drift():
    # There is deliberately no npm three dependency: the UI loads the
    # pinned CDN build and tests use the vendored file. If someone adds
    # one, this test forces the pins to be reconciled explicitly.
    pkg = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "package.json")
    if os.path.isfile(pkg):
        import json
        deps = {**json.load(open(pkg)).get("dependencies", {}),
                **json.load(open(pkg)).get("devDependencies", {})}
        three = [v for k, v in deps.items() if "three" in k.lower()]
        assert all("0.152" in v for v in three), three
