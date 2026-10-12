# tests/test_scene_isolation.py
# Scene mode and printer mode never share geometry: scene effects cannot
# leak into deposition paths, printer actions never appear in scene modules,
# and only explicitly-tagged printable nodes are eligible for printing.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scene_templates_only_tag_solids_printable():
    import morphix_scene_ir as S
    import morphix_scene_templates as T
    import morphix_three_knowledge as K
    allowed = set(K.printer_compatible_geometry())
    for name in T.TEMPLATE_NAMES:
        ir = T.build(name)
        for nid in S.printable_nodes(ir):
            n = next(c for c in ir["nodes"] if c["id"] == nid)
            assert (n.get("geometry") or {}).get("ctor") in allowed, (name, nid)
        # scene effects (points/instancing/fog) are never tagged printable
        for n in ir["nodes"]:
            g = n.get("geometry") or {}
            if g.get("generator") or n.get("kind") in ("points",):
                assert n["id"] not in S.printable_nodes(ir), (name, n["id"])


def test_scene_js_never_contains_printer_actions():
    import morphix_scene_compile as C
    import morphix_scene_templates as T
    for name in T.TEMPLATE_NAMES:
        js = C.compile_scene_ir(T.build(name))
        for marker in ("deposit_segment", "deposit_point", "move_to",
                       "complete_layer", "complete_path", "complete_object",
                       "bead_width"):
            assert marker not in js, (name, marker)


def test_printer_guard_rejects_scene_effect_code():
    from morphix_guard import validate_js
    import morphix_scene_compile as C
    import morphix_scene_templates as T
    js = C.compile_scene_ir(T.build("fog_benchmark"))
    box = {"action": "add_box", "width": 1, "height": 1, "depth": 1,
           "position": [0, 0.5, 0], "color": "0xFFAA33", "label": "probe"}
    assert validate_js(box, js) != []      # not atomic, not grounded
    foggy = js + "\nscene.add(new THREE.Points(g, m));"
    assert validate_js(box, foggy) != []


def test_printer_deposit_rendering_uses_only_known_apis():
    import morphix_three_knowledge as K
    from morphix_path import render_deposit
    rec = {"action": "deposit_segment", "from": [0, 0, 0], "to": [0.2, 0, 0],
           "bead_width": 0.2, "bead_height": 0.2, "material": "body",
           "layer": 0, "label": "p", "component": "c", "seq": 1}
    js = render_deposit(rec)
    assert K.check_js_apis(js) == []
