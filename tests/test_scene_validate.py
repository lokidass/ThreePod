# tests/test_scene_validate.py
# Geometric + visual validation (Phase 4.1): required presence, finite
# non-degenerate bboxes, bounds, declared relations, floating suspicion,
# framing/lighting/fog pixel checks. Unit tests use canned graphs; live
# checks use real renders.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _table_ir():
    import morphix_scene_plan as P
    ir, _ = P.compose_routed_scene(
        P.route_systems("a wooden table in a room"),
        {"objective": "table room"}, seed=11)
    return ir


def _graph(ir):
    # canned live-style graph: exact design bboxes from the IR geometry
    g = {}
    for n in ir["nodes"]:
        p = n["transform"]["position"]
        geo = n.get("geometry") or {}
        prm = geo.get("params", {})
        if geo.get("generator"):
            g[n["id"]] = {"present": True, "empty": False, "visible": True,
                          "bbox": [[-45, -3, -45], [45, 6, 45]]}
        elif geo.get("ctor") == "BoxGeometry":
            w, h, d = prm["w"], prm["h"], prm["d"]
            g[n["id"]] = {"present": True, "empty": False, "visible": True,
                          "bbox": [[p[0] - w / 2, p[1] - h / 2,
                                    p[2] - d / 2],
                                   [p[0] + w / 2, p[1] + h / 2,
                                    p[2] + d / 2]]}
        else:
            g[n["id"]] = {"present": True, "empty": False, "visible": True,
                          "bbox": [[p[0] - 1, p[1] - 1, p[2] - 1],
                                   [p[0] + 1, p[1] + 1, p[2] + 1]]}
    return g


def test_assembly_passes_on_coherent_scene():
    import morphix_scene_validate as V
    ir = _table_ir()
    out = V.validate_assembly(ir, _graph(ir))
    assert out["verified"], [c for c in out["checks"] if not c["ok"]]
    assert any(c["name"].startswith("relation:") and c["ok"]
               for c in out["checks"])


def test_missing_required_fails():
    import morphix_scene_validate as V
    ir = _table_ir()
    g = _graph(ir)
    del g["table_set_top"]
    out = V.validate_assembly(ir, g)
    assert not out["verified"]
    assert any(c["name"] == "present:table_set_top" and not c["ok"]
               for c in out["checks"])


def test_broken_contact_fails_but_overlap_passes():
    import morphix_scene_validate as V
    ir = _table_ir()
    g = _graph(ir)
    # lift the top 0.5: contact broken (tol 0.05), still overlapping
    lo, hi = g["table_set_top"]["bbox"]
    g["table_set_top"]["bbox"] = [[lo[0], lo[1] + 0.5, lo[2]],
                                  [hi[0], hi[1] + 0.5, hi[2]]]
    out = V.validate_assembly(ir, g)
    bad = [c for c in out["checks"] if not c["ok"]]
    assert any("table_set_top" in c["name"] and "relation" in c["name"]
               for c in bad)


def test_degenerate_and_infinite_bboxes_fail():
    import morphix_scene_validate as V
    ir = _table_ir()
    g = _graph(ir)
    g["table_set_top"]["bbox"] = [[0, 0, 0], [0, 0, 0]]
    out = V.validate_assembly(ir, g)
    assert not out["verified"]
    g["table_set_top"]["bbox"] = [[0, 0, 0], [1, float("inf"), 1]]
    out = V.validate_assembly(ir, g)
    assert not out["verified"]


def test_projection_math():
    import morphix_scene_validate as V
    # camera at origin looking down -z... via eye/target form
    p = V.project_point([0, 0, 5], [0, 0, 0], 90, 640 / 480, 640, 480,
                        (0, 0, 0))
    assert p == (320, 240), p
    assert V.project_point([0, 0, 5], [0, 0, 0], 90, 640 / 480, 640, 480,
                           (0, 0, 6)) is None      # behind camera
    r = V.subject_rect(
        {"a": {"bbox": [[-1, -1, -1], [1, 1, 1]]}}, ["a"],
        {"position": [0, 0, 5], "target": [0, 0, 0], "fov": 90})
    assert r is not None and r[2] - r[0] > 100


def test_live_table_room_validates():
    """End-to-end geometric + visual validation on a real render."""
    import morphix_scene_plan as P
    import morphix_scene_validate as V
    from scene_browser import render_scenes
    os.makedirs(ACTUAL, exist_ok=True)
    ir, _ = P.compose_routed_scene(
        P.route_systems("a wooden table in a room"),
        {"objective": "table room"}, seed=11)
    res = render_scenes([("geo_live", ir,
                          os.path.join(ACTUAL, "_geo_live.png"))])["geo_live"]
    assert res["ready"] and not res["scene_error"]
    out = V.validate_assembly(ir, res.get("graph") or {})
    assert out["verified"], [c for c in out["checks"] if not c["ok"]]
    vis = V.verify_visual(ir, res["png"], res.get("graph") or {})
    assert vis["verified"], vis["checks"]
