# tests/test_scene_parts.py
# Freestyle part assemblies: valid IR, compilable JS, real browser render,
# and documented contacts (legs grounded, tops supported, walls meet
# floors, bands stack). Parts are capabilities, not hardcoded answers:
# unknown names raise instead of substituting.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "visual_geometry", ".actual")


def _wrap(nodes, name="parts_probe"):
    import morphix_scene_ir as S
    ir = S.new_scene(name, seed=5)
    ir["nodes"].extend(nodes)
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 500, "position": [8, 6, 10], "target": [0, 1, 0]}
    return ir


def test_part_catalogue_builds_valid_ir():
    import morphix_scene_ir as S
    import morphix_scene_parts as P
    assert set(P.PART_NAMES) >= {"table_set", "room_shell",
                                 "lighthouse_tower", "cabin", "castle_keep",
                                 "tent", "boat", "lamp_post", "pier"}
    structural = {
        "table_set": ("top", "leg_front_left"),
        "room_shell": ("floor", "wall_back", "wall_side"),
        "lighthouse_tower": ("band_0", "cap"),
        "cabin": ("walls", "roof"),
        "castle_keep": ("wall_n", "turret_ne"),
        "tent": ("shell",),
        "boat": ("hull", "sail"),
        "lamp_post": ("pole",),
        "pier": ("deck",),
        "dunes": (),
        "coral_bed": ("cones",),
    }
    for name in P.PART_NAMES:
        part = P.build_part(name, prefix=f"p_{name}_", required=True)
        assert part["nodes"], name
        assert part["strategy"], name          # strategy documented
        ir = _wrap(part["nodes"], name=name)
        assert S.validate_scene_ir(ir) == [], name
        by_id = {n["id"].split("p_%s_" % name, 1)[-1]: n
                 for n in part["nodes"]}
        for role in structural[name]:
            assert by_id[role].get("required") is True, (name, role)


def test_unknown_part_is_an_error():
    import morphix_scene_parts as P
    with pytest.raises(KeyError):
        P.build_part("dragon_castle")


def _pos(n):
    return n["transform"]["position"]


def _par(n, key):
    return n["geometry"]["params"][key]


def test_table_contacts_hold():
    import morphix_scene_parts as P
    nodes = P.build_part("table_set")["nodes"]
    top = next(n for n in nodes if n["id"] == "top")
    legs = [n for n in nodes if "leg" in n["id"]]
    assert len(legs) == 4
    tops = top["geometry"]["params"]
    top_base = _pos(top)[1] - tops["h"] / 2
    assert tops["h"] <= 0.3 * min(tops["w"], tops["d"])   # thin slab
    for leg in legs:
        p = _pos(leg)
        h = _par(leg, "h")
        assert abs(p[1] - h / 2) <= 1e-9                  # grounded
        assert abs((p[1] + h / 2) - top_base) <= 1e-9     # supports top
        assert abs(p[0]) < tops["w"] / 2 and abs(p[2]) < tops["d"] / 2


def test_room_walls_meet_floor():
    import morphix_scene_parts as P
    by_id = {n["id"]: n for n in
             P.build_part("room_shell")["nodes"]}
    floor_top = _pos(by_id["floor"])[1] + _par(by_id["floor"], "h") / 2
    assert floor_top == pytest.approx(0.0)
    for wid in ("wall_back", "wall_side"):
        w = by_id[wid]
        assert _pos(w)[1] - _par(w, "h") / 2 == pytest.approx(floor_top)


def test_lighthouse_bands_stack_without_gaps():
    import morphix_scene_parts as P
    bands = sorted(
        (n for n in P.build_part("lighthouse_tower")["nodes"]
         if n["id"].startswith("band_")),
        key=lambda n: _pos(n)[1])
    assert len(bands) == 5
    for lo, hi in zip(bands, bands[1:]):
        lo_top = _pos(lo)[1] + _par(lo, "h") / 2
        hi_base = _pos(hi)[1] - _par(hi, "h") / 2
        assert abs(lo_top - hi_base) <= 0.03


def test_parts_showroom_renders():
    """All assemblies composed into one scene render with zero errors."""
    import morphix_scene_compile as C
    import morphix_scene_ir as S
    import morphix_scene_parts as P
    from scene_browser import render_scenes
    ir = S.new_scene("showroom", seed=9)
    ir["camera"] = {"kind": "perspective", "fov": 55, "near": 0.1,
                    "far": 500, "position": [14, 10, 18],
                    "target": [0, 1, 0]}
    placements = [("table_set", [-9, 0, -6]), ("room_shell", [9, 0, -6]),
                  ("lighthouse_tower", [-9, 0, 6]),
                  ("cabin", [0, 0, -7]), ("castle_keep", [9, 0, 6]),
                  ("tent", [0, 0, 6]), ("boat", [-4, 0.4, 0]),
                  ("lamp_post", [4, 0, 0]), ("pier", [0, 0.4, -2])]
    for i, (name, pos) in enumerate(placements):
        part = P.build_part(name, prefix=f"s{i}_{name}_")
        for n in part["nodes"]:
            t = n["transform"]
            t["position"] = [t["position"][0] + pos[0],
                             t["position"][1] + pos[1],
                             t["position"][2] + pos[2]]
            ir["nodes"].append(n)
    assert S.validate_scene_ir(ir) == []
    assert C.validate_scene_js(C.compile_scene_ir(ir)) == []
    os.makedirs(ACTUAL, exist_ok=True)
    res = render_scenes([("showroom", ir,
                          os.path.join(ACTUAL, "_scene_showroom.png"))])
    r = res["showroom"]
    assert r["ready"] and r["scene_error"] is None
    assert r["console_errors"] == []
    assert r["stats"]["drawCalls"] >= 20
