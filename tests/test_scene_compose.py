# tests/test_scene_compose.py
# Request-driven composition (Phase 4.1): deterministic keyword routing,
# freestyle validation against the knowledge registry, routed composition
# with required-element coverage, and loud failure on unresolvable plans.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BENCHMARK_PROMPTS = {
    "a wooden table in a room": ([], ["table_set", "room_shell"]),
    "a forested mountain landscape": (["forest", "mountains"], []),
    "a modern city street at night": (["city"], ["lamp_post"]),
    "a detailed sci-fi environment": (["scifi"], []),
    "a lighthouse on a cliff at dusk": (["mountains"], ["lighthouse_tower"]),
    "a desert caravan at noon": ([], ["tent", "dunes"]),
    "a snowy cabin in the woods": (["forest"], ["cabin"]),
    "a harbor with boats at sunrise": (["water"], ["boat", "pier"]),
    "a castle on a hill": (["mountains"], ["castle_keep"]),
    "a coral reef underwater": (["water", "particles"], ["coral_bed"]),
}


def test_router_covers_all_benchmark_prompts():
    import morphix_scene_plan as P
    for prompt, (templates, parts) in BENCHMARK_PROMPTS.items():
        r = P.route_systems(prompt)
        for t in templates:
            assert t in r["templates"], (prompt, r)
        for p in parts:
            assert p in r["parts"], (prompt, r)
        assert r["required"], prompt       # every route names requirements


def test_router_rejects_gibberish_and_avoids_false_hits():
    import morphix_scene_plan as P
    r = P.route_systems("a seat by the window")
    assert r["templates"] == [] and r["parts"] == []
    r = P.route_systems("some random xyzzy bildings")
    assert r["templates"] == [] and r["parts"] == []


def test_freestyle_validation_gates_constructors():
    import morphix_scene_plan as P
    good, errs = P.validate_freestyle({
        "name": "obelisk", "kind": "mesh",
        "geometry": {"ctor": "ConeGeometry", "params": {"r": 1, "h": 4}},
        "material": {"kind": "standard", "color": "0x888888"},
        "transform": {"position": [0, 2, 0]}, "required": True})
    assert errs == [] and good["id"] == "free_obelisk"
    assert good["required"] is True
    bad, errs = P.validate_freestyle({
        "name": "webgpu_thing", "kind": "mesh",
        "geometry": {"ctor": "WebGPURenderer", "params": {}},
        "material": {"kind": "standard"}, "transform": {}})
    assert bad is None and any("unsupported" in e for e in errs)
    bad, errs = P.validate_freestyle({
        "name": "far", "kind": "mesh",
        "geometry": {"ctor": "BoxGeometry",
                     "params": {"w": 1, "h": 1, "d": 1}},
        "material": {"kind": "standard"},
        "transform": {"position": [500, 0, 0]}})
    assert bad is None and any("bounds" in e for e in errs)


def test_composition_marks_required_and_refuses_gaps():
    import morphix_scene_ir as S
    import morphix_scene_plan as P
    for prompt in BENCHMARK_PROMPTS:
        routed = P.route_systems(prompt)
        ir, rep = P.compose_routed_scene(routed, {"objective": prompt},
                                         seed=11)
        assert S.validate_scene_ir(ir) == [], prompt
        assert rep["required_ids"], prompt
        assert S.summarize(ir)["required"], prompt
    # an unresolvable requirement fails loudly, never silently incomplete
    routed = P.route_systems("a lighthouse on a cliff at dusk")
    with pytest.raises(ValueError) as ei:
        P.compose_routed_scene(
            routed, {"objective": "x", "required": ["dragon_castle_xyz"]},
            seed=11)
    assert "dragon_castle_xyz" in str(ei.value)


def test_required_aliases_resolve_without_rewriting_compounds():
    import morphix_scene_plan as P
    routed = P.route_systems("a harbor with boats at sunrise")
    # bare synonym resolves to the boat system ...
    ir, rep = P.compose_routed_scene(
        routed, {"objective": "h", "required": ["ship"]}, seed=11)
    assert any("boat" in nid for nid in rep["required_ids"])
    assert any("aliased" in r for r in rep["resolutions"])
    # ... while a compound name is never rewritten ...
    ir, rep = P.compose_routed_scene(
        P.route_systems("a lighthouse on a cliff at dusk"),
        {"objective": "l", "required": ["lighthouse_tower"]}, seed=11)
    assert any(nid.startswith("lighthouse_tower_")
               for nid in rep["required_ids"])
    assert not [r for r in rep["resolutions"] if "aliased" in r]
    # ... and genuine unknowns still fail loudly
    import morphix_scene_ir as S
    with pytest.raises(ValueError):
        P.compose_routed_scene(
            routed, {"objective": "h",
                     "required": ["dragon_castle_xyz"]}, seed=11)


def test_malformed_freestyle_and_relations_cannot_crash():
    import morphix_scene_plan as P
    routed = P.route_systems("a harbor with boats at sunrise")
    # freestyle as bare strings + relations as a string: dropped/ignored
    # loudly, composition proceeds on the routed systems
    ir, rep = P.compose_routed_scene(
        routed, {"objective": "h", "freestyle": ["a tower", 42],
                 "relations": "near water"}, seed=11)
    assert any("dropped" in r for r in rep["resolutions"])
    import morphix_scene_ir as S
    assert S.validate_scene_ir(ir) == []


def test_composition_is_deterministic():
    import morphix_scene_compile as C
    import morphix_scene_plan as P
    routed = P.route_systems("a harbor with boats at sunrise")
    a, _ = P.compose_routed_scene(routed, {"objective": "h"}, seed=11)
    b, _ = P.compose_routed_scene(routed, {"objective": "h"}, seed=11)
    assert C.compile_scene_ir(a) == C.compile_scene_ir(b)
