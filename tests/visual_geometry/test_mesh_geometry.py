# tests/visual_geometry/test_mesh_geometry.py
# Geometry-before-rendering acceptance: every fixture compiles through the
# production chain, executes segment by segment, and the committed mesh is
# checked against intent (bbox, contacts, materials, fill, validity).
# No live model inference; plans are frozen literals.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from fixtures import get_fixture, FIXTURES
from morphix_mesh import (deposits_to_mesh, mesh_bbox, check_mesh_validity,
                  check_no_duplicate_deposits, check_same_material_overlaps,
                  component_aabb, aabbs_touch_or_overlap, check_round_fill,
                  check_box_fill)
from morphix_specs import SPECS, repair_plan_for_spec, check_plan
from morphix_path import (compile_plan, merge_layers, optimize_toolpath,
                          PathState, validate_geometry)
from morphix_validator import validate_actions


def _build(name):
    """Production chain: repair -> gate -> compile -> optimize -> execute."""
    fx = get_fixture(name)
    plan = fx["plan"]
    spec = None
    if fx["spec"]:
        spec = next(s for s in SPECS if s["name"] == fx["spec"])
        plan, _ = repair_plan_for_spec(fx["spec"], plan)
        errs, _ = check_plan(spec, plan)
        assert errs == [], (name, errs)
    compiled, order, errs, keep, warns = compile_plan(plan, {})
    assert errs == {}, (name, errs)
    script, layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    opt, stats = optimize_toolpath(script)
    assert stats["fallback"] is False, (name, stats)
    assert validate_geometry(opt) == [], name
    st = PathState(total_layers=layers)
    recs = []
    for a in opt:
        assert validate_actions([a]) + st.check(a) == [], (name, a)
        r = st.apply(a)
        if r.get("action") in ("deposit_segment", "deposit_point"):
            recs.append(r)
    return fx, plan, recs, layers


def _close(a, b, tol):
    return abs(a - b) <= tol


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_mesh_valid_no_dupes_no_overlaps(name):
    _, _, recs, _ = _build(name)
    mesh = deposits_to_mesh(recs)
    assert check_mesh_validity(mesh) == []
    assert check_no_duplicate_deposits(recs) == []
    assert check_same_material_overlaps(mesh) == []


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_bbox_matches_plan_footprint(name):
    fx, plan, recs, _ = _build(name)
    mesh = deposits_to_mesh(recs)
    lo, hi = mesh_bbox(mesh)
    xs, ys, zs = [], [], []
    for a in plan:
        p = a["position"]
        if a["action"] == "add_box":
            w, h, d = a["width"], a["height"], a["depth"]
            xs += [p[0] - w / 2, p[0] + w / 2]
            ys += [p[1] - h / 2, p[1] + h / 2]
            zs += [p[2] - d / 2, p[2] + d / 2]
        elif a["action"] == "add_cylinder":
            r = max(a.get("radiusTop", a.get("radius", 0)),
                    a.get("radiusBottom", a.get("radius", 0)))
            h = a["height"]
            xs += [p[0] - r, p[0] + r]
            ys += [p[1] - h / 2, p[1] + h / 2]
            zs += [p[2] - r, p[2] + r]
        elif a["action"] == "add_sphere":
            r = a["radius"]
            xs += [p[0] - r, p[0] + r]
            ys += [p[1] - r, p[1] + r]
            zs += [p[2] - r, p[2] + r]
    # bead courses straddle the solid by half a bead (0.1); the top course
    # plane sits up to one layer below the solid top.
    for got, want_lo, want_hi in ((lo[0], min(xs), None),
                                  (hi[0], None, max(xs)),
                                  (lo[1], min(ys), None),
                                  (hi[1], None, max(ys)),
                                  (lo[2], min(zs), None),
                                  (hi[2], None, max(zs))):
        if want_lo is not None:
            assert got >= want_lo - 0.25, (name, got, want_lo)
            assert got <= want_lo + 0.15, (name, got, want_lo)
        else:
            assert got >= want_hi - 0.25, (name, got, want_hi)
            assert got <= want_hi + 0.15, (name, got, want_hi)
    # grounded objects rest on the plate (first course straddles y=0)
    assert -0.15 <= lo[1] <= 0.05, (name, lo[1])


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_contacts_and_materials(name):
    fx, plan, recs, _ = _build(name)
    mesh = deposits_to_mesh(recs)
    for a, b in fx["contacts"]:
        if b == "ground":
            bb = component_aabb(mesh, a)
            assert bb is not None, (name, a)
            assert -0.15 <= bb[0][1] <= 0.05, (name, a, bb[0][1])
        else:
            aa, ab = component_aabb(mesh, a), component_aabb(mesh, b)
            assert aa is not None and ab is not None, (name, a, b)
            assert aabbs_touch_or_overlap(aa, ab), (name, a, b)
    want_mat = {}
    for a in plan:
        want_mat[a["label"]] = a.get("material")
    got_mat = {}
    for r in recs:
        c = r.get("component")
        if r.get("material") != want_mat.get(c):
            got_mat[c] = r.get("material")
    assert got_mat == {}, (name, got_mat)
    for comp, mat in fx["materials"].items():
        assert any(r.get("component") == comp for r in recs), (name, comp)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fill_coverage(name):
    fx, plan, recs, _ = _build(name)
    for a in plan:
        if a["action"] == "add_box":
            errs = check_box_fill(recs, a["label"], a["width"], a["depth"])
            assert errs == [], (name, errs)
        elif a["action"] in ("add_cylinder", "add_sphere"):
            p = a["position"]
            if a["action"] == "add_sphere":
                r = a["radius"]
                errs = check_round_fill(recs, a["label"], r, r,
                                        p[1] - r, p[1] + r,
                                        spherical=True)
            else:
                rt = a.get("radiusTop", a.get("radius", 0))
                rb = a.get("radiusBottom", a.get("radius", 0))
                errs = check_round_fill(recs, a["label"], rb, rt,
                                        p[1] - a["height"] / 2,
                                        p[1] + a["height"] / 2)
            assert errs == [], (name, errs)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_layers_complete_and_ordered(name):
    fx, plan, recs, layers = _build(name)
    got = sorted({r.get("layer") for r in recs})
    assert got == list(range(layers)), (name, got, layers)
