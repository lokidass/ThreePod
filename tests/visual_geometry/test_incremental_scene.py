# tests/visual_geometry/test_incremental_scene.py
# Section 4: the committed incremental scene must preserve the intended
# geometry. Runs the PRODUCTION executor (run_path_print, frozen plan, no
# LLM) for representative fixtures, then checks the committed records and
# report against intent — dimensions, positions, topology, materials.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import morphix_pipeline as P
from fixtures import get_fixture
from morphix_mesh import (deposits_to_mesh, mesh_bbox, check_mesh_validity,
                          check_no_duplicate_deposits,
                          check_same_material_overlaps, component_aabb,
                          aabbs_touch_or_overlap)


def _run(name, tmp_path):
    fx = get_fixture(name)
    out = str(tmp_path)
    rep = P.run_path_print(name, out_dir=out, plan_actions=fx["plan"],
                           spec_name=fx["spec"], stream_scene=False,
                           deadline_s=300)
    assert rep["success"] is True, (name, rep.get("fidelity_warnings"))
    ckpt = json.load(open(os.path.join(out, "path.json"), encoding="utf-8"))
    return fx, rep, ckpt


@pytest.mark.parametrize("name", ["chair", "car", "box_sphere"])
def test_production_executor_preserves_intent(name, tmp_path):
    fx, rep, ckpt = _run(name, tmp_path)
    recs = [r for r in ckpt["records"]]
    assert len(recs) > 10
    mesh = deposits_to_mesh(recs)
    assert check_mesh_validity(mesh) == []
    assert check_no_duplicate_deposits(recs) == []
    assert check_same_material_overlaps(mesh) == []
    # committed layers cover every planned layer, in order
    script_layers = sorted({a.get("layer") for a in ckpt["script"]
                            if isinstance(a.get("layer"), int)})
    committed_layers = sorted({r.get("layer") for r in recs})
    assert committed_layers == script_layers
    # contacts from the fixture hold on committed geometry
    for a, b in fx["contacts"]:
        if b == "ground":
            bb = component_aabb(mesh, a)
            assert bb is not None and -0.15 <= bb[0][1] <= 0.05, (name, a)
        else:
            aa, ab = component_aabb(mesh, a), component_aabb(mesh, b)
            assert aa is not None and ab is not None, (name, a, b)
            assert aabbs_touch_or_overlap(aa, ab), (name, a, b)
    # materials committed as planned
    for comp, mat in fx["materials"].items():
        got = {r.get("material") for r in recs
               if r.get("component") == comp}
        assert got == {mat}, (name, comp, got)


def test_committed_scene_matches_report_counts(tmp_path):
    fx, rep, ckpt = _run("chair", tmp_path)
    assert rep["deposits"] == len(ckpt["records"])
    assert rep["path_actions"] >= len(ckpt["records"])
