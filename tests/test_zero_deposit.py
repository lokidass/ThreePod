# tests/test_zero_deposit.py
# Regression suite for the "engine left components incomplete" failure class:
# required components that silently ended up with zero planned deposits.
#
# Root causes fixed (see module docstrings for the full trace):
#   1. _plan_for_print paired a FAILED attempt's plan labels with a STALE
#      attempt's compiled script (franken-plan): every renamed component read
#      as planned=0/committed=0/missing downstream. Now the omit path only
#      ever prints the attempt that actually reached the slicer.
#   2. repair_cat_plan killed the whole run (ValueError) over one part's
#      wrong primitive opcode; the small thinker emits boxes for necks/heads/
#      ears constantly. Now such parts are dropped LOUDLY and replaced by
#      canonical synthesis, mirroring the long-standing legs treatment.
#   3. compile_component could return EMPTY layer groups (sub-layer sliver
#      stacked high in a layer band) with no error. Now it raises naming the
#      component and the slicing stage.
#   4. Deterministic resume spliced a foreign checkpoint onto the current
#      script by position. Now it refuses loudly.
# All model access is mocked or bypassed (plan_actions); geometry is
# deterministic.
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _body_box(label="body"):
    return {"action": "add_box", "width": 1.3, "height": 0.7, "depth": 0.7,
            "position": [0, 0.75, 0], "color": "0xFFAA33", "label": label,
            "material": "body"}


def _plane(label="backdrop"):
    return {"action": "add_plane", "width": 8, "height": 3,
            "position": [0, 0.5, 0], "color": "0x888888", "label": label,
            "material": "body"}


def _sliver_on_slab():
    slab = {"action": "add_box", "width": 1.0, "height": 0.55, "depth": 1.0,
            "position": [0, 0.275, 0], "color": "0xFFAA33", "label": "slab",
            "material": "body"}
    # 0.03 tall, base y=0.55: single layer band 2 whose course plane (0.4,
    # bead reaches 0.5) sits below the base — no contour exists.
    sliver = {"action": "add_box", "width": 0.5, "height": 0.03,
              "depth": 0.5, "position": [0, 0.565, 0], "color": "0xFFAA33",
              "label": "sliver", "material": "body"}
    return [slab, sliver]


# ------------------------------------------------- canonical cat deposits
def test_every_required_cat_component_gets_valid_deposits():
    import morphix_ir as IR
    from morphix_path import compile_plan, merge_layers, validate_geometry

    ir = IR.ir_for_category("cat")
    by_role = {}
    for c in ir["components"]:
        by_role.setdefault(c["role"], []).append(c["id"])
    assert by_role.get("body") and by_role.get("head")
    assert len(by_role.get("ear", [])) == 2
    assert len(by_role.get("leg", [])) == 4
    assert by_role.get("tail")

    actions = IR.ir_to_actions(ir)
    compiled, order, errors, _keep, _warns = compile_plan(actions, {})
    assert errors == {}
    for lab in order:
        layers = sorted(compiled.get(lab, {}).keys())
        n = sum(1 for k in layers for a in compiled[lab][k]
                if a.get("action") in ("deposit_segment", "deposit_point"))
        assert layers, lab                       # real layer coverage ...
        assert n > 0, lab                        # ... with real deposits
    # tapered parts keep their profile through slicing
    tapers = {a["label"]: (a.get("radiusTop"), a.get("radiusBottom"))
              for a in actions if a.get("radiusTop") != a.get("radiusBottom")}
    assert len(tapers) >= 4                      # ears + neck + tail (+legs)
    script, _layers = merge_layers(
        [compiled[lab] for lab in order])
    assert validate_geometry(script) == []


# ------------------------------------------------- slicing disappearance
def test_sliver_slicing_error_names_component_and_stage():
    from morphix_path import compile_component, compile_plan

    occ = {}
    slab = _sliver_on_slab()[0]
    groups, foot, (lo, hi), _w = compile_component(slab, occ, prefix="p00")
    assert groups
    import math
    from morphix_path import LAYER_HEIGHT
    k0 = max(0, int(math.floor((lo + 1e-9) / LAYER_HEIGHT)))
    k1 = max(k0, int(math.ceil((hi - 1e-9) / LAYER_HEIGHT)) - 1)
    for k in range(k0, k1 + 1):
        occ.setdefault(k, []).append(foot + (hi,))
    with pytest.raises(ValueError) as ei:
        compile_component(_sliver_on_slab()[1], occ, prefix="p01")
    assert "sliver" in str(ei.value)
    assert "slicing" in str(ei.value)

    _c, _o, errors, _k, _w = compile_plan(_sliver_on_slab(), {})
    assert "sliver" in errors
    assert "slicing" in errors["sliver"]


def test_slicing_disappearance_fails_the_run_honestly(tmp_path):
    import morphix_pipeline as P
    from morphix_config import PipelineError

    with pytest.raises(PipelineError) as ei:
        P.run_path_print("sliver test", out_dir=str(tmp_path),
                         plan_actions=_sliver_on_slab(),
                         spec_name="generic", stream_scene=False)
    assert "sliver" in str(ei.value)
    assert "slicing" in str(ei.value)


# ------------------------------------------------- repair robustness
def _garbage_cat_plan():
    """The small thinker's greatest hits: right labels, wrong opcodes."""
    return [
        {"action": "add_cylinder", "radiusTop": 0.09, "radiusBottom": 0.09,
         "height": 0.4, "position": [0, 0.75, 0], "color": "0xFFAA33",
         "label": "body_a1", "material": "body"},
        {"action": "add_box", "width": 0.4, "height": 0.1, "depth": 0.2,
         "position": [0, 0, 0], "color": "0x777777",
         "label": "neck_a1", "material": "body"},
        {"action": "add_box", "width": 0.1, "height": 0.2, "depth": 0.2,
         "position": [0, 0.2, 0], "color": "0x777777",
         "label": "head_a1", "material": "body"},
        {"action": "add_box", "width": 0.06, "height": 0.1, "depth": 0.2,
         "position": [0, 0, 0.2], "color": "0xFF0000",
         "label": "ear_left_a1", "material": "body"},
        {"action": "add_box", "width": 0.08, "height": 0.1, "depth": 0.2,
         "position": [0, 0.6, 0.3], "color": "0x777777",
         "label": "tail_a1", "material": "body"},
    ]


def test_wrong_primitive_parts_are_replaced_loudly_not_fatally():
    from morphix_specs import (CAT_SPEC, check_plan, repair_plan_for_spec)

    out, repairs = repair_plan_for_spec("cat", _garbage_cat_plan())
    reasons = " ".join(r["reason"] for r in repairs)
    assert "not a printable box" in reasons      # body was a cylinder
    assert "not a printable cylinder" in reasons  # neck was a box
    assert "not a printable sphere" in reasons    # head was a box
    labels = [a["label"] for a in out]
    assert "body" in labels and "neck" in labels and "head" in labels
    assert "tail" in labels
    errs, warns = check_plan(CAT_SPEC, out)
    assert errs == [], errs


def test_mislabeled_ear_does_not_hijack_the_tail_slot():
    from morphix_specs import (CAT_SPEC, check_plan, repair_plan_for_spec)

    plan = [_body_box("body_a1"),
            {"action": "add_cylinder", "radiusTop": 0.03,
             "radiusBottom": 0.12, "height": 0.28,
             "position": [0.45, 2.26, 0.25], "color": "0xFFAA33",
             "label": "tail_a1", "material": "body"}]
    out, repairs = repair_plan_for_spec("cat", plan)
    assert any("not at the body rear" in r["reason"] for r in repairs)
    tails = [a for a in out if "tail" in a.get("label", "")]
    assert len(tails) == 1
    assert tails[0]["position"][0] < 0            # canonical rear tail
    errs, warns = check_plan(CAT_SPEC, out)
    assert errs == [], errs
    assert not [w for w in warns if "not at the body rear" in w]


# ------------------------------------------------- attempt-identity (franken)
def _think_box_plus_plane():
    return [_body_box("body"), _plane()]


def test_stale_script_never_pairs_with_a_new_plan(monkeypatch, tmp_path):
    """Attempt 1 compiles a subset; attempt 2 dies BEFORE slicing. The run
    must print attempt 1's consistent subset — never attempt 2's labels
    over attempt 1's script (the screenshot's zero-deposit franken-plan)."""
    import morphix_pipeline as P
    import morphix_specs as S

    real_repair = S.repair_plan_for_spec
    calls = {"n": 0}

    def flaky_repair(spec, actions):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise ValueError("simulated attempt-2 repair failure")
        return real_repair(spec, actions)

    monkeypatch.setattr(S, "repair_plan_for_spec", flaky_repair)
    monkeypatch.setattr(P, "think",
                        lambda *a, **k: (_think_box_plus_plane(), "MOCK"))

    rep = P.run_path_print("cat", out_dir=str(tmp_path), guided=False,
                           stream_scene=False, deadline_s=300)
    assert calls["n"] == 2
    # the unprintable plane is reported loudly ...
    omitted = [o["component"] for o in rep.get("omitted", [])]
    assert "backdrop" in omitted
    # ... the ONLY zero-deposit row is the loudly-omitted plane: no phantom
    # phantoms from a stale script paired with a new plan ...
    zeroes = [c["component"] for c in rep["components"]
              if c["planned_deposits"] == 0]
    assert zeroes == ["backdrop"], zeroes
    for c in rep["components"]:
        if c["component"] not in omitted:
            assert c["planned_deposits"] > 0, c
            assert c["status"] == "completed", c
            assert c["planned_layer_range"] == [c["layers"][0],
                                                c["layers"][-1]], c
    # ... and the partial print is reported, never celebrated.
    assert rep["complete"] is False
    assert rep["success"] is False
    assert rep["ok"] is True                     # execution itself was valid


def test_two_preface_failures_raise_instead_of_printing_franken(
        monkeypatch, tmp_path):
    import morphix_pipeline as P
    from morphix_config import PipelineError

    monkeypatch.setattr(P, "think", lambda *a, **k: ([], "MOCK-EMPTY"))
    with pytest.raises(PipelineError):
        P.run_path_print("cat", out_dir=str(tmp_path), guided=False,
                         stream_scene=False, deadline_s=300)


# ------------------------------------------------- accounting integrity
def test_accounting_matches_committed_actions(tmp_path):
    import morphix_ir as IR
    import morphix_pipeline as P
    from morphix_path import PathState

    out = str(tmp_path)
    rep = P.run_path_print("cat", out_dir=out,
                           plan_actions=IR.ir_to_actions(
                               IR.ir_for_category("cat")),
                           spec_name="cat", stream_scene=False,
                           deadline_s=300)
    assert rep["success"] is True
    actions = json.load(open(os.path.join(out, "actions.json"),
                             encoding="utf-8"))
    st = PathState(total_layers=rep["layers"])
    per_comp = {}
    first, last = {}, {}
    for i, act in enumerate(actions):
        rec = st.apply(act)
        if rec.get("action") in ("deposit_segment", "deposit_point"):
            comp = rec.get("component")
            per_comp[comp] = per_comp.get(comp, 0) + 1
            first.setdefault(comp, i)
            last[comp] = i
    assert st.object_done
    for c in rep["components"]:
        assert c["planned_deposits"] > 0, c          # invariant: no zero plan
        assert c["committed_deposits"] == per_comp.get(c["component"], 0)
        assert c["committed_deposits"] >= c["planned_deposits"]
        assert c["status"] == "completed"
        assert c["planned_layer_range"] == [c["layers"][0], c["layers"][-1]]
        assert c["first_committed_index"] == first[c["component"]]
        assert c["last_committed_index"] == last[c["component"]]


def test_missing_tail_is_an_honest_failure(tmp_path):
    import morphix_ir as IR
    import morphix_pipeline as P
    from morphix_config import PipelineError

    actions = [a for a in IR.ir_to_actions(IR.ir_for_category("cat"))
               if "tail" not in a["label"]]
    with pytest.raises(PipelineError) as ei:
        P.run_path_print("cat", out_dir=str(tmp_path), plan_actions=actions,
                         spec_name="cat", stream_scene=False,
                         deadline_s=300)
    assert "tail" in str(ei.value).lower()


# ------------------------------------------------- cancel + resume
def test_cancel_then_resume_preserves_state(tmp_path):
    import morphix_ir as IR
    import morphix_pipeline as P
    from morphix_config import Cancelled, PipelineError

    plan = IR.ir_to_actions(IR.ir_for_category("cat"))
    out = str(tmp_path / "run")
    seen = {"n": 0}

    def stop_early():
        seen["n"] += 1
        return seen["n"] > 60

    with pytest.raises(Cancelled):
        P.run_path_print("cat", out_dir=out, plan_actions=plan,
                         spec_name="cat", stream_scene=True,
                         should_stop=stop_early, deadline_s=300)
    assert os.path.isfile(os.path.join(out, "path.json"))

    resumed = P.run_path_print("cat", out_dir=out, plan_actions=plan,
                               spec_name="cat", stream_scene=False,
                               resume=True, deadline_s=300)
    assert resumed["success"] is True
    assert all(c["status"] == "completed" for c in resumed["components"])

    fresh = P.run_path_print("cat", out_dir=str(tmp_path / "fresh"),
                             plan_actions=plan, spec_name="cat",
                             stream_scene=False, deadline_s=300)
    assert resumed["deposits"] == fresh["deposits"]

    # ... but resuming a FOREIGN checkpoint onto a new plan fails honestly.
    with pytest.raises(PipelineError) as ei:
        P.run_path_print("cat", out_dir=out,
                         plan_actions=IR.ir_to_actions(
                             IR.ir_for_category("chair")),
                         spec_name="chair", stream_scene=False,
                         resume=True, deadline_s=300)
    assert "resume refused" in str(ei.value)
