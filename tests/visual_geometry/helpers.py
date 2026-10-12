# tests/visual_geometry/helpers.py
# Shared deterministic build chain for visual-geometry tests: frozen plan
# -> repair/gate (when the fixture has a spec) -> compile -> optimize ->
# validate -> real segment executor -> committed records + scene JS.
# No live model inference anywhere in this module.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from fixtures import get_fixture
from morphix_specs import SPECS, repair_plan_for_spec, check_plan
from morphix_path import (compile_plan, merge_layers, optimize_toolpath,
                          PathState, validate_geometry, render_deposit)
from morphix_validator import validate_actions


def build_scene(name):
    """Returns (scene_js, records, layers, plan, script). Raises loudly on
    any defect — fixtures must be clean by construction."""
    fx = get_fixture(name)
    plan = fx["plan"]
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
    recs, finals = [], []
    for a in opt:
        assert validate_actions([a]) + st.check(a) == [], (name, a)
        r = st.apply(a)
        if r.get("action") in ("deposit_segment", "deposit_point"):
            recs.append(r)
            finals.append(render_deposit(r))
    scene_js = "\n\n".join(finals) + "\n" if finals else "// empty\n"
    return scene_js, recs, layers, plan, opt
