# tests/test_car_wheels.py
# A thinker-emitted stubby wheel next to full-size siblings once printed
# lopsided with success=True and zero warnings: the car gate checked count,
# radius range, spread and separation, but never cross-wheel coherence.
#
# Design contract (do not "fix" by breaking it): uniform wheel sets of ANY
# sane proportion are the thinker's design freedom — the frozen fixture
# itself runs r=0.35/h=0.5 wheels, and the suite requires kept geometry to
# be preserved. So repair keeps a valid stubby wheel, and the gate reports
# the inconsistency loudly (vetoing quality) instead of celebrating it.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _thinker_car(stubby=True):
    wheels = []
    for lab, x, z in (("wheel_front_left", 1.0, 0.65),
                      ("wheel_front_right", 1.0, -0.65),
                      ("wheel_rear_left", -1.0, 0.7),
                      ("wheel_rear_right", -1.0, -0.65)):
        h = 0.4 if (stubby and lab == "wheel_rear_left") else 0.6
        y = h / 2
        wheels.append({"action": "add_cylinder", "radiusTop": 0.3,
                       "radiusBottom": 0.3, "height": h,
                       "position": [x, y, z], "color": "0x222222",
                       "label": lab, "material": "tire"})
    return [
        {"action": "add_box", "width": 3.0, "height": 0.6, "depth": 1.6,
         "position": [0, 0.3, 0], "color": "0xFF0000", "label": "car_body",
         "material": "body"},
        {"action": "add_box", "width": 1.6, "height": 0.5, "depth": 1.4,
         "position": [0, 0.85, 0], "color": "0x44CCFF", "label": "cabin_top",
         "material": "accent"},
        *wheels,
    ]


def test_stubby_wheel_is_kept_but_reported():
    # Valid geometry is preserved (established contract); nothing is
    # silently reshaped. The gate below is what makes it loud.
    from morphix_specs import repair_plan_for_spec
    out, _repairs = repair_plan_for_spec("car", _thinker_car(stubby=True))
    kept = [a for a in out if a.get("label") == "wheel_rear_left"]
    assert kept and kept[0]["height"] == pytest.approx(0.4)


def test_gate_warns_on_inconsistent_wheels():
    from morphix_specs import CAR_SPEC, check_plan
    errs, warns = check_plan(CAR_SPEC, _thinker_car(stubby=True))
    assert errs == []
    assert any("inconsistent heights" in w for w in warns), warns


def test_gate_stays_quiet_on_uniform_sets():
    from morphix_specs import CAR_SPEC, check_plan
    plan = _thinker_car(stubby=False)
    for a in plan:
        if "wheel" in a.get("label", ""):
            a["height"] = 0.55
            a["position"][1] = 0.275
    _errs, warns = check_plan(CAR_SPEC, plan)
    assert not [w for w in warns if "inconsistent" in w], warns


def test_lopsided_car_prints_honestly_never_silently(monkeypatch, tmp_path):
    # All four wheels print with real deposits (nothing missing), but the
    # run is reported complete-with-warnings — success stays False.
    import morphix_pipeline as P
    monkeypatch.setattr(P, "think",
                        lambda *a, **k: (_thinker_car(stubby=True), "MOCK"))
    rep = P.run_path_print("car", out_dir=str(tmp_path), guided=False,
                           stream_scene=False, deadline_s=300)
    wheels = [c for c in rep["components"] if "wheel" in c["component"]]
    assert len(wheels) == 4
    assert all(c["status"] == "completed" for c in wheels)
    assert all(c["planned_deposits"] > 0 and
               c["committed_deposits"] >= c["planned_deposits"]
               for c in wheels)
    assert any("inconsistent heights" in w
               for w in rep.get("fidelity_warnings", []))
    assert rep["complete"] is True
    assert rep["success"] is False
