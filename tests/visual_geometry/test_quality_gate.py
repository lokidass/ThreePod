# tests/visual_geometry/test_quality_gate.py
# Unit tests for the final quality gate interface: stage ordering,
# failure classification, and the visual-failure distinction.
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from quality import quality_gate, STAGE_ORDER, STAGE_CLASS


def test_all_clear_accepts():
    v = quality_gate({})
    assert v == {"success": True, "stage": "accept",
                 "classification": "accept", "failures": []}


def test_first_failure_wins_in_pipeline_order():
    v = quality_gate({"mesh": ["bad tris"], "spec": ["no body"]})
    assert v["success"] is False
    assert v["stage"] == "spec"
    assert v["classification"] == "specification"
    assert len(v["failures"]) == 2


@pytest.mark.parametrize("stage,cls", [
    ("spec", "specification"), ("category", "specification"),
    ("mesh", "compiler"), ("continuity", "compiler"),
    ("completeness", "executor"), ("fidelity", "specification"),
    ("visual", "rendering"),
])
def test_stage_classification(stage, cls):
    others = {s: [] for s in STAGE_ORDER if s != stage}
    v = quality_gate({**others, stage: ["boom"]})
    assert v["success"] is False
    assert v["stage"] == stage
    assert v["classification"] == cls
    assert v["failures"] == [f"{stage}: boom"]


def test_visual_failure_is_distinct_from_geometry_failure():
    v = quality_gate({"spec": [], "category": [], "mesh": [],
                      "continuity": [], "completeness": [], "fidelity": [],
                      "visual": ["front SSIM 0.91 < 0.985"]})
    assert v["success"] is False
    assert v["stage"] == "visual"
    assert v["classification"] == "rendering"
    assert "front SSIM" in v["failures"][0]


def test_gate_never_marks_partial_success():
    for stage in STAGE_ORDER:
        v = quality_gate({stage: ["x"]})
        assert v["success"] is False
