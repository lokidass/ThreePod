# tests/visual_geometry/quality.py
# Final quality gate: one structured verdict combining every independent
# status. Kept separate from the schema validator (validate_actions) and
# the per-action guard (validate_js) — those prove individual actions are
# legal; this proves the OBJECT is right.
#
# Stage -> failure classification (for "where did it break" reports):
#   spec         -> "specification"  (intent/plan wrong)
#   category     -> "specification"  (category constraints violated)
#   mesh         -> "compiler"       (sliced geometry defective)
#   continuity   -> "compiler"       (toolpath disconnected)
#   completeness -> "executor"       (committed != planned)
#   fidelity     -> "specification"  (reconstruction vs intent)
#   visual       -> "rendering"      (pixels differ while geometry passes)
#
# A run reports success ONLY when every stage passes. A syntactically
# valid scene that fails any stage reports a distinct, classified failure
# — never success, never a silent repair.
STAGE_ORDER = ("spec", "category", "mesh", "continuity", "completeness",
               "fidelity", "visual")

STAGE_CLASS = {
    "spec": "specification",
    "category": "specification",
    "mesh": "compiler",
    "continuity": "compiler",
    "completeness": "executor",
    "fidelity": "specification",
    "visual": "rendering",
}


def quality_gate(stages: dict) -> dict:
    """stages: {stage_name: [error strings]} (missing stage = pass).
    Returns {"success": bool, "stage": first-failing stage or "accept",
    "classification": ..., "failures": [...]}."""
    failures = []
    first = None
    for stage in STAGE_ORDER:
        errs = stages.get(stage) or []
        if errs:
            if first is None:
                first = stage
            failures.extend(f"{stage}: {e}" for e in errs)
    if first is None:
        return {"success": True, "stage": "accept",
                "classification": "accept", "failures": []}
    return {"success": False, "stage": first,
            "classification": STAGE_CLASS[first], "failures": failures}
