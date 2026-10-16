# morphix_scene_verify.py
# Render-and-repair feedback loop (scene mode). Never claims success from
# compilation alone: every verdict requires an actual browser render, and
# every repair is bounded, logged, and re-rendered.
#
#   validate IR -> compile -> render -> evaluate -> repair? -> re-render ...
#
# Findings name the failing stage precisely (not-ready, scene error,
# console errors, empty render, instance shortfall). The only automatic
# repairs are ones that cannot change scene design: retrying a failed WebGL
# initialization in safe mode (pixelRatio 1, no antialias). Anything else
# (bad APIs, shader errors, empty geometry) is reported, never patched
# over — a failed repair budget ends in an explicit failure verdict.

import morphix_scene_compile as _C
import morphix_scene_ir as _S

MAX_REPAIRS = 2


def evaluate_render(ir, result):
    """Judge one rendered scene. Returns (ok, findings[]) where each
    finding is {"stage", "detail"}. ok requires: ready, zero errors,
    at least one draw call, and real geometry (triangles, unless the
    scene is points-only by design). Instance shortfalls are warnings,
    never silent."""
    findings = []
    result = result or {}
    if not result.get("ready"):
        findings.append({"stage": "render",
                         "detail": "renderer never signalled ready"})
    if result.get("scene_error"):
        findings.append({"stage": "render",
                         "detail": f"scene error: {result['scene_error']}"})
    for ce in result.get("console_errors", [])[:5]:
        findings.append({"stage": "render",
                         "detail": f"console error: {ce[:200]}"})
    stats = result.get("stats") or {}
    if result.get("ready") and stats.get("drawCalls", 0) < 1:
        findings.append({"stage": "render",
                         "detail": "ready but zero draw calls (empty render)"})
    kinds = {n.get("kind") for n in ir.get("nodes", [])
             if isinstance(n, dict)}
    if result.get("ready") and stats.get("triangles", 0) <= 0 \
            and kinds - {"points"}:
        findings.append({"stage": "render",
                         "detail": "ready but zero triangles with mesh "
                                   "nodes present"})
    planned = _S.instance_total(ir)
    placed = stats.get("instances")
    if planned and isinstance(placed, int) and placed < planned:
        findings.append({"stage": "content",
                         "detail": f"placed {placed} of {planned} instances "
                                   f"(scatter rejection); scene is sparser "
                                   f"than planned",
                         "warning": True})
    fatal = [f for f in findings if not f.get("warning")]
    return (not fatal, findings)


def repair_for_findings(ir, findings, safe=False):
    """Return (repaired_ir|None, note). The only automatic repair retries
    failed WebGL initialization in safe mode. safe=True means safe mode is
    already on: nothing left to try."""
    texts = " ".join(f.get("detail", "") for f in findings).lower()
    if any(k in texts for k in ("webgl", "context", "antialias",
                                "framebuffer")) and not safe:
        return dict(ir), ("renderer init failed: retrying safe mode "
                          "(pixelRatio 1, antialias off)")
    return None, ""


def render_and_repair(ir, render_one, max_repairs=MAX_REPAIRS,
                      pixel_ratio=1, antialias=False):
    """Full loop. render_one(js, pixel_ratio, antialias) -> result dict
    (ready/scene_error/console_errors/stats/elapsed_s). Returns a verdict
    dict that is ok ONLY on a clean rendered frame."""
    errs = _S.validate_scene_ir(ir)
    if errs:
        return {"ok": False, "stage": "ir",
                "error": "scene IR invalid: " + "; ".join(errs[:5]),
                "repairs": [], "attempts": 0}
    try:
        js = _C.compile_scene_ir(ir, pixel_ratio=pixel_ratio,
                                 antialias=antialias)
    except ValueError as e:
        return {"ok": False, "stage": "compile", "error": str(e),
                "repairs": [], "attempts": 0}
    js_errs = _C.validate_scene_js(js)
    if js_errs:
        return {"ok": False, "stage": "compile",
                "error": "; ".join(js_errs[:5]), "repairs": [], "attempts": 0}
    repairs, attempts = [], 0
    safe = (pixel_ratio == 1 and not antialias)
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            result = render_one(js, pixel_ratio, antialias)
        except Exception as e:
            result = {"ready": False, "scene_error": f"harness: {e}",
                      "console_errors": [], "stats": {}, "elapsed_s": 0.0}
        ok, findings = evaluate_render(ir, result)
        if ok:
            return {"ok": True, "stage": "render", "error": "",
                    "findings": findings, "repairs": repairs,
                    "attempts": attempts, "stats": result.get("stats", {}),
                    "summary": _S.summarize(ir)}
        fixed, note = repair_for_findings(ir, findings, safe=safe)
        if fixed is None:
            return {"ok": False, "stage": "render",
                    "error": "; ".join(f["detail"]
                                       for f in findings
                                       if not f.get("warning")) or
                    "render failed",
                    "findings": findings, "repairs": repairs,
                    "attempts": attempts}
        repairs.append(note)
        pixel_ratio, antialias, safe = 1, False, True
    return {"ok": False, "stage": "render",
            "error": "repair budget exhausted without a clean render",
            "repairs": repairs, "attempts": attempts}
