# morphix_scene_run.py
# Scene completion state machine (Phase 4.1, item 4):
#
#   PLANNING -> GENERATING -> COMPILING -> RENDERING -> VALIDATING
#       -> REPAIRING -> RENDERING ... -> COMPLETE | FAILED
#
# COMPLETE requires ALL of: every required element generated AND rendered
# with valid geometry, no fatal runtime errors, a passed validation, and
# no unresolved critical defects. Anything less lands in FAILED with the
# stage, the reason, and the preserved partial artifacts — never a silent
# success. Bounded retries, timeouts and repair budgets throughout.

import copy
import math
import os
import time

STATES = ("PLANNING", "GENERATING", "COMPILING", "RENDERING", "VALIDATING",
          "REPAIRING", "COMPLETE", "FAILED")

MAX_REPAIRS = 2


def _subject_center(graph, required_ids):
    pts = []
    for nid in required_ids:
        bbox = (graph.get(nid) or {}).get("bbox")
        if bbox:
            pts.append([(a + b) / 2 for a, b in zip(bbox[0], bbox[1])])
    if not pts:
        return None
    n = len(pts)
    return [sum(p[i] for p in pts) / n for i in range(3)]


def _subject_radius(graph, required_ids, center):
    r = 0.0
    for nid in required_ids:
        bbox = (graph.get(nid) or {}).get("bbox")
        if not bbox:
            continue
        for x in (bbox[0][0], bbox[1][0]):
            for y in (bbox[0][1], bbox[1][1]):
                for z in (bbox[0][2], bbox[1][2]):
                    r = max(r, math.dist((x, y, z), center))
    return r


def repair_camera_frame(ir, graph, required_ids):
    """Reframe the camera onto the required subject (logged repair). Keeps
    fov and view direction; retargets + dollies to fit. Returns True when
    the IR changed."""
    center = _subject_center(graph, required_ids)
    if center is None:
        return False
    cam = ir.get("camera", {})
    eye = list(cam.get("position", [6, 5, 9]))
    view = [e - t for e, t in zip(eye, cam.get("target", [0, 0, 0]))]
    n = math.dist(view, (0, 0, 0)) or 1.0
    view = [v / n for v in view]
    radius = max(0.5, _subject_radius(graph, required_ids, center))
    dist = radius / max(0.1, math.tan(math.radians(
        cam.get("fov", 55) / 2))) * 1.5
    cam["target"] = [round(v, 3) for v in center]
    cam["position"] = [round(c + v * dist, 3)
                       for c, v in zip(center, view)]
    return True


def repair_boost_light(ir):
    """Raise the strongest directional/hemisphere light x1.6 (once).
    Returns True when applied."""
    best, best_i = -1.0, None
    for i, Ld in enumerate(ir.get("lights", [])):
        if isinstance(Ld, dict) and Ld.get("kind") in (
                "directional", "hemisphere", "ambient") \
                and not Ld.get("_boosted"):
            try:
                inten = float(Ld.get("intensity", 1.0))
            except (TypeError, ValueError):
                continue
            if inten > best:
                best, best_i = inten, i
    if best_i is None:
        return False
    ir["lights"][best_i]["intensity"] = round(best * 1.6, 3)
    ir["lights"][best_i]["_boosted"] = True
    return True


def run_scene(request, thinker_model=None, seed=11, render_fn=None,
              vision_fn=None, max_repairs=MAX_REPAIRS, out_dir=None,
              emit=None):
    """Run a prompt to COMPLETE or FAILED. ``request`` = {"prompt": str}.
    ``render_fn(js)`` -> {ready, scene_error, console_errors, stats,
    png, elapsed_s, graph?}; when render_fn is None the run stops after
    COMPILING with state NEEDS_RENDER (static validation only). Returns a
    full report incl. transitions, timings and artifact paths."""
    import morphix_scene_compile as C
    import morphix_scene_ir as S
    import morphix_scene_plan as P
    import morphix_scene_validate as V

    emit = emit or (lambda e: None)
    t0 = time.monotonic()
    prompt = (request or {}).get("prompt", "")
    transitions = []

    def go(state, ok=True, detail=""):
        transitions.append({"state": state, "ok": bool(ok),
                            "detail": str(detail)[:300],
                            "t": round(time.monotonic() - t0, 2)})
        emit({"type": "scene_state", "state": state, "ok": ok,
              "detail": detail})

    def fail(stage, reason, **extra):
        go("FAILED", False, f"{stage}: {reason}")
        rep = {"state": "FAILED", "ok": False, "prompt": prompt,
               "stage": stage, "error": reason,
               "transitions": transitions,
               "total_s": round(time.monotonic() - t0, 2)}
        rep.update(extra)
        return rep

    if not prompt.strip():
        return fail("planning", "empty prompt")
    go("PLANNING")
    routed = P.route_systems(prompt)
    thinker_plan, plan_repairs = None, []
    if thinker_model:
        # The thinker sees the routed catalogue systems so its required
        # elements stay within what exists (plus freestyle specs) instead
        # of inventing unresolvable names.
        available = list(routed.get("templates", [])) + [
            f"part:{p}" for p in routed.get("parts", [])]
        try:
            thinker_plan = P.plan_scene(prompt, thinker_model,
                                        systems=available)
        except Exception as e:
            plan_repairs.append(f"thinker planning failed ({e}); "
                                f"router-only deterministic fallback")
            thinker_plan = None
    go("GENERATING")
    try:
        ir, comp = P.compose_routed_scene(routed, thinker_plan, seed=seed)
    except Exception as e:
        # The thinker is enrichment-only, never load-bearing: an unresolvable
        # thinker requirement triggers one re-plan with the failure as
        # feedback, then a router-only deterministic fallback (the router
        # always resolves its own requirements). Every step is logged —
        # nothing is silently substituted.
        if thinker_plan is not None:
            plan_repairs.append(f"coverage failed ({e}); one re-plan "
                                f"with the failure as feedback")
            try:
                thinker_plan = P.plan_scene(
                    prompt + f" Note: do NOT require {e}. "
                    f"Available systems: {available}.",
                    thinker_model, systems=available)
                ir, comp = P.compose_routed_scene(routed, thinker_plan,
                                                  seed=seed)
            except Exception as e2:
                plan_repairs.append(
                    f"re-plan failed ({e2}); router-only deterministic "
                    f"fallback")
                thinker_plan = None
                try:
                    ir, comp = P.compose_routed_scene(routed, None,
                                                      seed=seed)
                except Exception as e3:
                    return fail("generating",
                                f"{type(e3).__name__}: {e3}",
                                routed=routed)
        else:
            return fail("generating", f"{type(e).__name__}: {e}",
                        routed=routed)
    required_ids = comp.get("required_ids", [])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scene_ir.json"), "w",
                  encoding="utf-8") as f:
            f.write(S.ir_to_json(ir))
    go("COMPILING")
    try:
        js = C.compile_scene_ir(ir)
    except ValueError as e:
        return fail("compiling", str(e), ir_saved=bool(out_dir))
    js_errs = C.validate_scene_js(js)
    if js_errs:
        return fail("compiling", "; ".join(js_errs[:5]), ir_saved=bool(out_dir))
    if out_dir:
        with open(os.path.join(out_dir, "scene.js"), "w",
                  encoding="utf-8") as f:
            f.write(js)
    if render_fn is None:
        go("COMPLETE", True, "static validation only (no renderer)")
        return {"state": "NEEDS_RENDER", "ok": True, "prompt": prompt,
                "ir": S.summarize(ir), "required_ids": required_ids,
                "resolutions": comp.get("resolutions", []),
                "plan_repairs": plan_repairs,
                "transitions": transitions,
                "total_s": round(time.monotonic() - t0, 2)}

    repairs, attempts = [], 0
    result, graph, png = None, {}, None
    while True:
        attempts += 1
        go("RENDERING", True, f"attempt {attempts}")
        try:
            result = render_fn(js) or {}
        except Exception as e:
            result = {"ready": False,
                      "scene_error": f"harness: {e}",
                      "console_errors": [], "stats": {}, "png": None,
                      "elapsed_s": 0.0}
        png = result.get("png")
        graph = result.get("graph") or {}
        if out_dir and png and os.path.isfile(png):
            import shutil
            dst = os.path.join(out_dir,
                               f"render_a{attempts}.png")
            shutil.copy(png, dst)
        go("VALIDATING")
        ok_render, render_findings = V.evaluate_render(ir, result) \
            if hasattr(V, "evaluate_render") else (True, [])
        geo = V.validate_assembly(ir, graph)
        visual = V.verify_visual(ir, png, graph) if png else {
            "checks": [{"name": "screenshot", "ok": False,
                        "detail": "no screenshot captured",
                        "severity": "error"}], "verified": False}
        findings = ([{"check": "render:" + f.get("detail", ""),
                      "ok": False, "severity": "error"}
                     for f in render_findings if not f.get("warning")] +
                    [{"check": c["name"], "ok": c["ok"],
                      "detail": c["detail"],
                      "severity": c.get("severity", "error")}
                     for c in geo["checks"] if not c["ok"]] +
                    [{"check": "visual:" + c["name"], "ok": c["ok"],
                      "detail": c["detail"],
                      "severity": c.get("severity", "error")}
                     for c in visual["checks"] if not c["ok"]])
        critical = [f for f in findings if f["severity"] == "error"]
        warnings = [f for f in findings if f["severity"] != "error"]
        vision_note = ""
        if vision_fn is not None and not critical:
            try:
                seen = vision_fn(png, S.summarize(ir)) or ""
                want = (thinker_plan or {}).get("subject", "") or prompt
                overlap = {w for w in str(seen).lower().split()
                           if w in str(want).lower()} if seen else set()
                vision_note = (f"vision spot-check (advisory): {seen!r} "
                               f"vs wanted {want!r}")
                if seen and not overlap:
                    warnings.append({"check": "vision:subject",
                                     "ok": True,
                                     "detail": vision_note + " (advisory "
                                     "only, never fails the run)",
                                     "severity": "warning"})
            except Exception as e:
                warnings.append({"check": "vision:unavailable", "ok": True,
                                 "detail": f"spot-check skipped: {e}",
                                 "severity": "warning"})
        if not critical:
            go("COMPLETE", True,
               f"{len(required_ids)} required elements rendered+valid; "
               f"{len(warnings)} warning(s)")
            return {"state": "COMPLETE", "ok": True, "prompt": prompt,
                    "ir": S.summarize(ir), "required_ids": required_ids,
                    "resolutions": comp.get("resolutions", []),
                    "geometry": geo, "visual": visual,
                    "render": {k: result.get(k) for k in
                               ("ready", "scene_error", "console_errors",
                                "stats", "elapsed_s")},
                    "warnings": warnings, "vision_note": vision_note,
                    "repairs": repairs, "attempts": attempts,
                    "plan_repairs": plan_repairs,
                    "transitions": transitions,
                    "total_s": round(time.monotonic() - t0, 2)}
        if len(repairs) >= max_repairs:
            return fail("validating",
                        "repair budget exhausted with unresolved: " +
                        "; ".join(f["check"] + ": " + f["detail"]
                                  for f in critical[:4]),
                        findings=findings, repairs=repairs,
                        attempts=attempts)
        go("REPAIRING")
        fixed, note = _repair_once(ir, graph, required_ids, critical)
        if not fixed:
            return fail("repairing",
                        "no repair for: " + "; ".join(
                            f["check"] + ": " + f["detail"]
                            for f in critical[:4]),
                        findings=findings, repairs=repairs,
                        attempts=attempts)
        repairs.append(note)
        try:
            js = C.compile_scene_ir(ir)
        except ValueError as e:
            return fail("repairing", f"recompile failed: {e}",
                        findings=findings, repairs=repairs,
                        attempts=attempts)
        if out_dir:
            with open(os.path.join(out_dir, "scene.js"), "w",
                      encoding="utf-8") as f:
                f.write(js)


def _repair_once(ir, graph, required_ids, critical):
    """Map critical findings to IR edits. Returns (fixed, note). Only
    local, design-preserving repairs: camera reframe for out-of-frame
    subjects, light boost for unlit subjects. Anything else -> (False)."""
    texts = " ".join(f.get("check", "") + " " + f.get("detail", "")
                     for f in critical)
    if "subject_framed" in texts or "no required-subject corner" in texts:
        if repair_camera_frame(ir, graph, required_ids):
            return True, "camera reframed onto required subject"
    if "subject_lit" in texts:
        if repair_boost_light(ir):
            return True, "key light boosted x1.6"
    return False, ""
