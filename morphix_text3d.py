# morphix_text3d.py
# Text -> realistic 3D for subjects with a shape builder.
#
#   "a red sports car with big wheels" / "a tall wooden chair" / "a cat"
#
# The prompt's subject picks a family builder (vehicle, table, chair, stool,
# house, quadruped — the same shape-first builders image reconstruction
# fits to photos); style words pick its proportions preset; colour, material
# and size words become validated edit operations applied to the built
# shape. The result is printed, rendered (print + realistic model) and the
# print is checked against the model, exactly like any other object, and it
# can be edited by prompt afterwards.
#
# Subjects WITHOUT a builder (a skull, a guitar, a robot) are refused with a
# clear message: local text models cannot design arbitrary shapes reliably,
# so the honest route is a photo (Image mode) — never a stack of boxes.

import datetime
import json
import os
import re

import morphix_image_shapes as SH
import morphix_image_vision as VIS
from morphix_config import PipelineError

DEFAULT_PRESET = {"vehicle": "sedan", "table": "dining", "chair": "dining",
                  "stool": "bar", "house": "cottage"}
DEFAULT_PAINT = {
    "vehicle": {"body": "0x1F4E8C", "glass": "0x1B2530", "light": "0xF3EED9"},
    "table": {"body": "0x8B5A2B"}, "chair": {"body": "0x8B5A2B"},
    "stool": {"body": "0x8B5A2B"},
    "house": {"body": "0xE8DCC0", "second": "0xA0402F", "glass": "0x7DB9E8"},
    "quadruped": {"body": "0xB07A45"},
}
ANIMAL_COLORS = {"cat": "0xE08A3C", "dog": "0x8B5A2B", "horse": "0x6B4423",
                 "cow": "0xEDEDED", "sheep": "0xF2EFE6", "fox": "0xD2691E",
                 "pig": "0xF4B6B0", "goat": "0xD8CFC0", "deer": "0xA0703C",
                 "lion": "0xC8A15A", "tiger": "0xE08A2E", "wolf": "0x8A8A8A"}


def understand(prompt):
    """(category, family, label) of the prompt's subject."""
    p = (prompt or "").strip()
    if not p:
        raise PipelineError("describe an object, e.g. 'a red sports car'")
    np_ = VIS.subject_phrase(p if re.match(r"^(a|an|the)\b", p, re.I)
                             else "The image shows a " + p) or p
    cat, term, fam = VIS.resolve_category(np_)
    if not fam:  # e.g. "make me a car": search the whole prompt
        cat2, _t, fam2 = VIS.resolve_category(p)
        if fam2:
            cat, fam = cat2, fam2
    return cat, fam, np_


def _modifiers(prompt, ir):
    """Colour / material / size words attached to the noun they modify:
    'red car' -> whole object, 'blue roof' -> roof, 'big wheels' -> wheels."""
    import morphix_edit as E
    colors = E._colors()
    words = re.findall(r"#[0-9a-f]{6}|[a-z]+", prompt.lower())
    roles = set(E.targets_of(ir)["roles"])

    def target_after(i):
        for w in words[i + 1:i + 4]:
            base = w[:-1] if w.endswith("s") and w[:-1] in E.TARGET_SYNONYMS else w
            t = E.TARGET_SYNONYMS.get(base, base)
            if t.startswith("material:") or t in roles:
                return t
        return "all"
    ops = []
    i = 0
    while i < len(words):
        w = words[i]
        two = " ".join(words[i:i + 2])
        if two in colors:
            ops.append({"op": "color", "target": target_after(i + 1),
                        "color": colors[two]})
            i += 2
            continue
        if w.startswith("#"):
            ops.append({"op": "color", "target": target_after(i),
                        "color": "0x" + w[1:].upper()})
        elif w in E.MATERIAL_WORDS:
            mat, col = E.MATERIAL_WORDS[w]
            t = target_after(i)
            ops += [{"op": "material", "target": t, "material": mat},
                    {"op": "color", "target": t, "color": col}]
        elif w in colors:
            ops.append({"op": "color", "target": target_after(i),
                        "color": colors[w]})
        elif w in E.SCALE_WORDS and w not in ("long", "longer"):
            axis, f = E.SCALE_WORDS[w]
            t = target_after(i)
            ops.append({"op": "scale", "target": t,
                        "axis": "all" if t != "all" else axis, "factor": f})
        i += 1
    return ops


def build(prompt):
    """IR for the prompt (raises PipelineError if no builder fits)."""
    cat, fam, label = understand(prompt)
    if fam not in SH.BUILDERS:
        what = cat or "this object"
        raise PipelineError(
            f"no text-only shape for '{what}' yet. Text mode builds vehicles, "
            f"tables, chairs, stools, houses and four-legged animals; for "
            f"anything else use Image mode with a photo of a {what} — it "
            f"reconstructs any subject from its real outline.")
    builder, presets = SH.BUILDERS[fam]
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    hinted = [SH.STYLE_HINTS.get(fam, {}).get(w) for w in words]
    hinted = [h for h in hinted if h in presets]
    if fam == "quadruped":
        preset = cat if cat in presets else (hinted[0] if hinted else "dog")
    else:
        preset = hinted[0] if hinted else DEFAULT_PRESET.get(fam, next(iter(presets)))
    paint = {"body": "0x888888", "glass": "0x2A3540", "light": "0xF3EED9",
             "second": "0x6B4423"}
    paint.update(DEFAULT_PAINT.get(fam, {}))
    if fam == "quadruped" and cat in ANIMAL_COLORS:
        paint["body"] = ANIMAL_COLORS[cat]
    params = dict(presets[preset])
    comps = builder(params, paint)
    if fam == "vehicle":
        # "big wheels" for a vehicle is a BUILD parameter (wheel radius), so
        # the wheel arches grow with the wheels instead of being outgrown
        import morphix_edit as E
        probe = {"components": comps}
        wheel = [o for o in _modifiers(prompt, probe)
                 if o["op"] == "scale" and o["target"] == "wheel"]
        if wheel:
            f = wheel[0]["factor"]
            params["wr"] = min(0.11, max(0.045, params["wr"] * f))
            # bigger wheels ride higher: the body grows with them
            params["hr"] = min(0.5, params["hr"] * max(1.0, f))
            comps = builder(params, paint)
            del E
    # animals read best side-on (a nose-first view hides the body)
    yaw = 72.0 if fam == "quadruped" else 40.0
    cam = SH.camera_for(comps, yaw, 16.0)
    import morphix_image3d as I3
    from morphix_ir import _object_bbox
    ir = {
        "ir_version": I3.IR_VERSION, "kind": "text-object",
        "category": cat, "source": "text", "units": "scene", "up": "y",
        "front": "+x", "confidence": 1.0,
        "subject": {"category": cat, "label": label, "family": fam,
                    "identified": True, "confidence": 1.0,
                    "decided_by": f"prompt subject {label!r}",
                    "vision_status": "text"},
        "method": {"builder": "family", "kind": fam, "preset": preset,
                   "params": params, "build_order": list(SH.STAGES),
                   "log": [f"text prompt -> {fam} builder, preset {preset!r}"]},
        "viewpoint": {"yaw_deg": yaw, "elevation_deg": 16.0,
                      "basis": "default presentation view (no photo)"},
        "camera": cam,
        "appearance": {"body_color": paint["body"], "palette": {}},
        "environment": {"mode": "object-only", "elements": []},
        "uncertainty": {"occluded": [], "ambiguous": [], "assumptions": [
            f"proportions from the {preset!r} {fam} preset (not measured — "
            f"no photo); refine with prompt edits"]},
        "components": comps,
        "assumptions": [], "observed": {},
        "constraints": ["y is up; the object rests on the ground plane y=0"],
    }
    ir["bbox"] = _object_bbox(comps)
    ir["appearance"]["palette"] = I3.palette_for(comps)
    import morphix_edit as E
    ops = _modifiers(prompt, ir)
    if fam == "vehicle":  # already built into the wheel radius
        ops = [o for o in ops if not (o["op"] == "scale"
                                      and o["target"] == "wheel")]
    log = []
    if ops:
        ops = E.validate_ops(ops, ir)
        ir, log = E.apply_ops(ir, ops)
    return ir, ops, log


def run(prompt, runs_dir, emit=None):
    """Build, print, render and check. Returns the report (with run_id)."""
    import morphix_image3d as I3
    import morphix_image_display as DSP
    from morphix_ir import validate_ir
    ir, ops, log = build(prompt)
    errs = validate_ir(ir)
    if errs:
        raise PipelineError("built object is invalid: " + "; ".join(errs[:4]))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(runs_dir, stamp)
    trace = os.path.join(out, "trace")
    os.makedirs(trace, exist_ok=True)
    report, actions, records = I3._print(ir, out, emit or (lambda e: None))
    # render the print and the realistic model from the same camera; the
    # print must show the same shape as the model it was sliced from
    bg = "0xD8DDE2" if SH._luma(ir["appearance"]["body_color"]) < 110 else "0x1C2128"
    mats = sorted(ir["appearance"]["palette"])
    shaded = {"mode": "shaded", "bg": bg, "camera": ir["camera"],
              "colors": [ir["appearance"]["palette"][m] for m in mats],
              "deposits": I3._deposit_rows(records, lambda r: mats.index(
                  r.get("material")) if r.get("material") in mats else 0),
              "comps": [], "env": []}
    pages = [("render", shaded, os.path.join(trace, "render.png")),
             ("model", DSP.display_html(ir, verify=True,
                                        three_url="./three.module.js",
                                        addons_url="./", bg=bg),
              os.path.join(trace, "model.png"))]
    checks = []
    try:
        res = I3.render_pages(pages)
        ok = all(v.get("ok") and not v.get("errors") for v in res.values())
        checks.append({"name": "render_completed", "ok": ok,
                       "severity": "critical",
                       "detail": "print and model rendered" if ok else
                       str({k: v.get("errors") for k, v in res.items()})})
        if ok:
            import morphix_image_seg as SEG
            a, _ = I3.render_mask(res["render"]["png"], bg)
            b, _ = I3.render_mask(res["model"]["png"], bg)
            iou = SEG.mask_iou_normalized(a, b)
            checks.append({"name": "print_matches_model", "ok": iou >= 0.8,
                           "severity": "critical", "value": round(iou, 3),
                           "detail": f"printed vs realistic silhouette IoU "
                                     f"{iou:.3f} (>= 0.8)"})
            from PIL import Image
            ims = [Image.open(res[k]["png"]).convert("RGB") for k in
                   ("model", "render")]
            sheet = Image.new("RGB", (1280, 480))
            for i, im in enumerate(ims):
                sheet.paste(im, (i * 640, 0))
            sheet.save(os.path.join(trace, "compare.png"))
    except Exception as e:
        checks.append({"name": "render_completed", "ok": False,
                       "severity": "critical",
                       "detail": f"browser unavailable: {e}"})
    good = all(c["ok"] for c in checks if c["severity"] == "critical")
    report.update({
        "mode": "text-3d", "prompt": prompt, "ir_full": ir,
        "image": {"subject": ir["subject"],
                  "builder": {"method": "family",
                              "kind": ir["method"]["kind"],
                              "preset": ir["method"]["preset"]}},
        "text_ops": ops, "text_log": log,
        "visual_verification": {"checks": checks, "critical_ok": good,
                                "metrics": {}},
        "success": bool(report.get("ok") and report.get("complete") and good),
    })
    with open(os.path.join(out, "object.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out, "model.html"), "w", encoding="utf-8") as f:
        f.write(DSP.display_html(ir))
    report["run_id"] = stamp
    return report
