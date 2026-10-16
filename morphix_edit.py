# morphix_edit.py
# Edit a generated object with a natural-language prompt.
#
#   "make the wheels bigger and paint it red"
#   "remove the tail", "make it 30% taller", "make the roof blue"
#
# The prompt is turned into a small list of STRUCTURED operations, validated
# against the object's real components, applied to its validated IR, and the
# result is re-printed and re-verified as a NEW run (the original run is never
# modified). Operations:
#
#   {"op": "color",    "target": T, "color": "0xRRGGBB"}
#   {"op": "scale",    "target": T, "axis": "all|x|y|z", "factor": f}
#   {"op": "material", "target": T, "material": "<printer material>"}
#   {"op": "remove",   "target": T}
#
# T is "all", a role ("wheel", "roof", ...), a component id, or
# "material:<name>" (e.g. all glass). Common phrasings are parsed by
# deterministic rules; anything else is translated by the local builder
# model into the same JSON, which is validated exactly like rule output —
# an unknown target or value is an error, never guessed.

import copy
import datetime
import json
import os
import re
import shutil

from morphix_config import (BUILDER_MODEL, ModelUnavailableError,
                            PipelineError, ollama_generate)

EXTRA_COLORS = {"navy": "0x1F2F5C", "maroon": "0x6E1A1A", "olive": "0x6B7A2A",
                "lime": "0x84CC16", "magenta": "0xD02090", "violet": "0x8B5CF6",
                "indigo": "0x4338CA", "turquoise": "0x2EC4B6",
                "cream": "0xF3E9D2", "ivory": "0xF6F1E3", "bronze": "0x8C6239",
                "copper": "0xB87333", "chrome": "0xC9CED6", "matte black": "0x1B1B1D",
                "dark grey": "0x3C3F44", "dark gray": "0x3C3F44",
                "light grey": "0xC8CCD2", "light gray": "0xC8CCD2",
                "dark blue": "0x1E3A8A", "light blue": "0x7DB9E8",
                "dark green": "0x1F5130", "light green": "0x86D69B",
                "dark red": "0x7F1D1D", "sky blue": "0x87CEEB"}

# word -> (role, material) synonyms for targeting parts
TARGET_SYNONYMS = {
    "wheel": "wheel", "tire": "wheel", "tyre": "wheel", "rim": "wheel",
    "light": "light", "headlight": "light", "lamp": "light",
    "window": "material:glass", "windshield": "material:glass",
    "glass": "material:glass", "windscreen": "material:glass",
    "cabin": "cabin", "roof": "roof", "body": "body", "hood": "body",
    "leg": "leg", "seat": "seat", "backrest": "backrest", "back": "backrest",
    "top": "tabletop", "tabletop": "tabletop", "table top": "tabletop",
    "wall": "wall", "door": "door", "chimney": "chimney", "head": "head",
    "ear": "ear", "tail": "tail", "torso": "body", "neck": "neck",
}
WHOLE = {"it", "object", "model", "everything", "whole", "all", "car",
         "vehicle", "thing", "this", "skull", "chair", "table", "house"}

SCALE_WORDS = {
    "bigger": ("all", 1.25), "larger": ("all", 1.25), "enlarge": ("all", 1.25),
    "grow": ("all", 1.25), "smaller": ("all", 0.8), "shrink": ("all", 0.8),
    "tiny": ("all", 0.6), "huge": ("all", 1.6),
    "taller": ("y", 1.25), "higher": ("y", 1.25), "shorter": ("y", 0.8),
    "lower": ("y", 0.8), "flatter": ("y", 0.75),
    "wider": ("z", 1.25), "broader": ("z", 1.25), "fatter": ("z", 1.25),
    "thicker": ("z", 1.25), "narrower": ("z", 0.8), "thinner": ("z", 0.8),
    "slimmer": ("z", 0.8), "longer": ("x", 1.25),
    # base forms last ("twice as big", "make it small")
    "big": ("all", 1.25), "large": ("all", 1.25), "small": ("all", 0.8),
    "tall": ("y", 1.25), "wide": ("z", 1.25), "long": ("x", 1.25),
}
MATERIAL_WORDS = {"wooden": ("wood", "0x9C6B3F"), "wood": ("wood", "0x9C6B3F"),
                  "glass": ("glass", "0x99DDFF"),
                  "transparent": ("glass", "0x99DDFF"),
                  "rubber": ("tire", "0x222222")}
REMOVE_WORDS = ("remove", "delete", "without", "drop", "get rid of", "take off",
                "no ")

MODEL_SYSTEM = (
    "You translate a request to edit a 3D object into JSON operations. "
    "Reply with ONLY a JSON array. Allowed operations:\n"
    '{"op":"color","target":T,"color":"0xRRGGBB"}\n'
    '{"op":"scale","target":T,"axis":"all|x|y|z","factor":0.2-4.0}\n'
    '{"op":"material","target":T,"material":M}\n'
    '{"op":"remove","target":T}\n'
    "T is \"all\" or one of the listed targets. x = length, y = height, "
    "z = width. M is one of the listed materials. Use only listed targets.")


def _colors():
    from morphix_image_vision import NAMED_COLORS
    out = dict(NAMED_COLORS)
    out.update(EXTRA_COLORS)
    return out


# ------------------------------------------------------------------ targets
def targets_of(ir):
    """Every valid target for this object: roles (+ how many parts carry
    each role), ids, materials."""
    roles, ids, mats = {}, set(), set()
    for c in ir.get("components", []):
        r = c.get("role") or ""
        if r:
            roles[r] = roles.get(r, 0) + 1
        ids.add(c["id"])
        mats.add(c.get("material") or "")
    return {"roles": sorted(roles), "counts": roles,
            "ids": sorted(ids),
            "materials": sorted(m for m in mats if m)}


def select(ir, target):
    """Components matched by a target, or raises PipelineError."""
    comps = ir.get("components", [])
    if target in (None, "", "all"):
        return list(comps)
    if target.startswith("material:"):
        m = target.split(":", 1)[1]
        hit = [c for c in comps if c.get("material") == m]
    else:
        hit = [c for c in comps if c.get("role") == target
               or c["id"] == target]
    if not hit:
        t = targets_of(ir)
        raise PipelineError(
            f"this object has no part {target!r} (parts: "
            f"{', '.join(t['roles'])})")
    return hit


# ------------------------------------------------------------------ parsing
def _factor(clause, default):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", clause)
    if m:
        return float(m.group(1))
    if re.search(r"\b(twice|double)\b", clause):
        return 2.0
    if re.search(r"\b(half|halve)\b", clause):
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", clause)
    if m:
        p = float(m.group(1)) / 100.0
        return 1 + p if default >= 1 else max(0.05, 1 - p)
    if re.search(r"\b(much|a lot|way)\b", clause):
        return default ** 2
    if re.search(r"\b(slightly|a bit|a little)\b", clause):
        return 1 + (default - 1) * 0.4
    return default


def _target_in(clause, ir):
    """First part named in the clause, else 'all'."""
    roles = set(targets_of(ir)["roles"])
    for w in re.findall(r"[a-z]+", clause):
        base = w[:-1] if w.endswith("s") and w[:-1] in TARGET_SYNONYMS else w
        if base in TARGET_SYNONYMS:
            t = TARGET_SYNONYMS[base]
            if t.startswith("material:") or t in roles:
                return t
        if base in roles:
            return base
    return "all"


def parse_rules(prompt, ir):
    """Deterministic parse. Returns ops ([] if nothing understood)."""
    ops = []
    colors = _colors()
    text = prompt.lower().strip()
    for clause in re.split(r"\s*(?:,|;|\band\b|\bthen\b|\balso\b)\s*", text):
        if not clause:
            continue
        hexm = re.search(r"#([0-9a-f]{6})\b|0x([0-9a-f]{6})\b", clause)
        col, rest = None, clause
        if hexm:
            col = "0x" + (hexm.group(1) or hexm.group(2)).upper()
            rest = clause.replace(hexm.group(0), " ")
        else:
            for name in sorted(colors, key=len, reverse=True):
                if re.search(r"\b" + re.escape(name) + r"\b", clause):
                    col = colors[name]
                    # "light blue" is a colour, not the headlights
                    rest = re.sub(r"\b" + re.escape(name) + r"\b", " ",
                                  clause)
                    break
        target = _target_in(rest, ir)
        if any(w in clause for w in REMOVE_WORDS) and target != "all":
            ops.append({"op": "remove", "target": target})
            continue
        mat = next((MATERIAL_WORDS[w] for w in MATERIAL_WORDS
                    if re.search(r"\b" + w + r"\b", clause)), None)
        if mat and not (target.startswith("material:")
                        and target.endswith(mat[0])):
            ops.append({"op": "material", "target": target,
                        "material": mat[0]})
            if not col:
                ops.append({"op": "color", "target": target, "color": mat[1]})
        if col:
            ops.append({"op": "color", "target": target, "color": col})
        for word, (axis, f) in SCALE_WORDS.items():
            if re.search(r"\b" + word + r"\b", clause):
                ops.append({"op": "scale", "target": target, "axis": axis,
                            "factor": round(_factor(clause, f), 3)})
                break
        else:
            if re.search(r"\b(scale|resize|size)\b", clause):
                f = _factor(clause, 1.0)
                if f != 1.0:
                    ops.append({"op": "scale", "target": target,
                                "axis": "all", "factor": f})
    return ops


def parse_model(prompt, ir, model=BUILDER_MODEL):
    """Builder-model translation into the same ops JSON (validated)."""
    t = targets_of(ir)
    user = (f"Object parts (targets): {', '.join(t['roles'])}; "
            f"materials: {', '.join(sorted(set(t['materials']) | {'wood', 'glass', 'body', 'accent', 'tire'}))}.\n"
            f"Request: {prompt}\nJSON array:")
    raw = ollama_generate(model, MODEL_SYSTEM, user,
                          options={"temperature": 0.0, "num_ctx": 2048,
                                   "num_predict": 300})
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise PipelineError(f"the model did not return edit operations "
                            f"(reply: {(raw or '')[:160]!r})")
    try:
        ops = json.loads(m.group(0))
    except ValueError:
        raise PipelineError(f"the model returned invalid JSON: "
                            f"{m.group(0)[:160]!r}")
    return ops


def validate_ops(ops, ir):
    """Strict schema + targets. Returns normalized ops or raises."""
    from morphix_path import MATERIAL_COLORS
    if not isinstance(ops, list) or not ops:
        raise PipelineError("no edit operations")
    out = []
    for op in ops:
        if not isinstance(op, dict) or op.get("op") not in (
                "color", "scale", "material", "remove"):
            raise PipelineError(f"unknown edit operation {op!r}")
        tgt = str(op.get("target") or "all").lower()
        tgt = TARGET_SYNONYMS.get(tgt, tgt)
        select(ir, tgt)  # raises on an unknown part
        o = {"op": op["op"], "target": tgt}
        if op["op"] == "color":
            c = str(op.get("color", "")).strip()
            c = "0x" + c[1:] if c.startswith("#") else c
            if not re.fullmatch(r"0x[0-9A-Fa-f]{6}", c):
                raise PipelineError(f"bad colour {op.get('color')!r}")
            o["color"] = "0x" + c[2:].upper()
        elif op["op"] == "scale":
            try:
                f = float(op.get("factor"))
            except (TypeError, ValueError):
                raise PipelineError(f"bad scale factor {op.get('factor')!r}")
            if not 0.2 <= f <= 4.0:
                raise PipelineError(f"scale factor {f} outside 0.2..4")
            ax = op.get("axis", "all")
            if ax not in ("all", "x", "y", "z"):
                raise PipelineError(f"bad axis {ax!r}")
            o.update(axis=ax, factor=round(f, 4))
        elif op["op"] == "material":
            m = op.get("material")
            if m not in MATERIAL_COLORS:
                raise PipelineError(f"material must be one of "
                                    f"{sorted(MATERIAL_COLORS)} (got {m!r})")
            o["material"] = m
        elif tgt == "all":
            raise PipelineError("cannot remove the whole object")
        out.append(o)
    return out


# ------------------------------------------------------------------ apply
def _bounds(comps):
    import morphix_image_shapes as SH
    return SH.scene_bounds(comps)


def _scale_component(c, f, anchor):
    """Scale one component by (fx, fy, fz) about anchor (world point)."""
    fx, fy, fz = f
    p = c["position"]
    c["position"] = [round(anchor[i] + (p[i] - anchor[i]) * f[i], 4)
                     for i in range(3)]
    d = c["dims"]
    prim = c["primitive"]
    rad = (fx + fz) / 2.0
    if prim == "box":
        c["dims"] = [round(d[0] * fx, 4), round(d[1] * fy, 4),
                     round(d[2] * fz, 4)]
    elif prim in ("cylinder", "sphere"):
        r = rad if prim == "cylinder" else (fx + fy + fz) / 3.0
        h = fy if prim == "cylinder" else r
        c["dims"] = [round(d[0] * r, 4), round(d[1] * h, 4),
                     round(d[2] * r, 4)]
        if c.get("taper"):
            c["taper"] = [round(v * r, 4) for v in c["taper"]]
    elif prim == "profile":
        fu, fd = (fx, fz) if c.get("axis", "z") == "z" else (fz, fx)
        c["profile"] = [[round(u * fu, 4), round(y * fy, 4)]
                        for u, y in c["profile"]]
        c["depth"] = round(c["depth"] * fd, 4)
        if "depth_top" in c:
            c["depth_top"] = round(c["depth_top"] * fd, 4)
        c["dims"] = [round(d[0] * fx, 4), round(d[1] * fy, 4),
                     round(d[2] * fz, 4)]
    elif prim == "lathe":
        c["profile"] = [[round(r_ * rad, 4), round(y * fy, 4)]
                        for r_, y in c["profile"]]
        c["dims"] = [round(d[0] * rad, 4), round(d[1] * fy, 4),
                     round(d[2] * rad, 4)]
    elif prim == "inflate":
        # the grid has one cell size for x and y: scale it by their mean;
        # thickness follows z
        g = c["grid"]
        fxy = (fx * fy) ** 0.5
        g["cell"] = round(g["cell"] * fxy, 5)
        g["half"] = [[round(v * fz, 4) for v in row] for row in g["half"]]
        c["dims"] = [round(d[0] * fxy, 4), round(d[1] * fxy, 4),
                     round(d[2] * fz, 4)]


def apply_ops(ir, ops):
    """New IR with the operations applied. Returns (ir, log)."""
    ir = copy.deepcopy(ir)
    log = []
    for op in ops:
        sel = select(ir, op["target"])
        ids = {c["id"] for c in sel}
        if op["op"] == "remove":
            ir["components"] = [c for c in ir["components"]
                                if c["id"] not in ids]
            if not ir["components"]:
                raise PipelineError("the edit would remove every part")
            log.append(f"removed {', '.join(sorted(ids))}")
        elif op["op"] == "color":
            if op["target"] == "all":
                # "paint it red" repaints the object's paint, not its tires,
                # glass or lamps: the material of its largest part
                big = max(ir["components"], key=lambda c: c["dims"][0]
                          * c["dims"][1] * c["dims"][2])
                sel = [c for c in ir["components"]
                       if c.get("material") == big.get("material")]
            for c in sel:
                c["color"] = op["color"]
                if c.get("texture") == "photo":
                    # keep the photographed detail: the paint tints it
                    c["tint"] = op["color"]
            log.append(f"painted {op['target']} {op['color']}")
        elif op["op"] == "material":
            for c in sel:
                c["material"] = op["material"]
            log.append(f"{op['target']} material -> {op['material']}")
        else:
            f = op["factor"]
            fac = {"all": (f, f, f), "x": (f, 1, 1), "y": (1, f, 1),
                   "z": (1, 1, f)}[op["axis"]]
            if op["target"] == "all":
                lo, hi = _bounds(ir["components"])
                anchor = [(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2]
                for c in ir["components"]:
                    _scale_component(c, fac, anchor)
            else:
                for c in sel:
                    # scale in place, keeping the part's base where it is
                    lo, hi = _bounds([c])
                    anchor = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
                    _scale_component(c, fac, anchor)
            log.append(f"scaled {op['target']} {op['axis']} x{f}")
    # keep it standing on the ground
    lo, _hi = _bounds(ir["components"])
    if abs(lo[1]) > 1e-3:
        for c in ir["components"]:
            c["position"][1] = round(c["position"][1] - lo[1], 4)
    from morphix_ir import _object_bbox
    ir["bbox"] = _object_bbox(ir["components"])
    import morphix_image3d as I3
    ir.setdefault("appearance", {})["palette"] = I3.palette_for(
        ir["components"])
    return ir, log


# ------------------------------------------------------------------ run
def edit_run(runs_dir, run_id, prompt, model=BUILDER_MODEL, emit=None):
    """Apply a prompt to run_id's object; print + verify it as a new run.
    Returns the new run's report. Raises PipelineError with a clear,
    user-facing reason when the request cannot be applied."""
    import morphix_image_display as DSP
    import morphix_pipeline as P
    from morphix_ir import ir_to_actions, validate_ir
    from morphix_path import MAX_Y
    src = os.path.join(runs_dir, run_id)
    obj = os.path.join(src, "object.json")
    if not os.path.isfile(obj):
        raise PipelineError("this run has no editable object (scene "
                            "environments and reliefs cannot be edited)")
    with open(obj, encoding="utf-8") as f:
        ir = json.load(f)
    if not (prompt or "").strip():
        raise PipelineError("describe the change, e.g. 'make it red'")
    ops = parse_rules(prompt, ir)
    parser = "rules"
    if not ops:
        try:
            ops = parse_model(prompt, ir, model)
        except ModelUnavailableError as e:
            raise PipelineError(f"could not understand the request and the "
                                f"builder model is unavailable: {e}")
        parser = f"model {model}"
        try:
            ops = validate_ops(ops, ir)
        except PipelineError as e:
            t = targets_of(ir)
            raise PipelineError(
                f"couldn't turn \"{prompt}\" into an edit for this "
                f"{ir.get('category', 'object')} ({e}). Try e.g. \"paint it "
                f"red\", \"make it 20% taller\" or name a part: "
                f"{', '.join(t['roles'])}")
    ops = validate_ops(ops, ir)
    new_ir, log = apply_ops(ir, ops)
    errs = validate_ir(new_ir)
    if errs:
        raise PipelineError("the edited object is invalid: "
                            + "; ".join(errs[:4]))
    if new_ir["bbox"][1] > min(MAX_Y, 6.2):
        raise PipelineError(f"the edited object is {new_ir['bbox'][1]:.1f} "
                            f"tall — above the printer's layer budget")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(runs_dir, stamp)
    os.makedirs(os.path.join(out, "trace"), exist_ok=True)
    tex = os.path.join(src, "trace", "texture.png")
    has_tex = os.path.exists(tex) and any(
        c.get("texture") == "photo" for c in new_ir["components"])
    if has_tex:
        shutil.copy(tex, os.path.join(out, "trace", "texture.png"))
    label = (new_ir.get("subject") or {}).get("label") or new_ir.get(
        "category", "object")
    report = P.run_path_print(
        f"edited {label}", out_dir=out, emit=emit, segment_cap=6000,
        layer_cap=32, plan_actions=ir_to_actions(new_ir),
        plan_status="EDIT", spec_name="image",
        plan_source=new_ir.get("source", "text"), stream_scene=False,
        palette=new_ir["appearance"]["palette"])
    history = list(ir.get("edit_history", [])) + [
        {"run": run_id, "prompt": prompt, "ops": ops}]
    new_ir["edit_history"] = history
    report.update({
        "mode": "edit", "base_run": run_id, "prompt": prompt,
        "edit_ops": ops, "edit_parser": parser, "edit_log": log,
        "edit_history": history, "ir_full": new_ir,
        "image": {"subject": new_ir.get("subject")},
        "success": bool(report.get("ok") and report.get("complete")),
    })
    with open(os.path.join(out, "object.json"), "w", encoding="utf-8") as f:
        json.dump(new_ir, f, indent=2)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out, "model.html"), "w", encoding="utf-8") as f:
        f.write(DSP.display_html(new_ir, texture="trace/texture.png"
                                 if has_tex else None))
    report["run_id"] = stamp
    return report
