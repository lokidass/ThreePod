# morphix_validator.py
# Morphix pipeline validator: action schema + per-block JS guard + real
# JavaScript syntax check (node --check) + assembled-scene checks.
#
# Usage:
#   python morphix_validator.py --actions actions.json --js out/scene.js
#   from morphix_validator import validate_actions, validate_js_syntax, validate_scene

import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphix_guard import validate_js

_ACTION_SCHEMAS = {
    "add_sphere": {"required": ["radius", "position", "color", "label"],
                   "types": {"radius": (int, float), "position": list,
                             "color": str, "label": str}},
    "add_box": {"required": ["width", "height", "depth", "position", "color", "label"],
                "types": {"width": (int, float), "height": (int, float),
                          "depth": (int, float), "position": list,
                          "color": str, "label": str}},
    "add_cylinder": {"required": ["height", "position", "color", "label"],
                     "types": {"height": (int, float), "position": list,
                               "color": str, "label": str}},
    "add_cone": {"required": ["radius", "height", "position", "color", "label"],
                 "types": {"radius": (int, float), "height": (int, float),
                           "position": list, "color": str, "label": str}},
    "add_torus": {"required": ["radius", "tube", "position", "color", "label"],
                  "types": {"radius": (int, float), "tube": (int, float),
                            "position": list, "color": str, "label": str}},
    "add_plane": {"required": ["width", "height", "position", "color", "label"],
                  "types": dict()},
    "add_extrude": {"required": ["depth", "color", "label"],
                    "types": {"depth": (int, float), "color": str, "label": str}},
    "add_terrain": {"required": ["width", "color", "label"],
                    "types": {"width": (int, float), "color": str, "label": str}},
    "add_light": {"required": ["type", "intensity", "label"],
                  "types": {"type": str, "intensity": (int, float), "label": str}},
    "add_shader": {"required": ["preset", "label"],
                   "types": {"preset": str, "label": str}},
    "add_particles": {"required": ["count", "label"],
                      "types": {"count": int, "label": str}},
    "add_animation": {"required": ["preset", "label"],
                      "types": {"preset": str, "label": str}},
    # point-by-point printer protocol (morphix_path.py): one movement or
    # deposition per action, y-locked to the active layer. Lowercase only:
    # SCREAMING_CASE and whole-object primitives are refused here.
    "move_to": {"required": ["position", "layer", "label"],
                "types": {"position": list, "layer": int, "label": str}},
    "deposit_segment": {"required": ["from", "to", "bead_width",
                                     "bead_height", "material",
                                     "layer", "label"],
                        "types": {"from": list, "to": list,
                                  "bead_width": (int, float),
                                  "bead_height": (int, float),
                                  "material": str,
                                  "layer": int, "label": str}},
    "deposit_point": {"required": ["position", "size", "material",
                                   "layer", "label"],
                      "types": {"position": list, "size": (int, float),
                                "material": str,
                                "layer": int, "label": str}},
    "complete_path": {"required": ["layer", "label"],
                      "types": {"layer": int, "label": str}},
    "complete_layer": {"required": ["layer", "label"],
                       "types": {"layer": int, "label": str}},
    "complete_object": {"required": ["label"],
                        "types": {"label": str}},
}

_PATH_MATERIALS = ("body", "accent", "blue", "support", "glass", "tire",
                   "wood")

_COLOR_RE = re.compile(r"^0x[0-9a-fA-F]{6}$")
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")
_POSITIVE_KEYS = ("radius", "radiusTop", "radiusBottom", "width", "height",
                  "depth", "tube", "intensity", "count", "size", "angle",
                  "segments", "amplitude", "frequency")


def validate_actions(actions: list) -> list:
    """Check action schema, types, label uniqueness. Returns error strings."""
    errs = []
    if not isinstance(actions, list) or not actions:
        return ["no actions"]
    seen = set()
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            errs.append(f"[{i}] not an object")
            continue
        act = a.get("action")
        schema = _ACTION_SCHEMAS.get(act)
        if schema is None:
            errs.append(f"[{i}] unknown action {act!r}")
            continue
        for field in schema["required"]:
            if field not in a:
                errs.append(f"[{i}] {act} missing {field}")
        for field, types in schema["types"].items():
            if field in a and not isinstance(a[field], types):
                errs.append(f"[{i}] {act}.{field} wrong type "
                            f"(got {type(a[field]).__name__})")
        pos = a.get("position")
        if "position" in a and not (isinstance(pos, list) and len(pos) == 3
                                    and all(isinstance(v, (int, float)) for v in pos)):
            errs.append(f"[{i}] {act}.position must be [x,y,z] numbers")
        for field in ("from", "to"):
            v = a.get(field)
            if field in a and not (
                    isinstance(v, list) and len(v) == 3
                    and all(isinstance(x, (int, float))
                            and not isinstance(x, bool) for x in v)):
                errs.append(f"[{i}] {act}.{field} must be [x,y,z] numbers")
        if "layer" in a and (
                isinstance(a["layer"], bool)
                or not isinstance(a["layer"], int)
                or a["layer"] < 0):
            errs.append(f"[{i}] {act}.layer must be an int >= 0 "
                        f"(got {a['layer']!r})")
        if "material" in a and a["material"] not in _PATH_MATERIALS:
            errs.append(f"[{i}] {act}.material must be one of "
                        f"{list(_PATH_MATERIALS)} (got {a['material']!r})")
        color = a.get("color")
        if "color" in a and isinstance(color, str) and not _COLOR_RE.match(color):
            errs.append(f"[{i}] {act}.color must be 0xRRGGBB (got {color!r})")
        for field in _POSITIVE_KEYS:
            v = a.get(field)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(f"[{i}] {act}.{field} must be a number "
                            f"(got {type(v).__name__})")
            elif v <= 0:
                errs.append(f"[{i}] {act}.{field} must be > 0 (got {v})")
        label = a.get("label")
        if isinstance(label, str):
            if label in seen:
                errs.append(f"[{i}] duplicate label {label!r}")
            elif not _LABEL_RE.match(label):
                errs.append(f"[{i}] invalid label {label!r}: use snake_case "
                            f"letters/digits/underscores, max 40 chars "
                            f"(e.g. foundation_a1)")
            seen.add(label)
    return errs


def validate_js_syntax(code: str) -> list:
    """Real JS syntax check via node --check. Falls back to balance check."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", "--check", path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return []
        return ["node --check: " + (r.stderr.strip() or r.stdout.strip())[:400]]
    except FileNotFoundError:
        errs = []
        for o, c, name in (("(", ")", "parens"), ("[", "]", "brackets"),
                           ("{", "}", "braces")):
            if code.count(o) != code.count(c):
                errs.append(f"unbalanced {name} (node unavailable)")
        return errs
    except subprocess.TimeoutExpired:
        return ["node --check timed out"]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def validate_against_state(action: dict, state) -> list:
    """Geometry/state checks: bounds, collision, stacking refs, dup labels,
    plus printer discipline: the first piece starts at the origin on the
    ground, and every later piece rests on the ground or on existing geometry
    (no floating, no sinking). `state` is a morphix_state.SceneState (or
    anything with labels/get/top_y/support_y/collides; has_physical() is
    used when present so lights/shaders committed first don't waive the
    start-at-origin rule for the first physical piece)."""
    from morphix_state import (BOUND_XZ, MAX_Y, ATTACH_DIST, ORIGIN_TOL,
                               SUPPORT_TOL, SINK_TOL, _NO_FOOTPRINT,
                               _GROUND_LIKE, aabb_for)
    has_physical = (state.has_physical() if hasattr(state, "has_physical")
                    else bool(state.labels()))
    errs = []
    if action.get("action") == "done":
        return errs
    label = action.get("label", "")
    if label in state.labels():
        errs.append(f"duplicate label {label!r} (already in scene)")
    ref = action.get("on_top_of")
    if ref and state.get(ref) is None:
        errs.append(f"on_top_of unknown label {ref!r}")
    pos = action.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        if not (abs(pos[0]) <= BOUND_XZ and abs(pos[2]) <= BOUND_XZ):
            errs.append(f"position {pos} out of bounds (+/-{BOUND_XZ})")
        if not (0 <= pos[1] <= MAX_Y):
            errs.append(f"height y={pos[1]} out of range [0,{MAX_Y}]")
        if action.get("action") not in _NO_FOOTPRINT:
            box = aabb_for(action, pos)
            hit = state.collides(box)
            if hit:
                errs.append(f"placement collides with {hit!r}")
            elif action.get("action") in _GROUND_LIKE:
                pass  # terrain is landscape ground: no placement rules
            elif not has_physical:
                # first physical piece: must start at the origin on the ground
                mn, mx = box["min"], box["max"]
                dx = 0.0 if mn[0] <= 0 <= mx[0] else min(abs(mn[0]), abs(mx[0]))
                dz = 0.0 if mn[2] <= 0 <= mx[2] else min(abs(mn[2]), abs(mx[2]))
                reach = (dx * dx + dz * dz) ** 0.5
                if reach > ORIGIN_TOL:
                    errs.append(
                        f"first piece must start at the origin (within "
                        f"{ORIGIN_TOL} of [0,0,0]), got position {pos}; "
                        f"center it near the origin resting on the ground")
                elif box["min"][1] > SUPPORT_TOL:
                    errs.append(
                        f"first piece floats (base y={box['min'][1]:.2f}); "
                        f"rest it on the ground (y=0)")
                elif box["min"][1] < -SINK_TOL:
                    errs.append(
                        f"first piece sinks below the ground "
                        f"(base y={box['min'][1]:.2f})")
            else:
                if box["min"][1] < -SINK_TOL:
                    errs.append(
                        f"placement sinks below the ground "
                        f"(base y={box['min'][1]:.2f})")
                else:
                    gap = box["min"][1] - state.support_y(box)
                    if gap > SUPPORT_TOL:
                        errs.append(
                            f"placement floats {gap:.2f} above its support; "
                            f"rest it on the ground or on existing geometry "
                            f"(use \"on_top_of\" to stack)")
                d = state.nearest_dist(pos)
                if d > ATTACH_DIST:
                    errs.append(f"detached from scene ({d:.1f} > {ATTACH_DIST}); "
                                f"grow from the print head near {state.head}")
    return errs


def validate_scene(actions: list, blocks: list) -> dict:
    """Full pipeline report: actions + per-block guard + syntax + assembly."""
    report = {"ok": True, "action_errors": [], "blocks": [], "scene_errors": []}
    report["action_errors"] = validate_actions(actions)
    labels = [a.get("label") for a in actions if isinstance(a, dict)]
    if len(labels) != len(set(labels)):
        report["scene_errors"].append("duplicate labels across actions")
    for i, (act, code) in enumerate(zip(actions, blocks)):
        guard_errs = validate_js(act, code or "")
        syntax_errs = validate_js_syntax(code or "")
        b = {"index": i, "action": act.get("action") if isinstance(act, dict) else "?",
             "label": act.get("label") if isinstance(act, dict) else "?",
             "guard_errors": guard_errs, "syntax_errors": syntax_errs,
             "ok": not guard_errs and not syntax_errs}
        report["blocks"].append(b)
    # assembly: no duplicate const declarations (uid collision across blocks)
    consts = re.findall(r"const\s+([A-Za-z_$][\w$]*)", "\n".join(blocks))
    dupes = {c for c in consts if consts.count(c) > 1}
    if dupes:
        report["scene_errors"].append(
            f"duplicate const declarations: {sorted(dupes)[:5]}")
    report["ok"] = not report["action_errors"] and not report["scene_errors"] \
        and all(b["ok"] for b in report["blocks"])
    return report


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--actions", required=True)
    p.add_argument("--js", required=True)
    args = p.parse_args()
    actions = json.load(open(args.actions, encoding="utf-8"))
    code = open(args.js, encoding="utf-8").read()
    # split assembled scene.js back into blocks on scene.add/push terminators
    term = re.compile(r"scene\.add\([^;]*\);|morphix_\w+\.push\([^;]*\);")
    blocks, last = [], 0
    for m in term.finditer(code):
        blocks.append(code[last:m.end()].strip())
        last = m.end()
    tail = code[last:].strip()
    if tail:
        blocks.append(tail)
    blocks = [b for b in blocks if b]
    if len(blocks) != len(actions):  # fall back: validate whole file as one unit
        report = {"ok": True, "action_errors": validate_actions(actions),
                  "blocks": [], "scene_errors": []}
        syn = validate_js_syntax(code)
        report["scene_errors"].extend(syn)
        report["ok"] = not report["action_errors"] and not syn
    else:
        report = validate_scene(actions, blocks)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
