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
}

_COLOR_RE = re.compile(r"^0x[0-9a-fA-F]{6}$")
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
    """Geometry/state checks: bounds, collision, stacking refs, dup labels.
    `state` is a morphix_state.SceneState (or anything with labels/get/top_y/collides)."""
    from morphix_state import (BOUND_XZ, MAX_Y, ATTACH_DIST, _NO_FOOTPRINT,
                               aabb_for)
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
            hit = state.collides(aabb_for(action, pos))
            if hit:
                errs.append(f"placement collides with {hit!r}")
            elif state.labels():
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
