# morphix_scene_plan.py
# Scene-mode planner: a structured plan (objective, style, backend,
# template systems, overrides) becomes Scene IR by composing TESTED
# templates — the small thinker selects and parameterizes systems, it never
# invents vertices or shaders. Plans are schema-validated with a bounded
# repair retry; failures raise loudly (strict mode), never silently
# substituted.
#
# Hierarchy: Scene -> Environment (base template) -> Major systems
# (additional templates) -> Components (IR nodes).

import json
import re

from morphix_scene_ir import validate_scene_ir
from morphix_scene_templates import TEMPLATE_NAMES, build

PLAN_VERSION = 1

SCENE_PLANNER_SYSTEM = (
    "You are a Three.js scene planner. Reply with ONLY a JSON object, no "
    "prose, no markdown:\n"
    '{"objective": "<what the viewer should see>", '
    '"style": "<visual style in a few words>", '
    '"backend": "webgl", '
    '"templates": ["<base>", "<system>", ...], '
    '"seed": <integer>, '
    '"animations": true}\n'
    "Rules: backend is ALWAYS \"webgl\" (the engine runs Three.js r152, "
    "which has no WebGPU renderer). templates lists 1-3 names from: "
    + ", ".join(TEMPLATE_NAMES) + ". "
    "The first template owns the environment and camera; the rest add "
    "object systems. If unsure, answer with a single best template and "
    "animations true."
)

REQUIRED_PLAN_KEYS = ("objective", "style", "backend", "templates", "seed")


def parse_plan_response(raw):
    """Extract + validate a planner reply. Returns (plan, errors)."""
    if not raw:
        return None, ["empty planner reply"]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, ["reply was not a JSON object"]
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None, ["reply was not valid JSON"]
    if not isinstance(obj, dict):
        return None, ["plan must be a JSON object"]
    errors = []
    for key in REQUIRED_PLAN_KEYS:
        if key not in obj:
            errors.append(f"plan missing {key!r}")
    if obj.get("backend") != "webgl":
        errors.append(f"backend must be 'webgl' on r152 "
                      f"(got {obj.get('backend')!r})")
    ts = obj.get("templates")
    if not isinstance(ts, list) or not ts:
        errors.append("templates must be a non-empty list")
    else:
        for t in ts:
            if t not in TEMPLATE_NAMES:
                errors.append(f"unknown template {t!r} "
                              f"(available: {', '.join(TEMPLATE_NAMES)})")
        if len(ts) > 3:
            errors.append("at most 3 template systems per scene")
    if not isinstance(obj.get("seed"), int):
        errors.append("seed must be an integer")
    if errors:
        return None, errors
    plan = {"version": PLAN_VERSION,
            "objective": str(obj["objective"])[:200],
            "style": str(obj["style"])[:120],
            "backend": "webgl",
            "templates": [str(t) for t in ts[:3]],
            "seed": int(obj["seed"]),
            "animations": bool(obj.get("animations", True))}
    for key in ("camera", "fog"):
        if key in obj and isinstance(obj[key], dict):
            plan[key] = obj[key]
    return plan, []


def plan_scene(prompt, thinker_model, num_predict=800, retries=1):
    """Plan a scene with the thinker (bounded retries). Returns a validated
    plan dict; raises PipelineError on failure."""
    from morphix_config import PipelineError, ollama_generate
    last = "no attempt"
    for attempt in range(retries + 1):
        if attempt:
            sys_msg = (SCENE_PLANNER_SYSTEM +
                       f"\nYour previous reply failed validation: {last}. "
                       f"Reply with JSON only.")
        else:
            sys_msg = SCENE_PLANNER_SYSTEM
        raw = ollama_generate(thinker_model, sys_msg,
                              f"Plan a scene: {prompt}. JSON only.")
        plan, errors = parse_plan_response(raw)
        if plan is not None:
            return plan
        last = "; ".join(errors)
    raise PipelineError("scene planner failed validation: " + last)


def plan_to_scene_ir(plan):
    """Compose a Scene IR from a validated plan: the first template owns
    the world (environment, camera); the rest contribute object systems
    with prefixed IDs. Deterministic; raises on budget/spec violations."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")
    names = plan.get("templates") or []
    if not names:
        raise ValueError("plan has no template systems")
    for t in names:
        if t not in TEMPLATE_NAMES:
            raise ValueError(f"unknown template {t!r}")
    seed = int(plan.get("seed", 1))
    base = build(names[0], seed=seed)
    base["meta"]["name"] = str(plan.get("objective", names[0]))[:80]
    if plan.get("animations") is False:
        base["animations"] = []
    if isinstance(plan.get("camera"), dict):
        for k in ("position", "target", "fov"):
            if k in plan["camera"]:
                base["camera"][k] = plan["camera"][k]
    if isinstance(plan.get("fog"), dict):
        base["environment"]["fog"] = plan["fog"]

    for extra in names[1:]:
        _merge_system(base, build(extra, seed=seed + 7919))
    errs = validate_scene_ir(base)
    if errs:
        raise ValueError("composed scene IR invalid: " + "; ".join(errs[:5]))
    return base


def _merge_system(base, other):
    """Merge another template's object systems into base (prefixed IDs).
    Base keeps its environment and camera. Two passes so child references
    always resolve inside the merged set."""
    tag = "".join(c if (c.isalnum() or c == "_") else "_"
                  for c in other["meta"]["name"])[:16] or "sys"
    existing = {n.get("id") for n in base.get("nodes", [])}
    idmap = {}
    for n in other.get("nodes", []):
        new = f"{tag}_{n['id']}"
        i = 2
        while new in existing or new in idmap.values():
            new = f"{tag}_{n['id']}_{i}"
            i += 1
        idmap[n["id"]] = new
    for n in other.get("nodes", []):
        c = dict(n)
        c["id"] = idmap[n["id"]]
        c["children"] = [idmap[k] for k in n.get("children", [])]
        base["nodes"].append(c)
    for Ld in other.get("lights", []):
        if len(base["lights"]) >= int(
                base["budgets"].get("max_lights", 8)):
            break
        c = dict(Ld)
        new = f"{tag}_{Ld['id']}"
        i = 2
        while new in existing:
            new = f"{tag}_{Ld['id']}_{i}"
            i += 1
        c["id"] = new
        base["lights"].append(c)
    for a in other.get("animations", []):
        c = dict(a)
        c["target"] = idmap.get(c.get("target", ""), c.get("target", ""))
        base["animations"].append(c)
