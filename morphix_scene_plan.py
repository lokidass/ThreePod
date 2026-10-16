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

PLAN_VERSION = 2

# Deterministic keyword router: prompt words -> concrete systems. The
# router (not the thinker) owns system SELECTION so benchmark prompts
# resolve reproducibly; the thinker owns semantics (subject, required
# elements, mood) and may ADD freestyle elements. Every route is unit-
# tested against the benchmark prompts.
# A system entry: {"source": "template"|"part", "name": str,
#                  "required": bool, "params": {...}}.
_ROUTES = [
    (("lighthouse", "beacon"),
     {"templates": [], "required": ["lighthouse_tower"],
      "parts": ["lighthouse_tower"],
      "environment": {"background": {
          "kind": "gradient", "top": "0x2C4258", "mid": "0xD89A5A",
          "bottom": "0x1C2634"}},
      "camera": {"kind": "perspective", "fov": 55, "near": 0.1,
                 "far": 600, "position": [16, 9, 22],
                 "target": [0, 3, 0]}}),
    (("table", "desk", "dining", "furniture"),
     {"templates": [], "required": ["table_set", "room_shell"],
      "parts": ["table_set", "room_shell"],
      "environment": {"background": {"kind": "color",
                                     "color": "0x2A2620"}},
      "camera": {"kind": "perspective", "fov": 50, "near": 0.1,
                 "far": 200, "position": [4.2, 2.8, 5.2],
                 "target": [0, 0.8, 0]}}),
    (("room", "interior", "bedroom", "living"),
     {"templates": [], "required": ["room_shell"],
      "parts": ["room_shell"],
      "environment": {"background": {"kind": "color",
                                     "color": "0x2A2620"}},
      "camera": {"kind": "perspective", "fov": 50, "near": 0.1,
                 "far": 200, "position": [4.2, 2.8, 5.2],
                 "target": [0, 0.8, 0]}}),
    (("forest", "tree", "trees", "woods", "wood", "jungle"),
     {"templates": ["forest"], "required": ["forest"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("mountain", "hill", "peak", "landscape", "terrain", "cliff",
      "valley"),
     {"templates": ["mountains"], "required": ["mountains"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("city", "street", "urban", "downtown", "night"),
     {"templates": ["city"], "required": ["city"],
      "parts": ["lamp_post"],
      "environment": {}, "camera": {}}),
    (("scifi", "sci-fi", "space", "hangar", "robot", "futuristic"),
     {"templates": ["scifi"], "required": ["scifi"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("water", "lake", "river", "ocean", "sea", "pond"),
     {"templates": ["water"], "required": ["water"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("harbor", "harbour", "boats", "boat", "ship", "dock", "pier",
      "sail"),
     {"templates": ["water"], "required": ["water", "boat", "pier"],
      "parts": ["boat", "pier"],
      "environment": {}, "camera": {}}),
    (("fog", "mist", "atmospheric", "haze"),
     {"templates": ["sky_fog"], "required": ["sky_fog"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("cabin", "hut", "lodge", "snow", "winter"),
     {"templates": [], "required": ["cabin"],
      "parts": ["cabin"],
      "environment": {"background": {
          "kind": "gradient", "top": "0x2C4258", "mid": "0xA8BECF",
          "bottom": "0x1C2634"}},
      "camera": {"kind": "perspective", "fov": 55, "near": 0.1,
                 "far": 400, "position": [10, 5, 12],
                 "target": [0, 1.5, 0]}}),
    (("castle", "fortress", "fort", "keep", "palace"),
     {"templates": [], "required": ["castle_keep"],
      "parts": ["castle_keep"],
      "environment": {"background": {
          "kind": "gradient", "top": "0x2C4258", "mid": "0xA8BECF",
          "bottom": "0x1C2634"}},
      "camera": {"kind": "perspective", "fov": 55, "near": 0.1,
                 "far": 500, "position": [16, 9, 20],
                 "target": [0, 2.5, 0]}}),
    (("tent", "camp", "campsite", "caravan", "desert", "dune", "sand"),
     {"templates": [], "required": ["tent", "dunes"],
      "parts": ["tent", "dunes"],
      "environment": {"background": {
          "kind": "gradient", "top": "0x3A6A9E", "mid": "0xE0C090",
          "bottom": "0x8A6F4D"}},
      "camera": {"kind": "perspective", "fov": 55, "near": 0.1,
                 "far": 400, "position": [10, 5, 12],
                 "target": [0, 1, 0]}}),
    (("coral", "reef", "underwater", "ocean floor"),
     {"templates": ["water", "particles"],
      "required": ["water", "coral_bed"],
      "parts": ["coral_bed"],
      "environment": {}, "camera": {}}),
    (("studio", "product", "showcase", "showroom", "pedestal"),
     {"templates": ["studio"], "required": ["studio"],
      "parts": [],
      "environment": {}, "camera": {}}),
    (("particle", "stars", "embers", "fireflies", "snowfall"),
     {"templates": ["particles"], "required": ["particles"],
      "parts": [],
      "environment": {}, "camera": {}}),
]


def _stems(word):
    """Keyword-matching forms: exact plus light de-inflection. Only stems
    that ARE catalogue keywords ever match, so 'seat' can never drag in
    'sea' and 'glass' never becomes a keyword."""
    out = {word}
    for suf, cut in (("ies", 3), ("ing", 3), ("ed", 2), ("s", 1),
                     ("y", 1)):
        if word.endswith(suf) and len(word) - cut >= 3:
            out.add(word[:len(word) - cut])
    # ies -> y (cities -> city)
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    return out


def route_systems(prompt):
    """Deterministic prompt -> systems. Returns
    {"templates": [...], "parts": [...], "required": [...],
     "environment": {...}, "camera": {...}, "hits": [...]}, merging every
    matching route (deduplicated, order-stable). Empty when nothing
    matches — the thinker or freestyle must then supply systems, else the
    run fails loudly (never a silent default scene)."""
    words = set(re.findall(r"[a-z][a-z\-]*", (prompt or "").lower()))
    forms = set()
    for w in words:
        forms.update(_stems(w))
    templates, parts, required, hits = [], [], [], []
    environment, camera = {}, {}
    for keywords, route in _ROUTES:
        hit = sorted(set(keywords) & forms)
        if hit:
            hits.append(hit)
            for t in route["templates"]:
                if t not in templates:
                    templates.append(t)
            for p in route["parts"]:
                if p not in parts:
                    parts.append(p)
            for r in route["required"]:
                if r not in required:
                    required.append(r)
            if route["environment"]:
                environment = route["environment"]
            if route["camera"]:
                camera = route["camera"]
    return {"templates": templates, "parts": parts, "required": required,
            "environment": environment, "camera": camera, "hits": hits}

SCENE_PLANNER_SYSTEM = (
    "You are a Three.js scene planner. Reply with ONLY a JSON object, no "
    "prose, no markdown:\n"
    '{"objective": "<what the viewer should see>", '
    '"style": "<visual style in a few words>", '
    '"subject": "<main subject words>", '
    '"scene_type": "<interior|landscape|urban|scifi|seascape|other>", '
    '"backend": "webgl", '
    '"templates": ["<base>", "<system>", ...], '
    '"required": ["must-have element names"], '
    '"freestyle": [{"name": "<snake>", "kind": "mesh", '
    '"geometry": {"ctor": "<BoxGeometry|SphereGeometry|CylinderGeometry|'
    'ConeGeometry|TorusGeometry|PlaneGeometry|CircleGeometry|RingGeometry>", '
    '"params": {"w": 1}}, '
    '"material": {"kind": "standard", "color": "0xFFFFFF"}, '
    '"transform": {"position": [0,0,0]}, '
    '"strategy": "<why this technique>", "required": true}], '
    '"seed": <integer>, '
    '"animations": true}\n'
    "Rules: backend is ALWAYS \"webgl\" (the engine runs Three.js r152, "
    "which has no WebGPU renderer). templates lists 0-3 names from: "
    + ", ".join(TEMPLATE_NAMES) + ". "
    "Name every must-have element in required. Use freestyle ONLY for an "
    "element no template covers, with real r152 constructors and finite "
    "numbers. Omit freestyle when everything is covered. If unsure, answer "
    "with subject, scene_type, one best template, and animations true."
)

REQUIRED_PLAN_KEYS = ("objective", "style", "backend", "seed")
VALID_SCENE_TYPES = ("interior", "landscape", "urban", "scifi", "seascape",
                     "other")


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
    substitutions = {}
    if not isinstance(ts, list) or not ts:
        errors.append("templates must be a non-empty list")
    else:
        import difflib
        fixed = []
        for t in ts:
            if t in TEMPLATE_NAMES:
                fixed.append(t)
                continue
            near = difflib.get_close_matches(str(t), TEMPLATE_NAMES,
                                             n=1, cutoff=0.6)
            if near:
                # Near-miss ("tables", "forrest") maps to the catalogue
                # entry loudly instead of refusing an ordinary request.
                substitutions[str(t)] = near[0]
                fixed.append(near[0])
            else:
                errors.append(f"unknown template {t!r} "
                              f"(available: {', '.join(TEMPLATE_NAMES)})")
        ts = fixed
        if len(ts) > 3:
            errors.append("at most 3 template systems per scene")
    if errors:
        return None, errors
    if obj.get("scene_type") not in VALID_SCENE_TYPES \
            and obj.get("scene_type") is not None:
        errors.append(f"scene_type must be one of "
                      f"{list(VALID_SCENE_TYPES)} "
                      f"(got {obj.get('scene_type')!r})")
    freestyle = obj.get("freestyle", [])
    if freestyle is not None and not isinstance(freestyle, list):
        errors.append("freestyle must be a list")
        freestyle = []
    required = obj.get("required", [])
    if required is not None and not isinstance(required, list):
        errors.append("required must be a list")
        required = []
    notes = []
    raw_seed = obj.get("seed", 11)
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        seed = None
    if seed is None or seed != raw_seed:
        # The seed carries no semantics (any int reproduces); coercing a
        # malformed one with a logged note beats failing a whole run.
        notes.append(f"malformed seed {raw_seed!r} coerced to 11")
        seed = 11
    if errors:
        return None, errors
    plan = {"version": PLAN_VERSION,
            "objective": str(obj["objective"])[:200],
            "style": str(obj["style"])[:120],
            "subject": str(obj.get("subject", ""))[:120],
            "scene_type": obj.get("scene_type") or "other",
            "backend": "webgl",
            "templates": [str(t) for t in ts[:3]],
            "required": [str(r)[:64] for r in (required or [])],
            "freestyle": freestyle or [],
            "seed": seed,
            "animations": bool(obj.get("animations", True)),
            "substitutions": substitutions,
            "notes": notes}
    for key in ("camera", "fog"):
        if key in obj and isinstance(obj[key], dict):
            plan[key] = obj[key]
    return plan, []


# Known synonyms when matching thinker-required names to systems.
# Anything else unresolvable fails loudly (never invented).
REQUIRED_ALIASES = {"ship": "boat", "ships": "boat", "table": "table_set",
                    "room": "room_shell", "tower": "lighthouse_tower",
                    "house": "cabin", "hut": "cabin", "quay": "pier",
                    "dock": "pier", "wharf": "pier", "harbor": "pier",
                    "jetty": "pier", "streetlamp": "lamp_post",
                    "street lamp": "lamp_post", "trees": "forest"}


def plan_scene(prompt, thinker_model, num_predict=800, retries=1,
               systems=()):
    """Plan a scene with the thinker (bounded retries). Returns a validated
    plan dict; raises PipelineError on failure. ``systems`` lists the
    already-routed catalogue systems so required elements stay within
    what exists (plus freestyle specs) instead of inventing names."""
    from morphix_config import PipelineError, ollama_generate
    last = "no attempt"
    context = ""
    if systems:
        context = (" Available systems you may reference in required: "
                   + ", ".join(systems) + ". Name required elements ONLY "
                   "from these systems (or describe them as freestyle).")
    for attempt in range(retries + 1):
        if attempt:
            sys_msg = (SCENE_PLANNER_SYSTEM + context +
                       f"\nYour previous reply failed validation: {last}. "
                       f"Reply with JSON only.")
        else:
            sys_msg = SCENE_PLANNER_SYSTEM + context
        raw = ollama_generate(thinker_model, sys_msg,
                              f"Plan a scene: {prompt}. JSON only.")
        plan, errors = parse_plan_response(raw)
        if plan is not None:
            return plan
        last = "; ".join(errors)
    raise PipelineError("scene planner failed validation: " + last)


def validate_freestyle(spec):
    """Validate a thinker-proposed freestyle element against the knowledge
    registry. Returns (node_dict|None, errors). Unknown constructors,
    non-finite params, or bad materials fail loudly here — never as a
    broken mesh in the browser. Optional elements with errors are dropped
    by the caller (logged); required ones fail the run."""
    from morphix_three_knowledge import api_supported
    import morphix_scene_ir as S
    if not isinstance(spec, dict):
        return None, ["freestyle element must be an object"]
    errors = []
    name = spec.get("name", "")
    if not (isinstance(name, str) and name.replace("_", "").isalnum()):
        errors.append(f"freestyle element needs a snake_case name "
                      f"(got {name!r})")
    kind = spec.get("kind", "mesh")
    if kind not in ("mesh", "instanced", "points"):
        errors.append(f"freestyle {name!r}: kind must be "
                      f"mesh|instanced|points (got {kind!r})")
    geo = spec.get("geometry") or {}
    ctor = geo.get("ctor", "")
    ok, why = api_supported(ctor)
    if not ok:
        errors.append(f"freestyle {name!r}: geometry {ctor!r} unsupported "
                      f"({why})")
    params = geo.get("params") or {}
    if not isinstance(params, dict) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            and abs(v) != float("inf") and v == v
            for v in params.values()):
        errors.append(f"freestyle {name!r}: params must be finite numbers")
    mat = spec.get("material") or {"kind": "standard"}
    if mat.get("kind") not in ("standard", "basic", "points"):
        errors.append(f"freestyle {name!r}: material kind must be "
                      f"standard|basic|points (got {mat.get('kind')!r})")
    t = spec.get("transform") or {}
    for f in ("position", "rotation", "scale"):
        v = t.get(f, [0, 0, 0] if f != "scale" else [1, 1, 1])
        if not (isinstance(v, list) and len(v) == 3 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                and abs(x) != float("inf") and x == x for x in v)):
            errors.append(f"freestyle {name!r}: transform.{f} must be "
                          f"[x,y,z] numbers")
    if abs((t.get("position") or [0, 0, 0])[0]) > 200 or \
            abs((t.get("position") or [0, 0, 0])[2]) > 200:
        errors.append(f"freestyle {name!r}: position outside scene bounds")
    if errors:
        return None, errors
    node = S.node(f"free_{name}", kind=kind,
                  geometry=S.geometry(ctor, params),
                  material=S.material(mat.get("kind", "standard"),
                                      color=mat.get("color", "0xFFFFFF")),
                  transform=S.transform(
                      position=t.get("position", [0, 0, 0]),
                      rotation=t.get("rotation", [0, 0, 0]),
                      scale=t.get("scale", [1, 1, 1])),
                  count=int(spec.get("count", 0)),
                  distribution=dict(spec.get("distribution", {})))
    node["required"] = bool(spec.get("required", False))
    return node, []


def compose_routed_scene(routed, thinker_plan=None, seed=11):
    """Compose a Scene IR from routed systems + an optional thinker plan.

    Order of authority: router systems (deterministic) + thinker-added
    systems/freestyle (validated). The first TEMPLATE owns environment and
    camera; systems without any template get a default light rig (logged).
    Required elements (route + thinker) become required nodes; anything
    unresolvable fails loudly with the missing element named. Returns
    (ir, report{systems, required_ids, resolutions}).
    """
    import morphix_scene_ir as S
    from morphix_scene_templates import build
    thinker_plan = thinker_plan or {}
    systems = list(routed.get("templates", []))
    for extra in thinker_plan.get("systems_add", []) or []:
        if extra in TEMPLATE_NAMES and extra not in systems:
            systems.append(extra)
    if not systems and not routed.get("parts") \
            and not thinker_plan.get("freestyle"):
        raise ValueError("no scene systems resolved: router found nothing "
                         "and the thinker proposed nothing usable")
    base_name = systems[0] if systems else "freestyle"
    if systems:
        ir = build(systems[0], seed=seed)
        ir["meta"]["name"] = str(thinker_plan.get("objective") or base_name)[:80]
        # Route-level environment/camera win over the base template: the
        # route author chose them for this prompt (e.g. dusk for a
        # lighthouse); empty route env/camera keeps template defaults.
        if routed.get("environment"):
            ir["environment"].update(routed["environment"])
        if routed.get("camera"):
            ir["camera"].update(routed["camera"])
    else:
        ir = S.new_scene(str(thinker_plan.get("objective") or "freestyle"),
                         seed=seed)
        env, cam = routed.get("environment") or {}, routed.get("camera") or {}
        if env:
            ir["environment"].update(env)
        if cam:
            ir["camera"].update(cam)
        ir["lights"].append({"id": "fill", "kind": "hemisphere",
                             "color": "0xBDD7F2", "intensity": 0.7,
                             "ground": "0x334455"})
        ir["lights"].append({"id": "key", "kind": "directional",
                             "color": "0xFFF2E0", "intensity": 1.4,
                             "position": [6, 10, 7]})
    if thinker_plan.get("animations") is False:
        ir["animations"] = []
    if isinstance(thinker_plan.get("camera"), dict):
        for k in ("position", "target", "fov"):
            if k in thinker_plan["camera"]:
                ir["camera"][k] = thinker_plan["camera"][k]
    if isinstance(thinker_plan.get("fog"), dict):
        ir["environment"]["fog"] = thinker_plan["fog"]

    required_ids, resolutions = [], []
    # track which node ids each system contributed (for required marking)
    system_nodes = {}
    if systems:
        system_nodes[systems[0]] = [n["id"] for n in ir.get("nodes", [])]
    for extra in systems[1:]:
        idmap = _merge_system(ir, build(extra, seed=seed + 7919))
        system_nodes[extra] = list(idmap.values())
        resolutions.append(f"system template {extra}")
    from morphix_scene_parts import build_part
    for part in routed.get("parts", []):
        piece = build_part(part, prefix=f"{part}_",
                           required=(part in routed.get("required", [])))
        ir["nodes"].extend(piece["nodes"])
        ir["lights"].extend(piece.get("lights", []))
        for rel in piece.get("relations", []):
            if all(k in rel for k in ("a", "rel", "b")):
                entry = {"a": rel["a"], "rel": rel["rel"],
                         "b": rel["b"], "tol": rel.get("tol", 0.15)}
                if entry not in ir["relations"]:
                    ir["relations"].append(entry)
        system_nodes[f"part:{part}"] = [n["id"] for n in piece["nodes"]]
        required_ids.extend(n["id"] for n in piece["nodes"]
                            if n.get("required"))
        resolutions.append(f"part {part} ({piece['strategy']})")
    for spec in thinker_plan.get("freestyle", []) or []:
        if not isinstance(spec, dict):
            # A bare string cannot carry validity or required-ness: drop it
            # loudly rather than crashing the composition.
            resolutions.append(f"freestyle {spec!r} is not an object: "
                               f"dropped")
            continue
        node, errs = validate_freestyle(spec)
        if node is None:
            if spec.get("required"):
                raise ValueError("required freestyle element "
                                 f"{spec.get('name')!r} invalid: "
                                 + "; ".join(errs))
            resolutions.append(f"freestyle {spec.get('name')!r} dropped: "
                               + "; ".join(errs))
            continue
        ir["nodes"].append(node)
        if node.get("required"):
            required_ids.append(node["id"])
        resolutions.append(f"freestyle {spec.get('name')} "
                           f"({spec.get('strategy', 'model-composed')})")
    # Coverage gate: every required element name must resolve to at least
    # one system (template/part/freestyle). Unresolved requirements fail
    # loudly here — never as a silently incomplete scene downstream.
    wanted = list(routed.get("required", [])) + list(
        thinker_plan.get("required", []) or [])
    by_id = {n["id"]: n for n in ir.get("nodes", [])}
    _rels = thinker_plan.get("relations", [])
    if _rels is not None and not isinstance(_rels, list):
        resolutions.append(f"thinker relations malformed "
                           f"({type(_rels).__name__}): ignored")
        _rels = []
    for i, rel in enumerate(_rels or []):
        # Thinker relations reference element names; resolve to node ids
        # (exact, then substring). Unresolvable relations fail loudly.
        if not isinstance(rel, dict):
            raise ValueError(f"relation[{i}] must be an object")
        sides = {}
        for side in ("a", "b"):
            want = str(rel.get(side, "")).lower()
            hit = [nid for nid in by_id
                   if nid.lower() == want or want in nid.lower()
                   or nid.lower() in want]
            if not hit:
                raise ValueError(
                    f"relation[{i}]: {side}={rel.get(side)!r} matches no "
                    f"scene node")
            sides[side] = sorted(hit, key=len)[0]
        entry = {"a": sides["a"], "rel": rel.get("rel", "near"),
                 "b": sides["b"], "tol": rel.get("tol", 0.15)}
        if entry not in ir["relations"]:
            ir["relations"].append(entry)
            resolutions.append(f"relation {sides['a']} "
                               f"{entry['rel']} {sides['b']}")
    # Matchable tokens per system (template names, part names).
    tokens = {}
    for s in system_nodes:
        tokens[s] = s.split("part:")[-1].lower()
    required_systems = []
    for want in wanted:
        w = str(want).lower()

        def _hits(word):
            out = [s for s, tok in tokens.items()
                   if word in tok or tok in word]
            if "freestyle" in word:
                out.append("freestyle")
            return out

        # Phase 1: direct match (compound names must never be rewritten:
        # "tower" inside "lighthouse_tower" is not an alias occasion).
        hit_systems = _hits(w)
        if not hit_systems:
            # Phase 2: alias substitution for bare synonyms ("ship" ->
            # "boat", "harbor_walls" -> "pier_walls").
            aw = w
            for alias, target in REQUIRED_ALIASES.items():
                if aw == alias:
                    aw = target
                    break
            else:
                for alias, target in REQUIRED_ALIASES.items():
                    if alias in aw:
                        aw = aw.replace(alias, target)
                        break
            if aw != w:
                hit_systems = _hits(aw)
                if hit_systems:
                    resolutions.append(f"required {want!r} aliased "
                                       f"to {aw!r}")
        # freestyle nodes match by their own names
        free = [nid for nid, n in by_id.items()
                if nid.startswith("free_") and
                (w in nid.lower() or nid.lower() in w)]
        if not hit_systems and not free:
            raise ValueError(
                f"required element {want!r} resolved to no scene system "
                f"(systems: {sorted(system_nodes)}); cannot complete")
        for s in hit_systems:
            if s != "freestyle" and s not in required_systems:
                required_systems.append(s)
        for s in hit_systems:
            if s == "freestyle":
                continue
            for nid in system_nodes.get(s, []):
                if nid not in by_id:
                    continue
                if (by_id[nid].get("geometry") or {}).get("generator"):
                    # Procedural nodes (terrain) of a required system are
                    # verified live at render, never flagged static-required
                    # (generators are scene effects by construction).
                    continue
                by_id[nid]["required"] = True
                if nid not in required_ids:
                    required_ids.append(nid)
        for nid in free:
            by_id[nid]["required"] = True
            if nid not in required_ids:
                required_ids.append(nid)
        resolutions.append(f"required {want!r} -> "
                           f"{sorted(hit_systems) + free}")
    errs = S.validate_scene_ir(ir)
    if errs:
        raise ValueError("composed scene IR invalid: " + "; ".join(errs[:5]))
    return ir, {"systems": systems, "required_ids": required_ids,
                "required_systems": required_systems,
                "resolutions": resolutions}


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
    return idmap
