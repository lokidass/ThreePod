# morphix_thinker.py
# Morphix Thinker — natural-language scene description -> atomic action JSONs.
# Backend: DeepSeek-R1 via local Ollama, configured in morphix_config.
#
# In STRICT mode (MORPHIX_STRICT_MODELS=true, the deployment default) the
# thinker ALWAYS uses the real model: a missing model or unusable output
# raises ModelUnavailableError so the user sees the failure. The deterministic
# rule planner only runs when strict mode is turned off.
#
# Usage:
#   python morphix_thinker.py "a red sphere on a blue box with a warm sun light"
#   python morphix_thinker.py --prompt "..." --model deepseek-r1:1.5b --out actions.json

import json
import re
import sys

from morphix_config import (ModelUnavailableError, STRICT_MODELS, THINKER_MODEL,
                            ollama_generate)

ALLOWED_ACTIONS = [
    "add_sphere", "add_box", "add_cylinder", "add_cone", "add_torus",
    "add_torus_knot", "add_plane", "add_extrude", "add_lathe", "add_tube",
    "add_terrain", "add_instanced_mesh", "add_light", "add_shader",
    "set_material", "add_particles", "add_animation", "add_post",
    "add_environment",
]

THINKER_SYSTEM = """You are the Morphix Thinker, a 3D scene planner.
Decompose the user's scene description into a list of ATOMIC action JSONs.
Rules:
- Output ONLY a JSON array, no markdown, no explanation, no <think> discussion.
- Each item is ONE atomic operation with exactly the fields below.
- Allowed actions and their required fields:
  add_sphere: radius, position [x,y,z], color (0xRRGGBB), label
  add_box: width, height, depth, position, color, label
  add_cylinder: radiusTop, radiusBottom, height, position, color, label
  add_cone: radius, height, position, color, label
  add_torus: radius, tube, position, color, label
  add_plane: width, height, position, color, label
  add_extrude: depth, position, color, label (roofs, prisms)
  add_terrain: width, segments, amplitude, frequency, color, label
  add_light: type (directional|point|ambient|spot|hemisphere), intensity, position, label
  add_shader: preset (fresnel|noise_displacement|hologram|water|fire), color, intensity, label
  add_particles: count, size, color, label
- color is ALWAYS 6 hex digits like 0xFF0000, never 8 digits, never a string in a numeric field.
- Numbers are JSON numbers, never strings. intensity is a number like 2.5.
- Word mapping: ring/donut -> add_torus, ball/head -> add_sphere, sun/lamp -> add_light, glow -> add_shader fresnel.
- Labels must be unique snake_case with a short uid suffix (e.g. sphere_a1b2c3).
- Positions sit on/above the ground (y >= 0). Keep 1-8 actions for simple
  scenes, up to 16 for complex ones (house, figure, landscape).
- Complex scenes are BUILT UP in dependency order: foundation -> walls ->
  roof -> door/windows -> lights. Use on_top_of stacking where heights depend
  on earlier parts.
- Word mapping: ring/donut -> add_torus, ball/head -> add_sphere, sun/lamp -> add_light, glow -> add_shader fresnel, house -> foundation+walls+roof+door+windows chain, roof -> add_extrude.
- Example: [{"action":"add_box","width":8,"height":0.3,"depth":6,"position":[0,0,0],"color":"0x777777","label":"base_a1"}, {"action":"add_sphere","radius":0.5,"position":[0,1,0],"color":"0xFF0000","label":"ball_b2"}]"""


def _strip_think(text: str) -> str:
    """DeepSeek-R1 wraps reasoning in <think>...</think>; keep only the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def _fix_trailing_commas(text: str) -> str:
    """DeepSeek loves `},]` / `,}`. Make that valid before json.loads."""
    return _TRAILING_COMMA.sub(r"\1", text)


def _extract_json_array(text: str):
    """Robustly pull the JSON array out of a DeepSeek reply. Tolerates a
    trailing comma and stray text by scanning for the largest valid array."""
    text = _strip_think(text)
    # strip markdown fences if the model adds them despite instructions
    text = re.sub(r"```(?:json)?", "", text).strip()
    for lo, ch in enumerate(text):
        if ch != "[":
            continue
        for hi in range(len(text), lo, -1):
            cand = _fix_trailing_commas(text[lo:hi]).rstrip()
            if cand.count("[") != cand.count("]"):
                continue
            try:
                val = json.loads(cand)
                if isinstance(val, list):
                    return val
            except Exception:
                continue
    raise ValueError("no JSON array in thinker output")


# ---------------------------------------------------------------- fallback
_COLOR_WORDS = {
    "red": "0xFF0000", "blue": "0x2244AA", "green": "0x22AA44",
    "gold": "0xFFD700", "white": "0xFFFFFF", "black": "0x222222",
    "warm": "0xFFEECC", "orange": "0xFF8800", "teal": "0x00ffcc",
}

_uid_counter = [0]


def _uid(prefix: str) -> str:
    _uid_counter[0] += 1
    return f"{prefix}_{_uid_counter[0]:02d}"


def rule_based_plan(prompt: str) -> list:
    """Deterministic keyword planner. Always valid, always grounded."""
    p = prompt.lower()
    actions = []
    color = "0x888888"
    for word, hexv in _COLOR_WORDS.items():
        if word in p:
            color = hexv
            break
    m = re.search(r"0x[0-9a-fA-F]{6}", prompt)
    if m:
        color = m.group(0)
    if "box" in p or "cube" in p or "base" in p or "floor" in p or "table" in p:
        actions.append({"action": "add_box", "width": 2, "height": 0.3, "depth": 2,
                        "position": [0, 0, 0], "color": color, "label": _uid("base")})
        y = 0.8
    else:
        y = 1.0
    if "sphere" in p or "ball" in p or "head" in p:
        actions.append({"action": "add_sphere", "radius": 0.5, "position": [0, y, 0],
                        "color": color, "label": _uid("ball")})
    if "cylinder" in p or "pillar" in p or "trunk" in p:
        actions.append({"action": "add_cylinder", "radiusTop": 0.3, "radiusBottom": 0.4,
                        "height": 1.5, "position": [1.2, 0.75, 0], "color": color,
                        "label": _uid("cyl")})
    if "cone" in p:
        actions.append({"action": "add_cone", "radius": 0.5, "height": 1.2,
                        "position": [-1.2, 0.6, 0], "color": color, "label": _uid("cone")})
    if "torus" in p or "ring" in p or "donut" in p:
        actions.append({"action": "add_torus", "radius": 0.8, "tube": 0.2,
                        "position": [0, y, 0], "color": color, "label": _uid("ring")})
    if "glow" in p or "fresnel" in p or "shader" in p:
        actions.append({"action": "add_shader", "preset": "fresnel", "color": "0x00ffcc",
                        "intensity": 2.0, "label": _uid("glow")})
    if "sun" in p or "light" in p or "lamp" in p or "warm" in p:
        actions.append({"action": "add_light", "type": "directional", "intensity": 2.5,
                        "position": [10, 15, 8], "color": "0xFFEECC", "label": _uid("sun")})
    if "particle" in p or "snow" in p or "rain" in p or "spark" in p:
        actions.append({"action": "add_particles", "count": 500, "size": 0.05,
                        "color": color, "label": _uid("parts")})
    if not actions:  # unknown prompt -> minimal valid scene, never empty
        actions = [{"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0],
                    "color": color, "label": _uid("ball")}]
    if "house" in p or "building" in p or "cabin" in p:
        actions = [
            {"action": "add_box", "width": 8, "height": 0.3, "depth": 6,
             "position": [0, 0, 0], "color": "0x777777", "label": _uid("base")},
            {"action": "add_box", "width": 8, "height": 4, "depth": 6,
             "position": [0, 2.15, 0],
             "color": "0xF5DEB3", "label": _uid("walls")},
            {"action": "add_extrude", "width": 8.8, "height": 2.2, "depth": 6.4,
             "position": [0, 4.15, -3.2], "color": "0x8B0000", "label": _uid("roof")},
            {"action": "add_box", "width": 1.2, "height": 2, "depth": 0.12,
             "position": [0, 1.15, 3.1], "color": "0x8B4513", "label": _uid("door")},
            {"action": "add_box", "width": 1.1, "height": 1.1, "depth": 0.06,
             "position": [2.5, 2.2, 3.1], "color": "0xADD8E6", "label": _uid("winR")},
            {"action": "add_box", "width": 1.1, "height": 1.1, "depth": 0.06,
             "position": [-2.5, 2.2, 3.1], "color": "0xADD8E6", "label": _uid("winL")},
            {"action": "add_light", "type": "directional", "intensity": 2.5,
             "position": [10, 15, 8], "color": "0xFFEECC", "label": _uid("sun")},
        ]
        # walls stack on the foundation regardless of absolute guess
        actions[1]["on_top_of"] = actions[0]["label"]
        return actions
    if "tree" in p:
        trunk = {"action": "add_cylinder", "radiusTop": 0.12, "radiusBottom": 0.2,
                 "height": 2.0, "position": [0, 1.0, 0], "color": "0x6B3A2A",
                 "label": _uid("trunk")}
        actions.append(trunk)
        actions.append({"action": "add_sphere", "radius": 1.2,
                        "position": [0, 3.2, 0], "on_top_of": trunk["label"],
                        "color": "0x2E7D32", "label": _uid("leaves")})
        return actions
    return actions


def think(prompt: str, model: str = THINKER_MODEL) -> tuple:
    """Return (actions, status). In strict mode a model failure raises
    ModelUnavailableError instead of silently using the rule planner."""
    try:
        raw = ollama_generate(model, THINKER_SYSTEM, prompt,
                              options={"temperature": 0.0, "num_ctx": 4096,
                                       "num_predict": 3000})
        actions = _extract_json_array(raw)
        if not isinstance(actions, list) or not actions:
            raise ValueError("empty action list")
        # keep only known actions; thinker must not invent opcodes
        actions = [a for a in actions if isinstance(a, dict)
                   and a.get("action") in ALLOWED_ACTIONS][:16]
        if not actions:
            raise ValueError("no allowed actions")
        return actions, f"MODEL {model}"
    except Exception as e:
        if STRICT_MODELS:
            raise ModelUnavailableError(
                f"thinker model {model!r} failed: {e}") from e
        return rule_based_plan(prompt), f"FALLBACK_RULES ({e})"


THINK_NEXT_SYSTEM = THINKER_SYSTEM + """
INCREMENTAL MODE: output exactly ONE JSON object (not an array) — the single
next atomic action given the CURRENT SCENE below. Use "on_top_of": "<label>"
to stack on existing geometry instead of guessing heights. "on_top_of" is a
FIELD inside an action object, never an action name by itself. Use a fresh unique
label every step. If the scene already satisfies the goal, output
{"action":"done","reason":"..."} and nothing else.

The ONLY valid action names are: add_sphere, add_box, add_cylinder, add_cone,
add_torus, add_plane, add_extrude, add_terrain, add_light, add_shader,
add_particles. Never invent names like add_tree, add_roof or add_wall — map the
idea onto one of the actions above (tree = add_cylinder trunk + add_sphere
leaves, roof = add_extrude, wall = add_box)."""


def _extract_json_object(text: str) -> dict:
    """Tolerate sloppy-but-parseable model output: a lone object, or a whole
    JSON array (the thinker's default habit) from which we take the first real
    geometry/lighting action. Skips hallucinated on_top_of pseudo-actions."""
    text = _strip_think(text)
    text = re.sub(r"```(?:json)?", "", text).strip()
    for lo, ch in enumerate(text):
        if ch not in "{[":
            continue
        for hi in range(len(text), lo, -1):
            try:
                val = json.loads(_fix_trailing_commas(text[lo:hi]))
            except Exception:
                continue
            if isinstance(val, dict):
                return val
            if isinstance(val, list) and val:
                for a in val:
                    if isinstance(a, dict) and a.get("action") not in (None, "on_top_of"):
                        return a
                if isinstance(val[-1], dict):
                    return val[-1]
    raise ValueError("no valid JSON object in thinker output")


def think_plan(goal: str, model: str = THINKER_MODEL, _retried=False) -> tuple:
    """Decide the incremental step budget with the real model.

    A bare "how many steps?" question makes DeepSeek-R1 loop in its reasoning
    until the token budget runs out and return nothing. Instead we ask it for
    the actual atomic plan (the task it is good at) and use the number of
    actions it produced as the budget. Real model, no rule fallback; a failed
    or empty plan raises in strict mode."""
    try:
        actions, _status = think(goal, model)
        n = len(actions)
        if not (1 <= n <= 24):
            raise ValueError(f"planner returned {n} actions")
        return max(2, min(16, n)), f"MODEL {model}"
    except Exception as e:
        if not _retried:
            return think_plan(goal, model, _retried=True)
        if STRICT_MODELS:
            raise ModelUnavailableError(
                f"thinker model {model!r} could not budget the plan: {e}") from e
        n = len(rule_based_plan(goal))
        return max(2, min(16, n)), f"FALLBACK_RULES ({e})"


def think_next(goal: str, state_summary: str, step: int,
               feedback: str = "", model: str = THINKER_MODEL,
               _retried=False) -> tuple:
    """Propose ONE next action conditioned on live scene state.
    Returns (action, status) where action may be {"action":"done"}.
    A network outage of the real model is fatal in strict mode; an unparseable
    reply from a reachable model is returned as a {"action": None, "_error": ...}
    object so the loop can feed it back and try again (never a silent fallback)."""
    prompt = (f"GOAL: {goal}\nSTEP: {step}\n{state_summary}\n"
              + (f"LAST ERROR (you MUST fix this, do not repeat it): {feedback}\n"
                 if feedback else "")
              + "Do NOT reason. Do NOT discuss. Output ONLY the JSON object:\n"
                "Choose the single NEXT action. It MUST:\n"
                "- be one action name from the allowed list, with ALL of that "
                "action's required fields (never omit width/height/depth/radius);\n"
                "- use a brand-new snake_case label that is NOT already in the "
                "current scene above;\n"
                "- place new geometry beside or on top of existing parts, never "
                "overlapping them (or add a light/shader which has no footprint).\n"
                "Next single action as ONE JSON object:")
    try:
        raw = ollama_generate(model, THINK_NEXT_SYSTEM, prompt,
                              options={"temperature": 0.0, "num_ctx": 4096,
                                       "num_predict": 1500})
    except Exception as e:
        # unreachable / timed-out model is a real outage, not a bad answer
        if not _retried:
            return think_next(goal, state_summary, step, feedback, model,
                              _retried=True)
        if STRICT_MODELS:
            raise ModelUnavailableError(
                f"thinker model {model!r} unreachable at step {step}: {e}") from e
        return rule_next_fallback(goal, state_summary), f"FALLBACK_RULES ({e})"
    try:
        act = _extract_json_object(raw)
        if not isinstance(act, dict):
            raise ValueError("model did not return a JSON object")
    except Exception as e:
        # reachable model, unusable reply: hand it to the loop as a rejection
        if not _retried:
            return think_next(goal, state_summary, step, feedback, model,
                              _retried=True)
        return ({"action": None,
                 "_error": f"unparseable model output ({e})",
                 "_raw": (raw or "")[:200]},
                f"MODEL {model} (unparsed)")
    # Hand any object back to the loop: an unknown action, a missing field or a
    # bad placement is validated there and fed back to the model for a retry.
    return act, f"MODEL {model}"


_RULE_CACHE = {}


def rule_next_fallback(goal: str, state_summary: str) -> dict:
    """Offline single-step fallback: first rule-planned action whose label is
    new to the scene, else done. The plan is cached per goal so labels and
    on_top_of refs stay consistent across steps. Keeps the loop productive
    without a model."""
    placed = set(re.findall(r"\b\w+_\d+\b", state_summary))
    plan = _RULE_CACHE.get(goal)
    if plan is None:
        plan = rule_based_plan(goal)
        _RULE_CACHE[goal] = plan
    for a in plan:
        if a.get("label") not in placed:
            a = dict(a)
            ref = a.get("on_top_of")
            if ref and ref not in placed:
                # ref points at a same-batch label never committed; fall back
                # to the absolute coordinates (already consistent in rule plans)
                del a["on_top_of"]
            return a
    return {"action": "done", "reason": "rule plan exhausted"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompt = " ".join(args) if args else "a red sphere on a blue box with warm sun light"
    model = THINKER_MODEL
    for i, a in enumerate(sys.argv):
        if a == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
    actions, status = think(prompt, model)
    print(f"STATUS: {status}")
    print(json.dumps(actions, indent=2))
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
