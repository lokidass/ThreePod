# morphix_pipeline.py
# Full Morphix pipeline: prompt -> Thinker (DeepSeek) -> actions ->
# Builder (Qwen) -> Guard -> scene.js + scene.html + report.json
#
# Usage:
#   python morphix_pipeline.py "a red sphere on a blue box with warm sun light"
#   python morphix_pipeline.py "a gold ring with glow" --out ./out_demo
#   python morphix_pipeline.py "..." --incremental --out ./out_step
#
# Deployment policy (MORPHIX_STRICT_MODELS=true, the default):
#   * Real models only. The thinker MUST be a DeepSeek model and the builder
#     MUST be the Qwen model (or the local LoRA). The `template` builder and
#     the rule-based planner are rejected - a missing or unusable model fails
#     the run loudly instead of silently producing guessed geometry.
#   * Every step is validated AND reported: action schema, live-geometry state
#     checks (bounds/collision/stacking), the anti-hallucination guard, and a
#     real `node --check` syntax check. Failures are surfaced, never patched
#     over quietly.
#   * Each step can be pushed to an `emit(event)` callback so a web client can
#     render the run live (see server.py /api/jobs stream).

import argparse
import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphix_config import (BUILDER_MAX_RETRIES, BUILDER_MODEL, STRICT_MODELS,
                            THINKER_MODEL, Cancelled, ModelUnavailableError,
                            PipelineError, ollama_generate)
from morphix_thinker import (think, think_next, think_plan, rule_based_plan,
                             goal_parts, goal_coverage)
from morphix_specs import (check_plan, expected_of, repair_car_plan,
                           spec_for_prompt, verify_component, verify_solids)
from morphix_ir import (actions_to_ir, summary_of, ir_for_prompt,
                        validate_ir, ir_to_actions, _add_shape_checks,
                        template_key)
from morphix_path import (INSPECT_TEMPLATE, MATERIAL_COLORS, PATH_ACTIONS,
                          PathState, compile_plan, layer_y, merge_layers,
                          optimize_toolpath, render_deposit,
                          validate_geometry)
from morphix_guard import enforce, render_fallback, validate_js
from morphix_validator import (validate_scene, validate_js_syntax,
                               validate_actions, validate_against_state)
from morphix_state import SceneState

BUILDER_SYSTEM = (
    "You are Morphix Builder — a Three.js 3D printer agent.\n"
    "You receive ONE action JSON per tick describing a single atomic operation.\n"
    "Output ONLY raw JavaScript. No markdown, no backticks, no explanation, no comments.\n"
    "\n"
    "Rules:\n"
    "- Each output patches the EXISTING scene — never rebuild from scratch.\n"
    "- Use unique variable names based on the label field to avoid collisions.\n"
    "- Always use MeshStandardMaterial or MeshPhysicalMaterial for realism.\n"
    "- Set castShadow = true and receiveShadow = true on every mesh.\n"
    "- Code must execute inside a Three.js r152 scene with renderer, camera, scene already defined.\n"
    "- morphix_mixers, morphix_shaders, morphix_physics are global arrays for tracking.\n"
    "- Never output anything except valid executable JavaScript."
)


def _clean_js(text: str) -> str:
    """Strip chat markers/fences and trailing chatter from a model response."""
    for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", " thinking", " response"):
        if marker in text:
            text = text.split(marker)[0]
    text = re.sub(r"```(?:js|javascript)?", "", text).strip()
    ends = [m.end() for m in re.finditer(
        r"(?:scene\.add\([^;]*\);|morphix_\w+\.push\([^;]*\);)", text)]
    if ends:
        text = text[:ends[-1]]
    return text.strip()


def _prompt_for(action: dict, state=None) -> str:
    """Action JSON plus live scene context so each step depends on geometry."""
    base = json.dumps(action)
    if state is None or not state.objects:
        return base
    used = sorted(state.labels())
    return (base + f"\nCurrent scene: {state.summary()}\n"
            f"Used labels (do not reuse): {used}. "
            f"Keep new geometry inside +/-25, y in [0,30], no overlaps.")


def _ollama_build(model: str, action: dict, timeout: int = None,
                  state=None, feedback: str = "") -> str:
    """Ask the Qwen builder to emit JS for one action. Real model only."""
    prompt = _prompt_for(action, state)
    if feedback:
        prompt += "\nPrevious output was REJECTED by the validator with:\n" + feedback
    return ollama_generate(model, BUILDER_SYSTEM, prompt, timeout=timeout)


class LoraBuilder:
    """Client of the persistent LoRA worker subprocess (morphix_lora_worker).

    The user's fine-tuned 7B lives in the CHILD process, never in the server:
    a bitsandbytes native crash kills only the worker, which is detected here
    and reported as a model failure (the server keeps serving). One worker
    per server process, shared by all jobs; GPU requests are serialized."""

    def __init__(self):
        self._proc = None
        self._next_id = 0
        self._rqueue = None
        self._spawns = 0
        self._log_path = os.path.join(
            __import__("tempfile").gettempdir(), "morphix_lora_worker.log")

    def _spawn(self):
        import queue
        import subprocess
        import threading as _th
        self._spawns += 1
        _unload_ollama_models()
        try:
            logf = open(self._log_path, "w", encoding="utf-8",
                        errors="replace")
        except OSError:
            logf = None
        print("Spawning LoRA worker (fine-tuned Qwen) ...")
        self._proc = subprocess.Popen(
            [sys.executable, "-u",
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "morphix_lora_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=logf, text=True, bufsize=1)
        self._rqueue = queue.Queue()
        proc, q = self._proc, self._rqueue

        def _drain():
            try:
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)  # EOF marker: worker died

        _th.Thread(target=_drain, daemon=True).start()
        # handshake: skip any startup chatter, wait for the READY marker
        # (overall timeout so a wedged load fails loudly instead of hanging)
        import time as _time
        deadline = _time.time() + 900
        ready = False
        first = None
        while True:
            remaining = deadline - _time.time()
            if remaining <= 0:
                break
            first = self._recv(timeout=remaining)
            if first is None:
                break
            if first.strip() == "MORPHIX_LORA_READY":
                ready = True
                break
            # startup log line on stdout: ignore here (full text is in log file)
        if not ready:
            tail = ""
            try:
                with open(self._log_path, encoding="utf-8",
                          errors="replace") as f:
                    tail = f.read()[-800:]
            except OSError:
                pass
            raise ModelUnavailableError(
                "LoRA worker failed to start "
                f"(exit={self._proc.poll()}): {(first or '')[:200]}"
                f"{tail[-400:]}")

    def _recv(self, timeout):
        try:
            return self._rqueue.get(timeout=timeout)
        except Exception:
            return None

    def _ensure_alive(self):
        if self._proc is None:
            self._spawn()
        elif self._proc.poll() is not None:
            # worker died mid-run (native crash): respawn once-in-a-while so
            # a single segfault degrades to a slow retry, not a dead run.
            # Cap respawns: a permanently broken worker must fail loudly.
            if self._spawns >= 3:
                raise ModelUnavailableError(
                    f"LoRA worker keeps dying (exit={self._proc.poll()}); "
                    f"see {self._log_path}")
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
            print("LoRA worker died; respawning ...")
            self._spawn()
        return self._proc

    def build(self, action: dict, state=None, feedback: str = "") -> str:
        prompt = (f"<|im_start|>system\n{BUILDER_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{_prompt_for(action, state)}<|im_end|>\n")
        if feedback:
            prompt += f"\nPrevious output was REJECTED:\n{feedback}\n"
        prompt += "<|im_start|>assistant\n"
        budget = 384 if action.get("action") in (
            "add_shader", "add_post", "add_animation", "add_particles") else 256
        # serialized: one GPU, one request at a time across job threads
        with _LORA_GEN_LOCK:
            proc = self._ensure_alive()
            self._next_id += 1
            rid = self._next_id
            try:
                proc.stdin.write(json.dumps(
                    {"id": rid, "prompt": prompt,
                     "max_new_tokens": budget}) + "\n")
                proc.stdin.flush()
            except Exception as e:
                raise ModelUnavailableError(
                    f"LoRA worker unreachable: {e}") from e
            raw = self._recv(timeout=600)
            if raw is None:
                code = proc.poll()
                if code is not None:
                    raise ModelUnavailableError(
                        f"LoRA worker crashed during generate (exit={code}); "
                        f"see {self._log_path}")
                raise ModelUnavailableError(
                    "LoRA worker timed out during generate")
            try:
                resp = json.loads(raw)
            except Exception:
                raise ModelUnavailableError(
                    f"LoRA worker returned garbage: {raw[:200]}")
            if resp.get("id") != rid or "code" not in resp:
                raise ModelUnavailableError(
                    f"LoRA worker error: {resp.get('error', raw[:200])}")
            self._spawns = 0  # served: reset the respawn budget
            return resp["code"]


_LORA_SINGLETON = None
_LORA_SINGLETON_LOCK = threading.Lock()
_LORA_GEN_LOCK = threading.Lock()


def _unload_ollama_models():
    """Best-effort `ollama stop` for the known models. The local 7B LoRA and
    Ollama-resident models cannot share a 6GB card; the LoRA must load into
    VRAM that Ollama would otherwise hold. Failures are ignored (Ollama may
    simply have nothing loaded)."""
    import shutil
    import subprocess
    if shutil.which("ollama") is None:
        return
    for m in ("deepseek-r1:1.5b", "morphix-thinker",
              "qwen2.5-coder:7b", "moondream"):
        try:
            subprocess.run(["ollama", "stop", m], capture_output=True,
                           timeout=60)
        except Exception:
            pass


def get_lora_builder() -> LoraBuilder:
    """Process-wide singleton: one worker subprocess no matter how many jobs
    ask for builder='lora'. The worker spawns (and loads the 7B) lazily on
    its first build request."""
    global _LORA_SINGLETON
    with _LORA_SINGLETON_LOCK:
        if _LORA_SINGLETON is None:
            _LORA_SINGLETON = LoraBuilder()
        return _LORA_SINGLETON
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Morphix Scene</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#111}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.152.2/build/three.module.js","three/addons/":"https://unpkg.com/three@0.152.2/examples/jsm/"}}</script>
</head><body>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111622);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 500);
camera.position.set(6, 5, 9);
new OrbitControls(camera, renderer.domElement);
const morphix_mixers = [], morphix_shaders = [], morphix_physics = [];
const grid = new THREE.GridHelper(20, 20, 0x334455, 0x223344);
scene.add(grid);
const amb = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(amb);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(6, 10, 7);
sun.castShadow = true;
scene.add(sun);
// ---- generated blocks ----
__BLOCKS__
// ---- animate ----
const clock = new THREE.Clock();
function animate(){
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  for (const m of morphix_mixers) m.update(dt);
  for (const s of morphix_shaders) if (s.uniforms && s.uniforms.time) s.uniforms.time.value = t;
  renderer.render(scene, camera);
}
animate();
</script></body></html>
"""


def _think_options(builder: str, num_predict: int) -> dict:
    """Ollama options for thinker calls. On builder='lora' the fine-tuned 7B
    owns the GPU, so the thinker is pinned to CPU (num_gpu=0) and the two fit
    side by side; otherwise defaults (full GPU)."""
    opts = {"temperature": 0.0, "num_ctx": 4096, "num_predict": num_predict}
    if builder == "lora":
        opts["num_gpu"] = 0
    return opts


def _reject_builder(name: str) -> PipelineError:
    return PipelineError(
        f"'{name}' is not a real model. Deployment policy (MORPHIX_STRICT_MODELS="
        f"true) requires the actual DeepSeek + Qwen models; template/rule "
        f"geometry is disabled. Unset MORPHIX_STRICT_MODELS to opt out.")


def _write_run(out_dir: str, actions, finals, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    scene_js = "\n\n".join(finals) + "\n" if finals else "// empty scene\n"
    with open(os.path.join(out_dir, "scene.js"), "w", encoding="utf-8") as f:
        f.write(scene_js)
    with open(os.path.join(out_dir, "scene.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__BLOCKS__", scene_js))
    with open(os.path.join(out_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)
    if extra:
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2)
    return scene_js


def _write_partial_report(out_dir: str, *, reason: str, committed: int,
                          total: int, mode: str, guided: bool = False,
                          extra: dict = None) -> dict:
    """Persist an explicit INCOMPLETE report when a run stops early
    (deadline, reject budget, user cancel handled by caller). Committed
    geometry stays on disk via the last checkpoint; this records exactly
    what remains. Never marks incomplete work complete."""
    rep = {"ok": False, "complete": False, "success": False,
           "status": "incomplete", "reason": reason,
           "committed_actions": committed, "total_actions": total,
           "remaining_actions": max(0, total - committed),
           "mode": mode, "guided": guided}
    if extra:
        rep.update(extra)
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep


def _sanitize_plan(actions: list, think_status: str, emit) -> list:
    """Per-action schema check + label de-dup (renames preserve intent).
    In strict mode a wholly malformed plan is an error, not a rules fallback."""
    from morphix_validator import validate_actions as _va
    from morphix_thinker import ALLOWED_ACTIONS as _ALLOWED
    _valid = [a for a in actions
              if isinstance(a, dict) and a.get("action") in _ALLOWED
              and not _va([a])]
    _seen = {}
    for _a in _valid:
        _lab = _a.get("label", "m")
        if _lab in _seen:
            _seen[_lab] += 1
            _a["label"] = f"{_lab}_{_seen[_lab]:02d}"
        else:
            _seen[_lab] = 0
    if not _valid:
        raise PipelineError(
            "thinker returned no valid actions. Every returned action failed "
            f"schema validation: {_va(actions)}")
    if len(_valid) != len(actions):
        emit({"type": "warn", "phase": "sanitize",
              "message": f"{len(actions) - len(_valid)} actions dropped by validator"})
        print(f"THINK SANITIZE: {len(actions)} -> {len(_valid)} actions")
    print(f"THINK: {think_status} -> {len(_valid)} valid actions")
    return _valid


PATH_BUILDER_SYSTEM = (
    "You are Morphix Path Builder — a 3D-printer toolhead agent.\n"
    "You receive the construction plan, the active component and layer, the\n"
    "authoritative toolhead position, the last committed action, a compact\n"
    "committed-geometry summary, the remaining budget, and the allowed action\n"
    "schema. Propose exactly ONE next movement or deposition action as a\n"
    "single JSON object. No arrays, no markdown, no explanation, no comments.\n"
    "Allowed actions (lowercase only): move_to, deposit_segment,\n"
    "deposit_point, complete_path, complete_layer, complete_object.\n"
    "Action shapes:\n"
    "- {\"action\":\"deposit_segment\",\"from\":[x,y,z],\"to\":[x,y,z],\n"
    "   \"bead_width\":0.2,\"bead_height\":0.2,"
    "\"material\":\"body\"|\"accent\"|\"blue\"|\"support\"|\"glass\"|\"tire\"|"
    "\"wood\",\n"
    "   \"layer\":0,\"label\":\"...\"}\n"
    "- {\"action\":\"move_to\",\"position\":[x,y,z],\"layer\":0,\"label\":\"...\"}\n"
    "- {\"action\":\"complete_path\"|\"complete_layer\",\"layer\":0,\"label\":\"...\"}\n"
    "- {\"action\":\"complete_object\",\"label\":\"...\"}\n"
    "Rules:\n"
    "- deposit_segment.from MUST equal the current toolhead position exactly.\n"
    "- Deposit only on the active layer plane (y equals layer_y exactly).\n"
    "- layer MUST equal the active layer. Never skip layers.\n"
    "- Positioning travel is handled by the engine: propose deposit_*\n"
    "  and complete_* only. If you diverge from the plan, use move_to\n"
    "  yourself to reach your strikepoint.\n"
    "- Close paths and layers with complete_path / complete_layer when the\n"
    "  state shows the work is done.\n"
    "- Emit complete_object only when every planned component is completed.\n"
    "- Labels must be unique snake_case."
)

PRINT_PLAN_SUFFIX_BASE = (
    "\nPRINT MODE: reply with 1-6 grounded footprints ONLY, as a JSON array "
    "of add_box / add_cylinder / add_sphere actions (nothing else is "
    "printable — never add_extrude, add_particles or invented opcodes; roofs "
    "print as thin flat boxes). Each footprint: position is its CENTER; rest "
    "it on the ground with y = half its height (a 0.4-tall box sits at y=0.2) "
    "or stack it onto an earlier part; keep every part SMALL (widths/depths "
    "at most 8, heights at most 3, |x|,|z| <= 10); use unique "
    "snake_case labels. Give each part a material from body/accent/blue/"
    "support/glass/tire/wood (blue for a requested blue body, glass for "
    "windows, tire for wheels, wood for wooden furniture) and true "
    "proportions "
)

PRINT_PLAN_EXAMPLES = {
    "car": (
        "(car bodies long and low with four small corner wheels, cabins "
        "above bodies, windows/doors thin on side faces). Example car parts "
        "(copy the shapes, adapt freely): "
        '{"action":"add_box","width":3.0,"height":0.6,"depth":1.6,'
        '"position":[0,0.3,0],"color":"0xFF0000","label":"car_body",'
        '"material":"body"} '
        '{"action":"add_box","width":1.6,"height":0.5,"depth":1.4,'
        '"position":[0,0.85,0],"color":"0x44CCFF","label":"cabin_top",'
        '"material":"accent"} '
        '{"action":"add_cylinder","radiusTop":0.3,"radiusBottom":0.3,'
        '"height":0.4,"position":[-1.0,0.2,0.7],"color":"0x222222",'
        '"label":"wheel_0","material":"tire"}'
    ),
    "chair": (
        "(chairs have one horizontal seat, a backrest rising from the seat "
        "rear, and four grounded legs at the seat corners — never wheels, "
        "cabins or windshields). Example chair parts (copy the shapes, "
        "adapt freely): "
        '{"action":"add_box","width":1.1,"height":0.12,"depth":1.1,'
        '"position":[0,0.51,0],"color":"0x9C6B3F","label":"seat_a1",'
        '"material":"wood"} '
        '{"action":"add_box","width":0.12,"height":0.9,"depth":1.0,'
        '"position":[-0.49,1.02,0],"color":"0x9C6B3F","label":"backrest_a1",'
        '"material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[0.45,0.225,0.45],"color":"0x9C6B3F",'
        '"label":"leg_0","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[0.45,0.225,-0.45],"color":"0x9C6B3F",'
        '"label":"leg_1","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[-0.45,0.225,0.45],"color":"0x9C6B3F",'
        '"label":"leg_2","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[-0.45,0.225,-0.45],"color":"0x9C6B3F",'
        '"label":"leg_3","material":"wood"}'
    ),
    "stool": (
        "(stools have one horizontal seat on four grounded legs and NO "
        "backrest — never wheels or cabins). Example stool parts (copy the "
        "shapes, adapt freely): "
        '{"action":"add_box","width":1.0,"height":0.12,"depth":1.0,'
        '"position":[0,0.51,0],"color":"0x9C6B3F","label":"seat_a1",'
        '"material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[0.4,0.225,0.4],"color":"0x9C6B3F",'
        '"label":"leg_0","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[0.4,0.225,-0.4],"color":"0x9C6B3F",'
        '"label":"leg_1","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[-0.4,0.225,0.4],"color":"0x9C6B3F",'
        '"label":"leg_2","material":"wood"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.07,'
        '"height":0.45,"position":[-0.4,0.225,-0.4],"color":"0x9C6B3F",'
        '"label":"leg_3","material":"wood"}'
    ),
    "cat": (
        "(cats have one body, a tapered neck grafting a round head above "
        "the body front, two tapered ears on the head, four grounded "
        "tapered legs, and one tapered tail at the rear — never wheels, "
        "cabins or windshields). Example cat parts (copy the shapes, "
        "adapt freely): "
        '{"action":"add_box","width":1.3,"height":0.7,"depth":0.7,'
        '"position":[0,0.75,0],"color":"0xFFAA33","label":"body_a1",'
        '"material":"body"} '
        '{"action":"add_cylinder","radiusTop":0.2,"radiusBottom":0.26,'
        '"height":0.3,"position":[0.45,1.25,0],"color":"0xFFAA33",'
        '"label":"neck_a1","material":"body"} '
        '{"action":"add_sphere","radius":0.36,'
        '"position":[0.45,1.76,0],"color":"0xFFAA33","label":"head_a1",'
        '"material":"body"} '
        '{"action":"add_cylinder","radiusTop":0.03,"radiusBottom":0.12,'
        '"height":0.28,"position":[0.45,2.26,0.16],"color":"0xFFAA33",'
        '"label":"ear_left","material":"body"} '
        '{"action":"add_cylinder","radiusTop":0.03,"radiusBottom":0.12,'
        '"height":0.28,"position":[0.45,2.26,-0.16],"color":"0xFFAA33",'
        '"label":"ear_right","material":"body"} '
        '{"action":"add_cylinder","radiusTop":0.07,"radiusBottom":0.1,'
        '"height":0.4,"position":[0.5,0.2,0.25],"color":"0xFFAA33",'
        '"label":"leg_0","material":"body"}'
    ),
}

PRINT_PLAN_SUFFIX_NEUTRAL = (
    "Example part (copy the shape, adapt freely): "
    '{"action":"add_box","width":1.0,"height":0.5,"depth":1.0,'
    '"position":[0,0.25,0],"color":"0xFFAA33","label":"part_a1",'
    '"material":"body"}'
)

# Backwards-compatible alias: the full car-flavoured prompt used before
# category-conditioned examples existed.
PRINT_PLAN_SUFFIX = (PRINT_PLAN_SUFFIX_BASE + PRINT_PLAN_EXAMPLES["car"])


def print_suffix_for(spec_name: str) -> str:
    """Few-shot example block for the detected category. A shared car-only
    example made the small thinker copy car parts into unrelated prompts
    (observed: car_body/cabin_top/wheel_0 labels in a chair plan), so each
    supported category gets its own example and unknown prompts get a
    neutral single box."""
    return PRINT_PLAN_SUFFIX_BASE + PRINT_PLAN_EXAMPLES.get(
        spec_name or "", PRINT_PLAN_SUFFIX_NEUTRAL)


def _parse_single_action(text: str) -> dict:
    """Exactly one JSON object — arrays or trailing actions are rejected so a
    builder response can never smuggle future steps into the current one."""
    t = text.strip()
    for fence in ("```json", "```"):
        t = t.replace(fence, "")
    t = t.strip()
    try:
        val, end = json.JSONDecoder().raw_decode(t)
    except Exception as e:
        raise ValueError(f"no JSON object in builder output: {e}")
    if not isinstance(val, dict):
        raise ValueError("builder must return one JSON object, not an array")
    if t[end:].strip().rstrip("}").strip():
        # trailing content after the object -> more than one action
        raise ValueError("builder must return exactly one action")
    return val


def run_pipeline(prompt: str, builder: str = "ollama",
                 builder_model: str = BUILDER_MODEL,
                 thinker_model: str = THINKER_MODEL,
                 out_dir: str = "./out", emit=None, should_stop=None) -> dict:
    """Whole-scene pipeline (think once, build all blocks). Fails loudly in
    strict mode when a model is unavailable or a block cannot validate.
    `should_stop` is an optional zero-arg callable; when it returns True the
    run aborts with Cancelled and no further blocks are committed."""
    os.makedirs(out_dir, exist_ok=True)
    emit = emit or (lambda e: None)
    if STRICT_MODELS and builder == "template":
        raise _reject_builder("template")
    emit({"type": "start", "prompt": prompt, "builder": builder,
          "builder_model": builder_model, "thinker_model": thinker_model,
          "incremental": False, "strict": STRICT_MODELS})
    print(f"PROMPT: {prompt}")

    # 1) THINK with the real model
    actions, think_status = think(prompt, thinker_model,
                                  options=_think_options(builder, 3000))
    emit({"type": "think", "phase": "plan", "status": think_status,
          "actions": actions, "n_actions": len(actions)})

    # sanitize: per-action schema check, then de-duplicate labels (renames
    # preserve the thinker's intent and keep var names unique). In strict mode
    # a malformed action list from the model is an error, not a rules fallback.
    actions = _sanitize_plan(actions, think_status, emit)
    _cov, _miss = goal_coverage(goal_parts(prompt), actions)
    if _miss:
        emit({"type": "warn", "phase": "goal",
              "message": f"plan covers {len(_cov)}/{len(_cov) + len(_miss)} "
                         f"goal parts; missing: {_miss}"})
        print(f"GOAL: missing parts in plan: {_miss}")

    # 2) BUILD through the real model, guarded + validated per block
    # (shared singleton: one 7B in VRAM no matter how many jobs run)
    lora = get_lora_builder() if builder == "lora" else None
    finals, statuses = [], []
    for i, act in enumerate(actions):
        if should_stop is not None and should_stop():
            raise Cancelled(f"cancelled by user after {i}/{len(actions)} blocks")
        final, status, errors = _build_block(
            act, builder, builder_model, lora, emit=emit)
        if not final:
            raise PipelineError(
                f"block {i + 1} ({act.get('action')} {act.get('label')}) "
                f"failed validation after retries: {'; '.join(errors)}")
        finals.append(final)
        statuses.append(status)
        emit({"type": "block", "index": i, "action": act.get("action"),
              "label": act.get("label"), "status": status,
              "errors": errors, "validated": not errors})

    # 3) + 4) ASSEMBLE + VALIDATE whole scene
    scene_js = _write_run(out_dir, actions, finals)
    report = validate_scene(actions, finals)
    report["think_status"] = think_status
    report["block_statuses"] = statuses
    report["syntax_scene_js"] = validate_js_syntax(scene_js)
    report["mode"] = "pipeline"
    report["ok"] = report["ok"] and not report["syntax_scene_js"]
    _write_run(out_dir, actions, finals, extra=report)

    n_ok = sum(1 for b in report["blocks"] if b["ok"])
    print(f"RESULT: {'OK' if report['ok'] else 'FAIL'} "
          f"({n_ok}/{len(finals)} blocks clean) -> {out_dir}/")
    emit({"type": "done", "run_id": os.path.basename(out_dir),
          "report": report, "scene_js": scene_js, "ok": report["ok"]})
    return report


def _plan_for_print(prompt: str, builder: str, thinker_model: str, spec,
                    spec_name: str, emit) -> dict:
    """Plan + compile a text prompt for path printing (extracted so that
    image reconstruction can reuse the identical commit engine while supplying
    its own already-canonical component plan). Returns a result dict."""
    planner_attempts = 0
    think_status = "NONE"
    last_err = None
    keep = {}
    compiled, order, comp_errors, comp_warns = {}, [], {}, {}
    fidelity_warns = []
    plan_repairs = []
    actions = []
    # Last attempt that actually reached the slicer, with EVERYTHING from
    # that same attempt (plan, status, compile outputs, repairs, warnings).
    # Planning, slicing and accounting must share one identity: pairing
    # attempt N's plan labels with attempt M's script silently turns every
    # renamed component into a zero-deposit "missing" part downstream.
    last_consistent = None
    for attempt in (1, 2):
        if attempt > 1:
            # Clean re-think: small thinkers derail on appended failure text,
            # so resample fresh. The failure stays visible via warn + logs.
            # Compiled components from attempt 1 are kept by label below.
            emit({"type": "warn", "phase": "plan",
                  "message": f"re-thinking after: {last_err}"})
        planner_attempts += 1
        try:
            actions, think_status = think(
                prompt + print_suffix_for(spec_name), thinker_model,
                options=_think_options(builder, 3000))
            emit({"type": "think", "phase": "plan", "status": think_status,
                  "actions": actions, "n_actions": len(actions)})
            actions = _sanitize_plan(actions, think_status, emit)
            if spec_name in ("car", "chair", "stool", "cat"):
                # normalize + repair BEFORE the gate, against the SELECTED
                # category only: missing parts are derived deterministically,
                # valid model geometry is preserved, every fix is logged.
                # Car repair never touches a chair plan and vice versa.
                from morphix_specs import repair_plan_for_spec
                try:
                    actions, repairs = repair_plan_for_spec(
                        spec_name, actions)
                except ValueError as e:
                    raise PipelineError(f"{spec_name} plan schema error: {e}")
                plan_repairs = repairs
                for r in repairs:
                    emit({"type": "repair", "component": r["component"],
                          "reason": r["reason"]})
                    print(f"REPAIR {r['component']}: {r['reason']}")
            spec_errs, spec_warns = check_plan(spec, actions)
            fidelity_warns = list(spec_warns)
            emit({"type": "spec", "spec": spec_name, "errors": spec_errs,
                  "warnings": spec_warns})
            for w in spec_warns:
                emit({"type": "warn", "phase": "spec", "message": w})
            if spec_errs:
                raise PipelineError(
                    f"plan fails {spec_name} fidelity gate: "
                    f"{'; '.join(spec_errs)}")
            compiled, order, comp_errors, keep, comp_warns = compile_plan(
                actions, keep)
            # Snapshot the consistent triple even when it carries errors:
            # the omit path below may only ever print THIS attempt's plan.
            last_consistent = (list(actions), think_status, compiled, order,
                               dict(comp_errors), dict(keep), dict(comp_warns),
                               list(plan_repairs), list(fidelity_warns))
            for lab in order:
                emit({"type": "compile", "component": lab,
                      "status": "ok" if lab in compiled else "failed",
                      "error": comp_errors.get(lab, ""),
                      "warning": "; ".join(comp_warns.get(lab, "")),
                      "layers": sorted(compiled.get(lab, {}).keys())})
            if comp_errors:
                failed = "; ".join(f"{k}: {v}"
                                   for k, v in comp_errors.items())
                raise PipelineError(
                    f"{len(comp_errors)} component(s) failed compilation: "
                    f"{failed}")
            break
        except (PipelineError, ModelUnavailableError) as e:
            last_err = e
            print(f"PLAN attempt {attempt} failed: {e}")
    else:
        # Attempts exhausted. Print the compilable subset loudly instead of
        # failing valid work over one unprintable component — but only from
        # the attempt that actually reached the slicer, and never when
        # nothing compiled at all. Falling back to a STALE script while
        # keeping the failed attempt's plan labels would print one plan
        # while accounting for another (every renamed part reads as a
        # zero-deposit "missing" component downstream).
        if last_consistent is None:
            raise last_err
        (actions, think_status, compiled, order, comp_errors, keep,
         comp_warns, plan_repairs, fidelity_warns) = last_consistent
        if not compiled:
            raise last_err
        emit({"type": "warn", "phase": "plan",
              "message": f"printing the compilable subset of the plan that "
                         f"reached the slicer after {planner_attempts} "
                         f"attempt(s); the other attempt failed before "
                         f"slicing ({last_err})"})
        omitted = [(lab, comp_errors.get(lab, "")) for lab in order
                   if lab not in compiled]
        emit({"type": "warn", "phase": "plan",
              "message": f"omitting {len(omitted)} unprintable component(s): "
                         + ", ".join(lab for lab, _ in omitted)})
    omitted_comps = [(lab, comp_errors.get(lab, "")) for lab in order
                     if lab not in compiled]
    return {"actions": actions, "think_status": think_status,
            "compiled": compiled, "order": order, "comp_errors": comp_errors,
            "keep": keep, "comp_warns": comp_warns,
            "omitted_comps": omitted_comps, "fidelity_warns": fidelity_warns,
            "plan_repairs": plan_repairs,
            "planner_attempts": planner_attempts}


def run_path_print(prompt: str, builder: str = "ollama",
                   builder_model: str = BUILDER_MODEL,
                   thinker_model: str = THINKER_MODEL,
                   out_dir: str = "./out_path", guided: bool = False,
                   vision_every: int = 0, segment_cap: int = 4000,
                   layer_cap: int = 32, deadline_s: int = 1800,
                   retries: int = 3, max_rejects: int = 60,
                   build_one=None, emit=None, should_stop=None,
                   resume: bool = False, plan_actions: list = None,
                   plan_status: str = None, spec_name: str = None,
                   plan_source: str = "text", stream_scene: bool = True) -> dict:
    """Point-by-point printer run. The thinker plans footprint primitives
    once; the engine slices them into validated per-layer segments.
    Deterministic (default): the engine commits the sliced script one action
    at a time, each validated against live toolhead state and rendered
    immediately. Guided: the builder proposes exactly one movement/deposition
    per call against live toolhead state; the deterministic script is offered
    as correction feedback on rejects (never silently substituted) and the
    validator remains the sole authority. `build_one(ctx)->str` injects a
    fake single-step builder (tests). Checkpoints persist per commit.
    Hard budgets (all reported, none silently extended): planner attempts
    (2 think calls max), builder attempts + repair retries, segment_cap,
    layer_cap, and an overall deadline_s wall clock."""
    import time as _time
    t0 = _time.monotonic()

    def expired():
        return (_time.monotonic() - t0) > deadline_s
    os.makedirs(out_dir, exist_ok=True)
    emit = emit or (lambda e: None)
    if STRICT_MODELS and builder == "template":
        raise _reject_builder("template")
    emit({"type": "start", "prompt": prompt, "builder": builder,
          "builder_model": builder_model, "thinker_model": thinker_model,
          "incremental": True, "mode": "path", "guided": guided,
          "strict": STRICT_MODELS})
    print(f"PROMPT: {prompt}  (path print, guided={guided})")

    planner_attempts = 0
    last_err = None
    keep = {}
    compiled, order, comp_errors, comp_warns = {}, [], {}, {}
    omitted_comps = []
    fidelity_warns = []
    plan_repairs = []
    spec = spec_for_prompt(spec_name) if spec_name else spec_for_prompt(prompt)
    spec_name = spec["name"] if spec else (spec_name or "generic")
    if (plan_actions is None and spec is not None
            and spec["name"] == "box_sphere"):
        # Supported canonical prompt: the deterministic compiler owns the
        # geometry end-to-end, so the model is NOT consulted. A hallucinated
        # sphere (or a box sliced into strips) can never appear.
        canon = ir_for_prompt(prompt)
        canon_errs = validate_ir(canon)
        if canon_errs:
            raise PipelineError("canonical box+sphere IR invalid: "
                                + "; ".join(canon_errs))
        plan_actions = ir_to_actions(canon)
        plan_status = "CANONICAL-TEMPLATE"
        plan_source = "object-template"
        emit({"type": "think", "phase": "plan", "status": plan_status,
              "actions": plan_actions, "n_actions": len(plan_actions),
              "note": "deterministic canonical prompt; model not consulted"})
    if plan_actions is not None:
        # Image reconstruction (or any external canonical plan): the component
        # geometry was already frozen in the object IR, so the model is NOT
        # consulted and the builder cannot alter a single dimension. The
        # deterministic slicer below owns every remaining decision.
        actions = [dict(a) for a in plan_actions]
        think_status = plan_status or "FROZEN-PLAN"
        emit({"type": "think", "phase": "plan", "status": think_status,
              "actions": actions, "n_actions": len(actions)})
        spec_errs, spec_warns = check_plan(spec, actions)
        fidelity_warns = list(spec_warns)
        emit({"type": "spec", "spec": spec_name, "errors": spec_errs,
              "warnings": spec_warns})
        for w in spec_warns:
            emit({"type": "warn", "phase": "spec", "message": w})
        if spec_errs:
            raise PipelineError(
                f"plan fails {spec_name} fidelity gate: {'; '.join(spec_errs)}")
        compiled, order, comp_errors, keep, comp_warns = compile_plan(
            actions, {})
        for lab in order:
            emit({"type": "compile", "component": lab,
                  "status": "ok" if lab in compiled else "failed",
                  "error": comp_errors.get(lab, ""),
                  "warning": "; ".join(comp_warns.get(lab, "")),
                  "layers": sorted(compiled.get(lab, {}).keys())})
        if comp_errors:
            failed = "; ".join(f"{k}: {v}"
                               for k, v in comp_errors.items())
            raise PipelineError(
                f"{len(comp_errors)} component(s) failed compilation: {failed}")
    else:
        planned = _plan_for_print(prompt, builder, thinker_model, spec,
                                  spec_name, emit)
        actions = planned["actions"]
        think_status = planned["think_status"]
        compiled = planned["compiled"]
        order = planned["order"]
        comp_errors = planned["comp_errors"]
        keep = planned["keep"]
        comp_warns = planned["comp_warns"]
        omitted_comps = planned["omitted_comps"]
        fidelity_warns = planned["fidelity_warns"]
        plan_repairs = planned["plan_repairs"]
        planner_attempts = planned["planner_attempts"]
    # Canonical IR for the frozen plan (text or image source): report the
    # component hierarchy the slicer is about to realise.
    plan_ir = actions_to_ir(actions, spec_name, source=plan_source)
    emit({"type": "ir", "ir": summary_of(plan_ir)})
    # Acceptance-level shape gate for supported categories: a plan whose
    # parts are individually valid but jointly wrong (flat slab "body",
    # ring wheels, floating cabin, car parts in a chair) must veto quality
    # acceptance here, at plan time, before a single bead commits.
    if spec_name in ("car", "chair"):
        shape_fails = []
        _add_shape_checks(
            spec_name, plan_ir.get("components", []),
            plan_ir.get("bbox") or [0, 0, 0],
            lambda n, ok, d: None if ok else shape_fails.append(
                f"{spec_name} shape: {n}: {d}"))
        for msg in shape_fails:
            fidelity_warns.append(msg)
            emit({"type": "warn", "phase": "shape", "message": msg})
    parts = goal_parts(prompt)
    covered, missing = goal_coverage(parts, actions)
    emit({"type": "goal", "parts": parts, "covered": covered,
          "missing": missing})

    script, total_layers = merge_layers(
        [compiled[lab] for lab in order if lab in compiled])
    n_raw = len(script)
    script, opt_stats = optimize_toolpath(script)
    print(f"TOOLPATH: {n_raw} -> {len(script)} actions "
          f"(-{n_raw - len(script)}: {opt_stats['merged_segments']} merged, "
          f"{opt_stats['removed_travel']} redundant travel, "
          f"{opt_stats['removed_zero']} zero-length; travel "
          f"{opt_stats['travel_dist_before']} -> "
          f"{opt_stats['travel_dist_after']})"
          + (" [optimizer fallback: kept raw script]"
             if opt_stats["fallback"] else ""))
    emit({"type": "optimize", "before": n_raw, "after": len(script),
          "stats": opt_stats})
    if not script:
        raise PipelineError("nothing sliceable: path mode needs grounded "
                            "add_box/add_cylinder/add_sphere footprints in "
                            "the plan")
    geo_errs = validate_geometry(script)
    if geo_errs:
        # The sliced toolpath itself is broken (disconnected, unclosed, or
        # degenerate). Refuse to commit a single gapped segment.
        raise PipelineError(
            f"compiled toolpath has geometry defects: "
            f"{'; '.join(geo_errs[:5])}"
            + (f" (+{len(geo_errs) - 5} more)" if len(geo_errs) > 5 else ""))
    print(f"GEOMETRY: toolpath validated ({len(script)} actions, "
          f"{total_layers} layers)")
    if total_layers > layer_cap:
        raise PipelineError(
            f"{total_layers} layers exceed layer_cap={layer_cap}; "
            f"ask for something smaller")
    if len(script) > segment_cap:
        raise PipelineError(
            f"{len(script)} path actions exceed segment_cap={segment_cap}; "
            f"ask for something smaller")
    print(f"PATH: {len(script)} actions across {total_layers} layers "
          f"[{think_status}]")
    emit({"type": "plan", "max_steps": len(script), "status": "sliced",
          "decided_by": "engine", "layers": total_layers})

    ckpt_path = os.path.join(out_dir, "path.json")
    st = PathState(total_layers=total_layers)
    path_actions, finals, statuses, blocks = [], [], [], []
    if resume and os.path.isfile(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        st = PathState.restore(ckpt["state"])
        for rec in ckpt.get("records", []):
            finals.append(render_deposit(rec))
            statuses.append("path/OK")
            blocks.append({"index": len(blocks), "action": rec["action"],
                           "label": rec.get("label", "?"), "guard_errors": [],
                           "syntax_errors": [], "ok": True})
        for saved in ckpt.get("script", []):
            path_actions.append(saved)
        print(f"RESUME: {len(path_actions)} actions replayed from {ckpt_path}")

    def matched_prefix(saved, ref):
        n = 0
        for r, e in zip(saved, ref):
            if r == e:
                n += 1
            else:
                break
        return n

    committed_records = []
    # rebuild record list from replayed path actions for divergence tracking.
    # ANY replay failure (including foreign-plan geometry the state machine
    # cannot even index) ends the replay: the prefix comparison below then
    # reports the divergence instead of crashing on someone else's layers.
    _replay = PathState(total_layers=total_layers)
    for _a in path_actions:
        try:
            _r = _replay.apply(_a)
            if _r.get("action") in ("deposit_segment", "deposit_point"):
                committed_records.append(_r)
        except Exception:
            break
    ptr = matched_prefix(path_actions, script)
    # A checkpoint from a DIFFERENT plan must never be spliced onto this
    # script — but a legitimately finished run replays its whole script plus
    # the trailing complete_object bookkeeping marker, which is NOT part of
    # the script and must not count as divergence (idempotent re-resume).
    trailing = path_actions[len(script):]
    trailing_ok = all(isinstance(a, dict)
                      and a.get("action") == "complete_object"
                      for a in trailing)
    diverged = (ptr < min(len(path_actions), len(script))) or not trailing_ok
    if diverged and not guided and path_actions:
        # A checkpoint from a DIFFERENT plan (fresh re-think, edited prompt,
        # stale out_dir) must never be spliced onto this script: positional
        # slicing below would commit one plan's tail as another plan's work
        # and every renamed component would read as zero-deposit "missing".
        # Refuse loudly — restart without --resume for a clean run.
        got = path_actions[ptr].get("label", "?") \
            if ptr < len(path_actions) else "END"
        want = script[ptr].get("label", "?") \
            if ptr < len(script) else "END"
        raise PipelineError(
            f"resume refused: checkpoint holds {len(path_actions)} actions "
            f"from a different plan (first divergence at step {ptr + 1}: "
            f"replayed {got!r} vs planned {want!r}); restart without "
            f"--resume")

    def checkpoint():
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"state": st.checkpoint(), "records": committed_records,
                       "script": path_actions, "thought": actions}, f)

    vision_log, broken_streak, vision_dead = [], 0, False

    def maybe_inspect(force=False):
        """Screenshot the working scene and ask the vision model whether
        the print is going well. Advisory only: verdicts stream to the UI
        and land in the report; two straight BROKENs raise a visible warn
        suggesting Stop. Never fails the run."""
        nonlocal broken_streak, vision_dead
        if vision_dead or not vision_every or not st.seq:
            return
        if not force and st.seq % vision_every:
            return
        if force and vision_log and vision_log[-1]["deposits"] == st.seq:
            return
        from morphix_vision import (screenshot_scene, judge_print,
                                    track_verdict)
        shot = os.path.join(out_dir, f"shot_{st.seq:04d}.png")
        try:
            screenshot_scene("file://" + os.path.abspath(
                os.path.join(out_dir, "inspect.html")), shot)
            res = judge_print(shot, prompt, st.seq, st.active_layer + 1)
        except Exception as e:
            vision_dead = True
            emit({"type": "warn", "phase": "vision",
                  "message": f"vision inspection disabled: {e}"})
            print(f"VISION disabled: {e}")
            return
        if not res.get("available"):
            vision_dead = True
            emit({"type": "warn", "phase": "vision",
                  "message": "no vision model installed; inspections off"})
            return
        broken_streak, warn = track_verdict(broken_streak, res["verdict"])
        entry = {"step": len(path_actions), "deposits": st.seq,
                 "verdict": res["verdict"], "note": res["note"],
                 "model": res.get("model", ""),
                 "shot": os.path.basename(shot)}
        vision_log.append(entry)
        emit({"type": "vision", **entry})
        print(f"VISION [{res['verdict']}] {res['note']}")
        if warn:
            emit({"type": "warn", "phase": "vision", "message": warn})

    def commit(act, record):
        # Backstop: printer mode commits deposition-protocol actions only.
        # Whole-object primitives are rejected by the validator long before
        # this point; this raises instead of silently converting anything.
        if act.get("action") not in PATH_ACTIONS:
            raise PipelineError(
                f"refusing to commit forbidden action "
                f"{act.get('action')!r} in printer mode")
        committed_records.append(record) if record.get("action") in (
            "deposit_segment", "deposit_point") else None
        path_actions.append(act)
        if record.get("action") in ("deposit_segment", "deposit_point"):
            finals.append(render_deposit(record))
        statuses.append("path/OK")
        blocks.append({"index": len(blocks), "action": act.get("action"),
                       "label": act.get("label", "?"), "guard_errors": [],
                       "syntax_errors": [], "ok": True})
        scene_js = ""
        if stream_scene:
            # Live SSE mode refreshes the viewer on a throttled schedule:
            # every 25 commits and at every layer boundary. Rewriting the
            # whole scene + checkpoint per commit is O(n^2) disk churn that
            # dominates long prints (measured ~60s at ~1800 commits), while
            # validation, state advance and the commit event itself still
            # happen for EVERY action below — batching never swallows a
            # deposition commit.
            layer_now = act.get("layer")
            boundary = (act.get("action") == "complete_layer"
                        or layer_now != commit_tick["layer"])
            commit_tick["n"] += 1
            if boundary or commit_tick["n"] % 25 == 0:
                scene_js = _write_run(out_dir, path_actions, finals)
                with open(os.path.join(out_dir, "inspect.html"), "w",
                          encoding="utf-8") as f:
                    f.write(INSPECT_TEMPLATE.replace("__BLOCKS__", scene_js))
                checkpoint()
                if isinstance(layer_now, int):
                    commit_tick["layer"] = layer_now
                maybe_inspect()
        emit({"type": "commit", "step": len(path_actions), "action": act,
              "status": "path/OK", "scene_js": scene_js,
              "committed": len(path_actions), "budget": len(script),
              "run_id": os.path.basename(out_dir)})

    def stop_requested():
        return should_stop is not None and should_stop()

    builder_attempts = repair_total = loop_attempts = ceiling = 0
    commit_tick = {"n": 0, "layer": -1}
    from morphix_path import component_status

    def comp_report():
        return component_status(actions, script, path_actions)

    if not guided:
        done_comps = set()
        for act in script[len(path_actions):]:
            if stop_requested():
                raise Cancelled(
                    f"cancelled by user after {len(path_actions)} path actions")
            if expired():
                # The deadline is a safety limit, not a data loss event:
                # persist the checkpoint plus an explicit partial report so
                # the run can resume and no one mistakes it for complete.
                checkpoint()
                _write_partial_report(
                    out_dir, reason=f"deadline exceeded ({deadline_s}s)",
                    committed=len(path_actions), total=len(script),
                    mode="path", guided=False,
                    extra={"layers": total_layers,
                           "components": comp_report()})
                raise PipelineError(
                    f"deadline exceeded ({deadline_s}s) after "
                    f"{len(path_actions)}/{len(script)} path actions; "
                    f"partial state checkpointed, resume with --resume")
            errs = validate_actions([act]) + st.check(act)
            if errs:  # deterministic script must always validate; a failure
                # here is an engine bug, reported loudly, never skipped.
                raise PipelineError(
                    f"engine script failed validation at {act}: {errs}")
            emit({"type": "think", "step": len(path_actions) + 1,
                  "phase": "action", "action": act, "status": "ENGINE"})
            record = st.apply(act)
            print(f"PATH {len(path_actions) + 1}/{len(script)}: "
                  f"{act['action']} {act.get('label')} [path/OK]")
            commit(act, record)
            if act.get("action") == "complete_layer":
                emit({"type": "layer", "layer": act.get("layer"),
                      "committed": len(path_actions), "total": len(script)})
                for c in comp_report():
                    if c["status"] == "completed" \
                            and c["component"] not in done_comps:
                        done_comps.add(c["component"])
                        emit({"type": "component", "component": c["component"],
                              "status": "completed",
                              "deposits": c["committed_deposits"],
                              "layers": c["layers"]})
        st.apply({"action": "complete_object", "label": "obj_done"})
        if not path_actions or path_actions[-1].get("action") != "complete_object":
            path_actions.append({"action": "complete_object",
                                 "label": "obj_done"})
        checkpoint()
        leftover = [c for c in comp_report()
                    if c["status"] != "completed"
                    and c["component"] not in
                    {lab for lab, _ in omitted_comps}]
        if leftover:  # engine invariant: deterministic run finishes all parts
            raise PipelineError(
                f"engine left components incomplete: {leftover}")
    else:
        fails, loop_attempts, builder_attempts, repair_total = 0, 0, 0, 0
        ceiling = max(segment_cap * 4, segment_cap + 8)
        last = None
        while not st.object_done:
            if stop_requested():
                raise Cancelled(
                    f"cancelled by user after {len(path_actions)} path actions")
            if expired():
                checkpoint()
                _write_partial_report(
                    out_dir,
                    reason=f"deadline exceeded ({deadline_s}s)",
                    committed=len(path_actions), total=len(script),
                    mode="path", guided=True,
                    extra={"layers": total_layers,
                           "builder_attempts": builder_attempts,
                           "repair_retries": repair_total,
                           "components": comp_report()})
                raise PipelineError(
                    f"deadline exceeded ({deadline_s}s) after "
                    f"{len(path_actions)} path actions, "
                    f"{builder_attempts} builder attempts; partial state "
                    f"checkpointed, resume with --resume")
            if repair_total >= max_rejects:
                checkpoint()
                _write_partial_report(
                    out_dir,
                    reason=f"reject budget exhausted ({repair_total} "
                           f"rejected proposals, cap {max_rejects})",
                    committed=len(path_actions), total=len(script),
                    mode="path", guided=True,
                    extra={"layers": total_layers,
                           "builder_attempts": builder_attempts,
                           "components": comp_report()})
                raise PipelineError(
                    f"guided print rejected {repair_total}x without converging "
                    f"(cap {max_rejects}); partial state checkpointed")
            # deterministic travel: the engine owns positioning moves (they
            # deposit nothing and cannot corrupt geometry); the builder
            # authors every deposition and completion. Once diverged, the
            # builder must move_to itself (validator enforces it).
            while (not diverged and ptr < len(script)
                   and script[ptr].get("action") == "move_to"):
                trav = script[ptr]
                terrs = validate_actions([trav]) + st.check(trav)
                if terrs:  # engine script must always validate; loud if not
                    raise PipelineError(
                        f"engine travel failed validation: {terrs}")
                emit({"type": "think", "step": len(path_actions) + 1,
                      "phase": "action", "action": trav, "status": "ENGINE"})
                trec = st.apply(trav)
                print(f"PATH {len(path_actions) + 1}: move_to "
                      f"{trav.get('label')} [engine-travel]")
                commit(trav, trec)
                ptr += 1
            loop_attempts += 1
            if loop_attempts > ceiling:
                raise PipelineError(
                    f"guided print hit the attempt ceiling ({ceiling} "
                    f"attempts for {len(path_actions)} committed actions, "
                    f"{builder_attempts} builder calls) without finishing")
            if len(path_actions) >= segment_cap:
                raise PipelineError(
                    f"guided print exceeded segment_cap={segment_cap} "
                    f"without complete_object")
            comp = comp_report()
            active_comp = None
            if not diverged and ptr < len(script):
                active_comp = script[ptr].get("component")
            ctx = {"toolhead": st.pos, "active_layer": st.active_layer,
                   "layer_y": layer_y(st.active_layer),
                   "plan": actions, "active_component": active_comp,
                   "committed": st.seq, "previous": last,
                   "geometry": {"deposits": st.seq,
                                "layers_done": st.active_layer,
                                "components": [
                                    {"component": c["component"],
                                     "status": c["status"],
                                     "committed": c["committed_deposits"],
                                     "planned": c["planned_deposits"]}
                                    for c in comp]},
                   "remaining": {"actions": segment_cap - len(path_actions),
                                 "attempts": ceiling - loop_attempts,
                                 "seconds": round(deadline_s - (
                                     _time.monotonic() - t0))},
                   "goal_missing": goal_coverage(parts, actions)[1],
                   "constraints": {
                       "from_must_equal_toolhead": True,
                       "deposit_y_must_equal_layer_y": True,
                       "layer_must_equal_active_layer": True,
                       "one_action_only": True,
                       "materials": sorted(MATERIAL_COLORS)}}
            if not diverged and ptr < len(script):
                ctx["suggestion"] = script[ptr]
            if ptr >= len(script):
                ctx["next"] = ("all planned paths are printed — emit "
                               "complete_object now")
            emit({"type": "step", "step": len(path_actions) + 1,
                  "phase": "think", "committed": len(path_actions),
                  "budget": segment_cap,
                  "state": f"toolhead {st.pos} layer {st.active_layer}"})
            builder_attempts += 1
            try:
                raw = build_one(ctx) if build_one else ollama_generate(
                    builder_model, PATH_BUILDER_SYSTEM, json.dumps(ctx))
            except Exception as e:
                emit({"type": "block_error", "action": "?",
                      "attempt": fails + 1, "error": str(e)})
                fails += 1
                repair_total += 1
                if fails > retries:
                    raise PipelineError(f"builder failed {fails}x: {e}")
                continue
            try:
                act = _parse_single_action(raw)
            except ValueError as e:
                fails += 1
                repair_total += 1
                emit({"type": "reject", "step": len(path_actions) + 1,
                      "phase": "parse", "action": {"_raw": raw[:200]},
                      "errors": [str(e)]})
                if fails > retries:
                    raise PipelineError(
                        f"builder unparseable {fails}x: {e}")
                continue
            errs = validate_actions([act]) + st.check(act)
            if errs:
                fails += 1
                repair_total += 1
                fb = "; ".join(errs)
                if not diverged and ptr < len(script):
                    fb += f"; expected next: {json.dumps(script[ptr])}"
                emit({"type": "reject", "step": len(path_actions) + 1,
                      "phase": "path", "action": act, "errors": errs})
                print(f"PATH reject ({fb})")
                if fails > retries:
                    raise PipelineError(
                        f"guided step failed validation after retries: {fb}")
                continue
            fails = 0
            if act.get("action") == "complete_object":
                # components gate: no finish while required parts are missing.
                # (PathState already guarantees full layers; this guarantees
                # full components. Rejected proposals are never committed.)
                undone = [c for c in comp_report()
                          if c["status"] != "completed"]
                if undone:
                    fails += 1
                    repair_total += 1
                    fb = ("complete_object refused: components incomplete: "
                          + ", ".join(f"{c['component']}={c['status']} "
                                      f"({c['committed_deposits']}/"
                                      f"{c['planned_deposits']})"
                                      for c in undone))
                    emit({"type": "reject", "step": len(path_actions) + 1,
                          "phase": "goal", "action": act, "errors": [fb]})
                    print(f"PATH reject ({fb})")
                    if fails > retries:
                        raise PipelineError(
                            f"guided finish refused after retries: {fb}")
                    continue
            if (not diverged and ptr < len(script)
                    and act != script[ptr]):
                diverged = True
                print("PATH: builder diverged from engine script "
                      "(still valid — suggestions off)")
            elif not diverged:
                ptr += 1
            emit({"type": "think", "step": len(path_actions) + 1,
                  "phase": "action", "action": act,
                  "status": f"MODEL {builder_model}"})
            record = st.apply(act)
            last = act
            print(f"PATH {len(path_actions) + 1}: {act['action']} "
                  f"{act.get('label')} [guided/OK]")
            commit(act, record)

    maybe_inspect(force=True)
    # geometric verification: committed deposits vs planned component
    # geometry. A component is complete only when verified — action counts
    # alone never complete anything, and success requires verification.
    thought_by_label = {a.get("label"): a for a in actions
                        if isinstance(a, dict)}
    solid_checks = verify_solids(actions, committed_records)
    solid_by = {}
    for sc in solid_checks:
        solid_by.setdefault(sc["component"], []).append(sc)
    verified_comps = []
    for comp in comp_report():
        exp = expected_of(thought_by_label.get(comp["component"], {}))
        ver = verify_component(spec, {**comp, **exp}, committed_records)
        for sc in solid_by.get(comp["component"], []):
            ver["checks"].append({"name": sc["name"], "ok": sc["ok"],
                                  "detail": sc["detail"]})
        ver["verified"] = all(c["ok"] for c in ver["checks"])
        comp.update(verified=ver["verified"], checks=ver["checks"])
        verified_comps.append(comp)
        marks = ", ".join(f"{c['name']}={'OK' if c['ok'] else 'FAIL'}"
                          for c in ver["checks"])
        print(f"VERIFY {comp['component']}: "
              f"{'VERIFIED' if ver['verified'] else 'FAILED'} ({marks})")
    geometry_ok = all(c["ok"] for c in solid_checks)
    # As-built mesh checks on the committed deposits: duplicates, invalid
    # triangles and interpenetrating beads veto acceptance exactly like any
    # other fidelity warning (measured ~40ms; never a bottleneck).
    from morphix_mesh import (deposits_to_mesh, check_mesh_validity,
                              check_no_duplicate_deposits,
                              check_same_material_overlaps)
    _mesh = deposits_to_mesh(committed_records)
    for _err in (check_mesh_validity(_mesh)
                 + check_no_duplicate_deposits(committed_records)
                 + check_same_material_overlaps(_mesh)):
        fidelity_warns.append(f"mesh: {_err}")
        emit({"type": "warn", "phase": "mesh", "message": _err})
    complete = (not omitted_comps
                and bool(verified_comps)
                and all(c.get("verified") for c in verified_comps)
                and geometry_ok)
    scene_js = _write_run(out_dir, path_actions, finals)
    report = {"ok": True, "mode": "path", "guided": guided,
              "action_errors": [], "blocks": blocks, "scene_errors": [],
              "think_status": think_status, "block_statuses": statuses,
              "goal_parts": parts,
              "goal_covered": goal_coverage(parts, actions)[0],
              "goal_missing": goal_coverage(parts, actions)[1],
              "components": verified_comps,
              "geometry_checks": solid_checks,
              "omitted": [{"component": lab, "error": msg}
                          for lab, msg in omitted_comps],
              "complete": complete,
              "seat_warnings": comp_warns,
              "fidelity_warnings": fidelity_warns,
              "repairs": plan_repairs,
              "spec": spec_name,
              # Honest category labeling: "template" means a deterministic
              # category template backs the geometry; "generic" means the
              # thinker proposed free-form parts with no category backing.
              # Generic success only ever means "valid deposition", never
              # "recognized object".
              "category_support": ("template"
                                   if template_key(spec_name) != "object"
                                   else "generic"),
              "budgets": {"planner_attempts": planner_attempts,
                          "builder_attempts": builder_attempts,
                          "repair_retries": repair_total,
                          "loop_attempts": loop_attempts,
                          "attempt_ceiling": ceiling,
                          "committed_actions": len(path_actions),
                          "segment_cap": segment_cap,
                          "layers": total_layers, "layer_cap": layer_cap,
                          "deadline_s": deadline_s,
                          "path_optimization": opt_stats},
              "path_actions": len(path_actions),
              "deposits": st.seq, "layers": total_layers,
              "vision": vision_log,
              "ir": summary_of(plan_ir),
              "ir_full": plan_ir,
              "syntax_scene_js": validate_js_syntax(scene_js)}
    report["ok"] = not report["syntax_scene_js"]
    # success is separated: execution valid (ok) vs geometry verified and
    # goal met (complete) vs final quality accepted (both, and no fidelity
    # warnings — ugly-but-printed geometry is reported, never celebrated).
    report["quality_accepted"] = bool(
        report["ok"] and report.get("complete")
        and not report.get("fidelity_warnings"))
    # The single headline verdict: execution valid AND geometry verified
    # AND goal met AND no fidelity warnings. Anything less is reported,
    # never celebrated — and the CLI exits non-zero on it.
    report["success"] = bool(
        report["ok"] and report.get("complete")
        and report.get("quality_accepted"))
    _write_run(out_dir, path_actions, finals, extra=report)
    with open(os.path.join(out_dir, "object.json"), "w",
              encoding="utf-8") as f:
        json.dump(plan_ir, f, indent=2)
    print(f"RESULT: {'OK' if report['ok'] else 'FAIL'} "
          f"({st.seq} deposits, {total_layers} layers) -> {out_dir}/")
    emit({"type": "done", "run_id": os.path.basename(out_dir),
          "report": report, "scene_js": scene_js, "ok": report["ok"]})
    return report


def _build_block(act: dict, builder: str, builder_model: str, lora,
                 emit=None) -> tuple:
    """One action -> guarded, validate_js_syntax-checked JS.

    Strict mode: only the real model output is acceptable. On guard/syntax
    failure the builder is re-asked with the validator feedback; after
    MORPHIX_BUILDER_MAX_RETRIES the step FAILS (returns (None, '', errors)).
    Non-strict mode keeps the deterministic template as the final safety net.
    """
    emit = emit or (lambda e: None)
    action_name = act.get("action", "?")
    last_guardless = ""
    last_model_error = ""
    last_raw = ""
    method = builder if builder != "lora" else "lora"
    for attempt in range(1 + (BUILDER_MAX_RETRIES if STRICT_MODELS else 0)):
        try:
            if builder == "lora":
                raw = lora.build(act, feedback=last_guardless)
            elif builder == "ollama":
                raw = _ollama_build(builder_model, act, feedback=last_guardless)
            else:
                raw = render_fallback(act)  # non-strict only; rejected above
                method = "template"
        except Exception as e:
            last_model_error = str(e)
            emit({"type": "block_error", "action": action_name,
                  "attempt": attempt + 1, "error": last_model_error})
            if STRICT_MODELS:
                continue  # allows retries on transient network errors
            raw, method = render_fallback(act), "template-fallback"
        code = _clean_js(raw)
        last_raw = raw
        final, gstat, gerrs = enforce(act, code)
        syn_errs = validate_js_syntax(final) if gstat == "OK" else []
        if gstat == "OK" and not syn_errs:
            emit({"type": "validate", "action": action_name,
                  "label": act.get("label"), "attempt": attempt + 1,
                  "guard_errors": [], "syntax_errors": [],
                  "guard_status": "OK"})
            return final, f"{method}/OK", []
        guard_info = {"guard_errors": gerrs, "syntax_errors": syn_errs,
                      "guard_status": gstat}
        emit({"type": "validate", "action": action_name,
              "label": act.get("label"), "attempt": attempt + 1, **guard_info})
        last_guardless = "; ".join(gerrs + syn_errs) or "invalid JS"
        print(f"  guard{'/syntax' if syn_errs else ''} reject "
              f"{action_name}: {last_guardless}")
        if not STRICT_MODELS:
            # last resort: deterministic, always-valid template block
            final = render_fallback(act)
            if not validate_js_syntax(final):
                return final, "template/OK", []
            emit({"type": "block_error", "action": action_name,
                  "error": "template fallback itself failed syntax check"})
            return None, "template/FAIL", ["template fallback failed"]
    # strict mode, exhausted retries -> this step fails loudly.
    # Report the true cause: no model output at all (unreachable/OOM) is
    # different from output that failed guard validation.
    if last_model_error and not last_guardless:
        errs = f"builder model failed, no output to validate: {last_model_error}"
    else:
        errs = last_guardless or "model output failed guard validation"
        if last_raw.strip():
            # show what the model actually wrote (truncated, single "error"
            # element so the "; "-split downstream does not shred it).
            excerpt = " ".join(last_raw.split())[:300]
            errs += f" | builder wrote: {excerpt!r}"
    emit({"type": "block_fail", "action": action_name,
          "label": act.get("label"), "errors": errs.split("; ")})
    return None, f"{method}/FAIL", list(errs.split("; "))
def run_incremental(prompt: str, builder: str = "ollama",
                    builder_model: str = BUILDER_MODEL,
                    thinker_model: str = THINKER_MODEL,
                    out_dir: str = "./out_step", max_steps=None,
                    cap: int = 24, emit=None, should_stop=None) -> dict:
    """Incremental generation: ONE atomic action per tick, each conditioned on
    live scene geometry. The thinker decides the step budget up front
    (max_steps=None -> think_plan, clamped to cap); its per-step 'done' stops
    the loop early. Every step is streamed + validated for the client.
    `should_stop` is an optional zero-arg callable checked at the top of every
    step; when it returns True the run aborts with Cancelled, preserving all
    committed steps and the partial scene written so far."""
    emit = emit or (lambda e: None)
    os.makedirs(out_dir, exist_ok=True)
    if STRICT_MODELS and builder == "template":
        raise _reject_builder("template")
    emit({"type": "start", "prompt": prompt, "builder": builder,
          "builder_model": builder_model, "thinker_model": thinker_model,
          "incremental": True, "strict": STRICT_MODELS})

    if max_steps is None:
        max_steps, plan_status = think_plan(
            prompt, thinker_model,
            options=_think_options(builder, 3000))
        max_steps = max(1, min(cap, max_steps))
        emit({"type": "plan", "max_steps": max_steps, "status": plan_status,
              "decided_by": "thinker"})
        print(f"PLAN: thinker budgets {max_steps} steps (cap {cap}) [{plan_status}]")
    else:
        max_steps = max(1, min(cap, max_steps))
        emit({"type": "plan", "max_steps": max_steps, "status": "user",
              "decided_by": "user"})
    print(f"PROMPT: {prompt}  (incremental, max {max_steps} steps)")
    # goal coverage: component tokens the finished scene must contain. Every
    # step is steered toward the still-missing parts, and a premature "done"
    # is refused while parts are missing (bounded by max_attempts).
    parts = goal_parts(prompt)
    covered, missing = goal_coverage(parts, [])
    emit({"type": "goal", "parts": parts, "covered": covered,
          "missing": missing})
    if parts:
        print(f"GOAL: {len(parts)} parts expected: {parts}")
    state = SceneState()
    lora = get_lora_builder() if builder == "lora" else None
    actions, finals, statuses, trace = [], [], [], []
    feedback = ""
    step = 0          # attempt/event counter (drives live visibility)
    committed = 0     # successfully built actions (drives the budget)
    # A rejected proposal must not silently eat the action budget: the thinker
    # gets a few attempts per committed action, up to a hard ceiling.
    max_attempts = max(max_steps * 4, max_steps + 4)
    tstat = "NONE"
    consecutive_rejects = 0
    thinker_done = False
    while committed < max_steps and step < max_attempts:
        if should_stop is not None and should_stop():
            emit({"type": "cancelled", "step": step, "committed": committed,
                  "budget": max_steps})
            raise Cancelled(
                f"cancelled by user at step {step} "
                f"({committed}/{max_steps} steps committed)")
        step += 1
        emit({"type": "step", "step": step, "phase": "think",
              "committed": committed, "budget": max_steps,
              "state": state.summary()})
        summary = state.summary()
        if missing:
            summary += (f"\nFinal-goal parts still missing: "
                        f"{missing} — prefer building one of these next.")
        try:
            act, tstat = think_next(prompt, summary, step, feedback,
                                    thinker_model,
                                    options=_think_options(builder, 1500))
        except ModelUnavailableError:
            emit({"type": "error", "step": step, "phase": "think",
                  "message": f"thinker model failed at step {step}"})
            raise
        if act.get("action") == "done":
            if missing:
                # premature finish: goal parts still uncovered — refuse and
                # steer back. The attempt ceiling still bounds the loop, so
                # this cannot spin forever; the report stays complete=False.
                feedback = (f"goal incomplete, still missing parts: {missing}. "
                            f"Build one of them next, do not stop yet.")
                print(f"STEP {step}: done refused ({feedback}), continuing")
                emit({"type": "reject", "step": step, "phase": "goal",
                      "action": act, "errors": [feedback]})
                trace.append({"step": step, "thinker": tstat,
                              "error": feedback})
                continue
            print(f"STEP {step}: thinker done ({act.get('reason', '')}) [{tstat}]")
            emit({"type": "step", "step": step, "phase": "done",
                  "reason": act.get("reason", "")})
            thinker_done = True
            trace.append({"step": step, "thinker": tstat, "done": True})
            break
        if not act.get("action"):
            feedback = act.get("_error", "model did not return an action")
            if act.get("_raw"):
                feedback += f"; it wrote: {act['_raw']!r}"
            print(f"STEP {step}: unusable reply ({feedback}), retrying")
            emit({"type": "reject", "step": step, "phase": "parse",
                  "action": act, "errors": [feedback]})
            trace.append({"step": step, "thinker": tstat, "error": feedback})
            continue
        emit({"type": "think", "step": step, "phase": "action",
              "action": act, "status": tstat})
        # resolve state refs (on_top_of -> absolute) BEFORE validation/build
        try:
            act = state.resolve(act)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            feedback = f"invalid action fields: {e}"
            print(f"STEP {step}: bad action ({feedback}), retrying")
            emit({"type": "reject", "step": step, "phase": "resolve",
                  "action": act, "errors": [feedback]})
            trace.append({"step": step, "thinker": tstat, "error": feedback})
            continue
        errs = validate_actions([act]) + validate_against_state(act, state)
        if errs:
            feedback = "; ".join(errs)
            consecutive_rejects += 1
            emit({"type": "reject", "step": step, "phase": "state",
                  "action": act, "errors": errs})
            try:
                spot = state.suggest_spot(act)
            except (ValueError, TypeError, KeyError, IndexError):
                spot = None
            if spot and any("collides" in e or "floats" in e
                           or "detached" in e or "rest it" in e
                           or "sinks" in e for e in errs):
                feedback += f"; try position {spot}"
            if any("floats" in e or "rest it" in e for e in errs):
                feedback += ("; rest the piece on the ground (base y=0) or "
                             "stack it with \"on_top_of\" an existing label")
            if any("duplicate label" in e for e in errs) and act.get("label"):
                feedback += (f"; use a brand-new label such as "
                             f"{act['label']}_2 instead of reusing "
                             f"{act['label']!r}")
            if any("invalid label" in e for e in errs) and act.get("label"):
                sug = re.sub(r"\W+", "_", str(act["label"])).strip("_").lower()
                feedback += (f"; relabel in snake_case, e.g. "
                             f"'{(sug or 'part')}_a1'")
            print(f"STEP {step}: rejected ({feedback}), retrying")
            trace.append({"step": step, "thinker": tstat,
                          "action": act, "rejected": errs})
            continue
        # build through the real model, guarded + syntactically checked
        final, status, blk_errs = _build_block(act, builder, builder_model,
                                               lora, emit=emit)
        if not final:
            raise PipelineError(
                f"step {step} ({act.get('action')} {act.get('label')}) "
                f"failed validation after retries: {'; '.join(blk_errs)}")
        state.add(act)
        state.update_head(act.get("position", [0, 0, 0]))
        actions.append(act)
        covered, missing = goal_coverage(parts, actions)
        emit({"type": "goal", "step": step, "action": act.get("action"),
              "label": act.get("label"), "covered": covered,
              "missing": missing})
        finals.append(final)
        statuses.append(status)
        committed += 1
        trace.append({"step": step, "thinker": tstat, "action": act,
                      "status": status})
        consecutive_rejects = 0
        feedback = ""
        print(f"STEP {step}: {act['action']} {act.get('label')} [{status}] "
              f"({committed}/{max_steps})")
        scene_js = _write_run(out_dir, actions, finals)
        emit({"type": "commit", "step": step, "action": act,
              "status": status, "scene_js": scene_js,
              "committed": committed, "budget": max_steps,
              "run_id": os.path.basename(out_dir)})

    scene_js = _write_run(out_dir, actions, finals)
    report = validate_scene(actions, finals) if actions else {
        "ok": False, "action_errors": ["no actions"], "blocks": [],
        "scene_errors": ["empty scene"]}
    report["mode"] = "incremental"
    report["think_status"] = tstat if actions else "NONE"
    report["block_statuses"] = statuses
    report["trace"] = trace
    report["syntax_scene_js"] = validate_js_syntax(scene_js)
    report["planned_steps"] = max_steps
    report["committed_steps"] = committed
    report["attempts"] = step
    report["goal_parts"] = parts
    report["goal_covered"] = covered
    report["goal_missing"] = missing
    # complete = the thinker said "done", or we committed the whole budget.
    # A partial but valid scene is still ok; the flag makes the shortfall visible.
    report["complete"] = bool(thinker_done or committed >= max_steps)
    report["ok"] = report["ok"] and not report["syntax_scene_js"]
    _write_run(out_dir, actions, finals, extra=report)
    print(f"RESULT: {'OK' if report['ok'] else 'FAIL'} "
          f"({committed}/{max_steps} steps committed"
          f"{'' if report['complete'] else ', INCOMPLETE'}) -> {out_dir}/")
    emit({"type": "done", "run_id": os.path.basename(out_dir),
          "report": report, "scene_js": scene_js, "ok": report["ok"]})
    return report


def main():
    p = argparse.ArgumentParser(description="Morphix full pipeline")
    p.add_argument("prompt", nargs="?", default="a red sphere on a blue box with warm sun light")
    p.add_argument("--builder", default="lora", choices=["ollama", "lora"])
    p.add_argument("--builder-model", default=BUILDER_MODEL)
    p.add_argument("--thinker-model", default=THINKER_MODEL)
    p.add_argument("--out", default="./out")
    p.add_argument("--incremental", action="store_true",
                   help="one atomic action per tick, conditioned on live scene state")
    p.add_argument("--max-steps", type=int, default=None,
                   help="step budget; default None = thinker decides (think_plan)")
    p.add_argument("--cap", type=int, default=24,
                   help="hard ceiling for thinker-decided budgets")
    p.add_argument("--path", action="store_true",
                   help="point-by-point path print: slice footprints into "
                        "validated layer segments instead of whole primitives")
    p.add_argument("--guided", action="store_true",
                   help="path mode: ask the builder for one movement per call "
                        "(slow); default executes the deterministic script")
    p.add_argument("--resume", action="store_true",
                   help="path mode: resume from out_dir/path.json checkpoint")
    p.add_argument("--segment-cap", type=int, default=4000,
                   help="max path actions per print job")
    p.add_argument("--layer-cap", type=int, default=32,
                   help="max printed layers per print job")
    p.add_argument("--deadline-s", type=int, default=1800,
                   help="overall wall-clock budget per print job (seconds)")
    p.add_argument("--max-rejects", type=int, default=60,
                   help="guided mode: abort after this many rejected proposals "
                        "(partial state is checkpointed, never lost)")
    p.add_argument("--vision-every", type=int, default=5,
                   help="path mode: vision-inspect every N deposits (0 disables)")
    a = p.parse_args()
    if a.path:
        report = run_path_print(a.prompt, a.builder, a.builder_model,
                                a.thinker_model, a.out, guided=a.guided,
                                vision_every=a.vision_every,
                                segment_cap=a.segment_cap,
                                layer_cap=a.layer_cap,
                                deadline_s=a.deadline_s, resume=a.resume,
                                max_rejects=a.max_rejects)
    elif a.incremental:
        report = run_incremental(a.prompt, a.builder, a.builder_model,
                                 a.thinker_model, a.out, a.max_steps, a.cap)
    else:
        report = run_pipeline(a.prompt, a.builder, a.builder_model,
                              a.thinker_model, a.out)
    sys.exit(0 if report.get("success", report["ok"]) else 1)


if __name__ == "__main__":
    main()