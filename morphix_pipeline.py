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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphix_config import (BUILDER_MAX_RETRIES, BUILDER_MODEL, STRICT_MODELS,
                            THINKER_MODEL, ModelUnavailableError, PipelineError,
                            ollama_generate)
from morphix_thinker import think, think_next, think_plan, rule_based_plan
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
    """Local fine-tuned Qwen LoRA builder (GPU). Loaded lazily, once."""

    def __init__(self):
        self._ready = False

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import test_qwen7b as T
        self.T = T
        base = T.resolve_base()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        print(f"Loading LoRA builder ({base}) ...")
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map={"": 0},
            torch_dtype=torch.bfloat16, trust_remote_code=True)
        if os.path.isdir(T.ADAPTER):
            model = PeftModel.from_pretrained(model, T.ADAPTER)
        model.eval()
        self.model, self.torch = model, torch
        self._ready = True

    def build(self, action: dict, state=None, feedback: str = "") -> str:
        if not self._ready:
            self._load()
        prompt = (f"<|im_start|>system\n{BUILDER_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{_prompt_for(action, state)}<|im_end|>\n")
        if feedback:
            prompt += f"\nPrevious output was REJECTED:\n{feedback}\n"
        prompt += "<|im_start|>assistant\n"
        inputs = self.tok(prompt, return_tensors="pt").to(0)
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        budget = 384 if action.get("action") in (
            "add_shader", "add_post", "add_animation", "add_particles") else 256
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=budget, do_sample=False,
                repetition_penalty=1.15, no_repeat_ngram_size=8,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=[im_end, self.tok.eos_token_id])
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
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


def run_pipeline(prompt: str, builder: str = "ollama",
                 builder_model: str = BUILDER_MODEL,
                 thinker_model: str = THINKER_MODEL,
                 out_dir: str = "./out", emit=None) -> dict:
    """Whole-scene pipeline (think once, build all blocks). Fails loudly in
    strict mode when a model is unavailable or a block cannot validate."""
    os.makedirs(out_dir, exist_ok=True)
    emit = emit or (lambda e: None)
    if STRICT_MODELS and builder == "template":
        raise _reject_builder("template")
    emit({"type": "start", "prompt": prompt, "builder": builder,
          "builder_model": builder_model, "thinker_model": thinker_model,
          "incremental": False, "strict": STRICT_MODELS})
    print(f"PROMPT: {prompt}")

    # 1) THINK with the real model
    actions, think_status = think(prompt, thinker_model)
    emit({"type": "think", "phase": "plan", "status": think_status,
          "actions": actions, "n_actions": len(actions)})

    # sanitize: per-action schema check, then de-duplicate labels (renames
    # preserve the thinker's intent and keep var names unique). In strict mode
    # a malformed action list from the model is an error, not a rules fallback.
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
    actions = _valid
    print(f"THINK: {think_status} -> {len(actions)} valid actions")

    # 2) BUILD through the real model, guarded + validated per block
    lora = LoraBuilder() if builder == "lora" else None
    finals, statuses = [], []
    for i, act in enumerate(actions):
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
            emit({"type": "block_error", "action": action_name,
                  "attempt": attempt + 1, "error": str(e)})
            if STRICT_MODELS:
                continue  # allows retries on transient network errors
            raw, method = render_fallback(act), "template-fallback"
        code = _clean_js(raw)
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
    # strict mode, exhausted retries -> this step fails loudly
    errs = last_guardless or "model output failed guard validation"
    emit({"type": "block_fail", "action": action_name,
          "label": act.get("label"), "errors": errs.split("; ")})
    return None, f"{method}/FAIL", list(errs.split("; "))
def run_incremental(prompt: str, builder: str = "ollama",
                    builder_model: str = BUILDER_MODEL,
                    thinker_model: str = THINKER_MODEL,
                    out_dir: str = "./out_step", max_steps=None,
                    cap: int = 24, emit=None) -> dict:
    """Incremental generation: ONE atomic action per tick, each conditioned on
    live scene geometry. The thinker decides the step budget up front
    (max_steps=None -> think_plan, clamped to cap); its per-step 'done' stops
    the loop early. Every step is streamed + validated for the client."""
    emit = emit or (lambda e: None)
    os.makedirs(out_dir, exist_ok=True)
    if STRICT_MODELS and builder == "template":
        raise _reject_builder("template")
    emit({"type": "start", "prompt": prompt, "builder": builder,
          "builder_model": builder_model, "thinker_model": thinker_model,
          "incremental": True, "strict": STRICT_MODELS})

    if max_steps is None:
        max_steps, plan_status = think_plan(prompt, thinker_model)
        max_steps = max(1, min(cap, max_steps))
        emit({"type": "plan", "max_steps": max_steps, "status": plan_status,
              "decided_by": "thinker"})
        print(f"PLAN: thinker budgets {max_steps} steps (cap {cap}) [{plan_status}]")
    else:
        max_steps = max(1, min(cap, max_steps))
        emit({"type": "plan", "max_steps": max_steps, "status": "user",
              "decided_by": "user"})
    print(f"PROMPT: {prompt}  (incremental, max {max_steps} steps)")
    state = SceneState()
    lora = LoraBuilder() if builder == "lora" else None
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
        step += 1
        emit({"type": "step", "step": step, "phase": "think",
              "committed": committed, "budget": max_steps,
              "state": state.summary()})
        try:
            act, tstat = think_next(prompt, state.summary(), step, feedback,
                                    thinker_model)
        except ModelUnavailableError:
            emit({"type": "error", "step": step, "phase": "think",
                  "message": f"thinker model failed at step {step}"})
            raise
        if act.get("action") == "done":
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
            if spot and any("collides" in e for e in errs):
                feedback += f"; try position {spot}"
            if any("duplicate label" in e for e in errs) and act.get("label"):
                feedback += (f"; use a brand-new label such as "
                             f"{act['label']}_2 instead of reusing "
                             f"{act['label']!r}")
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
    p.add_argument("--builder", default="ollama", choices=["ollama", "lora"])
    p.add_argument("--builder-model", default=BUILDER_MODEL)
    p.add_argument("--thinker-model", default=THINKER_MODEL)
    p.add_argument("--out", default="./out")
    p.add_argument("--incremental", action="store_true",
                   help="one atomic action per tick, conditioned on live scene state")
    p.add_argument("--max-steps", type=int, default=None,
                   help="step budget; default None = thinker decides (think_plan)")
    p.add_argument("--cap", type=int, default=24,
                   help="hard ceiling for thinker-decided budgets")
    a = p.parse_args()
    if a.incremental:
        report = run_incremental(a.prompt, a.builder, a.builder_model,
                                 a.thinker_model, a.out, a.max_steps, a.cap)
    else:
        report = run_pipeline(a.prompt, a.builder, a.builder_model,
                              a.thinker_model, a.out)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()