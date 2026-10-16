# server.py
# Morphix production backend: serves the frontend + JSON API around the
# thinker -> builder -> guard -> validator pipeline, plus Server-Sent Events
# so every step is streamed live and validated in front of the user.
#
# Run locally:        python server.py [--port 5000]
# Run in production:  gunicorn --config gunicorn.conf.py server:app
#
# API:
#   GET  /api/health                             -> model readiness
#   POST /api/think   {prompt, thinker_model}    -> {actions, status}
#   POST /api/jobs    {prompt, builder, ...}     -> starts a run, returns job id
#   GET  /api/jobs/<id>                          -> job status + result
#   GET  /api/jobs/<id>/stream?after=N           -> SSE of every step
#   GET  /api/jobs/<id>/events                   -> pollable JSON of events
#   POST /api/generate ...                       -> sync run (kept for scripts)
#   POST /api/image   (multipart)                -> image-to-3D run
#   POST /api/scenes  {template|plan|prompt}     -> scene-mode 3D environment
#   GET  /api/scene-templates                    -> tested template catalogue
#   GET  /api/runs /api/runs/<id>                -> history / single run
#   GET  /runs/<id>/scene.html|scene.js|actions.json|report.json

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from morphix_config import (BUILDER_MODEL, OLLAMA_HOST, PORT, REQUIRED_MODELS,
                            STRICT_MODELS, THINKER_MODEL, ollama_models)
from morphix_thinker import think
from morphix_pipeline import run_pipeline, run_incremental
from morphix_jobs import RUNS_DIR, create_job, get_job, event_stream

# UI: the React app (web/, built to web/dist) when it has been built,
# otherwise the classic static frontend. The classic UI stays reachable at
# /classic/ either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIC_DIR = os.path.join(_HERE, "frontend")
WEB_DIST = os.path.join(_HERE, "web", "dist")
UI_DIR = (WEB_DIST if os.path.isfile(os.path.join(WEB_DIST, "index.html"))
          else CLASSIC_DIR)

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
os.makedirs(RUNS_DIR, exist_ok=True)


@app.after_request
def _run_assets_cors(resp):
    # Run pages render inside a sandboxed iframe (opaque origin); their
    # textures (trace/texture.png) and module assets need CORS to load.
    if request.path.startswith("/runs/"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health():
    models = ollama_models()
    from morphix_config import model_ready, VISION_MODEL
    from morphix_image import available_vision_model
    thinker_ready = model_ready(THINKER_MODEL)
    builder_ready = model_ready(BUILDER_MODEL)
    vision = VISION_MODEL or available_vision_model()
    ready = thinker_ready and builder_ready
    return jsonify({
        "ollama": OLLAMA_HOST if models else "unreachable",
        "models": sorted(models),
        "thinker_model": THINKER_MODEL,
        "builder_model": BUILDER_MODEL,
        "vision_model": vision,
        "vision_ready": bool(vision),
        "thinker_ready": thinker_ready,
        "builder_ready": builder_ready,
        "ready": ready,
        "strict": STRICT_MODELS,
        "required": REQUIRED_MODELS,
    })


# ------------------------------------------------------------------- think
@app.post("/api/think")
def api_think():
    body = request.get_json(force=True) or {}
    prompt = body.get("prompt", "")
    model = body.get("thinker_model", THINKER_MODEL)
    if not prompt.strip():
        return jsonify({"error": "empty prompt"}), 400
    actions, status = think(prompt, model)
    return jsonify({"actions": actions, "status": status})


# -------------------------------------------- live jobs (SSE step stream)
@app.post("/api/jobs")
def api_jobs_create():
    body = request.get_json(force=True) or {}
    prompt = body.get("prompt", "")
    if not prompt.strip():
        return jsonify({"error": "empty prompt"}), 400
    builder = body.get("builder", "lora")
    if builder not in ("ollama", "lora"):
        return jsonify({"error": f"bad builder {builder!r}; "
                                 "deployment policy requires a real model "
                                 "(ollama | lora)"}), 400
    incremental = bool(body.get("incremental", True))
    max_steps = body.get("max_steps")
    cap = int(body.get("cap", 24))
    builder_model = body.get("builder_model") or BUILDER_MODEL
    thinker_model = body.get("thinker_model") or THINKER_MODEL
    mode = body.get("mode", "primitive")
    if mode not in ("primitive", "path"):
        return jsonify({"error": f"bad mode {mode!r}; "
                                  "expected 'primitive' | 'path'"}), 400
    guided = bool(body.get("guided", False))
    try:
        vision_every = int(body.get("vision_every", 5))
    except (TypeError, ValueError):
        vision_every = 5
    job_id, out_dir = create_job(
        prompt, builder, builder_model, thinker_model,
        incremental, max_steps, cap, mode=mode, guided=guided,
        vision_every=vision_every)
    return jsonify({"job_id": job_id, "run_id": job_id,
                    "stream": f"/api/jobs/{job_id}/stream"})


@app.get("/api/jobs/<job_id>")
def api_jobs_get(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    info = job.snapshot()
    return jsonify({"job_id": job_id, "done": info["done"],
                    "error": info["error"], "result": info["result"],
                    "cancelled": info["cancelled"],
                    "event_count": len(info["events"])})


@app.get("/api/jobs/<job_id>/stream")
def api_jobs_stream(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    try:
        after = max(0, int(request.args.get("after", 0)))
    except ValueError:
        after = 0

    def gen():
        yield ": connected\n\n"
        yield from event_stream(job, after)

    return Response(stream_with_context(gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.get("/api/jobs/<job_id>/events")
def api_jobs_events(job_id):
    """Polling fallback: last event index + snapshot of all events."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    info = job.snapshot()
    return jsonify({"job_id": job_id, "done": info["done"],
                    "error": info["error"],
                    "cancelled": info["cancelled"],
                    "events": info["events"],
                    "result": info["result"]})


@app.post("/api/jobs/<job_id>/cancel")
def api_jobs_cancel(job_id):
    """Ask a running job to stop. The worker checks between steps; already
    committed layers are preserved and the partial scene stays viewable."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if job.done:
        return jsonify({"error": "job already finished"}), 409
    job.request_cancel()
    return jsonify({"job_id": job_id, "cancelled": True})


# ------------------------------------------------ sync run (for tooling)
@app.post("/api/generate")
def api_generate():
    body = request.get_json(force=True) or {}
    prompt = body.get("prompt", "")
    if not prompt.strip():
        return jsonify({"error": "empty prompt"}), 400
    builder = body.get("builder", "lora")
    if builder not in ("ollama", "lora"):
        return jsonify({"error": f"bad builder {builder!r}"}), 400
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_id)
    try:
        if body.get("incremental", True):
            report = run_incremental(
                prompt, builder, body.get("builder_model") or BUILDER_MODEL,
                body.get("thinker_model") or THINKER_MODEL,
                out_dir, body.get("max_steps"), int(body.get("cap", 24)))
        else:
            report = run_pipeline(
                prompt, builder, body.get("builder_model") or BUILDER_MODEL,
                body.get("thinker_model") or THINKER_MODEL, out_dir)
    except Exception as e:
        return jsonify({"error": f"pipeline failed: {e}"}), 500
    actions = json.load(open(os.path.join(out_dir, "actions.json"), encoding="utf-8"))
    scene_js = open(os.path.join(out_dir, "scene.js"), encoding="utf-8").read()
    return jsonify({"run_id": run_id, "actions": actions,
                    "scene_js": scene_js, "report": report})


# ----------------------------------------------------------------- history
@app.get("/api/runs")
def api_runs():
    ids = sorted((d for d in os.listdir(RUNS_DIR)
                  if os.path.isdir(os.path.join(RUNS_DIR, d))
                  and os.path.isfile(os.path.join(RUNS_DIR, d, "scene.html"))),
                 reverse=True)
    return jsonify({"runs": ids})


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/runs/<run_id>")
def api_run(run_id):
    d = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(d):
        return jsonify({"error": "unknown run"}), 404

    def load(name, default):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f) if name.endswith(".json") else f.read()
    return jsonify({"run_id": run_id, "actions": load("actions.json", []),
                    "scene_js": load("scene.js", ""),
                    "report": load("report.json", {})})


@app.get("/runs/<run_id>/<fname>")
def run_file(run_id, fname):
    if fname not in ("scene.html", "scene.js", "actions.json", "report.json",
                     "scene_ir.json", "model.html", "object.json"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.join(RUNS_DIR, run_id), fname)


@app.post("/api/text3d")
def api_text3d():
    # Text -> realistic object via the shape-first family builders.
    from morphix_config import PipelineError
    from morphix_text3d import run as text_run
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "describe an object"}), 400
    try:
        report = text_run(prompt, RUNS_DIR)
    except PipelineError as e:
        return jsonify({"error": str(e), "stage": "understand"}), 422
    except Exception as e:
        return jsonify({"error": f"text to 3D failed: {e}", "stage": "build"}), 500
    run_id = report.pop("run_id")
    with open(os.path.join(RUNS_DIR, run_id, "scene.js"), encoding="utf-8") as f:
        scene_js = f.read()
    return jsonify({"run_id": run_id, "report": report, "scene_js": scene_js,
                    "actions": []})


@app.post("/api/runs/<run_id>/edit")
def api_run_edit(run_id):
    # Edit an object with a prompt -> a NEW run (the original is untouched).
    from morphix_config import PipelineError
    from morphix_edit import edit_run
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "describe the change, e.g. 'make it red'"}), 400
    if not os.path.isdir(os.path.join(RUNS_DIR, run_id)):
        return jsonify({"error": "unknown run"}), 404
    try:
        report = edit_run(RUNS_DIR, run_id, prompt)
    except PipelineError as e:
        return jsonify({"error": str(e), "stage": "edit"}), 422
    except Exception as e:
        return jsonify({"error": f"edit failed: {e}", "stage": "edit"}), 500
    new_id = report.pop("run_id")
    with open(os.path.join(RUNS_DIR, new_id, "scene.js"), encoding="utf-8") as f:
        scene_js = f.read()
    return jsonify({"run_id": new_id, "report": report, "scene_js": scene_js,
                    "actions": []})


@app.get("/api/runs/<run_id>/parts")
def api_run_parts(run_id):
    # Editable parts of a run's object (drives the edit suggestions).
    p = os.path.join(RUNS_DIR, run_id, "object.json")
    if not os.path.isfile(p):
        return jsonify({"editable": False, "roles": []})
    from morphix_edit import targets_of
    with open(p, encoding="utf-8") as f:
        ir = json.load(f)
    t = targets_of(ir)
    return jsonify({"editable": True, "roles": t["roles"],
                    "counts": t["counts"],
                    "materials": t["materials"],
                    "category": ir.get("category")})


@app.get("/runs/<run_id>/trace/<fname>")
def run_trace_file(run_id, fname):
    # image -> 3D evidence: reference, mask, renders, comparison, JSON trace
    from werkzeug.utils import secure_filename
    if secure_filename(fname) != fname or not fname.endswith(
            (".png", ".jpg", ".jpeg", ".json")):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.join(RUNS_DIR, run_id, "trace"), fname)


# ------------------------------------------------------------------- image
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.post("/api/image")
def api_image():
    from werkzeug.utils import secure_filename
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "no image uploaded"}), 400
    mode = request.form.get("mode", "relief")
    if mode not in ("relief", "mosaic", "reconstruct"):
        return jsonify({"error": "bad mode (relief | mosaic | reconstruct)"}), 400
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src = os.path.join(UPLOADS_DIR, f"{stamp}_{secure_filename(f.filename)}")
    f.save(src)
    out_dir = os.path.join(RUNS_DIR, stamp)

    if mode == "reconstruct":
        # Open-ended image -> 3D (any subject): family builders where they
        # exist, the subject's measured silhouette otherwise, measured
        # visual verification. Vision problems are reported in the result,
        # never answered with "pick one of N categories".
        from morphix_image3d import ReconstructionError
        from morphix_reconstruct import run_reconstruction
        category = (request.form.get("category") or "").strip() or None
        vision_model = (request.form.get("vision_model") or "").strip()
        environment = request.form.get("environment", "object-only")
        if environment not in ("object-only", "preserve"):
            return jsonify({"error": "environment must be object-only | "
                                     "preserve"}), 400
        try:
            report = run_reconstruction(src, out_dir, category=category,
                                        vision_model=vision_model,
                                        environment=environment)
        except ReconstructionError as e:
            return jsonify({"error": str(e), "stage": e.stage}), 422
        except Exception as e:
            return jsonify({"error": f"reconstruction failed: {e}",
                            "stage": "internal"}), 500
        actions = json.load(open(os.path.join(out_dir, "actions.json"),
                                 encoding="utf-8"))
        scene_js = open(os.path.join(out_dir, "scene.js"),
                        encoding="utf-8").read()
        return jsonify({"run_id": stamp, "mode": "reconstruct",
                        "actions": actions, "scene_js": scene_js,
                        "report": report,
                        "ir": report.get("ir"),
                        "status": report.get("reconstruction_status"),
                        "success": report.get("success"),
                        "trace": f"/runs/{stamp}/trace/",
                        "warnings": report.get("warnings", [])})

    from morphix_image import run_image
    grid = max(4, min(24, int(request.form.get("grid", 12))))
    try:
        report = run_image(src, out_dir, grid, mode)
    except Exception as e:
        return jsonify({"error": f"image pipeline failed: {e}"}), 500
    actions = json.load(open(os.path.join(out_dir, "actions.json"), encoding="utf-8"))
    scene_js = open(os.path.join(out_dir, "scene.js"), encoding="utf-8").read()
    resp = {"run_id": stamp, "mode": f"image-{mode}", "actions": actions,
            "scene_js": scene_js, "report": report}
    if request.form.get("describe") == "1":
        from morphix_image import describe_image
        resp["description"] = describe_image(src) or "(no vision model)"
    return jsonify(resp)


# ------------------------------------------------- scene mode (Phase 3)
# Rich interactive Three.js environments. Explicitly NOT the printer
# pipeline: no slicing, no deposition, no component accounting — scene
# graphs, materials, fog, instancing, animation and renderer selection.
@app.get("/api/scene-templates")
def api_scene_templates():
    from morphix_scene_ir import summarize
    from morphix_scene_templates import TEMPLATE_NAMES, build
    catalogue = []
    for name in TEMPLATE_NAMES:
        try:
            catalogue.append({"name": name, **summarize(build(name))})
        except ValueError as e:
            catalogue.append({"name": name, "error": str(e)})
    return jsonify({"templates": catalogue,
                    "backend": "webgl", "three_revision": "152"})


@app.post("/api/scenes")
def api_scenes():
    from morphix_config import PipelineError
    from morphix_scene_compile import (compile_scene_ir, scene_html,
                                       validate_scene_js)
    from morphix_scene_ir import ir_from_json, summarize, validate_scene_ir
    body = request.get_json(force=True) or {}
    seed = body.get("seed", 11)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer"}), 400
    try:
        if body.get("template"):
            from morphix_scene_templates import build
            try:
                ir = build(body["template"], seed=seed)
            except KeyError as e:
                return jsonify({"error": str(e)}), 400
            plan = {"templates": [body["template"]], "seed": seed,
                    "backend": "webgl", "source": "template-direct"}
        elif body.get("plan"):
            from morphix_scene_plan import plan_to_scene_ir
            plan = body["plan"]
            ir = plan_to_scene_ir(plan)
            plan = {**plan, "source": "client-plan"}
        elif (body.get("prompt") or "").strip():
            from morphix_scene_plan import plan_scene, plan_to_scene_ir
            plan = plan_scene(body["prompt"],
                              body.get("thinker_model") or THINKER_MODEL)
            ir = plan_to_scene_ir(plan)
            plan = {**plan, "source": "thinker"}
        else:
            return jsonify({"error": "give one of: template | plan | "
                                     "prompt"}), 400
        errs = validate_scene_ir(ir)
        if errs:
            return jsonify({"error": "scene IR invalid: " + "; ".join(errs),
                            "stage": "ir_validation", "plan": plan}), 422
        scene_js = compile_scene_ir(ir)
        js_errs = validate_scene_js(scene_js)
        if js_errs:
            return jsonify({"error": "scene JS invalid: " + "; ".join(js_errs),
                            "stage": "compile", "plan": plan}), 422
    except PipelineError as e:
        # plan_scene failures land here (planner validation exhausted)
        return jsonify({"error": str(e), "stage": "planning"}), 422
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e), "stage": "compile"}), 422
    except Exception as e:
        return jsonify({"error": f"scene generation failed: {e}",
                        "stage": "render"}), 500
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scene.js"), "w",
              encoding="utf-8") as f:
        f.write(scene_js)
    with open(os.path.join(out_dir, "scene.html"), "w",
              encoding="utf-8") as f:
        f.write(scene_html(scene_js, title=f"Morphix scene — {stamp}"))
    with open(os.path.join(out_dir, "scene_ir.json"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps(ir, indent=2, sort_keys=True))
    summary = summarize(ir)
    report = {"ok": True, "mode": "scene", "plan": plan,
              "summary": summary, "backend": "webgl",
              "three_revision": "152",
              "validation": {"ir_errors": [], "js_errors": []}}
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "actions.json"), "w",
              encoding="utf-8") as f:
        json.dump([], f)
    return jsonify({"run_id": stamp, "mode": "scene", "scene_js": scene_js,
                    "ir": summary, "report": report})


@app.get("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/classic/")
@app.get("/classic/<path:fname>")
def classic(fname="index.html"):
    return send_from_directory(CLASSIC_DIR, fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT, help="port (env PORT)")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    print(f"Morphix server on http://{args.host}:{args.port} "
          f"(strict={STRICT_MODELS})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()