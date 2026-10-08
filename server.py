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

app = Flask(__name__, static_folder="frontend", static_url_path="")
os.makedirs(RUNS_DIR, exist_ok=True)


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health():
    models = ollama_models()
    from morphix_config import model_ready
    thinker_ready = model_ready(THINKER_MODEL)
    builder_ready = model_ready(BUILDER_MODEL)
    ready = thinker_ready and builder_ready
    return jsonify({
        "ollama": OLLAMA_HOST if models else "unreachable",
        "models": sorted(models),
        "thinker_model": THINKER_MODEL,
        "builder_model": BUILDER_MODEL,
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
    builder = body.get("builder", "ollama")
    if builder not in ("ollama", "lora"):
        return jsonify({"error": f"bad builder {builder!r}; "
                                 "deployment policy requires a real model "
                                 "(ollama | lora)"}), 400
    incremental = bool(body.get("incremental", True))
    max_steps = body.get("max_steps")
    cap = int(body.get("cap", 24))
    builder_model = body.get("builder_model") or BUILDER_MODEL
    thinker_model = body.get("thinker_model") or THINKER_MODEL
    job_id, out_dir = create_job(
        prompt, builder, builder_model, thinker_model,
        incremental, max_steps, cap)
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
                    "events": info["events"],
                    "result": info["result"]})


# ------------------------------------------------ sync run (for tooling)
@app.post("/api/generate")
def api_generate():
    body = request.get_json(force=True) or {}
    prompt = body.get("prompt", "")
    if not prompt.strip():
        return jsonify({"error": "empty prompt"}), 400
    builder = body.get("builder", "ollama")
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
    if fname not in ("scene.html", "scene.js", "actions.json", "report.json"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.join(RUNS_DIR, run_id), fname)


# ------------------------------------------------------------------- image
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.post("/api/image")
def api_image():
    from werkzeug.utils import secure_filename
    from morphix_image import run_image
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "no image uploaded"}), 400
    grid = max(4, min(24, int(request.form.get("grid", 12))))
    mode = request.form.get("mode", "relief")
    if mode not in ("relief", "mosaic"):
        return jsonify({"error": "bad mode"}), 400
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src = os.path.join(UPLOADS_DIR, f"{stamp}_{secure_filename(f.filename)}")
    f.save(src)
    out_dir = os.path.join(RUNS_DIR, stamp)
    try:
        report = run_image(src, out_dir, grid, mode)
    except Exception as e:
        return jsonify({"error": f"image pipeline failed: {e}"}), 500
    actions = json.load(open(os.path.join(out_dir, "actions.json"), encoding="utf-8"))
    scene_js = open(os.path.join(out_dir, "scene.js"), encoding="utf-8").read()
    resp = {"run_id": stamp, "actions": actions,
            "scene_js": scene_js, "report": report}
    if request.form.get("describe") == "1":
        from morphix_image import describe_image
        resp["description"] = describe_image(src) or "(no vision model)"
    return jsonify(resp)


@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")


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