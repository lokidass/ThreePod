# morphix_jobs.py
# Background job runner + Server-Sent Events (SSE) stream for live step
# visibility. POST /api/jobs starts the pipeline in a worker thread; each step
# (think / plan / build / guard / validate / commit) is pushed to an event list
# that clients consume over GET /api/jobs/<id>/stream.

import datetime
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morphix_pipeline import run_pipeline, run_incremental  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

MAX_JOBS_KEPT = 50
JOBS = {}
_jobs_lock = threading.Lock()


class Job:
    """One run. `emit` is called from the worker thread; the SSE stream
    drains the event list. `finish` is called when the run completes or dies."""

    def __init__(self, job_id: str):
        self.id = job_id
        self.events = []            # list of dicts (the source of truth)
        self.result = None          # final payload for GET /api/jobs/<id>
        self.error = None           # fatal error message, if any
        self.done = False
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self._cancel = threading.Event()

    def request_cancel(self):
        """Ask the worker to stop after the current step. Committed layers
        are preserved; the worker raises Cancelled at the next check."""
        self._cancel.set()
        self._wake.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def emit(self, ev: dict):
        ev.setdefault("type", "message")
        ev["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            self.events.append(ev)
        self._wake.set()

    def finish(self, result=None, error=None):
        with self.lock:
            self.result = result
            self.error = error
            self.done = True
        self._wake.set()

    def snapshot(self):
        with self.lock:
            return {"job_id": self.id, "events": list(self.events),
                    "result": _jsonable(self.result), "error": self.error,
                    "done": self.done,
                    "cancelled": self._cancel.is_set()}


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def create_job(prompt: str, builder: str, builder_model: str,
               thinker_model: str, incremental: bool, max_steps, cap: int,
               mode: str = "primitive", guided: bool = False,
               vision_every: int = 0) -> str:
    """Start a run in a worker thread. Returns the job id (= run id).
    mode='path' prints sliced layer paths point by point (guided=True asks
    the builder for one movement per call, otherwise the engine executes
    the deterministic script with full validation)."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_id)
    # The viewport must represent THIS run from second zero: publish an empty
    # scene upfront so a rejected first step shows an empty grid — never the
    # previous run's committed geometry.
    from morphix_pipeline import _write_run as _init_run
    _init_run(out_dir, [], [])
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ok": False, "status": "running", "mode": mode,
                   "guided": guided, "prompt": prompt}, f)
    job = Job(run_id)
    with _jobs_lock:
        JOBS[run_id] = job
        # prune oldest finished jobs so the dict never grows unbounded
        finished = sorted((j for j in JOBS.values() if j.done), key=lambda j: j.id)
        for old in finished[: max(0, len(JOBS) - MAX_JOBS_KEPT)]:
            JOBS.pop(old.id, None)

    def worker():
        from morphix_config import Cancelled
        from morphix_pipeline import run_path_print
        try:
            if mode == "path":
                report = run_path_print(
                    prompt, builder, builder_model, thinker_model, out_dir,
                    guided=guided, vision_every=vision_every,
                    emit=job.emit, should_stop=job.cancel_requested)
            elif incremental:
                report = run_incremental(prompt, builder, builder_model,
                                         thinker_model, out_dir, max_steps,
                                         cap, emit=job.emit,
                                         should_stop=job.cancel_requested)
            else:
                report = run_pipeline(prompt, builder, builder_model,
                                      thinker_model, out_dir, emit=job.emit,
                                      should_stop=job.cancel_requested)
            actions = json.load(open(os.path.join(out_dir, "actions.json"),
                                     encoding="utf-8"))
            scene_js = open(os.path.join(out_dir, "scene.js"),
                            encoding="utf-8").read()
            job.finish(result={"run_id": run_id, "actions": actions,
                               "scene_js": scene_js, "report": report,
                               "ok": report.get("ok", False)})
        except Exception as e:  # fatal -> report it as the last event
            from morphix_config import Cancelled as _Cancelled
            if isinstance(e, _Cancelled):
                job.emit({"type": "cancelled", "message": str(e)})
                job.finish(error="cancelled by user")
                return
            job.emit({"type": "error", "fatal": True,
                      "message": f"{type(e).__name__}: {e}"})
            job.finish(error=str(e))

    threading.Thread(target=worker, daemon=True, name=f"morphix-{run_id}").start()
    return run_id, out_dir


def get_job(job_id: str):
    with _jobs_lock:
        return JOBS.get(job_id)


def event_stream(job: Job, after: int = 0):
    """Generator yielding SSE frames. Replays events after `after`, then follows
    new ones until the job finishes. Sends heartbeats to keep the connection
    alive behind proxies."""
    sent = after
    while True:
        with job.lock:
            batch = job.events[sent:]
            sent = len(job.events)
            done = job.done
            error, result = job.error, job.result
        for ev in batch:
            name = ev.get("type", "step")
            yield (f"event: {name if name != 'message' else 'step'}\n"
                   f"data: {json.dumps(ev, default=str)}\n\n")
        if done:
            payload = {"type": "end", "job_id": job.id, "error": error,
                       "result": result if result is not None else None}
            yield f"event: end\ndata: {json.dumps(payload, default=str)}\n\n"
            return
        if not job._wake.wait(timeout=20):
            yield ": heartbeat\n\n"
        job._wake.clear()