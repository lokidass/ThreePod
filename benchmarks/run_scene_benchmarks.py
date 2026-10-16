"""Scene benchmark runner (Phase 4.1, item 9).

For each prompt: plan -> Scene IR -> compile -> browser render, saving all
evidence (IR, JS, screenshots, errors, stats, timings, retries) plus a
per-prompt verdict that NEVER claims more than measured:
  semantic (required elements present?), geometric (validation), runtime
  (browser errors), completion (state machine), visual (content checks).

Usage:
  python benchmarks/run_scene_benchmarks.py            # full live run
  python benchmarks/run_scene_benchmarks.py --list     # show prompts
  python benchmarks/run_scene_benchmarks.py --out DIR  # custom out dir

Live thinker calls make full runs nondeterministic across runs; fixed seeds
make each single run reproducible. Frozen-plan regression coverage lives in
tests/test_scene_benchmark.py (deterministic, no model).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))

PROMPTS = [
    {"id": "table_room", "prompt": "a wooden table in a room",
     "expect": ["table", "room"]},
    {"id": "forest_mountain", "prompt": "a forested mountain landscape",
     "expect": ["forest", "mountains"]},
    {"id": "city_street", "prompt": "a modern city street at night",
     "expect": ["city"]},
    {"id": "scifi_env", "prompt": "a detailed sci-fi environment",
     "expect": ["scifi"]},
    {"id": "lighthouse", "prompt": "a lighthouse on a cliff at dusk",
     "expect": []},  # unseen: no template; must compose, not refuse
    {"id": "desert_caravan", "prompt": "a desert caravan at noon",
     "expect": []},
    {"id": "snowy_cabin", "prompt": "a snowy cabin in the woods",
     "expect": ["forest"]},
    {"id": "harbor", "prompt": "a harbor with boats at sunrise",
     "expect": ["water"]},
    {"id": "castle", "prompt": "a castle on a hill",
     "expect": ["mountains"]},
    {"id": "coral_reef", "prompt": "a coral reef underwater",
     "expect": ["particles", "water"]},
]


def run_one(entry, out_dir, thinker_model, seed, vision_fn=None):
    """Run a single prompt through the completion state machine with a
    real browser renderer. Returns the evidence record:
    semantic elements, geometry validation, runtime status, completion,
    screenshot, timings, retries/repairs, fallback use, remaining defects.
    """
    import morphix_scene_run as R
    from scene_browser import render_scenes

    t0 = time.monotonic()

    def live_render(js):
        png = os.path.join(out_dir, f"{entry['id']}.png")
        try:
            return render_scenes(
                [(entry["id"], js, png)])[entry["id"]]
        except Exception as e:
            return {"ready": False,
                    "scene_error": f"harness: {type(e).__name__}: {e}",
                    "console_errors": [], "stats": {}, "graph": {},
                    "png": None, "elapsed_s": 0.0}

    rep = R.run_scene({"prompt": entry["prompt"]},
                      thinker_model=thinker_model, seed=seed,
                      render_fn=live_render, vision_fn=vision_fn,
                      out_dir=os.path.join(out_dir, entry["id"]))
    rec = {"id": entry["id"], "prompt": entry["prompt"],
           "expect_templates": entry["expect"], "seed": seed}
    rec["state_report"] = rep
    v = {"complete": bool(rep.get("ok") and rep.get("state") == "COMPLETE"),
         "stage": (rep.get("state") or "FAILED").lower(),
         "error": rep.get("error", "")}
    # semantic presence: required ids rendered (from the state report)
    geo = rep.get("geometry", {}) or {}
    present = [c["name"] for c in geo.get("checks", [])
               if c["name"].startswith("present:") and c["ok"]]
    missing = [c["name"] for c in geo.get("checks", [])
               if c["name"].startswith("present:") and not c["ok"]]
    rec["semantic"] = {"required_ids": rep.get("required_ids", []),
                       "present": present, "missing": missing}
    rec["verdict"] = v
    rec["total_s"] = round(time.monotonic() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="",
                    help="comma-separated prompt ids")
    a = ap.parse_args()
    if a.list:
        for e in PROMPTS:
            print(f"{e['id']:16s} {e['prompt']}")
        return
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = a.out or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), f"report_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    from morphix_config import THINKER_MODEL
    from morphix_image import available_vision_model, describe_image
    vm = available_vision_model()
    vision_fn = (lambda png, summary: describe_image(png, vm)) if vm else None
    if vm:
        print(f"vision spot-checks via {vm} (advisory only)")
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    records = []
    for entry in PROMPTS:
        if only and entry["id"] not in only:
            continue
        print(f"[{entry['id']}] {entry['prompt']} ...", flush=True)
        rec = run_one(dict(entry), out_dir, THINKER_MODEL, seed=11,
                      vision_fn=vision_fn)
        records.append(rec)
        v = rec.get("verdict", {})
        sem = rec.get("semantic", {})
        print(f"  -> complete={v.get('complete')} stage={v.get('stage')} "
              f"present={len(sem.get('present', []))} "
              f"missing={sem.get('missing')} "
              f"{(v.get('error') or '')[:100]}", flush=True)
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump({"prompts": records}, f, indent=2)
    done = sum(1 for r in records if r.get("verdict", {}).get("complete"))
    print(f"{done}/{len(records)} complete -> {out_dir}/report.json")


if __name__ == "__main__":
    main()
