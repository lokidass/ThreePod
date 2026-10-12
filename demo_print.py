# demo_print.py
# Deterministic point-by-point print demo. No models, no credentials.
# Prints a 2-layer plaque (perimeter + infill on layer 0, smaller perimeter
# on layer 1) through the real PathState validator + deterministic engine,
# then writes a viewable scene and state->action training pairs.
#
# Usage:
#   python demo_print.py --out ./out_print
#   python morphix_validator.py --actions out_print/actions.json --js out_print/scene.js

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morphix_path import (PathState, trace_rect, zigzag_infill,
                          render_deposit, layer_y)
from morphix_validator import validate_actions, validate_js_syntax

PLAN = {"plan": "demo-plaque", "layers": 2,
        "description": "2x1 perimeter + infill at y=0, 1x0.5 perimeter at y=0.2"}


def build() -> tuple:
    """Run the full print. Returns (path_actions, scene_js, train_rows)."""
    st = PathState(total_layers=2)
    script = []
    script += trace_rect(0, 0.0, 0.0, 2.0, 1.0, label="l0_perim")
    script += zigzag_infill(0, 0.0, 0.0, 2.0, 1.0, label="l0_fill")
    script.append({"action": "complete_layer", "layer": 0,
                   "label": "l0_done"})
    script += trace_rect(1, 0.0, 0.0, 1.0, 0.5, label="l1_perim")
    script.append({"action": "complete_layer", "layer": 1,
                   "label": "l1_done"})
    script.append({"action": "complete_object", "label": "obj_done"})

    blocks, train_rows = [], []
    for act in script:
        schema_errs = validate_actions([act])
        if schema_errs:
            raise AssertionError(f"demo action failed schema: {schema_errs}")
        state_errs = st.check(act)
        if state_errs:
            raise AssertionError(f"demo action failed path check: {state_errs}")
        row = {"state": st.summary(),
               "plan": PLAN,
               "layer": {"index": st.active_layer,
                         "y": layer_y(st.active_layer)},
               "action": act}
        rec = st.apply(act)
        row["committed"] = rec
        train_rows.append(row)
        if rec.get("action") in ("deposit_segment", "deposit_point"):
            blocks.append(render_deposit(rec))
    if not st.object_done:
        raise AssertionError("demo did not complete the object")
    syn = validate_js_syntax("\n".join(blocks))
    if syn:
        raise AssertionError(f"demo scene failed node --check: {syn}")
    return script, "\n\n".join(blocks) + "\n", train_rows


def main():
    p = argparse.ArgumentParser(description="deterministic print demo")
    p.add_argument("--out", default="./out_print")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    actions, scene_js, rows = build()
    with open(os.path.join(a.out, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=1)
    with open(os.path.join(a.out, "scene.js"), "w", encoding="utf-8") as f:
        f.write(scene_js)
    from morphix_pipeline import HTML_TEMPLATE
    with open(os.path.join(a.out, "scene.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__BLOCKS__", scene_js))
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report = {"ok": True, "mode": "print-demo", "plan": PLAN,
              "n_actions": len(actions),
              "n_deposits": sum(1 for r in rows
                                if r["committed"].get("action",
                                                      "").startswith("deposit")),
              "n_train_rows": len(rows)}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"DEMO OK: {len(actions)} actions, "
          f"{report['n_deposits']} deposits, {len(rows)} train rows -> {a.out}/")


if __name__ == "__main__":
    main()
