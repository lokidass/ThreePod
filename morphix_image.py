# morphix_image.py
# Image -> 3D: turn a picture into a printer-ordered voxel relief/mosaic.
# Cells are emitted in SERPENTINE row order from one corner, so the scene
# prints step by step from a single origin like a 3D printer.
# An optional vision model (moondream/llava/...) can first describe the image
# into a text prompt for the normal thinker pipeline; the deterministic CV
# path below always works offline and is the default.
#
# Usage:
#   python morphix_image.py photo.png --grid 12 --mode relief --out ./out_img

import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morphix_config import ollama_models, ollama_generate

VISION_MODELS = ("moondream", "llava", "qwen2.5vl", "bakllava",
                 "granite-vision", "minicpm-v")


def available_vision_model():
    for n in ollama_models():
        for vm in VISION_MODELS:
            if n.startswith(vm):
                return n
    return ""


def vision_model_supports_images(name: str) -> bool:
    """True only for models known to accept image inputs.

    A text-only model (e.g. a coder LLM) must never be handed an image and
    assumed to have looked at it; the caller reports that clearly instead.
    """
    n = (name or "").strip().lower()
    if not n:
        return False
    return any(n.split(":")[0] == vm or n.startswith(vm)
               for vm in VISION_MODELS)


def describe_image(path: str, model: str = "") -> str:
    """Ask a vision model to describe the image as a 3D scene prompt."""
    model = model or available_vision_model()
    if not model:
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return ollama_generate(model, "", "What is in this image? One short sentence.",
                           images=[b64])


def _load_grid(path: str, grid: int):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((grid, grid), Image.LANCZOS)
    canvas = Image.new("RGB", (grid, grid), img.getpixel((0, 0)))
    canvas.paste(img, ((grid - img.width) // 2, (grid - img.height) // 2))
    return canvas


def _hex(rgb) -> str:
    return f"0x{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def image_to_actions(path: str, grid: int = 12, mode: str = "relief",
                     cell: float = 0.6, max_h: float = 3.0,
                     skip_bg: bool = True) -> tuple:
    """Return (actions, meta). Actions are add_box voxels in serpentine print
    order starting at the far corner. Relief height follows luminance."""
    grid = max(4, min(24, grid))
    img = _load_grid(path, grid)
    px = img.load()
    corners = [px[0, 0], px[grid - 1, 0], px[0, grid - 1], px[grid - 1, grid - 1]]
    bg = max(set(corners), key=corners.count)

    def is_bg(rgb):
        return sum(abs(a - b) for a, b in zip(rgb, bg)) < 60

    actions = []
    for row in range(grid):
        cols = range(grid) if row % 2 == 0 else range(grid - 1, -1, -1)
        for col in cols:
            rgb = px[col, row]
            if skip_bg and is_bg(rgb):
                continue
            lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0
            h = round(0.15 if mode == "mosaic" else 0.15 + lum * max_h, 3)
            x = round((col - (grid - 1) / 2) * cell, 3)
            z = round((row - (grid - 1) / 2) * cell, 3)
            actions.append({
                "action": "add_box", "width": round(cell * 0.96, 3),
                "height": h, "depth": round(cell * 0.96, 3),
                "position": [x, round(h / 2, 3), z],
                "color": _hex(rgb), "label": f"px_r{row:02d}c{col:02d}",
            })
    meta = {"grid": grid, "mode": mode, "cells": len(actions),
            "skipped_bg": grid * grid - len(actions),
            "vision_model": available_vision_model()}
    return actions, meta


def run_image(path: str, out_dir: str = "./out_img", grid: int = 12,
              mode: str = "relief") -> dict:
    """Build + validate an image-derived scene. Returns the report dict."""
    from morphix_guard import enforce
    from morphix_validator import validate_scene, validate_js_syntax
    from morphix_state import SceneState
    os.makedirs(out_dir, exist_ok=True)
    actions, meta = image_to_actions(path, grid, mode)
    state = SceneState()
    finals, statuses = [], []
    for act in actions:  # already in printer order; commit through the guard
        final, gstat, _ = enforce(act, "")  # empty code -> deterministic template
        finals.append(final)
        statuses.append(f"image/{gstat}")
        state.add(act)
        state.update_head(act["position"])
    scene_js = "\n\n".join(finals) + "\n" if finals else "// empty\n"
    with open(os.path.join(out_dir, "scene.js"), "w", encoding="utf-8") as f:
        f.write(scene_js)
    from morphix_pipeline import HTML_TEMPLATE
    with open(os.path.join(out_dir, "scene.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__BLOCKS__", scene_js))
    with open(os.path.join(out_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f)
    report = validate_scene(actions, finals)
    report["mode"] = f"image-{mode}"
    report["image_meta"] = meta
    report["block_statuses"] = statuses
    report["syntax_scene_js"] = validate_js_syntax(scene_js)
    report["ok"] = report["ok"] and not report["syntax_scene_js"]
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"IMAGE: {len(actions)} cells ({meta['skipped_bg']} bg skipped) "
          f"-> {'OK' if report['ok'] else 'FAIL'} {out_dir}/")
    return report


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--mode", default="relief", choices=["relief", "mosaic"])
    p.add_argument("--out", default="./out_img")
    p.add_argument("--describe", action="store_true",
                   help="also ask the vision model to describe the image")
    a = p.parse_args()
    if a.describe:
        print("VLM:", describe_image(a.image) or "(no vision model)")
    run_image(a.image, a.out, a.grid, a.mode)


if __name__ == "__main__":
    main()
