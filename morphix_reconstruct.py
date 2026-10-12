# morphix_reconstruct.py
# Semantic image -> 3D. A photograph is treated as a reference for an OBJECT,
# not as a height map:
#
#   photo ─► vision adapter (category + visible parts, documented uncertainty)
#         ─┐
#          ├─► Object IR (validated) -> printer primitives -> layer slicing
#   stats ─┘
#
# This is deliberately NOT the relief/mosaic pipeline in morphix_image.py:
#   * relief/mosaic turns pixels into a grid of px_rNNcNN columns;
#   * semantic reconstruction turns one photo into a small set of named
#     components (body / cabin / wheels ...) via the canonical IR, then prints
#     them with the same deterministic slicer as text-to-3D.
#
# Honesty rules:
#   * with a vision model, the category comes from the photo;
#   * with no vision model, the run is clearly marked `image-demo` and uses a
#     documented deterministic template — never presented as a measurement;
#   * exact depth / occluded surfaces are reported as unsupported, not guessed.

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morphix_config import (ModelUnavailableError, PipelineError,
                            STRICT_MODELS, VISION_MODEL, ollama_generate)
from morphix_image import (VISION_MODELS, available_vision_model,
                           vision_model_supports_images)
from morphix_ir import (_object_bbox, SILHOUETTE_ASPECT_BAND,
                        SUPPORTED_CATEGORIES, image_fidelity, ir_for_category,
                        ir_to_actions, summary_of, template_key, validate_ir)

# Role whose width may follow the photo silhouette aspect when fitting the
# category template. `tabletop` is added so a table's length matches the photo
# while its thickness stays a fixed thin slab.
_ASPECT_SCALABLE_ROLES = ("body", "cabin", "wall", "window", "tabletop",
                          "urn")

# A vision answer below this confidence, that names a category with no
# template, or whose template shape contradicts the measured silhouette is not
# trusted enough to build geometry: it triggers a bounded re-analysis and then
# a request for the user to choose. It is never silently swapped.
VISION_MIN_CONFIDENCE = 0.4
VISION_MAX_ATTEMPTS = 2

VISION_SYSTEM = (
    "You are a vision analyst for a 3D printer. Identify the SINGLE main "
    "physical object in the photograph and describe it. Reply with ONLY a "
    "JSON object, no prose, no markdown:\n"
    '{"category": "<one simple lowercase english word>", '
    '"confidence": <number 0.0-1.0>, '
    '"proportions": {"width": <number>, "height": <number>}, '
    '"components": [{"name": "<visible part>", '
    '"position": "<top|bottom|left|right|center>"}], '
    '"materials": ["visible materials"], "colors": ["visible colors"], '
    '"occlusions": ["surfaces you cannot see"], '
    '"parts": ["visible","parts"], "hidden": ["assumed","parts"]}\n'
    "Use simple category words (car, house, table, chair, bottle, mug, urn, "
    "tree). A table has ONE thin flat top and FOUR separate legs; an urn is "
    "tall and round with NO legs. Describe only what you can actually see and "
    "be honest about uncertainty: if unsure, give your single best category "
    "with a LOW confidence."
)

VISION_RETRY_SYSTEM = (
    "You are a vision analyst for a 3D printer. Your previous answer about "
    "the photograph was rejected because it does not fit the measured shape. "
    "Look at the photograph again carefully and reply with ONLY a JSON object "
    "(same schema as before: category, confidence, proportions, components, "
    "materials, colors, occlusions, parts, hidden).\n"
    "Supported objects you may choose from: {candidates}. Prefer one of those "
    "exact words. If the object really is not one of them, still name your "
    "best category and give a LOW confidence."
)


class VisionError(PipelineError):
    """The vision stage could not reach a plausible, schema-valid category.

    Carries a machine-readable ``info`` dict (needs_category, detected
    category, confidence, attempts, candidates) so a client can ask the user to
    choose the category instead of guessing for them.
    """

    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = dict(info or {})



# ------------------------------------------------------------------ image CV
def _is_bg_factory(bg):
    def is_bg(c):
        return sum(abs(a - b) for a, b in zip(c, bg)) < 60
    return is_bg


def image_stats(path: str, grid: int = 96) -> dict:
    """Deterministic, offline image statistics: silhouette box, aspect ratio,
    dominant colour, dark-round-feature count. No model, no network."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((grid, grid), Image.LANCZOS)
    w, h = img.size
    px = img.load()
    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    bg = max(set(corners), key=corners.count)
    is_bg = _is_bg_factory(bg)

    xs, ys, colors, dark_pts = [], [], [], []
    for y in range(h):
        for x in range(w):
            c = px[x, y]
            if is_bg(c):
                continue
            xs.append(x)
            ys.append(y)
            colors.append(c)
            if max(c) < 90:
                dark_pts.append((x, y))
    total = max(1, w * h)
    if not xs:
        return {"width": w, "height": h, "aspect": round(w / max(1, h), 3),
                "coverage": 0.0, "dominant": "0x888888", "dark_fraction": 0.0,
                "dark_blobs": 0, "silhouette": None}
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    dom = [int(sum(c[i] for c in colors) / len(colors)) for i in range(3)]
    return {
        "width": w, "height": h,
        "aspect": round(bw / bh, 3) if bh else 0.0,
        "coverage": round(len(xs) / float(w * h), 3),
        "dominant": f"0x{dom[0]:02X}{dom[1]:02X}{dom[2]:02X}",
        "dark_fraction": round(len(dark_pts) / max(1, len(xs)), 3),
        "dark_blobs": _count_blobs(dark_pts, max(3, (w * h) // 200),
                                   max(1.0, min(w, h) * 0.18)),
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "size": [w, h],
        "aspect_ratio": round((max(xs) - min(xs) + 1) / max(1, max(ys) - min(ys) + 1), 3),
    }


def _count_blobs(points, min_size, min_dist):
    """Count separated clusters of dark pixels (cheap grid BFS). Used only by
    the offline demonstration path to notice wheel-like features."""
    pts = set(points)
    seen = set()
    centers = []
    for p in list(pts):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        blob = []
        while stack:
            cx, cy = stack.pop()
            blob.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (cx + dx, cy + dy)
                    if q in pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        if len(blob) >= min_size:
            bx = sum(a for a, _ in blob) / len(blob)
            by = sum(b for _, b in blob) / len(blob)
            if all((bx - ox) ** 2 + (by - oy) ** 2 >= min_dist ** 2
                   for ox, oy in centers):
                centers.append((bx, by))
    return len(centers)


def demo_category(stats: dict) -> str:
    """Offline demonstration classifier (documented, deterministic). A wide
    silhouette with >=2 separated dark blobs (wheels) -> car; else object."""
    aspect = stats.get("aspect") or 0.0
    blobs = stats.get("dark_blobs") or 0
    if blobs >= 2 and aspect >= 1.4:
        return "car"
    if blobs >= 4 and 0.6 <= aspect <= 1.4:
        return "house"
    return "object"


# ------------------------------------------------------------------ vision
def _first_json_object(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _clean_category(value) -> str:
    s = re.sub(r"[^A-Za-z]+", "", str(value or "")).lower()
    return s[:32]


def parse_vision_response(raw: str):
    """Parse + strictly validate a vision reply.

    Returns ``(det, errors)`` where ``det`` is the normalised dict the
    pipeline needs (or ``None`` when the reply is unusable) and ``errors`` is a
    list of human-readable schema problems. There is deliberately NO fallback
    to the first word of prose: a model that did not answer in the schema is
    reported as invalid rather than guessed from."""
    obj = _first_json_object(raw)
    if not obj:
        return None, ["reply was not a JSON object"]
    errors = []
    category = _clean_category(obj.get("category"))
    if not category:
        errors.append("missing category")
    try:
        confidence = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        errors.append("confidence was not a number")
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    def _strs(key):
        v = obj.get(key)
        if v is None:
            return []
        if not isinstance(v, list):
            errors.append(f"{key} must be a list")
            return []
        return [str(x) for x in v]

    proportions = obj.get("proportions")
    if isinstance(proportions, dict):
        proportions = {str(k): v for k, v in proportions.items()}
    else:
        if obj.get("proportions") is not None:
            errors.append("proportions must be an object")
        proportions = {}

    components = []
    if isinstance(obj.get("components"), list):
        for c in obj["components"]:
            if isinstance(c, dict):
                components.append({"name": str(c.get("name", "")),
                                   "position": str(c.get("position", ""))})
    elif obj.get("components") is not None:
        errors.append("components must be a list")

    if not category:
        return None, errors
    det = {"category": category, "confidence": confidence,
           "parts": _strs("parts"), "hidden": _strs("hidden"),
           "materials": _strs("materials"), "colors": _strs("colors"),
           "occlusions": _strs("occlusions"),
           "limitations": _strs("limitations"),
           "proportions": {str(k): v for k, v in proportions.items()},
           "components": components, "errors": errors, "raw": raw}
    return det, errors


def describe_object(path: str, model: str, system: str = VISION_SYSTEM,
                    prompt: str = "Identify the main object. JSON only."):
    """Ask a vision model to describe the photo, then schema-validate the reply.

    Returns the normalised detection dict, or ``None`` when the reply is not
    valid JSON in the required schema (the caller reports that clearly)."""
    with open(path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode("ascii")
    raw = ollama_generate(model, system, prompt, images=[b64])
    det, _errors = parse_vision_response(raw)
    return det


def _reconstruction_problem(category: str, confidence: float, stats: dict) -> str:
    """Return a reason string if the vision answer must not be trusted yet.

    A candidate is only accepted when it names a category with a deterministic
    template, its confidence clears the gate, and its template silhouette does
    not contradict the measured photo silhouette. An empty string means
    accepted."""
    if template_key(category) == "object":
        return (f"category {category!r} has no deterministic template "
                f"(supported: {', '.join(SUPPORTED_CATEGORIES)})")
    if confidence < VISION_MIN_CONFIDENCE:
        return (f"confidence {confidence:.2f} is below the "
                f"{VISION_MIN_CONFIDENCE} threshold")
    tmpl = ir_for_category(category)
    bbox = tmpl.get("bbox") or [0, 0, 0]
    tmpl_aspect = (bbox[0] / bbox[1]) if bbox[1] else 0.0
    photo_aspect = stats.get("aspect") if isinstance(stats, dict) else None
    lo, hi = SILHOUETTE_ASPECT_BAND
    if photo_aspect and tmpl_aspect > 0:
        ratio = tmpl_aspect / photo_aspect
        if ratio < lo or ratio > hi:
            return (f"the {category!r} template is {tmpl_aspect:.2f} "
                    f"wide-per-tall but the photo silhouette is "
                    f"{photo_aspect:.2f} (x{ratio:.2f}); the answer conflicts "
                    f"with the image")
    return ""


def analyze_image(path: str, model: str, emit=None) -> dict:
    """Identify the object in `path` with a bounded re-analysis loop.

    Attempt 1 uses the open schema prompt. If the answer is invalid, names an
    unsupported category, is too low-confidence, or conflicts with the measured
    silhouette, ONE targeted re-analysis is attempted (up to
    ``VISION_MAX_ATTEMPTS`` total). If the answer is still untrustworthy the
    caller is told to ask the user via :class:`VisionError` — the category is
    never silently substituted with a different object.

    Returns ``{"det", "stats", "attempts", "resolved"}``.
    """
    emit = emit or (lambda e: None)
    stats = image_stats(path)
    attempts = []
    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        if attempt == 1:
            det = describe_object(path, model)
        else:
            prev = attempts[-1]
            system = VISION_RETRY_SYSTEM.format(
                candidates=", ".join(SUPPORTED_CATEGORIES))
            prompt = (f"Your previous answer was "
                      f"{prev.get('detected') or 'unreadable'!r}. "
                      f"{prev.get('problem') or ''} Look at the photo again "
                      f"and reply with JSON only.")
            det = describe_object(path, model, system=system, prompt=prompt)
        if det is None:
            problem = "the vision model returned no usable JSON schema"
            attempts.append({"attempt": attempt, "detected": None,
                             "confidence": None, "problem": problem})
            emit({"type": "vision_retry", "attempt": attempt,
                  "detected": None, "problem": problem})
            continue
        problem = _reconstruction_problem(det["category"], det["confidence"],
                                          stats)
        attempts.append({"attempt": attempt, "detected": det["category"],
                         "confidence": det["confidence"], "problem": problem})
        if not problem:
            return {"det": det, "stats": stats, "attempts": attempts,
                    "resolved": True}
        emit({"type": "vision_retry", "attempt": attempt,
              "detected": det["category"], "confidence": det["confidence"],
              "problem": problem})
    last = attempts[-1] if attempts else {}
    info = {"needs_category": True,
            "detected_category": last.get("detected"),
            "confidence": last.get("confidence"),
            "attempts": attempts,
            "candidates": list(SUPPORTED_CATEGORIES)}
    raise VisionError(
        "could not identify the object reliably: "
        f"{last.get('problem') or 'no vision answer'}. Select the intended "
        f"category (one of: {', '.join(SUPPORTED_CATEGORIES)}) and retry.",
        info=info)


# ------------------------------------------------------------------ IR build
def ir_from_image(category: str, stats: dict, source: str, confidence=0.5,
                  observed_parts=None, hidden_parts=None,
                  warnings=None) -> dict:
    """Instantiate the canonical IR for `category` and fit it (bounded) to the
    photo silhouette. Hidden geometry is assumed from the template and marked
    unobserved; the assumption is always recorded."""
    base = ir_for_category(category)
    observed_parts = [str(p).lower() for p in (observed_parts or [])]
    hidden_parts = [str(p).lower() for p in (hidden_parts or [])]
    assumptions = list(base.get("assumptions", []))

    tmpl = base.get("bbox") or [0, 0, 0]
    tmpl_aspect = (tmpl[0] / tmpl[1]) if tmpl[1] else 0.0
    obs = stats.get("aspect") if isinstance(stats, dict) else None
    if obs and tmpl_aspect > 0:
        s = max(0.8, min(1.25, obs / tmpl_aspect))
        if abs(s - 1.0) > 0.05:
            assumptions.append(
                f"overall length scaled x{s:.2f} to match the photo silhouette "
                f"aspect ({obs:.2f} vs template {tmpl_aspect:.2f}); occluded "
                f"geometry assumed")
            for c in base["components"]:
                if c.get("role") in _ASPECT_SCALABLE_ROLES:
                    c["dims"][0] = round(c["dims"][0] * s, 4)
                    if c["primitive"] == "cylinder":
                        # keep the round cross-section circular (r_x == r_z)
                        c["dims"][2] = c["dims"][0]
                c["position"][0] = round(c["position"][0] * s, 4)
            base["bbox"] = _object_bbox(base["components"])

    for c in base["components"]:
        cid = str(c["id"]).lower()
        role = str(c.get("role", "")).lower()
        c["observed"] = bool(observed_parts) and (
            any(p in cid for p in observed_parts) or role in observed_parts)

    base["source"] = source
    base["confidence"] = round(float(confidence), 3)
    base["assumptions"] = assumptions + [
        "parts not visible in the photo are assumed from the category template"]
    base["observed"] = {
        "category": bool(observed_parts) or source == "image-manual",
        "visible_parts": observed_parts,
        "hidden_parts": hidden_parts,
    }
    return base


# ------------------------------------------------------------------ run
def run_reconstruction(image_path: str, out_dir: str = "./out_img",
                       category: str = None, vision_model: str = "",
                       emit=None, guided: bool = False,
                       segment_cap: int = 4000, layer_cap: int = 32) -> dict:
    """Semantic image -> 3D: identify the object (vision model, or a documented
    demonstration template), build + validate the canonical IR, freeze it into
    printer primitives, then print it with the shared deterministic slicer.
    Returns the path report augmented with the IR, provenance and fidelity."""
    import morphix_pipeline as P  # late import: avoids an import cycle
    emit = emit or (lambda e: None)
    os.makedirs(out_dir, exist_ok=True)

    stats = image_stats(image_path)
    auto_vision = available_vision_model()
    model = vision_model or VISION_MODEL or auto_vision
    warnings = []
    observed_parts, hidden_parts = [], []
    vision_info = {}

    if category and category.strip():
        cat = category.strip().lower()
        conf, source = 0.6, "image-manual"
        warnings.append("category supplied by the user: geometry comes from the "
                        "category template, not from measurements of the photo")
        # Manual choice still faces the silhouette gate BEFORE any path is
        # compiled: a template that contradicts the photo is a wrong
        # category, and printing 1000 beads of it first would only waste
        # the deadline to reach the same verdict.
        problem = _reconstruction_problem(cat, conf, stats)
        if problem:
            raise PipelineError(
                f"manual category {cat!r} rejected before compiling: "
                f"{problem}. Pick a category whose silhouette fits the "
                f"photo (supported: {', '.join(SUPPORTED_CATEGORIES)}).")
    elif model:
        if not vision_model_supports_images(model):
            raise PipelineError(
                f"configured vision model {model!r} is not a known "
                f"image-capable model (known: {', '.join(VISION_MODELS)}); "
                f"refusing to send the photo to a text-only model. Set "
                f"MORPHIX_VISION_MODEL to a vision model or pass a category "
                f"explicitly.")
        analysis = analyze_image(image_path, model, emit=emit)
        det = analysis["det"]
        stats = analysis["stats"]
        cat = det["category"]
        conf = det["confidence"]
        observed_parts = det["parts"]
        hidden_parts = det["hidden"]
        source = "image-vision"
        vision_info = {"attempts": analysis["attempts"],
                       "proportions": det.get("proportions", {}),
                       "components": det.get("components", []),
                       "materials": det.get("materials", []),
                       "colors": det.get("colors", []),
                       "occlusions": det.get("occlusions", []),
                       "limitations": det.get("limitations", [])}
        if len(vision_info["attempts"]) > 1:
            warnings.append(
                "the vision answer was re-analysed after an implausible first "
                "guess; reported limits: "
                + (", ".join(det.get("occlusions") or det.get("limitations"))
                   or "none"))
        elif det.get("occlusions") or det.get("limitations"):
            warnings.append(
                "vision reported reconstruction limits: "
                + "; ".join(det.get("occlusions", []) + det.get("limitations", [])))
    else:
        cat = demo_category(stats)
        conf, source = 0.25, "image-demo"
        warnings.append(
            "no vision model is configured: this is a deterministic "
            "DEMONSTRATION reconstruction from image statistics, not a "
            "measurement of the photo. Set MORPHIX_VISION_MODEL (e.g. "
            "moondream) for real semantic reconstruction.")

    ir = ir_from_image(cat, stats, source, confidence=conf,
                       observed_parts=observed_parts,
                       hidden_parts=hidden_parts, warnings=warnings)
    errs = validate_ir(ir)
    if errs:
        raise PipelineError("reconstructed object IR failed validation: "
                            + "; ".join(errs))

    emit({"type": "vision_input", "source": source, "category": cat,
          "vision_model": model, "vision_available": bool(auto_vision),
          "detected_parts": observed_parts,
          "confidence": conf,
          "proportions": vision_info.get("proportions", {}),
          "components": vision_info.get("components", []),
          "materials": vision_info.get("materials", []),
          "colors": vision_info.get("colors", []),
          "occlusions": vision_info.get("occlusions", []),
          "attempts": vision_info.get("attempts", [])})
    emit({"type": "ir", "ir": summary_of(ir)})

    actions = ir_to_actions(ir)
    report = P.run_path_print(
        f"reconstruct a {cat} from a photo", out_dir=out_dir, guided=guided,
        emit=emit, segment_cap=segment_cap, layer_cap=layer_cap,
        plan_actions=actions, plan_status=f"IMAGE {cat}",
        spec_name=ir.get("template"), plan_source=source,
        stream_scene=False)

    fidelity = image_fidelity(ir, stats, observed_parts)
    report["mode"] = "image-reconstruct"
    report["ir"] = summary_of(ir)
    report["ir_full"] = ir
    report["image"] = {"path": os.path.basename(image_path), "source": source,
                       "category": cat, "confidence": conf,
                       "vision_model": model,
                       "vision_available": bool(auto_vision),
                       "warnings": warnings, "stats": stats,
                       "vision_analysis": vision_info}
    # Technical action/geometry acceptance stays separate from this semantic
    # reconstruction-quality verdict; both are reported, never conflated.
    report["reconstruction_quality"] = fidelity
    report["image_fidelity"] = fidelity
    # image-specific final quality: geometry verified AND photo-consistent.
    report["quality_accepted"] = bool(
        report.get("quality_accepted") and fidelity.get("verified"))
    report["warnings"] = list(report.get("warnings", [])) + warnings

    with open(os.path.join(out_dir, "object.json"), "w",
              encoding="utf-8") as f:
        json.dump(ir, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"RECONSTRUCT: {cat} [{source}] -> "
          f"{'OK' if report['ok'] else 'FAIL'} "
          f"(quality {'accepted' if report['quality_accepted'] else 'rejected'})")
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Semantic image -> 3D object reconstruction (printer mode)")
    p.add_argument("image")
    p.add_argument("--out", default="./out_recon")
    p.add_argument("--category", default="",
                   help="override the object category (skips the vision model)")
    p.add_argument("--vision-model", default="")
    a = p.parse_args()
    rep = run_reconstruction(a.image, a.out, category=a.category or None,
                             vision_model=a.vision_model)
    print(json.dumps({"ok": rep["ok"], "complete": rep["complete"],
                      "quality_accepted": rep["quality_accepted"],
                      "source": rep["image"]["source"],
                      "category": rep["image"]["category"],
                      "warnings": rep["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
