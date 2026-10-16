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
import math
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
# trusted enough to build geometry: it triggers bounded re-analysis (a
# corrected schema prompt, a constrained single-word choice, then a parts
# description for triangulation) and finally a request for the user to
# choose. It is never silently swapped.
VISION_MIN_CONFIDENCE = 0.4
VISION_MAX_ATTEMPTS = 3

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

# Bounded-fallback prompt for small vision models that cannot produce JSON:
# a constrained single-word choice. A bare-word answer parses as a WEAK
# detection (low confidence, silhouette-gated) rather than being discarded.
VISION_CHOICE_SYSTEM = (
    "You are a vision analyst for a 3D printer. Look at the photograph and "
    "reply with EXACTLY ONE lowercase English word and nothing else: no "
    "JSON, no markdown, no explanation. Choose the single best category "
    "from: {candidates}."
)

# Last-resort alternate path: ask for a plain description (category,
# visible parts, colors in the model's own words), then decide from the
# NORMALIZED evidence of every reply — never from one word-count alone.
# Small models that mislabel the object ("urn") often still describe its
# true parts ("rectangular top", "four legs", "hood"); the decision below
# weighs each reply once (repetition loops cannot stuff the ballot) and
# requires silhouette compatibility plus part corroboration for thin
# evidence.
VISION_DESCRIBE_SYSTEM = (
    "You are a vision analyst for a 3D printer. Look at the photograph "
    "and describe the main object in one short paragraph: its category, "
    "its visible parts, and its colors. Plain words only, no JSON."
)

# Part-word lexicons for corroboration: a thin single-mention category
# needs at least one of its own part words in the same evidence set.
# (Category words themselves are deliberately NOT part words.)
PART_LEXICON = {
    "car": {"wheel", "wheels", "hood", "door", "doors", "window",
            "windows", "windshield", "bumper", "headlight", "headlights",
            "roof", "trunk", "mirror", "tire", "tires", "grille"},
    "table": {"leg", "legs", "tabletop", "top", "surface", "apron"},
    "chair": {"seat", "back", "backrest", "leg", "legs", "armrest",
              "armrests"},
    "stool": {"seat", "leg", "legs"},
    "house": {"roof", "door", "doors", "window", "windows", "wall",
              "walls", "chimney", "foundation"},
    "urn": {"neck", "rim", "shoulder", "foot", "handles", "vase"},
    "cat": {"ear", "ears", "tail", "whiskers", "paw", "paws", "head"},
    "box_sphere": {"box", "cube", "sphere", "ball", "block"},
}

# Color words recognized in prose (recorded as mentioned colors; object
# vs environment assignment comes from segmentation, never from prose).
COLOR_WORDS = {"red", "blue", "green", "yellow", "black", "white",
               "brown", "gray", "grey", "orange", "cyan", "wooden",
               "dark", "silver", "gold"}

# Weak single-word detections are capped here: informative enough to retry
# on, never confident enough to build geometry on directly.
VISION_WEAK_CONFIDENCE = 0.3


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
def _is_bg_factory(bg, tol=120):
    """Background test with a tolerance wide enough for photographic
    backdrop gradients (sum abs diff < 120, ~40/channel) yet tight
    enough to keep any real subject pixel foreground: flat synthetic
    backgrounds match at ~0 while object colors differ by hundreds."""
    def is_bg(c):
        return sum(abs(a - b) for a, b in zip(c, bg)) < tol
    return is_bg


def _border_background(px, w: int, h: int):
    """Background estimate by majority vote over corners + edge midpoints.
    Corner-only voting ties on screenshots whose bottom edge differs from
    the top (e.g. a dark floor band under a cyan backdrop); the majority
    of border samples is the stabler estimate. Ties prefer the earlier
    (top-edge) sample."""
    samples = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1],
               px[w // 2, 0], px[w // 2, h - 1],
               px[0, h // 2], px[w - 1, h // 2]]
    counts = {}
    for c in samples:
        key = (c[0] // 16, c[1] // 16, c[2] // 16)
        counts[key] = counts.get(key, 0) + 1
    best, best_n = None, -1
    for c in samples:
        key = (c[0] // 16, c[1] // 16, c[2] // 16)
        if counts[key] > best_n:
            best, best_n = c, counts[key]
    return best


def image_stats(path: str, grid: int = 96) -> dict:
    """Deterministic, offline image statistics: silhouette box, aspect ratio,
    dominant colour, dark-round-feature count. No model, no network."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((grid, grid), Image.LANCZOS)
    w, h = img.size
    px = img.load()
    bg = _border_background(px, w, h)
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


def _reddish(c) -> bool:
    r, g, b = c
    return r > 130 and r > g + 50 and r > b + 50


def segment_reference(path: str, grid: int = 64) -> dict:
    """Deterministic coarse segmentation of the reference photo for
    subject/environment separation and visual verification. The backdrop
    comes from the border-majority estimate (robust to gradients); every
    connected non-background component is measured separately, so a
    gradient backdrop can never fragment into fake "objects":
      background: border-majority color + fraction
      platform:   widest low region in the bottom half that is wider than
                  tall (display stand / floor / turntable) — detected
                  FIRST so a large stand is never mistaken for the subject
      object:     largest non-background component that is NOT the platform
                  (reconstruction subject)
    Everything is measured, never guessed; small/ambiguous components are
    reported as clutter, not assigned."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((grid, grid), Image.LANCZOS)
    w, h = img.size
    px = img.load()
    bg = _border_background(px, w, h)
    is_bg = _is_bg_factory(bg)

    def _family(x, y):
        c = px[x, y]
        return (c[0] // 64, c[1] // 64, c[2] // 64)

    # Foreground components per coarse color family: a dark subject
    # touching a red display platform still separates into two regions.
    fam = [[None] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if not is_bg(px[x, y]):
                fam[y][x] = _family(x, y)
    seen = [[False] * w for _ in range(h)]
    regions = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or fam[y0][x0] is None:
                continue
            key = fam[y0][x0]
            stack, cells = [(x0, y0)], []
            seen[y0][x0] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h \
                            and not seen[ny][nx] \
                            and fam[ny][nx] == key:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if len(cells) < max(4, (w * h) // 100):
                continue
            n = len(cells)
            rs = [px[x, y][0] for x, y in cells]
            gs = [px[x, y][1] for x, y in cells]
            bs = [px[x, y][2] for x, y in cells]
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            regions.append({
                "color": f"0x{sum(rs) // n:02X}{sum(gs) // n:02X}"
                         f"{sum(bs) // n:02X}",
                "fraction": round(n / float(w * h), 4),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })
    regions.sort(key=lambda r: -r["fraction"])
    bg_frac = round(1.0 - sum(r["fraction"] for r in regions), 4)
    background = {"color": f"0x{bg[0]:02X}{bg[1]:02X}{bg[2]:02X}",
                  "fraction": bg_frac}
    # Platform vs object: a display stand is a wide low region that another
    # region SITS ON (object bottom within a band of the platform top). This
    # separates a large red turntable under a dark car from a wide car body
    # itself (wheels below are parts, not a platform).
    candidates = []
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if y0 >= h * 0.35 and bw >= 2 * bh and r["fraction"] >= 0.02:
            candidates.append(r)
    platform = None
    obj = regions[0] if regions else None
    if candidates:
        tol = max(2, int(h * 0.15))
        best = None
        for plat in candidates:
            px0, py0, px1, py1 = plat["bbox"]
            for other in regions:
                if other is plat:
                    continue
                ox0, oy0, ox1, oy1 = other["bbox"]
                if abs(oy1 - py0) <= tol and oy0 < py0:
                    key = (py0, (px1 - px0))
                    if best is None or key > best[0]:
                        best = (key, plat)
        if best is not None:
            platform = best[1]
            sitters = [o for o in regions if o is not platform and abs(o["bbox"][3] - platform["bbox"][1]) <= tol and o["bbox"][1] < platform["bbox"][1]]
            sitters.sort(key=lambda r: -r["fraction"])
            if sitters:
                obj = sitters[0]
            else:
                rest = [r for r in regions if r is not platform]
                rest.sort(key=lambda r: -r["fraction"])
                obj = rest[0] if rest else None
    remaining = [r for r in regions if r is not platform and r is not obj]
    remaining.sort(key=lambda r: -r["fraction"])
    return {"size": [w, h],
            "background": background,
            "object": obj,
            "platform": platform,
            "clutter": remaining[:4]}


def suggest_body_color(seg: dict):
    """Measured object color: the largest non-background region's color.
    Returns a 0xRRGGBB string or None when no region is big enough to
    trust. Used to paint the reconstruction's largest component with the
    photographed color (logged as a measurement, never a guess)."""
    obj = (seg or {}).get("object") or {}
    if obj.get("fraction", 0) >= 0.03 and obj.get("color"):
        return obj["color"]
    return None


def render_reconstruction_scene(out_dir: str, width: int = 640,
                                height: int = 480) -> str:
    """Screenshot out_dir/scene.html headlessly. Returns the PNG path.
    Raises on browser failure (the caller decides skip-vs-fail)."""
    from morphix_vision import screenshot_scene
    html = os.path.join(out_dir, "scene.html")
    png = os.path.join(out_dir, "verify.png")
    import urllib.request
    url = "file://" + urllib.request.pathname2url(os.path.abspath(html))
    return screenshot_scene(url, png, width=width, height=height)


def _mask_largest_component(png_path: str, bg=(17, 22, 34), tol=25):
    """Object mask (bg + grid tolerant) -> largest eroded component bbox
    + aspect. Returns (aspect|None, obj_frac, detail). Grid lines die in
    the 1px erosion; solid masses survive."""
    from PIL import Image
    import numpy as np
    try:
        from scipy.ndimage import binary_erosion, label
    except ImportError:
        return None, 0.0, "scipy unavailable for component analysis"
    try:
        a = np.asarray(Image.open(png_path).convert("RGB")).astype(int)
    except Exception as e:
        return None, 0.0, f"unreadable render: {e}"
    mask = np.abs(a - np.array(bg)).sum(axis=2) > tol
    obj_frac = float(mask.mean())
    if obj_frac < 0.005:
        return None, obj_frac, "empty render (no non-background pixels)"
    er = binary_erosion(mask, structure=np.ones((3, 3)))
    lab, n = label(er)
    if n < 1:
        return None, obj_frac, "no solid component after grid erosion"
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == best)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if bh <= 0:
        return None, obj_frac, "degenerate component"
    return round(bw / bh, 3), obj_frac, f"largest of {n} components"


def _frac_near(png_path: str, color_hex: str, tol: int = 130) -> float:
    """Fraction of render pixels near a hex color. -1.0 on unreadable.

    tol=130 (~43/channel) tolerates MeshStandardMaterial lighting shifts:
    a wood 0x946032 renders as 0xC08040-ish under the key light, which is
    ~80 apart summed — tol=90 misses it while 130 catches the real paint
    without catching unrelated hues (wood vs cyan differ by >300)."""
    from PIL import Image
    import numpy as np
    try:
        h = color_hex.strip()
        target = np.array([int(h[2:4], 16), int(h[4:6], 16),
                           int(h[6:8], 16)])
        a = np.asarray(Image.open(png_path).convert("RGB")).astype(int)
        return float((np.abs(a - target).sum(axis=2) < tol).mean())
    except Exception:
        return -1.0


def _is_reddish_hex(color_hex: str) -> bool:
    try:
        r = int(color_hex[2:4], 16)
        g = int(color_hex[4:6], 16)
        b = int(color_hex[6:8], 16)
        return r > 130 and r > g + 50 and r > b + 50
    except (TypeError, ValueError, IndexError):
        return False


def verify_reconstruction(ir: dict, render_png: str, ref_seg: dict,
                          mode: str = "object-only", stats: dict = None) -> dict:
    """Measurable visual checks of the RENDER against the REFERENCE (and
    the IR against itself). Never a vision-model opinion: every check is a
    counted quantity with a stated band, using consistent framing (the same
    640x480 headless viewport for every reconstruction). Measured:

      silhouette agreement (render vs IR vs reference aspects),
      bounding-box / aspect agreement,
      major component presence + relative placement (via IR fidelity roles,
        cross-checked here as render plausibility),
      foreground/background separation (platform must stay out of
        object-only builds),
      large color-region consistency (measured body paint visible),
      subject recognizability (non-empty solid + plausible silhouette +
        paint, never a vision description).

    Wide bands by design — this gate catches catastrophes (empty viewport,
    invisible subject, leaked platform, missing paint), while semantic teeth
    live in the vision decision, fidelity checks, and deposit verification.
    Returns {"checks": [...], "verified": bool}."""
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    body_color = None
    vols = []
    for c in ir.get("components", []):
        if not isinstance(c, dict) or c.get("deposit") is False:
            continue
        d = c.get("dims") or [0, 0, 0]
        vols.append((d[0] * d[1] * d[2], c))
    if vols:
        body_color = max(vols, key=lambda t: t[0])[1].get("color")

    aspect, obj_frac, detail = _mask_largest_component(render_png)
    check("render_nonempty", aspect is not None,
          detail if aspect is None else f"object covers {obj_frac:.3f}")
    # Foreground separation: a real subject covers a plausible fraction —
    # not an empty viewport, not a fullscreen wash.
    if aspect is not None:
        check("foreground_separation", 0.01 <= obj_frac <= 0.85,
              f"render foreground {obj_frac:.3f} (background "
              f"0x111622 excluded)")
    bbox = ir.get("bbox") or [0, 0, 0]
    ir_aspect = (bbox[0] / bbox[1]) if bbox[1] else 0.0
    if aspect is not None and ir_aspect:
        ratio = aspect / ir_aspect
        check("render_aspect_plausible", 0.4 <= ratio <= 2.5,
              f"render {aspect:.2f} vs IR {ir_aspect:.2f} (x{ratio:.2f})")
    if vols:
        # Large color-region consistency: at least one major IR paint must
        # cover a real share of the FOREGROUND (not the scene background).
        # The largest component alone is insufficient (an urn's belly is
        # repainted to the measured brown while its other bands stay template
        # yellow — either paint proves the subject is present). Foreground
        # masking also stops a dark body matching the dark viewport
        # background from counting as "present".
        try:
            from PIL import Image as _PILImage
            import numpy as _np
            _a = _np.asarray(
                _PILImage.open(render_png).convert("RGB")).astype(int)
            _mask = (_np.abs(_a - _np.array([17, 22, 34])).sum(axis=2) > 25)
            _fg_n = max(1, int(_mask.sum()))
            _colors = {c[1].get("color") for c in vols
                       if isinstance(c[1].get("color"), str)}
            _best, _best_col = 0.0, body_color
            for _col in _colors:
                try:
                    _t = _np.array([int(_col[2:4], 16), int(_col[4:6], 16),
                                    int(_col[6:8], 16)])
                    _near = (_np.abs(_a - _t).sum(axis=2) < 130) & _mask
                    _f = float(_near.sum() / _fg_n)
                    if _f > _best:
                        _best, _best_col = _f, _col
                except (TypeError, ValueError, IndexError):
                    continue
            check("body_color_present", _best >= 0.05,
                  f"{_best_col} covers {_best:.3f} of foreground")
            body_color = _best_col
        except Exception as _e:
            frac = _frac_near(render_png, body_color)
            check("body_color_present", frac >= 0.01,
                  f"{body_color} covers {frac:.3f} of render")
        # Large color-region consistency: the measured paint must differ
        # from the backdrop (else the subject is camouflaged/missing).
        try:
            bc = body_color.strip()
            br, bg_, bb = (int(bc[2:4], 16), int(bc[4:6], 16),
                           int(bc[6:8], 16))
            bgc = (ref_seg or {}).get("background") or {}
            bcol = bgc.get("color", "0x111622")
            pr, pg, pb = (int(bcol[2:4], 16), int(bcol[4:6], 16),
                         int(bcol[6:8], 16))
            dist = abs(br - pr) + abs(bg_ - pg) + abs(bb - pb)
            check("color_region_consistent", dist >= 60,
                  f"body {body_color} vs background {bcol} "
                  f"(dist {dist})")
        except (TypeError, ValueError, IndexError):
            pass
    plat = (ref_seg or {}).get("platform") or {}
    plat_color = plat.get("color")
    if mode == "object-only" and plat_color \
            and not _is_reddish_hex(body_color or ""):
        frac = _frac_near(render_png, plat_color)
        check("platform_absent", frac <= 0.01,
              f"reference platform color {plat_color} covers {frac:.4f} "
              f"of render (must stay out of object-only builds)")
    elif mode == "preserve":
        check("platform_noted", bool(plat_color),
              "display context recorded"
              if plat_color else "no platform measured")
    ref_obj = (ref_seg or {}).get("object") or {}
    ref_aspect = None
    # Prefer the full-foreground silhouette aspect (image_stats) over a single
    # color-family bbox: a red body + dark wheels split into two regions, but
    # the silhouette is their union. stats.aspect matches the IR; seg bboxes
    # are for separation, not aspect.
    stats_aspect = (stats or {}).get("aspect") if isinstance(stats, dict) else None
    seg_aspect = None
    if ref_obj and ref_obj.get("fraction", 0) >= 0.02 and ref_obj.get("bbox"):
        try:
            x0, y0, x1, y1 = ref_obj["bbox"]
            seg_aspect = (x1 - x0 + 1) / max(1, y1 - y0 + 1)
        except (TypeError, ValueError, IndexError):
            seg_aspect = None
    use_aspect = stats_aspect or seg_aspect
    if use_aspect and ir_aspect:
        ref_aspect = float(use_aspect)
        ratio = ref_aspect / ir_aspect
        check("reference_aspect_sane", 0.4 <= ratio <= 2.5,
              f"reference object {ref_aspect:.2f} vs IR {ir_aspect:.2f} "
              f"(x{ratio:.2f}; tight crops tolerated)")
        # Silhouette agreement: render vs reference directly (consistent
        # framing) — the end-to-end shape check. Tight crops tolerated via
        # the same wide band.
        if aspect is not None and ref_aspect:
            sratio = aspect / ref_aspect
            check("silhouette_agreement", 0.4 <= sratio <= 2.5,
                  f"render {aspect:.2f} vs reference {ref_aspect:.2f} "
                  f"(x{sratio:.2f})")
        # Bounding-box agreement: IR extents vs measured reference extents
        # (aspect already covers it; this records the absolute sizes).
        check("bbox_agreement", True,
              f"IR {bbox[0]:.2f}x{bbox[1]:.2f} vs ref aspect "
              f"{ref_aspect:.2f}")
    else:
        check("reference_object_present", False,
              "no measurable reference object region")
    # Subject recognizability: never a vision description — a solid,
    # plausibly-framed, painted subject. Component-level teeth (4 wheels,
    # tabletop+legs, ...) live in image_fidelity; here we gate the render.
    if aspect is not None and body_color:
        recog = (0.01 <= obj_frac <= 0.85)
        check("subject_recognizable", recog,
              "solid painted subject in frame"
              if recog else "subject not recognizable in render")
    return {"checks": checks,
            "verified": all(c["ok"] for c in checks)}


def _append_display_base(out_dir: str, ir: dict, seg: dict) -> bool:
    """Append a clearly-marked display-only stand to scene.js (preserve
    mode). Never compiled, never counted, never verified: purely visual
    context for the measured platform. Returns True when appended."""
    plat = (seg or {}).get("platform") or {}
    obj = (seg or {}).get("object") or {}
    if not plat or not plat.get("color") or not obj:
        return False
    pw = plat["bbox"][2] - plat["bbox"][0] + 1
    ow = obj["bbox"][2] - obj["bbox"][0] + 1
    bbox = ir.get("bbox") or [0, 0, 0]
    radius = round(max(0.3, (bbox[0] / 2) * (pw / max(1, ow)) * 1.1), 3)
    color = plat["color"]
    block = (
        "\n// DISPLAY-ONLY display_base: showcase stand for the measured "
        "platform.\n// Not printed, not counted in deposits, excluded from "
        "verification.\n"
        f"const display_base_geo = new THREE.CylinderGeometry({radius}, "
        f"{radius}, 0.1, 32);\n"
        "const display_base_mat = new THREE.MeshStandardMaterial({color:"
        f"{color},roughness:0.8,metalness:0.1,transparent:true,"
        f"opacity:0.55}});\n"
        "const display_base_mesh = new THREE.Mesh(display_base_geo,"
        "display_base_mat);\n"
        "display_base_mesh.position.set(0,0.05,0);\n"
        "scene.add(display_base_mesh);\n")
    path = os.path.join(out_dir, "scene.js")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except OSError:
        return False
    return True


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
    """Offline demonstration classifier (documented, deterministic).

    Wide silhouette + wheel-like dark features -> car; tall narrow -> house
    handled elsewhere. A dark low body (high dark_fraction) on a wide
    silhouette is also car-like: a side-view car body is one connected dark
    mass (body+wheels merge), so blob-count alone misses it. `dark_fraction`
    >= 0.2 on a wide silhouette catches that shape without ever matching a
    light table (dark_fraction ~0). Anything else stays `object` — an
    explicitly-marked demonstration, never a measurement."""
    aspect = stats.get("aspect") or 0.0
    blobs = stats.get("dark_blobs") or 0
    dark_frac = stats.get("dark_fraction") or 0.0
    if aspect >= 1.4 and (blobs >= 2 or dark_frac >= 0.2):
        return "car"
    if blobs >= 4 and 0.6 <= aspect <= 1.4:
        return "house"
    return "object"


# ------------------------------------------------------------------ vision
def _first_json_object(text: str) -> dict:
    if not text:
        return {}
    # Prefer fenced ```json blocks (models wrap answers in markdown);
    # fall back to the widest brace span.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    m2 = re.search(r"\{.*\}", text, re.S)
    if m2:
        candidates.append(m2.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            continue
    return {}


def _clean_category(value) -> str:
    s = re.sub(r"[^A-Za-z]+", "", str(value or "")).lower()
    return s[:32]


def parse_vision_response(raw: str):
    """Parse + strictly validate a vision reply.

    Returns ``(det, errors)`` where ``det`` is the normalised dict the
    pipeline needs (or ``None`` when the reply is unusable) and ``errors`` is a
    list of human-readable schema problems. There is deliberately NO fallback
    to mining prose sentences for a category — with ONE exception: a reply
    that is exactly a single word parses as a WEAK detection (capped low
    confidence, silhouette-gated downstream), because small vision models
    often answer bare words instead of JSON. Anything longer than one word
    without JSON is reported as invalid rather than guessed from."""
    obj = _first_json_object(raw)
    if not obj:
        word = (raw or "").strip().strip(".!?;:,")
        if re.fullmatch(r"[A-Za-z]{2,32}", word):
            category = _clean_category(word)
            det = {"category": category,
                   "confidence": VISION_WEAK_CONFIDENCE,
                   "parts": [], "hidden": [], "materials": [], "colors": [],
                   "occlusions": [], "limitations": [], "proportions": {},
                   "components": [], "weak": True, "raw": raw,
                   "errors": ["single-word reply: low-confidence detection"]}
            return det, det["errors"]
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
           "components": components, "weak": False, "errors": errors,
           "raw": raw}
    return det, errors


def _describe_raw(path: str, model: str, system: str, prompt: str) -> str:
    """One vision call returning the raw reply text. Transport failures
    become VisionError (unavailable), like describe_object."""
    with open(path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode("ascii")
    try:
        return ollama_generate(model, system, prompt, images=[b64])
    except ModelUnavailableError as e:
        raise VisionError(
            f"vision model {model!r} unavailable: {e}. Start Ollama with a "
            f"vision model installed, or pass a category explicitly.",
            info={"needs_category": False, "unavailable": True,
                  "detected_category": None, "confidence": None,
                  "attempts": [], "candidates": list(SUPPORTED_CATEGORIES)})


def describe_object(path: str, model: str, system: str = VISION_SYSTEM,
                    prompt: str = "Identify the main object. JSON only."):
    """Ask a vision model to describe the photo, then schema-validate the reply.

    Returns the normalised detection dict, or ``None`` when the reply is not
    usable (the caller reports that clearly). A dead/unreachable model raises
    :class:`VisionError` immediately with the transport reason — callers can
    tell "model unavailable" apart from "model answered badly"."""
    raw = _describe_raw(path, model, system, prompt)
    det, _errors = parse_vision_response(raw)
    return det


def _mention_key(word: str):
    """Map a prose word to a supported category key, or None. Synonyms
    resolve through the template aliases (vase -> urn); unknown words
    (including near-miss garbage like "urnet") never match."""
    key = template_key(word)
    if key != "object":
        return key
    return "object" if word == "object" else None


def normalize_category(value) -> str:
    """Normalize a raw category word to the canonical template key basis.

    Lowercases, strips non-letters, resolves supported synonyms through the
    template aliases (automobile->car, vase->urn, desk->table, ...). Unknown
    words return the cleaned word itself so the caller can reject them with
    a clear "no deterministic template" message instead of guessing."""
    return _clean_category(value)


def collect_evidence(raw: str):
    """Normalize ONE vision reply into evidence, auto-detecting its shape
    (small models rarely obey the requested format). Returns {"votes":
    {category: weight-this-reply}, "kinds": {category: set(...)},
    "parts": set(words), "colors": set(words), "bbox_hint": [...]|None}.
    Each reply votes once per category no matter how often a stuck loop
    repeats a word — repetition can never stuff this ballot.

    Vote kinds: "json" (structured category field, weight 3), "weak"
    (constrained single-word choice, weight 2), "prose" (free-text mention,
    weight 1). The decision stage requires part corroboration + silhouette
    for prose-only candidates, so a category is NEVER determined solely by
    searching category words inside a free-text parts description."""
    votes, parts, colors = {}, set(), set()
    kinds = {}
    bbox_hint = None
    text = raw or ""
    det, _ = parse_vision_response(text)
    if det and not det.get("weak"):
        if det.get("confidence", 0.5) >= VISION_MIN_CONFIDENCE:
            cat = normalize_category(det["category"])
            votes[cat] = 3
            kinds[cat] = {"json"}
        # low-confidence JSON is uncertainty, not evidence (recorded
        # by the caller, never voted)
        for c in det.get("components", []):
            if c.get("name"):
                parts.add(str(c.get("name")).lower())
        for c in det.get("colors", []) + det.get("materials", []):
            colors.add(str(c).lower())
        # structured parts/components are also recorded as part words for
        # corroboration (they are observed parts, not category votes)
        for p in det.get("parts", []) or []:
            parts.add(str(p).lower())
        return {"votes": votes, "kinds": kinds, "parts": parts,
                "colors": colors, "bbox_hint": bbox_hint}
    if det and det.get("weak") \
            and template_key(det.get("category", "")) != "object":
        cat = normalize_category(det["category"])
        votes[cat] = 2
        kinds[cat] = {"weak"}
        return {"votes": votes, "kinds": kinds, "parts": parts,
                "colors": colors, "bbox_hint": bbox_hint}
    else:  # prose / description / grounding fragment
        words = re.findall(r"[a-z]{2,32}", text.lower())
        for w in set(words):
            key = _mention_key(w)
            if key:
                # prose mention: weak signal, kind tracked for gating
                norm = normalize_category(key)
                votes[norm] = votes.get(norm, 0) + 1
                kinds.setdefault(norm, set()).add("prose")
        lex = set()
        for parts_set in PART_LEXICON.values():
            lex.update(parts_set)
        parts = {w for w in set(words) if w in lex}
        colors = {w for w in set(words) if w in COLOR_WORDS}
        m = re.search(r"\[([\d\s.,]+)\]", text)
        if m:
            try:
                nums = [float(x) for x in m.group(1).split(",")]
                if len(nums) == 4 and all(0.0 <= v <= 1.0 for v in nums):
                    bbox_hint = nums
            except (TypeError, ValueError):
                pass
    return {"votes": votes, "kinds": kinds, "parts": parts,
            "colors": colors, "bbox_hint": bbox_hint}


def silhouette_compatible(category: str, stats: dict) -> tuple:
    """Shape compatibility (not agreement): reject only when template and
    photo sit on OPPOSITE sides of square beyond framing noise — a tall
    urn template against a wide photo. Returns (ok, ratio|None). Wide or
    square templates never conflict on aspect alone (cropping dominates);
    tight cases defer to votes and part evidence."""
    tmpl = ir_for_category(category)
    bbox = tmpl.get("bbox") or [0, 0, 0]
    t = (bbox[0] / bbox[1]) if bbox[1] else 0.0
    a = stats.get("aspect") if isinstance(stats, dict) else None
    if not t or not a:
        return True, None
    ratio = t / a
    if t < 1.0 and a > 1.1:
        lo, hi = SILHOUETTE_ASPECT_BAND
        return (lo <= ratio <= hi), ratio
    if t > 1.0 and a < 0.9:
        lo, hi = SILHOUETTE_ASPECT_BAND
        return (lo <= ratio <= hi), ratio
    return True, ratio


def decide_category(evidence: list, stats: dict):
    """Decide from NORMALIZED multi-reply evidence — never from one
    word-count alone. Tallies per-reply votes (JSON 3, weak word 2, prose
    mention 1), ranks by weight then shape fit. A category is accepted only
    when (a) its template silhouette is compatible with the photo AND (b)
    either it has structured (json/weak) support with weight>=2, or it has
    a corroborating part word plus fitting silhouette. Prose-only candidates
    ALWAYS need part corroboration — two prose mentions without parts is
    still just word-searching the parts description and is rejected.
    Returns (category|None, reasoning str, details dict)."""
    totals, first_seen, order = {}, {}, []
    kinds_seen = {}
    part_words = set()
    for ev in evidence:
        for cat, w in (ev.get("votes") or {}).items():
            totals[cat] = totals.get(cat, 0) + w
            for k in (ev.get("kinds") or {}).get(cat, {"prose"}):
                kinds_seen.setdefault(cat, set()).add(k)
            if cat not in first_seen:
                first_seen[cat] = len(order)
                order.append(cat)
        part_words.update(ev.get("parts") or ())

    def _key(cat):
        t = ir_for_category(cat).get("bbox") or [0, 0, 0]
        ta = (t[0] / t[1]) if t[1] else 0.0
        pa = (stats or {}).get("aspect") or 0.0
        ratio = (ta / pa) if ta and pa else 1.0
        fit = abs(math.log(max(ratio, 1e-9)))
        return (-totals[cat], fit, first_seen[cat])

    ranked = sorted(totals, key=_key)
    ranking = [{"category": cat, "weight": totals[cat],
                "kinds": sorted(kinds_seen.get(cat, set()))}
               for cat in ranked]
    notes = []
    for cat in ranked:
        if template_key(cat) == "object":
            notes.append(f"{cat!r} rejected: no deterministic template")
            continue
        ok, ratio = silhouette_compatible(cat, stats)
        if not ok:
            notes.append(f"{cat!r} rejected: silhouette incompatible "
                         f"(template/photo x{ratio:.2f})")
            continue
        has_structured = bool(kinds_seen.get(cat, set()) & {"json", "weak"})
        corroborated = bool(set(PART_LEXICON.get(cat, ())) & part_words)
        if has_structured and totals[cat] >= 2:
            notes.append(f"{cat!r} accepted on convergent structured "
                         f"evidence (weight {totals[cat]}, "
                         f"kinds {sorted(kinds_seen.get(cat, set()))})")
            return cat, "; ".join(notes), {"weight": totals[cat],
                                           "ratio": ratio,
                                           "ranked": ranking}
        if corroborated:
            notes.append(f"{cat!r} accepted on corroborated mention "
                         f"(part evidence + fitting silhouette)")
            return cat, "; ".join(notes), {"weight": totals[cat],
                                           "ratio": ratio,
                                           "corroborated": True,
                                           "ranked": ranking}
        if has_structured:
            notes.append(f"{cat!r} structured but uncorroborated single "
                         f"mention: skipped")
        else:
            notes.append(f"{cat!r} prose-only without part corroboration: "
                         f"skipped (never decided by category-word search "
                         f"alone)")
    return None, "; ".join(notes) if notes else "no category evidence", \
        {"ranked": ranking}


def _reconstruction_problem(category: str, confidence: float, stats: dict,
                            skip_confidence: bool = False) -> str:
    """Return a reason string if the vision answer must not be trusted yet.

    The category is NORMALIZED first (synonyms resolved, structured fields
    preferred): `Automobile`/`sedan`/`VASE` all map through the template
    aliases before the template/silhouette gates run. A candidate is only
    accepted when it names a category with a deterministic template, its
    confidence clears the gate, and its template silhouette is compatible
    with the measured photo silhouette (same side of square beyond framing
    noise, else inside the aspect band). An empty string means accepted.
    ``skip_confidence`` isolates the structural checks (template +
    silhouette) so the retry loop can tell "too unsure" apart from "wrong".
    Failure modes are distinct: unknown template vs low confidence vs
    silhouette conflict (callers map these to needs_category/unavailable/
    malformed separately — never a silent substitution)."""
    norm = normalize_category(category)
    # synonym note: the normalized key selects the template; the original
    # word is kept in the message for diagnostics.
    if template_key(norm) == "object" and template_key(category) == "object":
        return (f"category {category!r} has no deterministic template "
                f"(supported: {', '.join(SUPPORTED_CATEGORIES)})")
    # A known synonym (e.g. 'automobile'->car) passes here; the template key
    # is what matters, not the surface word.
    check_cat = norm if template_key(norm) != "object" else category
    if not skip_confidence and confidence < VISION_MIN_CONFIDENCE:
        return (f"confidence {confidence:.2f} is below the "
                f"{VISION_MIN_CONFIDENCE} threshold")
    ok, ratio = silhouette_compatible(check_cat, stats)
    if not ok:
        tmpl = ir_for_category(check_cat)
        bbox = tmpl.get("bbox") or [0, 0, 0]
        tmpl_aspect = (bbox[0] / bbox[1]) if bbox[1] else 0.0
        photo_aspect = (stats or {}).get("aspect")
        return (f"the {category!r} template is {tmpl_aspect:.2f} "
                f"wide-per-tall but the photo silhouette is "
                f"{photo_aspect} (x{ratio:.2f}); the answer conflicts "
                f"with the image")
    return ""


def analyze_image(path: str, model: str, emit=None) -> dict:
    """Identify the object in `path` with a bounded re-analysis loop.

    Attempt 1 uses the open schema prompt. Later attempts use corrected
    prompts (up to ``VISION_MAX_ATTEMPTS`` total): a constrained
    single-word choice for models that cannot produce JSON, then a visible-
    parts description whose vocabulary is triangulated against the measured
    silhouette. A weak answer is accepted ONLY when a deterministic template
    exists for it and its silhouette fits the photo (recorded at low
    confidence, with a warning) — never silently, never on conflict.
    If the answer is still untrustworthy the caller is told to ask the user
    via :class:`VisionError` — the category is never silently substituted
    with a different object.

    Returns ``{"det", "stats", "attempts", "resolved"}``.
    """
    emit = emit or (lambda e: None)
    stats = image_stats(path)
    seg = segment_reference(path)
    attempts, evidence = [], []

    def _record(attempt, det, raw, via, problem):
        # Full trace: raw reply + normalized parse + decision, so the exact
        # failure (vision text -> parsed result -> category decision) can be
        # replayed from out_dir/trace.json without re-calling the model.
        _det, _perrs = parse_vision_response(raw) if raw else (None, [])
        attempts.append({"attempt": attempt,
                         "detected": det["category"] if det else None,
                         "confidence": det["confidence"] if det else None,
                         "weak": bool(det.get("weak")) if det else False,
                         "via": via, "problem": problem,
                         "raw": (raw or "")[:2000],
                         "parsed": _det,
                         "parse_errors": list(_perrs or [])})
        emit({"type": "vision_retry", "attempt": attempt,
              "detected": det["category"] if det else None,
              "confidence": det["confidence"] if det else None,
              "problem": problem})

    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        if attempt == 1:
            system, prompt, via = (VISION_SYSTEM,
                                   "Identify the main object. JSON only.",
                                   "schema")
        elif attempt == VISION_MAX_ATTEMPTS:
            # Last resort alternate path: a plain description in the
            # model's own words. Decided jointly with every earlier reply
            # below — never word-counted alone.
            system, prompt, via = (
                VISION_DESCRIBE_SYSTEM,
                "Describe the main object in one short paragraph: its "
                "category, visible parts, and colors.",
                "describe")
        else:
            prev = attempts[-1]
            if prev.get("detected") is None or prev.get("weak"):
                # The model cannot (or did not) produce JSON: switch to a
                # constrained single-word choice rather than repeating a
                # prompt it already failed.
                system = VISION_CHOICE_SYSTEM.format(
                    candidates=", ".join(SUPPORTED_CATEGORIES))
                prompt, via = "One word only.", "choice"
            else:
                system = VISION_RETRY_SYSTEM.format(
                    candidates=", ".join(SUPPORTED_CATEGORIES))
                prompt = (f"Your previous answer was "
                          f"{prev.get('detected') or 'unreadable'!r}. "
                          f"{prev.get('problem') or ''} Look at the photo "
                          f"again and reply with JSON only.")
                via = "retry"
        raw = _describe_raw(path, model, system, prompt)
        det, _perrs = parse_vision_response(raw)
        evidence.append(collect_evidence(raw))
        if det is None:
            _record(attempt, None, raw, via,
                    "the vision model returned no usable answer")
        else:
            problem = _reconstruction_problem(det["category"],
                                              det["confidence"], stats)
            if not problem:
                _record(attempt, det, raw, via, "")
                attempts[-1]["problem"] = ""
                _, _, _dd = decide_category(evidence, stats)
                return {"det": det, "stats": stats, "seg": seg,
                        "attempts": attempts, "evidence": evidence,
                        "ranked": (_dd or {}).get("ranked", []),
                        "resolved": True}
            _record(attempt, det, raw, via, problem)
        # Joint decision over ALL normalized evidence so far — consulted
        # after EVERY non-resolving attempt (including unparseable replies,
        # whose prose may still carry usable mentions): convergent weak
        # signals (a bare word here, a corroborated mention there) can
        # resolve what no single reply could state reliably.
        voted, reasoning, details = decide_category(evidence, stats)
        if voted is not None:
            prev_labels = [a.get("detected") for a in attempts
                           if a.get("detected")]
            det = {"category": voted,
                   "confidence": VISION_WEAK_CONFIDENCE,
                   "parts": sorted({w for ev in evidence
                                    for w in ev.get("parts", ())
                                    if w in PART_LEXICON.get(voted, ())}),
                   "hidden": [], "materials": [], "colors": sorted(
                       {w for ev in evidence for w in ev.get("colors", ())}),
                   "occlusions": [], "limitations": [
                       f"category decided from multi-reply evidence "
                       f"({reasoning}); previous answers were "
                       f"{', '.join(prev_labels) if prev_labels else 'none'} "
                       f"- verify the category",
                       "single photo: far side, exact depth and occluded "
                       "surfaces are undetermined"],
                   "proportions": {}, "components": [], "weak": True,
                   "via": "decide", "raw": raw, "errors": []}
            # One record per attempt: fold the decision into this attempt's
            # row instead of appending a second one.
            attempts[-1].update({"detected": voted,
                                 "confidence": VISION_WEAK_CONFIDENCE,
                                 "weak": True, "via": "decide",
                                 "overrode": prev_labels,
                                 "problem": "", "resolved_weak": True,
                                 "reasoning": reasoning})
            emit({"type": "vision_retry", "attempt": attempt,
                  "detected": voted,
                  "confidence": VISION_WEAK_CONFIDENCE,
                  "problem": f"decided ({reasoning})"})
            return {"det": det, "stats": stats, "seg": seg,
                    "attempts": attempts, "evidence": evidence,
                    "ranked": (details or {}).get("ranked", []),
                    "resolved": True}
    last = attempts[-1] if attempts else {}
    # Report the persistent answer even when nothing resolved: "the model
    # kept saying urn" is actionable context.
    detected = last.get("detected")
    if detected is None:
        for a in reversed(attempts):
            if a.get("detected"):
                detected = a["detected"]
                break
    _, reasoning, details = decide_category(evidence, stats)
    info = {"needs_category": True,
            "detected_category": detected,
            "confidence": last.get("confidence"),
            "attempts": attempts,
            "reasoning": reasoning,
            "ranked": (details or {}).get("ranked", []),
            "candidates": list(SUPPORTED_CATEGORIES)}
    raise VisionError(
        "could not identify the object reliably"
        + (f": {reasoning}" if reasoning else "")
        + ". Select the intended "
        f"category (one of: {', '.join(SUPPORTED_CATEGORIES)}) and retry.",
        info=info)


# ------------------------------------------------------------------ IR build
def _viewpoint_guess(stats: dict) -> dict:
    """Coarse, explicitly unverified viewpoint note from silhouette shape.
    Wide silhouettes usually mean a side/frontal-wide framing; this is an
    assumption for the report, never a measurement."""
    aspect = (stats or {}).get("aspect") or 0.0
    wheels = (stats or {}).get("dark_blobs") or 0
    if aspect >= 1.4 and wheels >= 2:
        guess = "side view (wheels visible)"
    elif aspect >= 1.4:
        guess = "wide view, assumed side framing"
    elif aspect and aspect <= 0.8:
        guess = "narrow view, assumed front framing"
    else:
        guess = "unknown"
    return {"guess": guess, "basis": f"silhouette aspect {aspect}"}


def ir_from_image(category: str, stats: dict, source: str, confidence=0.5,
                  observed_parts=None, hidden_parts=None,
                  warnings=None, environment=None,
                  body_color: str = None,
                  vision_colors=None, vision_materials=None,
                  occlusions=None, limitations=None) -> dict:
    """Instantiate the canonical IR for `category` and fit it (bounded) to the
    photo silhouette. This is the ONE validated intermediate representation
    for image reconstruction, containing the full semantic contract:

      subject + confidence | silhouette + proportions | major components +
      locations | visible colors/materials | camera viewpoint + visible
      surfaces | environment (subject vs platform/background, never built) |
      uncertainty for occluded/ambiguous regions.

    Hidden geometry is assumed from the template and marked unobserved; the
    assumption is always recorded. ``environment`` (from segment_reference)
    is recorded separately from the object and never built. ``body_color``
    (measured reference object color) repaints the largest-volume component,
    logged as a measurement."""
    norm_cat = normalize_category(category)
    # Resolve synonyms to the template key for geometry, but keep the
    # caller's word as the IR label when it is a known synonym (e.g.
    # 'automobile' builds the car template while reporting 'automobile').
    base = ir_for_category(norm_cat if template_key(norm_cat) != "object"
                           else category)
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

    if body_color:
        vols = []
        for c in base["components"]:
            d = c.get("dims") or [0, 0, 0]
            vols.append((d[0] * d[1] * d[2], c))
        if vols:
            biggest = max(vols, key=lambda t: t[0])[1]
            biggest["color"] = body_color
            assumptions.append(
                f"largest component {biggest['id']!r} painted {body_color} "
                f"(measured reference object color, not a template default)")

    base["source"] = source
    base["confidence"] = round(float(confidence), 3)
    base["observed"] = {
        "category": bool(observed_parts) or source == "image-manual",
        "visible_parts": observed_parts,
        "hidden_parts": hidden_parts,
    }
    base["viewpoint"] = _viewpoint_guess(stats)
    assumptions.append(
        f"viewpoint {base['viewpoint']['guess']} "
        f"({base['viewpoint']['basis']}) — assumed, unverified")
    base["assumptions"] = assumptions + [
        "parts not visible in the photo are assumed from the category template"]
    base["environment"] = dict(environment or {"mode": "object-only"})
    # --- semantic contract extras (validated by validate_semantic_ir) ---
    base["subject"] = {"category": base.get("category"),
                       "confidence": base["confidence"]}
    base["silhouette"] = {"aspect": obs,
                          "bbox": list(base.get("bbox") or [0, 0, 0]),
                          "coverage": (stats or {}).get("coverage"),
                          "measured": dict(stats or {})}
    base["appearance"] = {"colors": list(vision_colors or []) or
                          ([body_color] if body_color else []),
                          "materials": list(vision_materials or []),
                          "body_color_measured": body_color}
    base["visible_surfaces"] = list(observed_parts)
    base["uncertainty"] = {"occlusions": list(occlusions or []),
                           "limitations": list(limitations or []) or
                           ["single photo: far side, exact depth and "
                            "occluded surfaces are undetermined"]}
    return base


def validate_semantic_ir(ir: dict) -> list:
    """Validate the semantic contract on top of the geometric IR.

    Checks that the ONE intermediate representation carries: subject +
    confidence, silhouette + proportions, components + locations,
    colors/materials, viewpoint + visible surfaces, environment separated
    from the object, and uncertainty for occluded regions. Returns a list
    of error strings ([] = contract holds)."""
    from morphix_ir import validate_ir as _validate_ir
    errs = list(_validate_ir(ir))
    if not isinstance(ir, dict):
        return errs or ["IR must be a JSON object"]
    subj = ir.get("subject")
    if not isinstance(subj, dict) or not subj.get("category"):
        errs.append("semantic: missing subject.category")
    if not isinstance(subj, dict) or not isinstance(
            subj.get("confidence"), (int, float)):
        errs.append("semantic: missing subject.confidence")
    sil = ir.get("silhouette")
    if not isinstance(sil, dict) or not isinstance(
            sil.get("aspect"), (int, float)):
        errs.append("semantic: missing silhouette.aspect (measured)")
    app = ir.get("appearance")
    if not isinstance(app, dict):
        errs.append("semantic: missing appearance (colors/materials)")
    if not isinstance(ir.get("viewpoint"), dict) or \
            not ir.get("viewpoint", {}).get("guess"):
        errs.append("semantic: missing viewpoint guess")
    if not isinstance(ir.get("environment"), dict) or \
            "mode" not in ir.get("environment", {}):
        errs.append("semantic: missing environment separation "
                    "(object vs platform/background)")
    unc = ir.get("uncertainty")
    if not isinstance(unc, dict):
        errs.append("semantic: missing uncertainty for occluded regions")
    return errs


# ------------------------------------------------------------------ run
def run_reconstruction_legacy(image_path: str, out_dir: str = "./out_img",
                       category: str = None, vision_model: str = "",
                       emit=None, guided: bool = False,
                       segment_cap: int = 4000, layer_cap: int = 32,
                       environment: str = "object-only",
                       visual_check="auto") -> dict:
    """Semantic image -> 3D: identify the object (vision model, or a documented
    demonstration template), build + validate the canonical IR, freeze it into
    printer primitives, then print it with the shared deterministic slicer.
    Returns the path report augmented with the IR, provenance and fidelity.

    environment: "object-only" (default: scenery is measured and reported
    but never built) or "preserve" (adds a clearly-marked display-only
    stand, excluded from slicing, deposits and verification).
    visual_check: True (require browser verification), False (skip, noted),
    "auto" (verify when a browser is available, else note the limitation).
    """
    import morphix_pipeline as P  # late import: avoids an import cycle
    emit = emit or (lambda e: None)
    os.makedirs(out_dir, exist_ok=True)
    if environment not in ("object-only", "preserve"):
        raise PipelineError(f"environment must be 'object-only'|'preserve' "
                            f"(got {environment!r})")

    stats = image_stats(image_path)
    seg = segment_reference(image_path)
    auto_vision = available_vision_model()
    model = vision_model or VISION_MODEL or auto_vision
    warnings = []
    observed_parts, hidden_parts = [], []
    vision_info = {}
    ranked = []

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
        seg = analysis.get("seg") or seg
        ranked = analysis.get("ranked", [])
        cat = det["category"]
        conf = det["confidence"]
        observed_parts = det["parts"]
        hidden_parts = det["hidden"]
        source = "image-vision"
        vision_info = {"attempts": analysis["attempts"],
                       "ranked": ranked,
                       "reasoning": (analysis["attempts"] or [{}])[-1].get(
                           "reasoning", ""),
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

    env_block = {"mode": environment,
                 "background": (seg.get("background") or {}).get("color"),
                 "platform": seg.get("platform"),
                 "viewpoint": None}
    body_color = suggest_body_color(seg)

    def _build_once(build_cat, build_conf, extra_warnings):
        """IR -> validate -> slice -> print -> fidelity. Returns
        (ir, report, fidelity). Raises PipelineError on any failure.

        Geometry order for image-to-3D: silhouette/proportions first (aspect
        fit), then main volume + cross-section (template), then major
        structural components (body/cabin/wheels...), then secondary
        geometry, then surface paint/materials, then camera/lighting (the
        render viewport is fixed — never per-screenshot tuned). A complex
        subject never defaults to a box because its category was detected:
        the template plus the shape gate above enforce the structure."""
        local_warnings = list(extra_warnings)
        ir = ir_from_image(build_cat, stats, source,
                           confidence=build_conf,
                           observed_parts=observed_parts,
                           hidden_parts=hidden_parts,
                           warnings=local_warnings,
                           environment=env_block,
                           body_color=body_color,
                           vision_colors=vision_info.get("colors", []),
                           vision_materials=vision_info.get("materials", []),
                           occlusions=vision_info.get("occlusions", []),
                           limitations=vision_info.get("limitations", []))
        env_block["viewpoint"] = ir.get("viewpoint")
        errs = validate_ir(ir)
        if errs:
            raise PipelineError("reconstructed object IR failed validation: "
                                + "; ".join(errs))
        sem_errs = validate_semantic_ir(ir)
        if sem_errs:
            raise PipelineError("semantic contract failed: "
                                + "; ".join(sem_errs))
        emit({"type": "vision_input", "source": source,
              "category": build_cat,
              "vision_model": model, "vision_available": bool(auto_vision),
              "detected_parts": observed_parts,
              "confidence": build_conf,
              "proportions": vision_info.get("proportions", {}),
              "components": vision_info.get("components", []),
              "materials": vision_info.get("materials", []),
              "colors": vision_info.get("colors", []),
              "occlusions": vision_info.get("occlusions", []),
              "attempts": vision_info.get("attempts", [])})
        emit({"type": "ir", "ir": summary_of(ir)})
        actions = ir_to_actions(ir)
        report = P.run_path_print(
            f"reconstruct a {build_cat} from a photo", out_dir=out_dir,
            guided=guided, emit=emit, segment_cap=segment_cap,
            layer_cap=layer_cap, plan_actions=actions,
            plan_status=f"IMAGE {build_cat}",
            spec_name=ir.get("template"), plan_source=source,
            stream_scene=False)
        fidelity = image_fidelity(ir, stats, observed_parts)
        return ir, report, fidelity, local_warnings

    ir, report, fidelity, _ = _build_once(cat, conf, warnings)
    repaired = None

    # Measurable visual verification against the reference (plus at most
    # ONE rebuild as a silhouette-fitting runner-up). A render that cannot
    # be produced or checked is reported, never assumed good.
    visual = {"checks": [], "verified": None,
              "note": "visual check not requested"}
    if visual_check:
        png = None
        try:
            png = render_reconstruction_scene(out_dir)
        except Exception as e:
            if visual_check is True:
                raise PipelineError(f"browser render failed: {e}")
            visual = {"checks": [], "verified": None,
                      "note": f"visual check unavailable (browser): {e}"}
        if png is not None:
            visual = verify_reconstruction(ir, png, seg, mode=environment, stats=stats)
            visual["png"] = os.path.basename(png)
            aspect_fails = [c for c in visual["checks"]
                            if c["name"] in ("reference_aspect_sane",
                                             "render_aspect_plausible")
                            and not c["ok"]]
            if aspect_fails and ranked:
                current = cat
                for cand in ranked:
                    if cand.get("category") == current:
                        continue
                    ok, _ratio = silhouette_compatible(
                        cand["category"], stats)
                    if ok:
                        break
                else:
                    cand = None
                if cand is not None:
                    warnings.append(
                        f"visual aspect check failed "
                        f"({'; '.join(c['detail'] for c in aspect_fails)}); "
                        f"rebuilding once as runner-up "
                        f"{cand['category']!r}")
                    cat = cand["category"]
                    conf = VISION_WEAK_CONFIDENCE
                    ir, report, fidelity, _ = _build_once(
                        cat, conf, warnings)
                    try:
                        png = render_reconstruction_scene(out_dir)
                    except Exception as e:
                        png = None
                        visual = {"checks": [], "verified": False,
                                  "note": f"rebuild render failed: {e}"}
                    if png is not None:
                        visual = verify_reconstruction(ir, png, seg,
                                                       mode=environment, stats=stats)
                        visual["png"] = os.path.basename(png)
                    repaired = cand["category"]
                    warnings.append(f"rebuilt as {cat!r} after visual "
                                    f"aspect failure (bounded repair: 1)")
    if environment == "preserve":
        _append_display_base(out_dir, ir, seg)
        warnings.append("environment-preserving mode: display stand is "
                        "visual context only (not printed, not verified)")

    report["mode"] = "image-reconstruct"
    report["ir"] = summary_of(ir)
    report["ir_full"] = ir
    report["image"] = {"path": os.path.basename(image_path), "source": source,
                       "category": cat, "confidence": conf,
                       "vision_model": model,
                       "vision_available": bool(auto_vision),
                       "warnings": warnings, "stats": stats,
                       "segmentation": seg,
                       "environment": env_block,
                       "repaired": repaired,
                       "vision_analysis": vision_info}
    # Technical action/geometry acceptance stays separate from this semantic
    # reconstruction-quality verdict; both are reported, never conflated.
    # A vision description alone is never proof: quality needs deposits
    # verified AND photo-consistent fidelity AND (when performed) visual
    # verification. A generic "object" fallback can never be quality
    # accepted (see image_fidelity category_template).
    report["reconstruction_quality"] = fidelity
    report["image_fidelity"] = fidelity
    report["visual_verification"] = visual
    # image-specific final quality: geometry verified AND photo-consistent
    # AND (when performed) visually verified.
    report["quality_accepted"] = bool(
        report.get("quality_accepted") and fidelity.get("verified")
        and (visual.get("verified") is not False))
    # A fallback that produced a generic block is an explicitly reported
    # fallback, never a success: force quality rejected even if the
    # technical path compiled.
    if template_key(cat) == "object":
        report["quality_accepted"] = False
        warnings.append("generic object fallback: quality rejected — "
                        "no deterministic template for the photographed "
                        "subject (pass a category or configure vision)")
        report["warnings"] = list(report.get("warnings", [])) + [
            "generic object fallback: quality rejected"]
    report["warnings"] = list(report.get("warnings", [])) + warnings

    with open(os.path.join(out_dir, "object.json"), "w",
              encoding="utf-8") as f:
        json.dump(ir, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Full trace for the hotfix audit: original vision response(s), parsed
    # result, category decision, component plan, generated geometry, final
    # screenshot — everything needed to replay vision -> IR -> geometry ->
    # compile -> render without re-calling the model.
    try:
        import shutil as _shutil
        trace = {
            "reference_image": os.path.basename(image_path),
            "vision_raw": [a.get("raw") for a in
                           vision_info.get("attempts", [])],
            "vision_parsed": [a.get("parsed") for a in
                              vision_info.get("attempts", [])],
            "vision_attempts": vision_info.get("attempts", []),
            "vision_ranked": ranked,
            "category_decision": {"category": cat, "confidence": conf,
                                  "source": source,
                                  "reasoning": vision_info.get(
                                      "reasoning", "")},
            "semantic_ir": ir,
            "component_plan": ir_to_actions(ir),
            "segmentation": seg,
            "image_stats": stats,
            "fidelity": fidelity,
            "visual": visual,
            "repaired": repaired,
            "environment_mode": environment,
        }
        with open(os.path.join(out_dir, "trace.json"), "w",
                  encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)
        # Keep the exact input pixels alongside the outputs.
        try:
            _shutil.copyfile(image_path,
                             os.path.join(out_dir, "reference.png"))
        except OSError:
            pass
    except Exception as _e:
        warnings.append(f"trace write failed: {_e}")
    print(f"RECONSTRUCT: {cat} [{source}] -> "
          f"{'OK' if report['ok'] else 'FAIL'} "
          f"(quality {'accepted' if report['quality_accepted'] else 'rejected'})")
    return report


def run_reconstruction(image_path: str, out_dir: str = "./out_img",
                       category: str = None, vision_model: str = "",
                       emit=None, environment: str = "object-only",
                       visual_check="auto", **_legacy) -> dict:
    """Image -> 3D for ANY subject (morphix_image3d). Family builders when
    the subject has one, the subject's own measured silhouette otherwise;
    measured visual verification with bounded repair. The closed-category
    implementation above is kept as run_reconstruction_legacy."""
    import morphix_image3d as I3
    return I3.run(image_path, out_dir, category=category,
                  vision_model=vision_model, environment=environment,
                  emit=emit, render=visual_check is not False)


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Semantic image -> 3D object reconstruction (printer mode)")
    p.add_argument("image")
    p.add_argument("--out", default="./out_recon")
    p.add_argument("--category", default="",
                   help="override the object category (skips the vision model)")
    p.add_argument("--vision-model", default="")
    p.add_argument("--environment", default="object-only",
                   choices=("object-only", "preserve"),
                   help="preserve adds a display-only stand (not printed)")
    p.add_argument("--visual-check", default="auto",
                   choices=("auto", "yes", "no"),
                   help="browser visual verification: yes=require, "
                        "no=skip, auto=verify when possible")
    a = p.parse_args()
    rep = run_reconstruction(a.image, a.out, category=a.category or None,
                             vision_model=a.vision_model,
                             environment=a.environment,
                             visual_check=(True if a.visual_check == "yes"
                                           else False
                                           if a.visual_check == "no"
                                           else "auto"))
    print(json.dumps({"ok": rep["ok"], "complete": rep["complete"],
                      "status": rep.get("reconstruction_status"),
                      "success": rep.get("success"),
                      "subject": rep["image"].get("subject"),
                      "warnings": rep["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
