# morphix_image_seg.py
# Subject / environment segmentation for image -> 3D reconstruction.
#
# A reference photo holds more than the object: a backdrop, a floor or a
# display platform, cast shadows, light pools and clutter. Reconstruction
# must model the SUBJECT only (object-only mode) or keep the rest as
# clearly separated, display-only context (environment-preserving mode) —
# never turn the background into the object, never drop object geometry
# because it was mistaken for scenery.
#
# Pipeline (deterministic, offline, no model):
#   1. border flood   — regions connected to the frame through smooth color
#                       steps are backdrop/floor candidates;
#   2. GrabCut        — colour-model refinement seeded by the flood (OpenCV;
#                       falls back to the flood mask, reported, if missing);
#   3. platform split — a wide, flat, differently coloured region under the
#                       subject that is attached to it is a display platform,
#                       not part of the object;
#   4. measurements   — subject bbox/aspect/coverage, paint clusters, ground
#                       contacts, left-right symmetry, and the environment
#                       regions (backdrop, floor, platform, shadow, clutter).
#
# Everything here is a measurement with a stated method; nothing is guessed
# from a vision model's prose.

import math
from collections import deque

import numpy as np
from PIL import Image

WORK_W = 200          # analysis width (px); aspect preserved
FLOOD_TOL = 30.0      # max summed RGB step between flood neighbours
MIN_SUBJECT = 0.003   # subject must cover >= 0.3% of the frame


def _load(path: str, work_w: int = WORK_W):
    img = Image.open(path).convert("RGB")
    scale = work_w / float(img.width)
    h = max(16, int(round(img.height * scale)))
    small = img.resize((work_w, h), Image.LANCZOS)
    return img, small, np.asarray(small).astype(np.float32)


def _smooth(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return np.stack([gaussian_filter(a[..., c], 0.8) for c in range(3)], -1)


BG_MODEL_TOL = 90.0   # max summed RGB distance to a border colour cluster


def border_model(a: np.ndarray, k: int = 6) -> np.ndarray:
    """Colour clusters of the frame border ring (backdrop, floor, light
    pools touching the frame)."""
    h, w, _ = a.shape
    b = max(1, h // 50)
    ring = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    C, _ = kmeans(ring.astype(np.float64), k, seed=4)
    return C


def border_flood(a: np.ndarray, tol: float = FLOOD_TOL) -> np.ndarray:
    """Pixels reachable from the frame border through neighbour steps whose
    summed RGB difference stays below ``tol`` (handles gradient backdrops
    and lit floors; stops at the object's outline) AND whose colour stays
    within BG_MODEL_TOL of a border colour cluster — so a soft,
    anti-aliased outline can never be crept through into the subject."""
    h, w, _ = a.shape
    C = border_model(a)
    dmin = np.min([np.abs(a - c).sum(2) for c in C], axis=0)
    okbg = dmin < BG_MODEL_TOL
    dr = np.abs(a[:, 1:] - a[:, :-1]).sum(2)
    dd = np.abs(a[1:, :] - a[:-1, :]).sum(2)
    dr = np.where(okbg[:, 1:] & okbg[:, :-1], dr, 1e9)
    dd = np.where(okbg[1:, :] & okbg[:-1, :], dd, 1e9)
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        if x + 1 < w and not seen[y, x + 1] and dr[y, x] < tol:
            seen[y, x + 1] = True
            q.append((y, x + 1))
        if x > 0 and not seen[y, x - 1] and dr[y, x - 1] < tol:
            seen[y, x - 1] = True
            q.append((y, x - 1))
        if y + 1 < h and not seen[y + 1, x] and dd[y, x] < tol:
            seen[y + 1, x] = True
            q.append((y + 1, x))
        if y > 0 and not seen[y - 1, x] and dd[y - 1, x] < tol:
            seen[y - 1, x] = True
            q.append((y - 1, x))
    return seen


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label
    lab, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _clean(mask: np.ndarray) -> np.ndarray:
    """Largest component, holes filled. A closing first re-seals thin
    frames (roof rails, pillars) so glass that mirrors the backdrop is
    kept as part of the subject instead of leaking into the background."""
    from scipy.ndimage import (binary_closing, binary_fill_holes,
                               binary_opening)
    m = binary_opening(mask, structure=np.ones((2, 2)))
    m = largest_component(m)
    closed = binary_closing(np.pad(m, 4), structure=np.ones((3, 3)),
                            iterations=2)[4:-4, 4:-4]
    return binary_fill_holes(closed | m)


def kmeans(X: np.ndarray, k: int, iters: int = 15, seed: int = 0):
    """Deterministic k-means (k-means++ init, fixed seed). Returns
    (centers[k,3], labels[n]). Small and dependency-free on purpose."""
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        tot = d2.sum()
        if tot <= 0:
            break
        centers.append(X[rng.choice(n, p=d2 / tot)])
    C = np.array(centers, dtype=np.float64)
    lab = np.zeros(n, int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        newC = np.array([X[lab == i].mean(0) if np.any(lab == i) else C[i]
                         for i in range(len(C))])
        if np.allclose(newC, C):
            break
        C = newC
    return C, lab


def _hex(c) -> str:
    r, g, b = (int(max(0, min(255, round(v)))) for v in c[:3])
    return f"0x{r:02X}{g:02X}{b:02X}"


def _grabcut(small: Image.Image, seed_fg: np.ndarray):
    """GrabCut refinement seeded by the flood core. Returns mask or None
    when OpenCV is unavailable or the result is degenerate."""
    try:
        import cv2
    except ImportError:
        return None
    bgr = np.asarray(small)[:, :, ::-1].copy()
    h, w = seed_fg.shape
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[seed_fg] = cv2.GC_PR_FGD
    b = max(2, h // 40)
    m[:b, :] = cv2.GC_BGD
    m[-b:, :] = cv2.GC_BGD
    m[:, :b] = cv2.GC_BGD
    m[:, -b:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(bgr, m, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    if out.sum() < MIN_SUBJECT * h * w:
        return None
    return _clean(out)


def _split_platform(a: np.ndarray, mask: np.ndarray):
    """Separate a display platform glued to the subject's base: a colour
    cluster inside the mask whose region is wide (>= 60% of the mask
    width), flat (<= 45% of its height), sits at the mask bottom and is
    attached under the rest. Returns (subject_mask, platform_info|None)."""
    from scipy.ndimage import label
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return mask, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mw, mh = x1 - x0 + 1, y1 - y0 + 1
    X = a[mask].astype(np.float64)
    C, lab = kmeans(X, 4, seed=1)
    full = np.full(mask.shape, -1, int)
    full[mask] = lab
    best = None
    for i in range(len(C)):
        comp, n = label(full == i)
        for j in range(1, n + 1):
            r = comp == j
            cnt = int(r.sum())
            if cnt < 0.08 * mask.sum():
                continue
            ry, rx = np.where(r)
            bw, bh = rx.max() - rx.min() + 1, ry.max() - ry.min() + 1
            reaches_bottom = ry.max() >= y1 - max(2, 0.05 * mh)
            wide = bw >= 0.6 * mw and bw >= 2.2 * bh
            flat = bh <= 0.65 * mh
            # the rest of the object must sit ON it (above its top band)
            rest = mask & ~r
            if not (reaches_bottom and wide and flat and rest.any()):
                continue
            rest_y = np.where(rest)[0]
            above = np.mean(rest_y < ry.min() + 0.5 * bh)
            if above < 0.6:
                continue
            score = cnt
            if best is None or score > best[0]:
                best = (score, r, C[i], (rx.min(), ry.min(), rx.max(),
                                         ry.max()))
    if best is None:
        return mask, None
    _, r, col, bb = best
    subject = largest_component(mask & ~r)
    h, w = mask.shape
    return subject, {"color": _hex(col), "fraction": round(float(r.mean()), 4),
                     "bbox": [round(bb[0] / w, 4), round(bb[1] / h, 4),
                              round(bb[2] / w, 4), round(bb[3] / h, 4)],
                     "method": "attached-flat-region"}


def _ground_contacts(mask: np.ndarray) -> list:
    """Runs of columns where the subject reaches its own bottom band —
    wheels, legs, feet or a flat base. Returned as normalized [x0, x1]
    spans of the subject width (left->right)."""
    ys, xs = np.where(mask)
    x0, x1, y1 = xs.min(), xs.max(), ys.max()
    h = ys.max() - ys.min() + 1
    band = max(1, int(round(0.06 * h)))
    low = np.zeros(x1 - x0 + 1, bool)
    for x in range(x0, x1 + 1):
        col = np.where(mask[:, x])[0]
        if len(col) and col.max() >= y1 - band:
            low[x - x0] = True
    spans, start = [], None
    for i, v in enumerate(list(low) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    W = float(x1 - x0 + 1)
    out = [[round(s / W, 3), round((e + 1) / W, 3)] for s, e in spans
           if (e - s + 1) >= max(1, 0.015 * W)]
    return out


def _symmetry(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    flip = crop[:, ::-1]
    inter = np.logical_and(crop, flip).sum()
    union = np.logical_or(crop, flip).sum()
    return round(float(inter / max(1, union)), 3)


def _environment(a: np.ndarray, subject: np.ndarray, flood: np.ndarray,
                 platform) -> dict:
    """Describe everything that is NOT the subject: backdrop (touches the
    top border), floor (spans the bottom band), shadow (darkened floor
    just under the subject), platform/light pool (distinct region under
    the subject) and clutter (other compact non-background regions)."""
    from scipy.ndimage import label
    h, w, _ = a.shape
    rest = ~subject
    out = {"backdrop": None, "floor": None, "platform": platform,
           "surface_light": None, "shadow": None, "clutter": []}
    X = a[rest].astype(np.float64)
    if len(X) < 10:
        return out
    C, lab = kmeans(X, 4, seed=2)
    full = np.full((h, w), -1, int)
    full[rest] = lab
    ys, xs = np.where(subject)
    sx0, sx1, sy0, sy1 = xs.min(), xs.max(), ys.min(), ys.max()
    regions = []
    for i in range(len(C)):
        r = full == i
        frac = float(r.mean())
        if frac < 0.005:
            continue
        ry, rx = np.where(r)
        regions.append({"color": _hex(C[i]), "fraction": round(frac, 4),
                        "top_touch": bool(r[0, :].mean() > 0.25),
                        "bottom_touch": bool(r[-1, :].mean() > 0.25),
                        "cy": round(float(ry.mean() / h), 3),
                        "mask": r})
    if regions:
        tops = [r for r in regions if r["top_touch"]]
        if tops:
            bd = max(tops, key=lambda r: r["fraction"])
            out["backdrop"] = {k: bd[k] for k in ("color", "fraction")}
        bots = [r for r in regions if r["bottom_touch"]
                and r["cy"] > sy0 / h and out["backdrop"] is not None
                and r["color"] != out["backdrop"]["color"]]
        if bots:
            fl = max(bots, key=lambda r: r["fraction"])
            out["floor"] = {k: fl[k] for k in ("color", "fraction")}
        # Light pool / platform NOT attached to the subject: a distinct,
        # saturated region overlapping the subject's x-range around its base.
        if True:
            band = np.zeros((h, w), bool)
            sh = sy1 - sy0 + 1
            band[max(0, sy1 - int(0.25 * sh)):min(h, sy1 + int(0.6 * sh)),
                 max(0, sx0 - (sx1 - sx0) // 3):
                 min(w, sx1 + (sx1 - sx0) // 3)] = True
            cands = []
            for r in regions:
                if out["backdrop"] and r["color"] == out["backdrop"]["color"]:
                    continue
                if out["floor"] and r["color"] == out["floor"]["color"]:
                    continue
                if platform and lab_distance(r["color"],
                                             platform["color"]) < 12:
                    continue  # the attached platform itself, already split
                inband = (r["mask"] & band).sum()
                if inband < 0.25 * r["mask"].sum():
                    continue
                c = np.array([int(r["color"][i:i + 2], 16)
                              for i in (2, 4, 6)], float)
                sat = (c.max() - c.min()) / max(1.0, c.max())
                if sat < 0.35:
                    continue
                cands.append(r)
            if cands:
                pl = max(cands, key=lambda r: r["fraction"])
                ry, rx = np.where(pl["mask"])
                key = "platform" if platform is None else "surface_light"
                out[key] = {
                    "color": pl["color"], "fraction": pl["fraction"],
                    "bbox": [round(rx.min() / w, 4), round(ry.min() / h, 4),
                             round(rx.max() / w, 4), round(ry.max() / h, 4)],
                    "method": "distinct-region-under-subject"}
    # Shadow: floor pixels right under the subject that are much darker
    # than the floor further away (contact shadow).
    sh_band = np.zeros((h, w), bool)
    sh_band[sy1:min(h, sy1 + max(2, (sy1 - sy0) // 6)), sx0:sx1 + 1] = True
    cand = sh_band & rest
    if cand.sum() > 5:
        lum = a.mean(2)
        far = rest.copy()
        far[max(0, sy0):min(h, sy1 + (sy1 - sy0) // 3),
            max(0, sx0 - 5):min(w, sx1 + 5)] = False
        ref = np.median(lum[far]) if far.any() else np.median(lum[rest])
        dark = cand & (lum < 0.6 * ref)
        if dark.sum() > 0.2 * cand.sum():
            out["shadow"] = {"fraction": round(float(dark.mean()), 4),
                             "method": "darkened band under subject"}
    # Clutter: compact non-flood regions that are neither subject nor
    # platform (objects lying around, labels, overlays).
    lab2, n = label(rest & ~flood)
    plat_bbox = (out["platform"] or {}).get("bbox")
    for j in range(1, n + 1):
        r = lab2 == j
        cnt = int(r.sum())
        if cnt < 0.002 * h * w:
            continue
        ry, rx = np.where(r)
        bb = [round(rx.min() / w, 4), round(ry.min() / h, 4),
              round(rx.max() / w, 4), round(ry.max() / h, 4)]
        if plat_bbox and bb[0] >= plat_bbox[0] - 0.02 \
                and bb[2] <= plat_bbox[2] + 0.02 \
                and bb[1] >= plat_bbox[1] - 0.05:
            continue
        out["clutter"].append({"bbox": bb,
                               "fraction": round(cnt / float(h * w), 4),
                               "color": _hex(a[r].mean(0))})
    out["clutter"] = sorted(out["clutter"],
                            key=lambda c: -c["fraction"])[:6]
    return out


def segment(path: str) -> dict:
    """Measure subject + environment of a reference image. Returns a
    JSON-serializable dict plus ``_mask`` (bool ndarray, analysis scale)
    and ``_rgb`` (float ndarray) for downstream comparison."""
    _img, small, a = _load(path)
    h, w, _ = a.shape
    sm = _smooth(a)
    flood = border_flood(sm)
    core = largest_component(~flood)
    notes = []
    method = "flood"
    mask = _clean(core) if core.sum() else core
    if mask.sum() >= MIN_SUBJECT * h * w:
        gc = _grabcut(small, mask)
        if gc is not None and gc.sum() >= 0.5 * mask.sum():
            mask, method = gc, "grabcut"
        elif gc is None:
            notes.append("OpenCV GrabCut unavailable or degenerate: subject "
                         "mask comes from the border flood only")
    if mask.sum() < MIN_SUBJECT * h * w:
        return {"ok": False, "method": method, "size": [w, h],
                "error": "no subject separable from the background "
                         "(subject touches every border or has no contrast)",
                "notes": notes, "_mask": mask, "_rgb": a}
    subject, platform = _split_platform(a, mask)
    if platform is not None:
        notes.append("display platform attached to the subject base was "
                     "split off (environment, not object)")
    ys, xs = np.where(subject)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    touches = bool(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2)
    if touches:
        notes.append("subject touches the frame border: it may be cropped "
                     "(extent along that side is uncertain)")
    X = a[subject].astype(np.float64)
    C, lab = kmeans(X, 4, seed=3)
    counts = np.bincount(lab, minlength=len(C))
    clusters = []
    for i in np.argsort(-counts):
        sel = lab == i
        if not sel.any():
            continue
        yy = ys[sel]
        clusters.append({"color": _hex(C[i]),
                         "fraction": round(float(counts[i] / len(X)), 4),
                         "cy": round(float((yy.mean() - y0) / bh), 3),
                         "luma": round(float(C[i].mean()), 1)})
    env = _environment(a, subject, flood, platform)
    return {
        "ok": True,
        "method": method,
        "size": [w, h],
        "subject": {
            "bbox": [round(x0 / w, 4), round(y0 / h, 4),
                     round((x1 + 1) / w, 4), round((y1 + 1) / h, 4)],
            "bbox_px": [int(x0), int(y0), int(x1), int(y1)],
            "aspect": round(bw / float(bh), 3),
            "coverage": round(float(subject.mean()), 4),
            "touches_border": touches,
            "colors": clusters,
            "body_color": clusters[0]["color"] if clusters else None,
            "ground_contacts": _ground_contacts(subject),
            "symmetry": _symmetry(subject),
        },
        "environment": env if env else {},
        "notes": notes,
        "_mask": subject,
        "_rgb": a,
    }


def public(seg: dict) -> dict:
    """JSON-safe copy (drops the ndarray fields and region masks)."""
    out = {k: v for k, v in seg.items() if not k.startswith("_")}
    return out


def save_mask(seg: dict, path: str) -> str:
    m = seg.get("_mask")
    if m is None:
        return ""
    Image.fromarray((m.astype(np.uint8) * 255)).save(path)
    return path


def tolerant(m: np.ndarray) -> np.ndarray:
    """1-px boundary tolerance at the normalized 64x64 scale (~1.5% of the
    subject): thin parts (legs, rails) are not failed for sub-pixel
    placement, while gross shape differences still count fully."""
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, structure=np.ones((3, 3)))


def mask_iou_normalized(a: np.ndarray, b: np.ndarray, size: int = 64):
    """Shape agreement independent of position/scale: each mask is cropped
    to its own bbox and resampled to size x size before IoU. Aspect is
    compared separately (see aspect_of)."""
    def norm(m):
        ys, xs = np.where(m)
        if len(xs) == 0:
            return np.zeros((size, size), bool)
        c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = Image.fromarray(c.astype(np.uint8) * 255).resize(
            (size, size), Image.BILINEAR)
        return np.asarray(img) >= 128
    na, nb = tolerant(norm(a)), tolerant(norm(b))
    inter = np.logical_and(na, nb).sum()
    union = np.logical_or(na, nb).sum()
    return float(inter / union) if union else 0.0


def aspect_of(m: np.ndarray):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)


def lab_distance(c1: str, c2: str) -> float:
    """Perceptual distance (CIE76 ΔE) between two 0xRRGGBB colors."""
    def to_lab(hx):
        rgb = [int(hx[i:i + 2], 16) / 255.0 for i in (2, 4, 6)]
        lin = [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92
               for v in rgb]
        X = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
        Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        Z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883

        def f(t):
            return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        return (116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z)))
    a, b = to_lab(c1), to_lab(c2)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def subject_colors(rgb: np.ndarray, mask: np.ndarray) -> list:
    """Paint clusters of a (visible) subject mask, sorted by area."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    y0, bh = ys.min(), ys.max() - ys.min() + 1
    X = rgb[mask].astype(np.float64)
    C, lab = kmeans(X, 4, seed=3)
    counts = np.bincount(lab, minlength=len(C))
    out = []
    for i in np.argsort(-counts):
        sel = lab == i
        if not sel.any():
            continue
        out.append({"color": _hex(C[i]),
                    "fraction": round(float(counts[i] / len(X)), 4),
                    "cy": round(float((ys[sel].mean() - y0) / bh), 3),
                    "luma": round(float(C[i].mean()), 1)})
    return out


def occluder_candidates(seg: dict) -> list:
    """Regions inside the subject mask that look like a separate object in
    front of (or beside) it: one uniform colour, a near-solid rectangle
    (bbox fill >= 0.9), >= 8% of the mask, at the mask's left or right
    extreme. Candidates only: run() accepts one as an occluder solely when
    ignoring it measurably improves the shape fit."""
    from scipy.ndimage import label
    mask, a = seg["_mask"], seg["_rgb"]
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    X = a[mask].astype(np.float64)
    C, lab = kmeans(X, 4, seed=5)
    full = np.full(mask.shape, -1, int)
    full[mask] = lab
    out = []
    for i in range(len(C)):
        comp, n = label(full == i)
        for j in range(1, n + 1):
            r = comp == j
            cnt = int(r.sum())
            if cnt < 0.08 * mask.sum() or cnt > 0.6 * mask.sum():
                continue
            ry, rx = np.where(r)
            bw, bh = rx.max() - rx.min() + 1, ry.max() - ry.min() + 1
            fill = cnt / float(bw * bh)
            at_edge = rx.min() <= x0 + 2 or rx.max() >= x1 - 2
            if fill >= 0.9 and at_edge:
                rest = a[mask & ~r].mean(0)
                if lab_distance(_hex(C[i]), _hex(rest)) >= 25:
                    out.append({"mask": r, "color": _hex(C[i]),
                                "fraction": round(cnt / float(mask.sum()), 3),
                                "fill": round(fill, 3)})
    return out
