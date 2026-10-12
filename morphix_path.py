# morphix_path.py
# Point-by-point printer core: toolhead state, atomic movement validation,
# deterministic path generators (ground truth for training + demo), engine
# rendering of validated segments, component tracking, and checkpoints.
#
# Printer-mode protocol — these six actions are the ONLY permitted
# geometry-construction actions in printer mode. Whole-object primitives
# (add_box, add_sphere, ...) are rejected here, never translated:
#   move_to         {position, layer, label}   travel, no deposition
#   deposit_segment {from, to, bead_width, bead_height, material,
#                    layer, label}
#   deposit_point   {position, size, material, layer, label}
#   complete_path   {layer, label}
#   complete_layer  {layer, label}
#   complete_object {label}
#
# Validated materials (engine-owned palette, never free-form colors):
#   body    structural filament      accent   highlight filament
#   blue    deep blue filament       support  temporary support
#   glass   translucent filament     tire     dark rubber filament
#   wood    brown wood-like filament
#   (colors in MATERIAL_COLORS)
#
# Bead placement convention (explicit): deposits are drawn centered on the
# active layer plane — i.e. the action's y IS the toolpath centerline, the
# same convention slicer previews use. No hidden offsets are ever applied;
# the validator checks the coordinates exactly as proposed.
#
# Bead cross-section (fixed, not per-call guesswork): the bead is as tall as
# one layer (BEAD_HEIGHT == LAYER_HEIGHT) so consecutive courses share a face
# and fuse, and is BEAD_WIDTH wide so solid infill lines placed BEAD_WIDTH
# apart touch instead of leaving bare gaps. Widening a single segment never
# "fixes" a gap here — the schedule itself is consistent.
#
# Rules enforced here (beyond schema checks in morphix_validator):
#   - one authoritative toolhead position; a segment/point must start AT it
#     (within POS_TOL); a failed step never advances the position.
#   - deposition is y-locked to the active layer plane (horizontal XZ slabs
#     along Y). layer must equal the active layer (no skipping ahead).
#   - complete_layer needs >= 1 committed segment in the layer.
#   - complete_object additionally needs every planned component completed
#     (checked by the caller via component_status; see run_path_print).
#
# The deterministic generators below produce ground-truth
# (state, action) sequences for training data and the offline demo.

import math

LAYER_HEIGHT = 0.2   # vertical spacing between printed plan layers
POS_TOL = 1e-3       # toolhead continuity tolerance
LAYER_TOL = 0.05     # deposition must sit on the active layer plane (y)
SEG_MAX_LEN = 5.0    # longest single deposited segment
SEG_MAX_DIM = 0.5    # max bead width/height/point size of one deposit
BOUND_XZ = 25.0
MAX_Y = 30.0
NGON_SIDES = 12  # cylinder footprints print as 12-gon perimeters
GROUND_TOL = 0.1   # bottom at/below this rests on the ground plane
SUPPORT_EPS = 0.02  # XZ overlap tolerance when stacking on lower geometry
MAX_LINEAR = 30.0  # no single dimension may exceed the scene itself; larger
# is always a thinker hallucination (observed: width 100, 50-layer roofs)

PATH_ACTIONS = {"move_to", "deposit_segment", "deposit_point",
                "complete_path", "complete_layer", "complete_object"}

MATERIAL_COLORS = {"body": 0xFFAA33, "accent": 0x44CCFF,
                   "blue": 0x2A6FDB, "support": 0x888888,
                   "glass": 0x99DDFF, "tire": 0x222222,
                   "wood": 0x9C6B3F}

# Primitives the slicer understands. Anything else fails loudly: path mode
# prints real sliced footprints, never a substituted mystery shape.
SLICEABLE = {"add_box", "add_cylinder", "add_sphere"}


def _vec(v) -> bool:
    return (isinstance(v, list) and len(v) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in v))


def _dim(v, cap: float = SEG_MAX_DIM) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and 0 < v <= cap)


def layer_y(layer: int) -> float:
    return layer * LAYER_HEIGHT


class PathState:
    """Authoritative toolhead + layer construction state. Serializable.
    The model proposes; only apply() of a CHECKED action mutates state."""

    def __init__(self, total_layers: int = 1,
                 start: list = None):
        self.total_layers = max(1, int(total_layers))
        self.active_layer = 0
        self.pos = [float(v) for v in (start or [0.0, 0.0, 0.0])]
        self.layers = [{"segments": [], "path_open": False,
                        "paths_done": 0} for _ in range(self.total_layers)]
        self.seq = 0            # committed deposition counter
        self.object_done = False

    # ------------------------------------------------ serialization
    def checkpoint(self) -> dict:
        return {"total_layers": self.total_layers,
                "active_layer": self.active_layer,
                "pos": list(self.pos),
                "layers": [{"segments": [dict(s) for s in L["segments"]],
                            "path_open": L["path_open"],
                            "paths_done": L["paths_done"]}
                           for L in self.layers],
                "seq": self.seq, "object_done": self.object_done}

    @classmethod
    def restore(cls, data: dict) -> "PathState":
        st = cls(total_layers=data.get("total_layers", 1))
        st.active_layer = int(data.get("active_layer", 0))
        st.pos = [float(v) for v in data.get("pos", [0, 0, 0])]
        st.layers = [{"segments": [dict(s) for s in L.get("segments", [])],
                      "path_open": bool(L.get("path_open", False)),
                      "paths_done": int(L.get("paths_done", 0))}
                     for L in data.get("layers", [])] or st.layers
        st.seq = int(data.get("seq", 0))
        st.object_done = bool(data.get("object_done", False))
        return st

    def summary(self) -> dict:
        return {"pos": list(self.pos), "active_layer": self.active_layer,
                "layer_y": layer_y(self.active_layer),
                "segments_committed": self.seq,
                "paths_done": self.layers[self.active_layer]["paths_done"]
                if self.active_layer < self.total_layers else 0}

    # ------------------------------------------------ validation
    def check(self, act: dict) -> list:
        """Validate one path action against live state. Returns error strings.
        Anything outside the six protocol actions is refused outright."""
        errs = []
        name = act.get("action", "?")
        if name not in PATH_ACTIONS:
            return [f"forbidden action {name!r} in printer mode: only "
                    f"{sorted(PATH_ACTIONS)} construct geometry here"]
        if name == "complete_object":
            if self.active_layer < self.total_layers or not all(
                    L["segments"] for L in self.layers):
                errs.append("complete_object refused: not all layers complete")
            return errs
        layer = act.get("layer")
        if (isinstance(layer, bool) or not isinstance(layer, int)
                or layer != self.active_layer):
            errs.append(f"layer must equal active layer "
                        f"{self.active_layer} (got {layer!r}); layers advance "
                        f"only via an authorized complete_layer")
            return errs
        if name == "complete_layer":
            if not self.layers[self.active_layer]["segments"]:
                errs.append("complete_layer refused: no segments in layer")
            return errs
        if name == "complete_path":
            if not self.layers[self.active_layer]["path_open"]:
                errs.append("complete_path refused: no open path in layer")
            return errs
        if name == "move_to":
            if not _vec(act.get("position")):
                errs.append("move_to.position must be [x,y,z] finite numbers")
            elif (abs(act["position"][0]) > BOUND_XZ
                  or abs(act["position"][2]) > BOUND_XZ
                  or not (0 <= act["position"][1] <= MAX_Y)):
                errs.append(f"move_to target {act['position']} out of bounds")
            return errs
        mat = act.get("material")
        if mat not in MATERIAL_COLORS:
            errs.append(f"material must be one of "
                        f"{sorted(MATERIAL_COLORS)} (got {mat!r})")
            return errs
        if name == "deposit_point":
            if not _vec(act.get("position")):
                return ["deposit_point.position must be [x,y,z] finite numbers"]
            if math.dist(act["position"], self.pos) > POS_TOL:
                errs.append(f"deposit_point must be at toolhead {self.pos} "
                            f"(got {act['position']})")
            if not _dim(act.get("size")):
                errs.append(f"size must be in (0,{SEG_MAX_DIM}] "
                            f"(got {act.get('size')!r})")
            errs.extend(self._deposit_common(act["position"]))
            return errs
        if name == "deposit_segment":
            if not _vec(act.get("from")) or not _vec(act.get("to")):
                return ["deposit_segment from/to must be [x,y,z] finite numbers"]
            if math.dist(act["from"], self.pos) > POS_TOL:
                errs.append(f"segment must start at toolhead {self.pos} "
                            f"(got {act['from']})")
            seg_len = math.dist(act["from"], act["to"])
            if seg_len < POS_TOL:
                errs.append("segment has zero length "
                            "(use deposit_point for beads)")
            elif seg_len > SEG_MAX_LEN:
                errs.append(f"segment too long ({seg_len:.2f} > {SEG_MAX_LEN})")
            for dim in ("bead_width", "bead_height"):
                if not _dim(act.get(dim)):
                    errs.append(f"{dim} must be in (0,{SEG_MAX_DIM}] "
                                f"(got {act.get(dim)!r})")
            to = act["to"]
            if (abs(to[0]) > BOUND_XZ or abs(to[2]) > BOUND_XZ
                    or not (0 <= to[1] <= MAX_Y)):
                errs.append(f"segment target {to} out of bounds")
            errs.extend(self._deposit_common(to))
            return errs
        return [f"unhandled path action {name!r}"]

    def _deposit_common(self, point: list) -> list:
        """Y-lock: deposition happens on the active layer plane only."""
        want = layer_y(self.active_layer)
        if abs(point[1] - want) > LAYER_TOL:
            return [f"deposition y={point[1]} not on active layer "
                    f"{self.active_layer} (y={want})"]
        return []

    # ------------------------------------------------ execution
    def apply(self, act: dict) -> dict:
        """Execute a CHECKED action. Returns the committed record. Raises
        ValueError if check() reports errors (position never advances then)."""
        errs = self.check(act)
        if errs:
            raise ValueError("; ".join(errs))
        name = act["action"]
        if name == "move_to":
            self.pos = [float(v) for v in act["position"]]
            return {"action": name, "pos": list(self.pos)}
        if name in ("deposit_segment", "deposit_point"):
            self.seq += 1
            rec = {"seq": self.seq, "action": name,
                   "layer": self.active_layer,
                   "material": act.get("material", "body"),
                   "component": act.get("component"),
                   "label": act.get("label", f"seg_{self.seq:04d}")}
            if name == "deposit_segment":
                rec.update({"from": list(act["from"]), "to": list(act["to"]),
                            "bead_width": act["bead_width"],
                            "bead_height": act["bead_height"]})
                self.pos = [float(v) for v in act["to"]]
            else:
                rec.update({"at": list(act["position"]), "size": act["size"]})
                self.pos = [float(v) for v in act["position"]]
            self.layers[self.active_layer]["segments"].append(rec)
            self.layers[self.active_layer]["path_open"] = True
            return rec
        if name == "complete_path":
            self.layers[self.active_layer]["path_open"] = False
            self.layers[self.active_layer]["paths_done"] += 1
            return {"action": name, "layer": self.active_layer}
        if name == "complete_layer":
            self.active_layer += 1
            return {"action": name, "layer": self.active_layer - 1}
        if name == "complete_object":
            self.object_done = True
            return {"action": name}
        raise ValueError(f"unhandled path action {name!r}")


# ------------------------------------------------ component tracking
# Completion invariant (one identity per component across planning, slicing,
# toolpath generation, execution and reporting):
#   completed  <=>  planned_deposits > 0 and committed_deposits >= planned.
# A required component is NEVER complete with zero planned deposits (nothing
# was sliced for it) — unless the canonical spec explicitly permits a
# non-deposited component. planned==0 with no compile error means the part
# never reached the slicer under this identity (wrong plan paired with the
# script); committed<planned means execution stopped early. Both are
# reported, never inferred from unrelated scene activity.
def component_status(thought: list, script: list, committed: list) -> list:
    """Lifecycle per planned component (one thought primitive each):
    missing (0 committed) / partial / completed, plus started flag.
    Matching is by the explicit component tag the slicer stamps on every
    generated action — never by guessing from free-form labels. Each row
    also carries the planned layer range and the first/last committed
    deposit indices so a missing part names WHERE it vanished (no planned
    layers at all => never sliced; layers planned but nothing committed
    => execution never reached it)."""
    planned = {}
    for act in script:
        comp = act.get("component")
        if comp and act.get("action") in ("deposit_segment", "deposit_point"):
            planned[comp] = planned.get(comp, 0) + 1
    done = {}
    first_idx, last_idx = {}, {}
    for i, act in enumerate(committed):
        comp = act.get("component")
        if comp and act.get("action") in ("deposit_segment", "deposit_point"):
            done[comp] = done.get(comp, 0) + 1
            if comp not in first_idx:
                first_idx[comp] = i
            last_idx[comp] = i
    out = []
    for t in thought:
        label = t.get("label", "?") if isinstance(t, dict) else "?"
        n = planned.get(label, 0)
        k = done.get(label, 0)
        status = ("completed" if n and k >= n
                  else "partial" if k else "missing")
        layers = sorted({a.get("layer") for a in script
                         if a.get("component") == label
                         and a.get("action") in ("deposit_segment",
                                                 "deposit_point")
                         and isinstance(a.get("layer"), int)})
        out.append({"component": label,
                    "action": t.get("action", "?") if isinstance(t, dict) else "?",
                    "planned_deposits": n, "committed_deposits": k,
                    "started": k > 0, "status": status, "layers": layers,
                    "planned_layer_range": [layers[0], layers[-1]]
                    if layers else [],
                    "first_committed_index": first_idx.get(label),
                    "last_committed_index": last_idx.get(label)})
    return out


# ------------------------------------------------ deterministic generators
# Ground truth for training data + the offline demo. Pure functions of
# (layer, geometry) — no model involved.

# One physical bead cross-section for the whole printer. The bead is exactly
# one layer tall so stacked courses share a face (no vertical gap), and is as
# wide as the infill spacing so neighbouring solid-fill lines touch. These are
# properties of the extrusion, never per-segment fudge factors.
BEAD_WIDTH = 0.2
BEAD_HEIGHT = LAYER_HEIGHT

_BEAD = {"bead_width": BEAD_WIDTH, "bead_height": BEAD_HEIGHT}


def trace_rect(layer: int, cx: float, cz: float, w: float, d: float,
               step: float = BEAD_WIDTH, label: str = "path",
               material: str = "body", joins: bool = True):
    """Perimeter of a w×d rectangle centred at (cx,cz) on layer's plane.

    The toolhead is continuous end-to-end: every segment starts exactly where
    the previous ended and the path returns to its start (closed). A small
    join bead is dropped on each of the four corners so a right-angle turn is
    filled rather than leaving a bead-half notch."""
    y = layer_y(layer)
    x0, x1, z0, z1 = cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2
    defpts = []
    x = x0
    while x < x1 - 1e-9:
        x = min(x + step, x1)
        defpts.append([round(x, 3), y, z0])
    c1 = len(defpts) - 1
    z = z0
    while z < z1 - 1e-9:
        z = min(z + step, z1)
        defpts.append([x1, y, round(z, 3)])
    c2 = len(defpts) - 1
    x = x1
    while x > x0 + 1e-9:
        x = max(x - step, x0)
        defpts.append([round(x, 3), y, z1])
    c3 = len(defpts) - 1
    z = z1
    while z > z0 + 1e-9:
        z = max(z - step, z0)
        defpts.append([x0, y, round(z, 3)])
    c4 = len(defpts) - 1
    corners = {c for c in (c1, c2, c3, c4) if c >= 0}
    acts = [{"action": "move_to", "position": [x0, y, z0],
             "layer": layer, "label": f"{label}_start"}]
    prev = [x0, y, z0]
    for i, p in enumerate(defpts):
        acts.append({"action": "deposit_segment", "from": prev, "to": p,
                     **_BEAD, "material": material, "layer": layer,
                     "label": f"{label}_{i:03d}"})
        prev = p
        if joins and i in corners:
            acts.append({"action": "deposit_point", "position": list(p),
                         "size": BEAD_WIDTH, "material": material,
                         "layer": layer, "label": f"{label}_join{i:03d}"})
    acts.append({"action": "complete_path",
                 "layer": layer, "label": f"{label}_done"})
    return acts


def zigzag_infill(layer: int, cx: float, cz: float, w: float, d: float,
                  spacing: float = BEAD_WIDTH, step: float = BEAD_WIDTH,
                  label: str = "fill", material: str = "body"):
    """Back-and-forth infill lines inside the rectangle (serpentine).

    spacing defaults to the bead width so adjacent lines abut and the layer
    is solid, not a hollow frame."""
    y = layer_y(layer)
    x0, x1, z0, z1 = cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2
    acts = []
    n = 0
    z = z0 + spacing
    flip = False
    prev = None
    while z < z1 - 1e-9:
        xs = [x0, x1] if not flip else [x1, x0]
        acts.append({"action": "move_to",
                     "position": [xs[0], y, round(z, 3)],
                     "layer": layer,
                     "label": f"{label}_start" if prev is None
                     else f"{label}_mv{n:03d}"})
        prev = [xs[0], y, round(z, 3)]
        x = xs[0]
        while abs(x - xs[1]) > 1e-9:
            x = min(x + step, xs[1]) if xs[1] > x else max(x - step, xs[1])
            to = [round(x, 3), y, round(z, 3)]
            acts.append({"action": "deposit_segment",
                         "from": prev, "to": to,
                         **_BEAD, "material": material, "layer": layer,
                         "label": f"{label}_{n:03d}"})
            prev = to
            n += 1
        flip = not flip
        z = round(z + spacing, 3)
    if n == 0:
        return []  # footprint too small for infill lines: perimeter suffices
    acts.append({"action": "complete_path",
                 "layer": layer, "label": f"{label}_done"})
    return acts


def trace_ngon(layer: int, cx: float, cz: float, r: float,
               nsides: int = NGON_SIDES, step_len: float = BEAD_WIDTH,
               label: str = "path", material: str = "body",
               joins: bool = True):
    """Regular-polygon perimeter on the layer plane (cylinder/sphere
    footprints). The closed loop returns to its start; a join bead lands on
    each vertex so sharp turns are filled."""
    y = layer_y(layer)
    verts = [[round(cx + r * math.cos(2 * math.pi * k / nsides), 3),
              y,
              round(cz + r * math.sin(2 * math.pi * k / nsides), 3)]
             for k in range(nsides + 1)]
    acts = [{"action": "move_to", "position": verts[0],
             "layer": layer, "label": f"{label}_start"}]
    prev = verts[0]
    n = 0
    for v in verts[1:]:
        edge = math.dist(prev, v)
        steps = max(1, int(math.ceil(edge / step_len)))
        for s in range(1, steps + 1):
            p = [round(prev[0] + (v[0] - prev[0]) * s / steps, 3), y,
                 round(prev[2] + (v[2] - prev[2]) * s / steps, 3)]
            acts.append({"action": "deposit_segment", "from": prev, "to": p,
                         **_BEAD, "material": material, "layer": layer,
                         "label": f"{label}_{n:03d}"})
            prev = p
            n += 1
        if joins:
            acts.append({"action": "deposit_point", "position": list(prev),
                         "size": BEAD_WIDTH, "material": material,
                         "layer": layer, "label": f"{label}_join{n:03d}"})
    acts.append({"action": "complete_path",
                 "layer": layer, "label": f"{label}_done"})
    return acts


def zigzag_disc(layer: int, cx: float, cz: float, r: float,
                spacing: float = BEAD_WIDTH, step: float = BEAD_WIDTH,
                label: str = "fill", material: str = "body"):
    """Back-and-forth infill lines clipped to a circle of radius r. Used to
    fill the interior of a round cross-section (cylinder / sphere) so each
    layer is a slice of the solid, not an empty hoop."""
    y = layer_y(layer)
    acts = []
    n = 0
    z = cz - r + spacing
    flip = False
    prev = None
    while z < cz + r - 1e-9:
        half = math.sqrt(max(0.0, r * r - (z - cz) ** 2))
        if half * 2 < step:  # too thin to print a line here
            z = round(z + spacing, 3)
            flip = not flip
            continue
        xs = [cx - half, cx + half] if not flip else [cx + half, cx - half]
        start = [round(xs[0], 3), y, round(z, 3)]
        acts.append({"action": "move_to", "position": start, "layer": layer,
                     "label": f"{label}_start" if prev is None
                     else f"{label}_mv{n:03d}"})
        prev = start
        x = xs[0]
        while abs(x - xs[1]) > 1e-9:
            x = min(x + step, xs[1]) if xs[1] > x else max(x - step, xs[1])
            to = [round(x, 3), y, round(z, 3)]
            acts.append({"action": "deposit_segment", "from": prev, "to": to,
                         **_BEAD, "material": material, "layer": layer,
                         "label": f"{label}_{n:03d}"})
            prev = to
            n += 1
        flip = not flip
        z = round(z + spacing, 3)
    if n == 0:
        return []
    acts.append({"action": "complete_path",
                 "layer": layer, "label": f"{label}_done"})
    return acts


def trace_cap(layer: int, cx: float, cz: float, size: float,
              label: str = "cap", material: str = "body"):
    """A single round bead on the axis — the very top/bottom slice of a
    sphere, where the cross-section collapses to a point."""
    y = layer_y(layer)
    p = [round(cx, 3), y, round(cz, 3)]
    size = min(SEG_MAX_DIM, max(size, 0.02))
    return [{"action": "move_to", "position": p, "layer": layer,
             "label": f"{label}_start"},
            {"action": "deposit_point", "position": p, "size": size,
             "material": material, "layer": layer, "label": f"{label}_pt"},
            {"action": "complete_path", "layer": layer,
             "label": f"{label}_done"}]


# ------------------------------------------------ compiler
# High-level primitive plan -> deterministic per-layer toolpaths. The thinker
# owns WHAT to build; this compiler owns HOW it prints: real cross-sections
# per horizontal layer (constant for boxes, interpolated radius for tapered
# cylinders), support validation (ground or overlapping lower construction —
# elevated doors, bodies and wheels are valid when supported), and precise
# per-component errors. Components that compile keep their paths across a
# bounded plan retry; only failed components are recompiled.

def _dims(act: dict) -> dict:
    """Validated dimensions. Raises ValueError naming the component."""
    name = act.get("action", "?")
    lab = act.get("label", "?")

    def num(key, minimum=0.0, maximum=MAX_LINEAR):
        v = act.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"{name} {lab!r}: {key} must be a number (got {v!r})")
        if not math.isfinite(v) or v < minimum:
            raise ValueError(
                f"{name} {lab!r}: {key} must be finite and >= {minimum} "
                f"(got {v!r})")
        if v > maximum:
            raise ValueError(
                f"{name} {lab!r}: {key}={v} exceeds the {maximum} scene "
                f"limit — dimensions must fit the build volume")
        return float(v)

    if name == "add_box":
        for key in ("width", "height", "depth"):
            if key not in act:
                raise ValueError(f"{name} {lab!r}: missing {key}")
        w, h, d = num("width", 1e-6), num("height", 1e-6), num("depth", 1e-6)
        return {"w": w, "h": h, "d": d, "half": h / 2.0}
    if name == "add_cylinder":
        if "height" not in act:
            raise ValueError(f"{name} {lab!r}: missing height")
        h = num("height", 1e-6)
        rt = num("radiusTop", 0.0) if "radiusTop" in act else None
        rb = num("radiusBottom", 0.0) if "radiusBottom" in act else None
        if rt is None and rb is None:
            if "radius" not in act:
                raise ValueError(f"{name} {lab!r}: missing radius")
            rt = rb = num("radius", 1e-6)
        else:
            rt = num("radiusTop", 0.0) if rt is not None else rb
            rb = num("radiusBottom", 0.0) if rb is not None else rt
        if max(rt, rb) <= 0:
            raise ValueError(f"{name} {lab!r}: radius must be > 0")
        return {"rt": rt, "rb": rb, "h": h, "half": h / 2.0}
    if name == "add_sphere":
        if "radius" not in act:
            raise ValueError(f"{name} {lab!r}: missing radius")
        r = num("radius", 1e-6)
        return {"r": r, "half": r}
    raise ValueError(f"path mode cannot slice {name!r} (label {lab!r}); "
                     f"supported footprints: {sorted(SLICEABLE)}")


def _radius_at(dims: dict, name: str, y: float, min_y: float,
               max_y: float) -> float:
    """Actual cross-section radius at height y — the real slice of the solid:
    constant for a box, linearly interpolated for a tapered cylinder, and the
    circle r(y)=sqrt(R^2-(y-c)^2) for a sphere."""
    if name == "add_sphere":
        r = dims["r"]
        c = (min_y + max_y) / 2.0
        return math.sqrt(max(0.0, r * r - (y - c) ** 2))
    if name != "add_cylinder" or max_y <= min_y:
        return max(dims.get("rt", 0.5), dims.get("rb", 0.5))
    t = min(1.0, max(0.0, (y - min_y) / (max_y - min_y)))
    return dims["rb"] + (dims["rt"] - dims["rb"]) * t


def _check_support(label: str, foot: tuple, min_y: float, k0: int,
                   occupancy: dict) -> str:
    """Support for a component footprint. Returns an error string or ''."""
    if min_y <= GROUND_TOL:
        return ""
    if k0 <= 0:
        return (f"{label!r} floats (base y={min_y:.2f}) with nothing below; "
                f"rest it on the ground or stack it on earlier construction")
    minx, maxx, minz, maxz = foot[:4]
    for entry in occupancy.get(k0 - 1, []):
        ox0, ox1, oz0, oz1 = entry[:4]
        if (minx < ox1 + SUPPORT_EPS and ox0 < maxx + SUPPORT_EPS
                and minz < oz1 + SUPPORT_EPS and oz0 < maxz + SUPPORT_EPS):
            return ""
    # Exact-contact support: lower construction whose top IS the base plane.
    # Thin single-layer supporters (a 0.12 seat inside one 0.2 course) never
    # reach layer k0-1, yet stacking exactly on their top is valid printing.
    if abs(_support_height(foot, min_y, occupancy) - min_y) <= SUPPORT_EPS:
        return ""
    return (f"{label!r} floats (base y={min_y:.2f}): no supporting geometry "
            f"in layer {k0 - 1} beneath its footprint")


def _support_height(foot: tuple, min_y: float, occupancy: dict) -> float:
    """Highest supporting surface at or below the footprint: the max top of
    overlapping lower footprints, else the ground (0.0)."""
    best = 0.0
    minx, maxx, minz, maxz = foot[:4]
    for layer, entries in occupancy.items():
        for entry in entries:
            ox0, ox1, oz0, oz1, top = entry
            if not (minx < ox1 + SUPPORT_EPS and ox0 < maxx + SUPPORT_EPS
                    and minz < oz1 + SUPPORT_EPS and oz0 < maxz + SUPPORT_EPS):
                continue
            if top <= min_y + SUPPORT_EPS:
                best = max(best, top)
    return best


def compile_component(act: dict, occupancy: dict,
                      prefix: str = "p", allow_seating: bool = True) -> tuple:
    """Compile ONE primitive into (groups, footprint, (min_y, max_y),
    warnings). Bottom layer gets perimeter + infill; upper layers perimeter.
    Every path action is stamped with component=<primitive label>.
    Sunk or floating parts are seated onto their support (ground or lower
    construction) — dimensions, XZ and IDs untouched — and the seating is
    reported in warnings, never silent. Raises ValueError with a
    component-specific message (unsupported action, bad dims, unknown
    on_top_of ref, or support that cannot be established)."""
    from morphix_state import SINK_TOL
    name = act.get("action", "?")
    comp = act.get("label", "?")
    if name not in SLICEABLE:
        raise ValueError(
            f"path mode cannot slice {name!r} (label {comp!r}); "
            f"supported footprints: {sorted(SLICEABLE)}")
    dims = _dims(act)
    mat = act.get("material", "body")
    pos = act.get("position", [0, 0, 0])
    try:
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{name} {comp!r}: position must be [x,y,z] numbers")
    if name == "add_box":
        w, d = dims["w"], dims["d"]
        foot = (cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2)

        def trace(layer, lab, yy=None):
            return trace_rect(layer, cx, cz, w, d, label=lab,
                              material=mat)
    elif name == "add_sphere":
        R = dims["r"]
        foot = (cx - R, cx + R, cz - R, cz + R)

        def trace(layer, lab, yy=None):
            yv = layer_y(layer) if yy is None else yy
            r = _radius_at(dims, name, yv, cy - R, cy + R)
            if r < BEAD_WIDTH:
                # cross-section collapsed to (near) a point: a bead, not a
                # zero-length ring. Size bridges the apex and the contact.
                return trace_cap(layer, cx, cz, max(2 * r, BEAD_WIDTH),
                                 label=lab, material=mat)
            return trace_ngon(layer, cx, cz, r, label=lab, material=mat)
    else:
        rmax = max(dims["rt"], dims["rb"])
        foot = (cx - rmax, cx + rmax, cz - rmax, cz + rmax)

        def trace(layer, lab, yy=None):
            r = _radius_at(dims, name,
                           layer_y(layer) if yy is None else yy,
                           cy - dims["half"], cy + dims["half"])
            return trace_ngon(layer, cx, cz, max(r, 0.05), label=lab,
                              material=mat)
    warnings = []
    min_y, max_y = cy - dims["half"], cy + dims["half"]
    k0 = max(0, int(math.floor((min_y + 1e-9) / LAYER_HEIGHT)))
    sunk = min_y < -SINK_TOL
    err = _check_support(comp, foot, min_y, k0, occupancy)
    if (sunk or err) and not allow_seating:
        # fixpoint pass: supporters may simply not exist yet — defer to the
        # final pass instead of auto-seating valid geometry onto nothing.
        if sunk:
            raise ValueError(f"{name} {comp!r} sinks below ground "
                             f"(base y={min_y:.2f})")
        raise ValueError(err)
    if sunk or err:
        sup_h = _support_height(foot, min_y, occupancy)
        if abs(sup_h - min_y) > 1e-9:
            warnings.append(f"seated base y={min_y:.2f} onto support "
                            f"y={sup_h:.2f}")
            cy += sup_h - min_y
            min_y, max_y = cy - dims["half"], cy + dims["half"]
            k0 = max(0, int(math.floor((min_y + 1e-9) / LAYER_HEIGHT)))
            err = _check_support(comp, foot, min_y, k0, occupancy)
            if min_y < -SINK_TOL or err:
                raise ValueError(err or f"{name} {comp!r} sinks below "
                                 f"ground (base y={min_y:.2f})")
        elif sunk:
            raise ValueError(f"{name} {comp!r} sinks below ground "
                             f"(base y={min_y:.2f})")
        else:
            raise ValueError(err)
    top = max(k0, int(math.ceil((max_y - 1e-9) / LAYER_HEIGHT)) - 1)
    if name == "add_sphere":
        # a sphere's apex belongs to its silhouette: keep the last layer at or
        # below max_y so the cross-section tapers to a cap, not a flat disc.
        top = max(k0, int(math.floor(max_y / LAYER_HEIGHT + 1e-9)))
    groups = {}
    for k in range(k0, top + 1):
        yk = layer_y(k)
        if yk + BEAD_HEIGHT / 2 < min_y - 1e-9:
            # Course plane entirely below the solid's base: printing here
            # would lay a phantom slab inside/below the supporting geometry
            # (observed: backrest course embedded in the seat courses).
            # The first course is the one whose bead reaches the base.
            continue
        per = trace(k, f"{prefix}_l{k}")
        if name == "add_box":
            # solid fill on every layer — a box is solid, not stacked hoops
            fill = zigzag_infill(k, cx, cz, foot[1] - foot[0],
                                 foot[3] - foot[2],
                                 label=f"{prefix}_l{k}_fill", material=mat)
            per = per + fill
        else:
            r = _radius_at(dims, name, yk, cy - dims["half"],
                           cy + dims["half"])
            if r >= BEAD_WIDTH * 1.5:
                per = per + zigzag_disc(k, cx, cz, r,
                                        label=f"{prefix}_l{k}_fill",
                                        material=mat)
        groups[k] = [dict(a, component=comp) for a in per]
    if not groups:
        # Slicing stage failure, reported LOUDLY with the component and the
        # layer band: without this, the part would compile "successfully"
        # with zero deposits and die far downstream as an unexplained
        # "missing" component. A sub-layer sliver stacked high in a layer
        # band (base above every course plane that spans it) is the usual
        # cause — genuinely too thin to print, never silently printed.
        raise ValueError(
            f"{name} {comp!r} sliced to zero layers "
            f"(slicing stage: solid y {min_y:.2f}..{max_y:.2f}, layer band "
            f"{k0}..{top} at {LAYER_HEIGHT:.2f}/course — every candidate "
            f"course plane sits below the solid base, so no contour "
            f"exists to print)")
    return groups, foot, (min_y, max_y), warnings


def compile_plan(actions: list, keep: dict = None) -> tuple:
    """Compile a whole thought plan. Returns (compiled, order, errors,
    keep_out, warns) where compiled[label] = {layer: trace}, order preserves
    plan order, errors[label] is a precise per-component message, keep_out
    carries reusable {action, groups, footprint, yrange, warnings} for a
    bounded retry, and warns[label] lists auto-seatings (loud, never silent).
    on_top_of refs resolve against earlier plan components (heights derived
    from their tops, never guessed). `keep` reuses byte-identical components
    without recompiling — but their support is re-validated every attempt."""
    compiled, order, errors, warns = {}, [], {}, {}
    keep = keep or {}
    keep_out = {}
    geo = {}        # label -> {"foot":..., "top":..., "bottom":...}
    occupancy = {}  # layer -> [(minx,maxx,minz,maxz,top)]

    def register(label, foot, min_y, max_y):
        geo[label] = {"foot": foot, "top": max_y, "bottom": min_y}
        k0 = max(0, int(math.floor((min_y + 1e-9) / LAYER_HEIGHT)))
        k1 = max(k0, int(math.ceil((max_y - 1e-9) / LAYER_HEIGHT)) - 1)
        for k in range(k0, k1 + 1):
            occupancy.setdefault(k, []).append(foot + (max_y,))

    def attempt(idx, label, act, final):
        """Try one component. Returns (ok, deferred, error). Only success
        mutates compiled/occupancy/keep_out/warns. During fixpoint passes
        (final=False) any support/geometry failure defers so lower
        components can compile first; the final pass (final=True) seats or
        fails loudly. Structural problems (non-sliceable) fail at once."""
        if not isinstance(act, dict) or act.get("action") not in SLICEABLE:
            return False, False, (
                f"unsupported action {act.get('action')!r}: printer mode "
                f"compiles {sorted(SLICEABLE)} footprints only")
        old = keep.get(label)
        if old is not None and old.get("action") == act:
            # identical work: reuse paths, re-validate support in this order
            foot, (lo, hi) = old["footprint"], old["yrange"]
            k0 = max(0, int(math.floor((lo + 1e-9) / LAYER_HEIGHT)))
            err = _check_support(label, foot, lo, k0, occupancy)
            if err:
                if final:
                    return False, False, \
                        err + " (plan changed underneath it)"
                return False, True, \
                    err + " (plan changed underneath it)"
            compiled[label] = old["groups"]
            keep_out[label] = old
            if old.get("warnings"):
                warns[label] = list(old["warnings"])
            register(label, foot, lo, hi)
            return True, False, ""
        # resolve stacking refs against already-compiled plan geometry
        work = dict(act)
        ref = work.get("on_top_of")
        if ref:
            sup = geo.get(ref)
            if sup is None:
                if final:
                    return False, False, (
                        f"on_top_of unknown label {ref!r}: "
                        f"stack onto an earlier plan component")
                return False, True, (
                    f"on_top_of unknown label {ref!r}: "
                    f"stack onto an earlier plan component")
            try:
                half = _dims(work)["half"]
                cx = ((sup["foot"][0] + sup["foot"][1]) / 2
                      + float(work.get("dx", 0)))
                cz = ((sup["foot"][2] + sup["foot"][3]) / 2
                      + float(work.get("dz", 0)))
            except (TypeError, ValueError) as e:
                return False, False, f"bad stacking fields: {e}"
            pos = list(work.get("position", [0, 0, 0]))
            pos[0], pos[1], pos[2] = cx, sup["top"] + half, cz
            work["position"] = pos
        try:
            groups, foot, (lo, hi), comp_warns = compile_component(
                work, occupancy, prefix=f"p{idx:02d}",
                allow_seating=final)
        except ValueError as e:
            if final:
                return False, False, str(e)
            return False, True, str(e)
        compiled[label] = groups
        keep_out[label] = {"action": dict(act), "groups": groups,
                           "footprint": foot, "yrange": (lo, hi),
                           "warnings": comp_warns}
        if comp_warns:
            warns[label] = list(comp_warns)
        register(label, foot, lo, hi)
        return True, False, ""

    for i, act in enumerate(actions):
        label = (act.get("label", f"part_{i:02d}")
                 if isinstance(act, dict) else f"part_{i:02d}")
        order.append(label)
    # Bottom-up fixpoint: support-unresolved components wait for their
    # supporters instead of failing on plan order. Truly broken components
    # surface the same errors once nothing else can progress.
    pending = [(i, label, actions[i]) for i, label in enumerate(order)]
    while pending:
        progressed = False
        still = []
        for idx, label, act in pending:
            ok, deferred, err = attempt(idx, label, act, final=False)
            if ok:
                progressed = True
            elif deferred:
                still.append((idx, label, act))
            else:
                errors[label] = err
        if not still:
            break
        if not progressed:
            for idx, label, act in still:
                ok, _, err = attempt(idx, label, act, final=True)
                if not ok:
                    errors[label] = err
            break
        pending = still
    return compiled, order, errors, keep_out, warns


def merge_layers(groups: list) -> tuple:
    """Merge per-primitive layer groups into one global script: for each
    global layer k, every primitive's trace, then one complete_layer k.
    Returns (script, total_layers)."""
    script = []
    top = max(([max(g) for g in groups if g] or [-1]))
    for k in range(top + 1):
        for g in groups:
            script += g.get(k, [])
        script.append({"action": "complete_layer", "layer": k,
                       "label": f"layer_{k:02d}_done"})
    return script, top + 1


# ------------------------------------------------ geometry-level validation
# Action-schema checks prove the protocol is legal; they do NOT prove the
# toolpath is connected. These checks re-derive the toolhead from the raw
# action list (independently of the generators) so a regression in
# trace_rect / trace_ngon / zigzag_* is caught before a single bead commits.

def validate_geometry(script: list) -> list:
    """Structural checks over a compiled script. Returns error strings; an
    empty list means: every deposit starts at the live toolhead (endpoint
    continuity), no zero-length or oversized segment exists, and every
    perimeter path closes on itself (no missing edge)."""
    errs = []
    head = None
    path_start = None
    for i, a in enumerate(script):
        if not isinstance(a, dict):
            errs.append(f"[{i}] action is not an object")
            continue
        name = a.get("action")
        label = a.get("label", "?")
        if name == "move_to":
            p = a.get("position")
            if not _vec(p):
                errs.append(f"[{i}] move_to {label}: bad position")
            else:
                head = [float(v) for v in p]
                path_start = list(head)
        elif name == "deposit_segment":
            frm, to = a.get("from"), a.get("to")
            if not _vec(frm) or not _vec(to):
                errs.append(f"[{i}] segment {label}: from/to must be [x,y,z]")
                continue
            if head is None or math.dist(frm, head) > POS_TOL:
                errs.append(f"[{i}] segment {label} starts at {frm} but the "
                            f"toolhead is {head} (broken continuity)")
            length = math.dist(frm, to)
            if length < POS_TOL:
                errs.append(f"[{i}] segment {label} has zero length")
            elif length > SEG_MAX_LEN:
                errs.append(f"[{i}] segment {label} length {length:.2f} "
                            f"exceeds {SEG_MAX_LEN}")
            head = [float(v) for v in to]
        elif name == "deposit_point":
            p = a.get("position")
            if not _vec(p):
                errs.append(f"[{i}] point {label}: bad position")
                continue
            if head is None or math.dist(p, head) > POS_TOL:
                errs.append(f"[{i}] point {label} at {p} is not at the "
                            f"toolhead {head} (broken continuity)")
            head = [float(v) for v in p]
        elif name == "complete_path":
            # infill is serpentine and may end anywhere; traced perimeters
            # must return to where they started (closed contour).
            if (path_start is not None and head is not None
                    and "fill" not in label
                    and math.dist(path_start, head) > POS_TOL):
                errs.append(f"[{i}] perimeter {label} is not closed: started "
                            f"at {path_start}, ended at {head}")
            path_start = None
    return errs


# ------------------------------------------------ toolpath optimization
# Post-pass over a compiled script: fewer actions, same geometry. The
# generators above are untouched (they are training-data ground truth);
# this pass only removes provably redundant work and reorders travel:
#   - zero-length segments and duplicate consecutive points are dropped;
#   - consecutive collinear same-layer/same-material/same-component segments
#     merge into single runs capped at SEG_MAX_LEN (a 3 m wall prints as one
#     3 m bead instead of fifteen 0.2 beads — identical endpoints);
#   - disconnected contours inside one layer are visited nearest-first so
#     the head stops criss-crossing the plate;
#   - move_to actions whose target IS the live toolhead are dropped.
# Contours are never merged across complete_path boundaries, layers are never
# reordered, bead dimensions/materials are never altered, and join beads,
# corners, holes and thin features are preserved (a join deposit_point always
# breaks a merge run). The result is re-validated with validate_geometry;
# any defect falls back to the unoptimized script loudly, never silently.
OPT_MIN_SPLIT = SEG_MAX_LEN


def _seg_len(a) -> float:
    return math.dist(a["from"], a["to"])


def _collinear_mergeable(prev_from, prev_to, nxt_to) -> bool:
    """prev_from->prev_to then prev_to->nxt_to fold into prev_from->nxt_to?"""
    if prev_from[1] != prev_to[1] or prev_to[1] != nxt_to[1]:
        return False
    ux, uz = prev_to[0] - prev_from[0], prev_to[2] - prev_from[2]
    vx, vz = nxt_to[0] - prev_to[0], nxt_to[2] - prev_to[2]
    if ux == 0 and uz == 0:
        return False
    cross = ux * vz - uz * vx
    if abs(cross) > 1e-9:
        return False
    return (ux * vx + uz * vz) > 0


def _same_run(a, b) -> bool:
    return (a.get("layer") == b.get("layer")
            and a.get("material") == b.get("material")
            and a.get("component") == b.get("component")
            and a.get("bead_width") == b.get("bead_width")
            and a.get("bead_height") == b.get("bead_height"))


def optimize_toolpath(script: list,
                      max_seg_len: float = SEG_MAX_LEN) -> tuple:
    """Returns (optimized_script, stats). Pure function; input untouched."""
    stats = {"before": len(script), "after": 0, "removed_zero": 0,
             "removed_dup_point": 0, "merged_segments": 0,
             "split_segments": 0, "removed_travel": 0,
             "travel_dist_before": 0.0, "travel_dist_after": 0.0,
             "fallback": False, "layers": {}}
    if not script:
        stats["after"] = 0
        return [], stats

    # -- split into layer chunks (complete_layer / complete_object stay put)
    chunks, cur = [], []
    for a in script:
        cur.append(a)
        if a.get("action") in ("complete_layer", "complete_object"):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    def travel_of(actions):
        d, head = 0.0, None
        for a in actions:
            if a.get("action") == "move_to" and _vec(a.get("position")):
                if head is not None:
                    d += math.dist(head, a["position"])
                head = list(a["position"])
            elif a.get("action") == "deposit_segment":
                head = list(a["to"])
            elif a.get("action") == "deposit_point":
                head = list(a["position"])
        return d

    stats["travel_dist_before"] = round(travel_of(script), 3)
    out = []
    for chunk in chunks:
        # -- split chunk into contour units (move_to..complete_path); other
        # -- actions (complete_layer/object) pin the chunk end in place.
        units, unit, tail = [], [], []
        for a in chunk:
            if a.get("action") == "complete_layer" \
                    or a.get("action") == "complete_object":
                tail.append(a)
            elif a.get("action") == "move_to" and unit:
                units.append(unit)
                unit = [a]
            else:
                unit.append(a)
        if unit:
            units.append(unit)

        def unit_start(u):
            for a in u:
                if a.get("action") == "move_to" and _vec(a.get("position")):
                    return list(a["position"])
                if a.get("action") == "deposit_segment" and _vec(a.get("from")):
                    return list(a["from"])
                if a.get("action") == "deposit_point" and _vec(a.get("position")):
                    return list(a["position"])
            return None

        def unit_end(u):
            end = None
            for a in u:
                if a.get("action") == "deposit_segment" and _vec(a.get("to")):
                    end = list(a["to"])
                elif a.get("action") == "deposit_point" \
                        and _vec(a.get("position")):
                    end = list(a["position"])
                elif a.get("action") == "move_to" \
                        and _vec(a.get("position")) and end is None:
                    end = list(a["position"])
            return end

        # -- greedy nearest-first visit order (deterministic: ties keep plan
        # -- order via stable sort on (distance, original index)).
        order = sorted(range(len(units)),
                       key=lambda i: (0 if i == 0 else 1, i)) if units else []
        if units:
            head = unit_end(units[0])
            remaining = [u for j, u in enumerate(units) if j != 0]
            ordered = [units[0]]
            while remaining:
                best = min(
                    range(len(remaining)),
                    key=lambda j: (
                        math.dist(head, unit_start(remaining[j]))
                        if head is not None and unit_start(remaining[j])
                        is not None else float("inf"), j))
                ordered.append(remaining.pop(best))
                end = unit_end(ordered[-1])
                if end is not None:
                    head = end
        else:
            ordered = []

        # -- stitch + merge pass: replay the toolhead, drop redundant
        # -- travel, fold collinear runs.
        head = None
        pending = None  # open merge run [first_action_dict, start, end]
        layer_tbl = stats["layers"]

        def layer_key(a):
            return a.get("layer", "?")

        def flush_pending():
            nonlocal pending
            if pending is not None:
                first, start, end = pending
                total = math.dist(start, end)
                if total <= max_seg_len + 1e-9:
                    seg = dict(first)
                    seg["from"], seg["to"] = start, end
                    out.append(seg)
                    lk = layer_key(first)
                    layer_tbl[lk] = layer_tbl.get(lk, 0) + 1
                else:
                    # oversize single runs are chopped into valid chunks
                    # (same line, same endpoints) rather than passed on to
                    # fail validation downstream.
                    n = int(math.ceil(total / max_seg_len))
                    for k in range(n):
                        a = [start[d] + (end[d] - start[d]) * k / n
                             for d in range(3)]
                        b = [start[d] + (end[d] - start[d]) * (k + 1) / n
                             for d in range(3)]
                        seg = dict(first)
                        seg["from"], seg["to"] = (
                            [round(v, 3) for v in a],
                            [round(v, 3) for v in b])
                        if k:
                            seg["label"] = f"{first.get('label', '?')}#c{k}"
                        out.append(seg)
                        stats["split_segments"] = \
                            stats.get("split_segments", 0) + 1
                        lk = layer_key(first)
                        layer_tbl[lk] = layer_tbl.get(lk, 0) + 1
                pending = None

        def note(a):
            lk = layer_key(a)
            layer_tbl[lk] = layer_tbl.get(lk, 0) + 1

        for u in ordered:
            for a in u:
                name = a.get("action")
                if name == "move_to":
                    tgt = a.get("position")
                    if head is not None and _vec(tgt) and \
                            math.dist(head, tgt) <= POS_TOL:
                        # redundant travel: drop it WITHOUT flushing the open
                        # merge run, so collinear segments on both sides of
                        # the dropped hop still fold into one bead.
                        stats["removed_travel"] += 1
                        continue
                    flush_pending()
                    out.append(a)
                    note(a)
                    if _vec(tgt):
                        head = list(tgt)
                elif name == "deposit_segment":
                    frm, to = a.get("from"), a.get("to")
                    if not (_vec(frm) and _vec(to)):
                        flush_pending()
                        out.append(a)
                        note(a)
                        continue
                    if math.dist(frm, to) < POS_TOL:
                        stats["removed_zero"] += 1
                        continue
                    if pending is not None:
                        pfirst, pstart, pend = pending
                        if _same_run(pfirst, a) and \
                                _collinear_mergeable(pstart, pend, to):
                            if math.dist(pstart, to) <= max_seg_len + 1e-9:
                                pending[2] = list(to)
                                stats["merged_segments"] += 1
                                head = list(to)
                                continue
                            flush_pending()
                    # start/continue run only if it chains on the live head
                    if pending is None:
                        pending = [a, list(frm), list(to)]
                    else:
                        flush_pending()
                        pending = [a, list(frm), list(to)]
                    head = list(to)
                else:
                    # deposit_point joins, complete_path, anything else:
                    # always break a merge run, always preserved.
                    if name == "deposit_point":
                        p = a.get("position")
                        if head is not None and _vec(p) and \
                                list(p) == list(head):
                            pass
                        elif head is not None and _vec(p) and \
                                math.dist(head, p) < POS_TOL:
                            stats["removed_dup_point"] += 1
                            out.append(a)
                            note(a)
                            continue
                    flush_pending()
                    out.append(a)
                    note(a)
                    if name == "deposit_point" and _vec(a.get("position")):
                        head = list(a["position"])
        flush_pending()
        out.extend(tail)
        for a in tail:
            note(a)

    stats["after"] = len(out)
    stats["travel_dist_after"] = round(travel_of(out), 3)
    problems = validate_geometry(out)
    if problems:
        stats["fallback"] = True
        stats["fallback_errors"] = problems[:5]
        stats["after"] = len(script)
        stats["travel_dist_after"] = stats["travel_dist_before"]
        return list(script), stats
    return out, stats


# ------------------------------------------------ deterministic rendering
# Engine-side Three.js for a VALIDATED segment/point. The model proposes the
# action; the engine draws exactly it, in the validated material color.

def render_deposit(rec: dict) -> str:
    """Three.js snippet for one committed deposition record: direction from
    A to B, length-derived box oriented along the segment at its midpoint
    (lookAt handles vertical and diagonal segments), point beads as spheres.
    Exactly one local segment per call — never a whole primitive."""
    color = MATERIAL_COLORS.get(rec.get("material", "body"),
                                MATERIAL_COLORS["body"])
    hexcolor = f"0x{color:06X}"
    lab = rec.get("label", f"seg_{rec.get('seq', 0):04d}")
    mat = (f"new THREE.MeshStandardMaterial({{color:{hexcolor},"
           f"roughness:0.6,metalness:0.1}})")
    if rec.get("action") == "deposit_point":
        x, y, z = rec["at"]
        r = rec["size"] / 2.0
        return (
            f"const bead_{lab} = new THREE.Mesh(\n"
            f"  new THREE.SphereGeometry({r!r}, 12, 12),\n"
            f"  {mat});\n"
            f"bead_{lab}.position.set({x!r}, {y!r}, {z!r});\n"
            f"bead_{lab}.castShadow = true;\n"
            f"bead_{lab}.receiveShadow = true;\n"
            f"scene.add(bead_{lab});"
        )
    fx, fy, fz = rec["from"]
    tx, ty, tz = rec["to"]
    dx, dy, dz = tx - fx, ty - fy, tz - fz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    w, h = rec["bead_width"], rec["bead_height"]
    mx, my, mz = (fx + tx) / 2, (fy + ty) / 2, (fz + tz) / 2
    return (
        f"const seg_{lab} = new THREE.Mesh(\n"
        f"  new THREE.BoxGeometry({w!r}, {h!r}, {length!r}),\n"
        f"  {mat});\n"
        f"seg_{lab}.position.set({mx!r}, {my!r}, {mz!r});\n"
        f"seg_{lab}.lookAt({tx!r}, {ty!r}, {tz!r});\n"
        f"seg_{lab}.castShadow = true;\n"
        f"seg_{lab}.receiveShadow = true;\n"
        f"scene.add(seg_{lab});"
    )


# ------------------------------------------------ deterministic rendering
INSPECT_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Morphix Inspect</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#111}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.152.2/build/three.module.js","three/addons/":"https://unpkg.com/three@0.152.2/examples/jsm/"}}</script>
</head><body>
<script type="module">
import * as THREE from 'three';
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111622);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 500);
camera.position.set(1.7, 1.5, 2.3);
camera.lookAt(0, 0.25, 0);
const morphix_mixers = [], morphix_shaders = [], morphix_physics = [];
const grid = new THREE.GridHelper(10, 10, 0x334455, 0x223344);
scene.add(grid);
const amb = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(amb);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(6, 10, 7);
sun.castShadow = true;
scene.add(sun);
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
