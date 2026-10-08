# morphix_state.py
# Live scene state for incremental generation. Each new action is resolved
# against current geometry (not blind absolute coordinates):
#   - {"on_top_of": "base_01"} places the object on that object's top surface
#   - collision / bounds / duplicate-label checks reject bad placements
#   - thinker receives a compact summary of what exists so far
#
# Usage:
#   from morphix_state import SceneState
#   st = SceneState()
#   act = st.resolve({"action":"add_sphere","radius":0.5,"on_top_of":"base_01",
#                     "color":"0xFF0000","label":"ball_02"})
#   st.add(act)

import math

GROUND_Y = 0.0
BOUND_XZ = 25.0
MAX_Y = 30.0
ATTACH_DIST = 6.0  # printer-style: new physical parts must grow near existing geometry
_NO_FOOTPRINT = {"add_light", "add_shader", "add_particles", "add_animation",
                 "add_post", "add_environment", "set_material"}


def _half_height(act: dict) -> float:
    a = act.get("action", "")
    g = lambda k, d=0.5: float(act.get(k, d))
    if a == "add_sphere":
        return g("radius")
    if a == "add_box":
        return g("height") / 2.0
    if a in ("add_cylinder", "add_cone"):
        return g("height") / 2.0
    if a == "add_torus":
        return g("tube", 0.2)
    if a == "add_extrude":
        return g("height", 1.5) / 2.0
    if a == "add_terrain":
        return g("amplitude", 1.5)
    if a == "add_plane":
        return 0.05
    return 0.25


def _half_footprint(act: dict) -> tuple:
    a = act.get("action", "")
    g = lambda k, d=0.5: float(act.get(k, d))
    if a == "add_sphere":
        r = g("radius")
        return r, r
    if a == "add_box":
        return g("width", 1) / 2.0, g("depth", 1) / 2.0
    if a == "add_cylinder":
        r = max(g("radiusTop", g("radius")), g("radiusBottom", g("radius")))
        return r, r
    if a == "add_cone":
        r = g("radius")
        return r, r
    if a == "add_torus":
        r = g("radius", 0.8) + g("tube", 0.2)
        return r, r
    if a == "add_torus_knot":
        r = g("radius", 0.6) + g("tube", 0.15)
        return r, r
    if a == "add_extrude":
        return g("width", 4) / 2.0, g("depth", 2) / 2.0
    if a == "add_terrain":
        r = g("width", 20) / 2.0
        return r, r
    if a == "add_plane":
        return g("width", 1) / 2.0, g("height", 1) / 2.0
    return 0.25, 0.25


def _aabb(pos: list, hx: float, hy: float, hz: float) -> dict:
    return {"min": [pos[0] - hx, pos[1] - hy, pos[2] - hz],
            "max": [pos[0] + hx, pos[1] + hy, pos[2] + hz]}


def aabb_for(act: dict, pos: list):
    """Action-aware bounding box. Most meshes are centered on position, but
    anchored geometry (extrude prisms grow +x/+y/+z from position) is not."""
    if act.get("action") == "add_extrude":
        w = float(act.get("width", 4))
        h = float(act.get("height", 1.5))
        d = float(act.get("depth", 2))
        return {"min": [pos[0] - w / 2, pos[1], pos[2]],
                "max": [pos[0] + w / 2, pos[1] + h, pos[2] + d]}
    hx, hz = _half_footprint(act)
    return _aabb(pos, hx, _half_height(act), hz)


def _overlaps(a: dict, b: dict, eps: float = 1e-6) -> bool:
    # strict overlap: merely touching faces is allowed (stacking)
    for i in range(3):
        if a["min"][i] >= b["max"][i] - eps or b["min"][i] >= a["max"][i] - eps:
            return False
    return True


class SceneState:
    def __init__(self):
        self.objects = []  # {label, action, pos, aabb, color}
        self.head = [0.0, 0.0, 0.0]  # print head: next piece grows from here

    # ------------------------------------------------ queries
    def labels(self) -> set:
        return {o["label"] for o in self.objects}

    def get(self, label: str):
        for o in self.objects:
            if o["label"] == label:
                return o
        return None

    def top_y(self, x: float, z: float) -> float:
        """Top surface height at (x,z): highest object footprint covering it, else ground."""
        top = GROUND_Y
        for o in self.objects:
            if o["aabb"] is None:
                continue
            mn, mx = o["aabb"]["min"], o["aabb"]["max"]
            if mn[0] - 1e-6 <= x <= mx[0] + 1e-6 and mn[2] - 1e-6 <= z <= mx[2] + 1e-6:
                top = max(top, mx[1])
        return top

    def collides(self, aabb: dict, ignore: str = "") -> str:
        """Return colliding object's label, or '' if free. Non-physical acts pass None."""
        if aabb is None:
            return ""
        for o in self.objects:
            if o["label"] == ignore or o["aabb"] is None:
                continue
            if _overlaps(aabb, o["aabb"]):
                return o["label"]
        return ""

    def suggest_spot(self, act: dict):
        """Find a free position for act's footprint, ordered by distance from
        the print head so the scene grows outward from one point, 3D-printer
        style, instead of jumping around."""
        if act.get("action") in _NO_FOOTPRINT:
            return None
        anchored = act.get("action") == "add_extrude"
        hy = 0.0 if anchored else _half_height(act)
        cands = []
        for r in (0, 2.5, 5, 7.5, 10, 15, 20):
            for cx, cz in ((r, 0), (0, r), (-r, 0), (0, -r),
                           (r, r), (-r, -r), (r, -r), (-r, r)):
                if abs(cx) > BOUND_XZ or abs(cz) > BOUND_XZ:
                    continue
                y = self.top_y(cx, cz) + hy
                if y > MAX_Y:
                    continue
                if not self.collides(aabb_for(act, [cx, y, cz])):
                    d = math.dist((cx, cz), (self.head[0], self.head[2]))
                    cands.append((d, [round(cx, 3), round(y, 3), round(cz, 3)]))
        if not cands:
            return None
        cands.sort(key=lambda t: t[0])
        return cands[0][1]

    def nearest_dist(self, pos: list) -> float:
        """Distance from pos to the closest existing physical object (inf if empty)."""
        best = math.inf
        for o in self.objects:
            if o["aabb"] is None:
                continue
            mn, mx = o["aabb"]["min"], o["aabb"]["max"]
            dx = max(mn[0] - pos[0], 0, pos[0] - mx[0])
            dy = max(mn[1] - pos[1], 0, pos[1] - mx[1])
            dz = max(mn[2] - pos[2], 0, pos[2] - mx[2])
            best = min(best, math.sqrt(dx * dx + dy * dy + dz * dz))
        return best

    def update_head(self, pos: list):
        self.head = [float(pos[0]), float(pos[1]), float(pos[2])]

    # ------------------------------------------------ mutation
    def resolve(self, act: dict) -> dict:
        """Resolve state-dependent refs to absolute coordinates. Returns a copy.
        Supports: {"on_top_of": "<label>", "dx": 0, "dz": 0}."""
        act = dict(act)
        ref = act.get("on_top_of")
        if ref:
            target = self.get(ref)
            if target is None:
                raise ValueError(f"on_top_of unknown label {ref!r}")
            if target["aabb"] is None:
                raise ValueError(f"on_top_of non-physical object {ref!r}")
            mn, mx = target["aabb"]["min"], target["aabb"]["max"]
            cx = (mn[0] + mx[0]) / 2.0 + float(act.get("dx", 0))
            cz = (mn[2] + mx[2]) / 2.0 + float(act.get("dz", 0))
            # anchored geometry (extrude) sits ON the surface; centered meshes
            # offset by half-height so their base touches it
            hy = 0.0 if act.get("action") == "add_extrude" else _half_height(act)
            y = self.top_y(cx, cz) + hy
            act["position"] = [round(cx, 3), round(y, 3), round(cz, 3)]
        act.setdefault("position", [0, _half_height(act), 0])
        p = act["position"]
        act["position"] = [float(p[0]), float(p[1]), float(p[2])]
        return act

    def add(self, act: dict):
        """Record a successfully built action."""
        pos = act.get("position", [0, 0, 0])
        aabb = None
        if act.get("action") not in _NO_FOOTPRINT:
            aabb = aabb_for(act, pos)
        self.objects.append({"label": act.get("label", "?"),
                             "action": act.get("action", "?"),
                             "pos": list(pos), "aabb": aabb,
                             "color": act.get("color", "")})

    # ------------------------------------------------ prompt context
    def summary(self) -> str:
        head = f"Print head at {self.head} — grow the next piece from here."
        if not self.objects:
            return f"Scene is empty. Ground is y=0. {head}"
        lines = []
        for o in self.objects:
            if o["aabb"] is None:
                lines.append(f"- {o['action']} {o['label']} (non-physical)")
            else:
                mx = o["aabb"]["max"]
                lines.append(f"- {o['action']} {o['label']} center {o['pos']} top_y={mx[1]:.2f}")
        return ("Current scene:\n" + "\n".join(lines)
                + f"\nGround is y=0. {head}")


if __name__ == "__main__":
    # self-test: stacking, collision, top query
    st = SceneState()
    assert st.top_y(0, 0) == 0.0
    box = st.resolve({"action": "add_box", "width": 2, "height": 0.3, "depth": 2,
                      "position": [0, 0, 0], "color": "0x777777", "label": "base_01"})
    st.add(box)
    assert abs(st.top_y(0, 0) - 0.15) < 1e-9, st.top_y(0, 0)
    ball = st.resolve({"action": "add_sphere", "radius": 0.5, "on_top_of": "base_01",
                       "color": "0xFF0000", "label": "ball_02"})
    assert ball["position"] == [0, 0.65, 0], ball["position"]  # 0.15 + 0.5
    st.add(ball)
    # overlapping box must collide with the ball (small probe above the base)
    bad = _aabb([0, 1.0, 0], 0.25, 0.25, 0.25)
    assert st.collides(bad) == "ball_02", st.collides(bad)
    # far away is free
    assert st.collides(_aabb([10, 1, 10], 1.0, 1.0, 1.0)) == ""
    try:
        st.resolve({"action": "add_sphere", "on_top_of": "nope", "label": "x"})
        raise AssertionError("unknown ref should raise")
    except ValueError:
        pass
    print("state self-test ok:", st.summary().replace("\n", " | "))
