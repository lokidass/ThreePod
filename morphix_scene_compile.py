# morphix_scene_compile.py
# Scene IR -> executable Three.js module (+ validation). The compiler only
# emits APIs present in the version-aware knowledge index for the selected
# backend; anything else is a loud error, never a fabrication.
#
# Determinism: every procedural placement flows from the IR seed through one
# shared mulberry32 stream (same IR -> same JS -> same pixels). The emitted
# module signals readiness via window.__scene_ready and reports
# window.__scene_stats (nodes, instances, draw calls, triangles); runtime
# failures land in window.__scene_error instead of failing silently.

import json

from morphix_scene_ir import validate_scene_ir
from morphix_three_knowledge import check_js_apis

THREE_VERSION_PIN = "0.152.2"

# Shared deterministic PRNG (mulberry32) + 2D value noise used by every
# procedural emitter. Fixed source text: templates never hand-roll randomness.
JS_RNG = """function __rng(seed) {
  let s = seed | 0;
  return function () {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function __hash2(x, y, seed) {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263)
    + Math.imul(seed, 1442695041);
  h = Math.imul(h ^ h >>> 13, 1274126177);
  return ((h ^ h >>> 16) >>> 0) / 4294967296;
}
function __vnoise(x, y, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = __hash2(xi, yi, seed), b = __hash2(xi + 1, yi, seed);
  const c = __hash2(xi, yi + 1, seed), d = __hash2(xi + 1, yi + 1, seed);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function __fbm(x, y, seed, octaves) {
  let amp = 0.5, f = 1, sum = 0, norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * __vnoise(x * f, y * f, seed + i * 101);
    norm += amp; amp *= 0.5; f *= 2.03;
  }
  return sum / norm;
}
"""

# Height fog injected into MeshStandardMaterial via onBeforeCompile
# (r152 chunk names, pinned). This is the WebGL "custom fog" analogue:
# distance fog PLUS altitude falloff, driven by world position.
JS_HEIGHT_FOG = """function __applyHeightFog(mat, color, density, heightFalloff) {
  mat.onBeforeCompile = function (shader) {
    shader.uniforms.uFogColor = { value: new THREE.Color(color) };
    shader.uniforms.uFogDensity = { value: density };
    shader.uniforms.uFogHeightFalloff = { value: heightFalloff };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\\nvarying vec3 vFogWorldPos;')
      .replace('#include <project_vertex>', '#include <project_vertex>\\nvFogWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\\nvarying vec3 vFogWorldPos;\\nuniform vec3 uFogColor;\\nuniform float uFogDensity;\\nuniform float uFogHeightFalloff;')
      .replace('#include <fog_fragment>', [
        'vec3 fogDir = vFogWorldPos - cameraPosition;',
        'float fogDist = length(fogDir);',
        'float fogH = exp(-max(vFogWorldPos.y, 0.0) * uFogHeightFalloff);',
        'float fogF = 1.0 - exp(-uFogDensity * uFogDensity * fogDist * fogDist * fogH);',
        'gl_FragColor.rgb = mix(gl_FragColor.rgb, uFogColor, clamp(fogF, 0.0, 1.0));'
      ].join('\\n'));
  };
  mat.needsUpdate = true;
}
"""

_GEOMETRY_EMITTERS = {
    "BoxGeometry": lambda p: f"new THREE.BoxGeometry({p.get('w', 1)}, {p.get('h', 1)}, {p.get('d', 1)})",
    "SphereGeometry": lambda p: f"new THREE.SphereGeometry({p.get('r', 0.5)}, {p.get('ws', 32)}, {p.get('hs', 32)})",
    "CylinderGeometry": lambda p: f"new THREE.CylinderGeometry({p.get('rt', 0.5)}, {p.get('rb', 0.5)}, {p.get('h', 1)}, {p.get('seg', 32)})",
    "ConeGeometry": lambda p: f"new THREE.ConeGeometry({p.get('r', 0.5)}, {p.get('h', 1)}, {p.get('seg', 32)})",
    "TorusGeometry": lambda p: f"new THREE.TorusGeometry({p.get('r', 0.5)}, {p.get('t', 0.2)}, {p.get('rs', 16)}, {p.get('ts', 64)})",
    "PlaneGeometry": lambda p: f"new THREE.PlaneGeometry({p.get('w', 10)}, {p.get('h', 10)}, {p.get('ws', 1)}, {p.get('hs', 1)})",
    "TorusKnotGeometry": lambda p: f"new THREE.TorusKnotGeometry({p.get('r', 0.6)}, {p.get('t', 0.2)}, {p.get('tub', 128)}, {p.get('rad', 16)}, {p.get('p', 2)}, {p.get('q', 3)})",
    "CircleGeometry": lambda p: f"new THREE.CircleGeometry({p.get('r', 1)}, {p.get('seg', 48)})",
    "RingGeometry": lambda p: f"new THREE.RingGeometry({p.get('ri', 0.8)}, {p.get('ro', 1.0)}, {p.get('seg', 48)})",
}


def _js_str(v):
    return json.dumps(v)


def _emit_material(m, fog_kind):
    kind = m.get("kind", "standard")
    color = m.get("color", "0xFFFFFF")
    if kind == "standard":
        extra = ""
        for key in ("roughness", "metalness", "emissiveIntensity",
                    "envMapIntensity"):
            if key in m:
                extra += f", {key}: {m[key]}"
        if "emissive" in m:
            extra += f", emissive: {m['emissive']}"
        if m.get("transparent"):
            extra += f", transparent: true, opacity: {m.get('opacity', 1)}"
        if m.get("side") in ("DoubleSide", "BackSide", "FrontSide"):
            extra += f", side: THREE.{m['side']}"
        return f"new THREE.MeshStandardMaterial({{ color: {color}{extra} }})"
    if kind == "basic":
        extra = ""
        if m.get("fog") is False:
            extra += ", fog: false"
        if m.get("side") in ("DoubleSide", "BackSide", "FrontSide"):
            extra += f", side: THREE.{m['side']}"
        if m.get("transparent"):
            extra += f", transparent: true, opacity: {m.get('opacity', 1)}"
        return f"new THREE.MeshBasicMaterial({{ color: {color}{extra} }})"
    if kind == "points":
        return (f"new THREE.PointsMaterial({{ color: {color}, "
                f"size: {m.get('size', 0.08)}, "
                f"sizeAttenuation: {str(bool(m.get('sizeAttenuation', True))).lower()}, "
                f"transparent: true, opacity: {m.get('opacity', 0.9)}, "
                f"depthWrite: false }})")
    raise ValueError(f"no tested WebGL emitter for material kind {kind!r} "
                     f"(fog={fog_kind!r}); add one before using it")


def _emit_transform(var, t):
    p, r, s = t["position"], t["rotation"], t["scale"]
    return (f"{var}.position.set({p[0]}, {p[1]}, {p[2]});\n"
            f"{var}.rotation.set({r[0]}, {r[1]}, {r[2]});\n"
            f"{var}.scale.set({s[0]}, {s[1]}, {s[2]});")


def compile_scene_ir(ir, three_url="three", pixel_ratio=1, antialias=False):
    """Compile a validated Scene IR to a Three.js module (string).

    Raises ValueError naming the first problem (IR invalid, unknown
    constructor, backend incompatibility). The output uses only
    knowledge-approved r152 core APIs for backend=webgl.
    """
    errs = validate_scene_ir(ir)
    if errs:
        raise ValueError("invalid scene IR: " + "; ".join(errs[:5]))
    meta = ir["meta"]
    if meta["backend"] != "webgl":
        raise ValueError(
            f"backend {meta['backend']!r} has no tested emitter on the "
            f"vendored r152 build (no WebGPURenderer/TSL); use 'webgl'")
    seed = int(meta["seed"])
    fog = (ir.get("environment") or {}).get("fog") or {"kind": "none"}
    fog_kind = fog.get("kind", "none")

    L = []
    L.append(f"import * as THREE from {_js_str(three_url)};")
    L.append(JS_RNG)
    if fog_kind == "custom":
        L.append(JS_HEIGHT_FOG)
    L.append("const __anims = [];")
    L.append("const renderer = new THREE.WebGLRenderer({ antialias: "
             + str(bool(antialias)).lower() + " });")
    L.append(f"renderer.setPixelRatio({pixel_ratio});")
    L.append("renderer.setSize(__W__, __H__);")
    L.append("renderer.shadowMap.enabled = true;")
    L.append("renderer.shadowMap.type = THREE.PCFSoftShadowMap;")
    L.append("renderer.toneMapping = THREE.ACESFilmicToneMapping;")
    L.append("renderer.toneMappingExposure = 1.0;")
    L.append("renderer.outputColorSpace = THREE.SRGBColorSpace;")
    L.append("document.body.appendChild(renderer.domElement);")
    L.append("const scene = new THREE.Scene();")

    # ---- environment: background / fog / environment map
    bg = (ir.get("environment") or {}).get("background") or {"kind": "color"}
    # The gradient-sky helper is emitted exactly once no matter how many
    # callers (background gradient and/or gradient_sky nodes) need it.
    sky_helper_emitted = False

    def need_sky_helper():
        nonlocal sky_helper_emitted
        if not sky_helper_emitted:
            L.append(_GRADIENT_HELPER)
            sky_helper_emitted = True

    def sky_call(top, mid, bot):
        need_sky_helper()
        L.append(f"__installGradient(scene, '{_css(top)}', '{_css(mid)}', "
                 f"'{_css(bot)}');")

    if bg.get("kind") == "color":
        L.append(f"scene.background = new THREE.Color({bg.get('color', '0x111622')});")
    elif bg.get("kind") == "gradient":
        sky_call(bg.get("top", "0x27435f"), bg.get("mid", "0x9db8d6"),
                 bg.get("bottom", "0x111622"))
    if fog_kind == "linear":
        L.append(f"scene.fog = new THREE.Fog({fog.get('color', '0x4080cc')}, "
                 f"{fog.get('near', 2.7)}, {fog.get('far', 4)});")
    elif fog_kind == "exp2":
        L.append(f"scene.fog = new THREE.FogExp2({fog.get('color', '0x9db8d6')}, "
                 f"{fog.get('density', 0.028)});")
    elif fog_kind == "custom":
        L.append(f"const __fogColor = {fog.get('color', '0x9db8d6')};")
        L.append(f"const __fogDensity = {fog.get('density', 0.028)};")
        L.append(f"const __fogHeightFalloff = {fog.get('height_falloff', 0.08)};")
    envmap = (ir.get("environment") or {}).get("environment_map")
    if envmap:
        L.append("__bakeEnvironment(scene, renderer);")

    # ---- camera
    cam = ir["camera"]
    if cam["kind"] == "perspective":
        L.append(f"const camera = new THREE.PerspectiveCamera({cam.get('fov', 55)}, "
                 f"__W__ / __H__, {cam.get('near', 0.1)}, {cam.get('far', 500)});")
    else:
        s = cam.get("half", 8)
        L.append(f"const camera = new THREE.OrthographicCamera(-{s}, {s}, "
                 f"{s} * __H__ / __W__, -{s} * __H__ / __W__, "
                 f"{cam.get('near', 0.1)}, {cam.get('far', 500)});")
    L.append(f"camera.position.set({cam['position'][0]}, {cam['position'][1]}, "
             f"{cam['position'][2]});")
    L.append(f"camera.lookAt({cam['target'][0]}, {cam['target'][1]}, "
             f"{cam['target'][2]});")

    # ---- lights (shadow only on the first shadow-casting directional)
    shadow_done = False
    for Ld in ir.get("lights", []):
        kind, lid = Ld["kind"], Ld["id"]
        v = f"__light_{lid}"
        col = Ld.get("color", "0xFFFFFF")
        inten = Ld.get("intensity", 1.0)
        if kind == "ambient":
            L.append(f"const {v} = new THREE.AmbientLight({col}, {inten});")
        elif kind == "hemisphere":
            L.append(f"const {v} = new THREE.HemisphereLight({col}, "
                     f"{Ld.get('ground', '0x223344')}, {inten});")
        elif kind == "directional":
            L.append(f"const {v} = new THREE.DirectionalLight({col}, {inten});")
            pos = Ld.get("position", [6, 10, 7])
            L.append(f"{v}.position.set({pos[0]}, {pos[1]}, {pos[2]});")
            if Ld.get("shadow") and not shadow_done:
                sh = Ld["shadow"]
                L.append(f"{v}.castShadow = true;")
                L.append(f"{v}.shadow.mapSize.set({sh.get('map', 1024)}, "
                         f"{sh.get('map', 1024)});")
                L.append(f"{v}.shadow.camera.left = {sh.get('left', -15)};")
                L.append(f"{v}.shadow.camera.right = {sh.get('right', 15)};")
                L.append(f"{v}.shadow.camera.top = {sh.get('top', 15)};")
                L.append(f"{v}.shadow.camera.bottom = {sh.get('bottom', -15)};")
                shadow_done = True
        elif kind == "point":
            L.append(f"const {v} = new THREE.PointLight({col}, {inten}, "
                     f"{Ld.get('distance', 0)}, {Ld.get('decay', 2)});")
            pos = Ld.get("position", [0, 3, 0])
            L.append(f"{v}.position.set({pos[0]}, {pos[1]}, {pos[2]});")
        elif kind == "spot":
            L.append(f"const {v} = new THREE.SpotLight({col}, {inten}, "
                     f"{Ld.get('distance', 0)}, {Ld.get('angle', 0.5)}, "
                     f"{Ld.get('penumbra', 0.4)});")
            pos = Ld.get("position", [0, 5, 0])
            L.append(f"{v}.position.set({pos[0]}, {pos[1]}, {pos[2]});")
        L.append(f"scene.add({v});")

    # ---- nodes
    by_id = {n["id"]: n for n in ir.get("nodes", [])}
    for n in ir.get("nodes", []):
        L.extend(_emit_node(n, by_id, seed, fog_kind, sky_call))

    # ---- procedural environment helpers referenced above
    if envmap:
        L.append("function __bakeEnvironment(scene, renderer) {")
        L.append("  const c = document.createElement('canvas');")
        L.append("  c.width = 64; c.height = 32;")
        L.append("  const g = c.getContext('2d');")
        L.append("  const grad = g.createLinearGradient(0, 0, 0, 32);")
        top = envmap.get("sky", "0x87ceeb")
        mid = envmap.get("horizon", "0xcfd8e6")
        bot = envmap.get("ground", "0x3a4a3a")
        L.append(f"  grad.addColorStop(0, '{_css(top)}');")
        L.append(f"  grad.addColorStop(0.55, '{_css(mid)}');")
        L.append(f"  grad.addColorStop(1, '{_css(bot)}');")
        L.append("  g.fillStyle = grad; g.fillRect(0, 0, 64, 32);")
        L.append("  const tex = new THREE.CanvasTexture(c);")
        L.append("  tex.mapping = THREE.EquirectangularReflectionMapping;")
        L.append("  tex.colorSpace = THREE.SRGBColorSpace;")
        L.append("  const pm = new THREE.PMREMGenerator(renderer);")
        L.append("  scene.environment = pm.fromEquirectangular(tex).texture;")
        L.append("  tex.dispose(); pm.dispose();")
        L.append("}")

    # ---- animations + interactions
    light_ids = {Ld.get("id") for Ld in ir.get("lights", [])
                 if isinstance(Ld, dict)}
    node_ids = {n.get("id") for n in ir.get("nodes", [])
                if isinstance(n, dict)}
    for a in ir.get("animations", []):
        L.extend(_emit_animation(a, light_ids, node_ids))
    for it in ir.get("interactions", []):
        if it.get("kind") == "orbit_controls":
            L.append("/* orbit_controls requested: needs an addons import "
                     "(network importmap). Omitted in the offline harness. */")

    # ---- frame loop with readiness + stats + error capture
    L.append("let __frames = 0;")
    L.append("window.__scene_error = null;")
    L.append("window.addEventListener('error', function (e) { "
             "window.__scene_error = String((e && e.message) || e); });")
    L.append("const __clock = new THREE.Clock();")
    L.append("function __tick() {")
    L.append("  const dt = Math.min(__clock.getDelta(), 0.05);")
    L.append("  const t = __clock.elapsedTime;")
    L.append("  for (const a of __anims) { a.update(dt, t); }")
    L.append("  renderer.render(scene, camera);")
    L.append("  if (++__frames >= 3) {")
    L.append("    window.__scene_stats = { frames: __frames, "
             "nodes: scene.children.length, "
             "drawCalls: renderer.info.render.calls, "
             "triangles: renderer.info.render.triangles };")
    L.append("    window.__scene_ready = true;")
    L.append("  } else { requestAnimationFrame(__tick); }")
    L.append("}")
    L.append("__tick();")
    return "\n".join(L) + "\n"


def _css(h):
    """0xRRGGBB -> #RRGGBB for canvas gradients."""
    h = str(h)
    return "#" + h[2:].lower() if h.startswith("0x") else h


_GRADIENT_HELPER = """
function __installGradient(scene, top, mid, bot) {
  const c = document.createElement('canvas');
  c.width = 4; c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 128);
  grad.addColorStop(0, top);
  grad.addColorStop(0.55, mid);
  grad.addColorStop(1, bot);
  g.fillStyle = grad; g.fillRect(0, 0, 4, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(220, 24, 16),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false }));
  scene.add(sky);
}
"""


def _emit_node(n, by_id, seed, fog_kind, sky_call=None):
    nid, kind = n["id"], n["kind"]
    v = f"__n_{nid}"
    L = []
    t = n.get("transform") or {"position": [0, 0, 0], "rotation": [0, 0, 0],
                               "scale": [1, 1, 1]}
    g = n.get("geometry") or {}
    if kind == "group":
        L.append(f"const {v} = new THREE.Group();")
        L.append(_emit_transform(v, t))
        for kid in n.get("children", []):
            L.append(f"{v}.add(__n_{kid});")
        L.append("scene.add(%s);" % v)
        return L
    mat = _emit_material(n.get("material") or {"kind": "standard"}, fog_kind)
    if kind == "mesh":
        if "generator" in g:
            L.extend(_emit_procedural_mesh(v, n, seed, sky_call))
            if g.get("generator") == "gradient_sky":
                return L  # helper-installed dome; no mesh variable exists
        else:
            ctor = g.get("ctor")
            if ctor not in _GEOMETRY_EMITTERS:
                raise ValueError(
                    f"node {nid!r}: no tested WebGL emitter for geometry "
                    f"{ctor!r}")
            L.append(f"const {v} = new THREE.Mesh("
                     f"{_GEOMETRY_EMITTERS[ctor](g.get('params', {}))}, {mat});")
        L.append(_emit_transform(v, t))
        if n.get("cast_shadow"):
            L.append(f"{v}.castShadow = true;")
        if n.get("receive_shadow"):
            L.append(f"{v}.receiveShadow = true;")
        if fog_kind == "custom":
            L.append(f"__applyHeightFog({v}.material, __fogColor, "
                     f"__fogDensity, __fogHeightFalloff);")
        L.append(f"scene.add({v});")
        return L
    if kind == "instanced":
        if "generator" in g:
            raise ValueError(
                f"node {nid!r}: instanced procedural generators are not "
                f"supported; use count + distribution on a fixed constructor")
        ctor = g.get("ctor")
        if ctor not in _GEOMETRY_EMITTERS:
            raise ValueError(
                f"node {nid!r}: no tested WebGL emitter for geometry "
                f"{ctor!r}")
        count = int(n.get("count", 0))
        dist = n.get("distribution") or {}
        L.append(f"const {v} = new THREE.InstancedMesh("
                 f"{_GEOMETRY_EMITTERS[ctor](g.get('params', {}))}, {mat}, {count});")
        L.append(f"{{ const __r = __rng({seed + _zseed(nid)});")
        L.append(f"  const __m = new THREE.Matrix4();")
        L.append(f"  const __p = new THREE.Vector3();")
        L.append(f"  const __q = new THREE.Quaternion();")
        L.append(f"  const __s = new THREE.Vector3();")
        L.append(f"  const __e = new THREE.Euler();")
        L.append(f"  let __placed = 0, __guard = 0;")
        L.append(f"  while (__placed < {count} && __guard++ < {count} * 40) {{")
        L.extend(_emit_scatter(dist, "__r", "__p", "__placed"))
        L.append(f"    __e.set(0, __r() * Math.PI * 2, 0);")
        L.append(f"    __q.setFromEuler(__e);")
        sc = dist.get("scale_jitter", 0.0)
        hv = dist.get("height_var", 0.0)
        if hv:
            # towers: seeded footprint + height, base resting on the ground
            # (overrides the scatter y: each tower stands on y=0).
            L.append(f"    const __sxz = 1.6 + __r() * 1.6;")
            L.append(f"    const __sy = {hv} * (0.25 + 0.75 * __r());")
            L.append(f"    __s.set(__sxz, __sy, __sxz);")
            L.append(f"    __p.y = __sy / 2;")
        elif sc:
            L.append(f"    const __sj = 1 + (__r() * 2 - 1) * {sc};")
            L.append(f"    __s.set(__sj, __sj, __sj);")
        else:
            L.append(f"    __s.set(1, 1, 1);")
        L.append(f"    __m.compose(__p, __q, __s);")
        L.append(f"    {v}.setMatrixAt(__placed, __m);")
        L.append(f"    __placed++;")
        L.append(f"  }}")
        L.append(f"  {v}.instanceMatrix.needsUpdate = true;")
        L.append(f"  {v}.count = __placed;")
        L.append(f"}}")
        if n.get("cast_shadow"):
            L.append(f"{v}.castShadow = true;")
        if n.get("receive_shadow"):
            L.append(f"{v}.receiveShadow = true;")
        if fog_kind == "custom":
            L.append(f"__applyHeightFog({v}.material, __fogColor, "
                     f"__fogDensity, __fogHeightFalloff);")
        L.append(f"scene.add({v});")
        return L
    if kind == "points":
        count = int(n.get("count", 0))
        dist = n.get("distribution") or {}
        spread = dist.get("spread", [20, 6, 20])
        L.append(f"const {v} = (function () {{")
        L.append(f"  const __r = __rng({seed + _zseed(nid)});")
        L.append(f"  const __pos = new Float32Array({count} * 3);")
        L.append(f"  for (let i = 0; i < {count}; i++) {{")
        L.append(f"    __pos[i * 3] = (__r() * 2 - 1) * {spread[0] / 2};")
        L.append(f"    __pos[i * 3 + 1] = __r() * {spread[1]};")
        L.append(f"    __pos[i * 3 + 2] = (__r() * 2 - 1) * {spread[2] / 2};")
        L.append(f"  }}")
        L.append(f"  const __g = new THREE.BufferGeometry();")
        L.append(f"  __g.setAttribute('position', new THREE.BufferAttribute(__pos, 3));")
        L.append(f"  return new THREE.Points(__g, {mat});")
        L.append(f"}})();")
        L.append(_emit_transform(v, t))
        L.append(f"scene.add({v});")
        return L
    raise ValueError(f"node {nid!r}: unknown kind {kind!r}")


def _zseed(s):
    """Small stable per-node seed offset (keeps streams independent)."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(str(s))) % 100003


def _emit_scatter(dist, rng, pvar, idxvar):
    """Position a scatter sample into pvar. Supports disc/ring/box/clearing."""
    L = []
    shape = dist.get("shape", "disc")
    if shape == "disc":
        L.append(f"    const __a = {rng}() * Math.PI * 2;")
        L.append(f"    const __rad = Math.sqrt({rng}()) * {dist.get('radius', 18)};")
        L.append(f"    {pvar}.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);")
    elif shape == "ring":
        L.append(f"    const __a = {rng}() * Math.PI * 2;")
        L.append(f"    const __rad = {dist.get('inner', 8)} + {rng}() * "
                 f"{dist.get('outer', 20) - dist.get('inner', 8)};")
        L.append(f"    {pvar}.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);")
    else:
        L.append(f"    {pvar}.set(({rng}() * 2 - 1) * {dist.get('hx', 15)}, 0, "
                 f"({rng}() * 2 - 1) * {dist.get('hz', 15)});")
    if dist.get("clearing"):
        L.append(f"    if (Math.hypot({pvar}.x, {pvar}.z) < {dist['clearing']}) continue;")
    if dist.get("ground") == "terrain":
        L.append(f"    {pvar}.y = __terrainH({pvar}.x, {pvar}.z) "
                 f"+ {dist.get('lift', 0)};")
    else:
        L.append(f"    {pvar}.y = {dist.get('y', 0)};")
    return L


def _emit_procedural_mesh(var, n, seed, sky_call=None):
    """Expand a procedural generator node into deterministic JS."""
    g = n["geometry"]
    gen, p = g["generator"], g.get("params", {})
    mat = _emit_material(n.get("material") or {"kind": "standard"}, "none")
    L = []
    if gen == "terrain":
        size = p.get("size", 60)
        seg = p.get("segments", 96)
        height = p.get("height", 6)
        L.append(f"function __terrainH(x, z) {{ return __fbm(x * 0.035 + 7.3, z * 0.035 + 2.9, {seed}, 4) * {height}; }}")
        L.append(f"const {var} = (function () {{")
        L.append(f"  const __g = new THREE.PlaneGeometry({size}, {size}, {seg}, {seg});")
        L.append(f"  __g.rotateX(-Math.PI / 2);")
        L.append(f"  const __pos = __g.attributes.position;")
        L.append(f"  for (let i = 0; i < __pos.count; i++) {{")
        L.append(f"    __pos.setY(i, __terrainH(__pos.getX(i), __pos.getZ(i)) - {height} * 0.35);")
        L.append(f"  }}")
        L.append(f"  __g.computeVertexNormals();")
        L.append(f"  return new THREE.Mesh(__g, {mat});")
        L.append(f"}})();")
    elif gen == "gradient_sky":
        if sky_call is None:
            raise ValueError("gradient_sky needs the shared sky helper")
        sky_call(p.get("top", "0x27435F"), p.get("mid", "0x9DB8D6"),
                 p.get("bot", "0x111622"))
        return L  # the shared helper installs the dome into the scene
    else:
        raise ValueError(f"unknown procedural generator {gen!r} "
                         f"(tested: terrain, gradient_sky)")
    t = n.get("transform") or {"position": [0, 0, 0], "rotation": [0, 0, 0],
                               "scale": [1, 1, 1]}
    L.append(_emit_transform(var, t))
    if n.get("receive_shadow"):
        L.append(f"{var}.receiveShadow = true;")
    L.append(f"scene.add({var});")
    return L


def _emit_animation(a, light_ids=(), node_ids=()):
    target = a.get("target", "")
    kind = a.get("kind", "rotate")
    if target in light_ids:
        # light effects (flicker); transforms are meaningless on lights.
        if kind != "flicker":
            raise ValueError(f"animation target {target!r} is a light: only "
                             f"'flicker' applies (got {kind!r})")
        v = f"__light_{target}"
    elif target in node_ids or not node_ids:
        v = f"__n_{target}"
    else:
        raise ValueError(f"animation target {target!r} names neither a "
                         f"node nor a light")
    if kind == "rotate":
        speed = a.get("speed", 0.3)
        return [f"__anims.push({{ update: function (dt, t) {{ {v}.rotation.y += dt * {speed}; }} }});"]
    if kind == "bob":
        amp = a.get("amplitude", 0.15)
        freq = a.get("frequency", 1.2)
        return [f"__anims.push({{ update: (function () {{ const __y0 = {v}.position.y; "
                f"return function (dt, t) {{ {v}.position.y = __y0 + Math.sin(t * {freq}) * {amp}; }}; }})() }});"]
    if kind == "flicker":
        base = a.get("base", 1.0)
        amp = a.get("amplitude", 0.4)
        freq = a.get("frequency", 7.0)
        return [f"__anims.push({{ update: function (dt, t) {{ {v}.intensity = {base} + Math.sin(t * {freq}) * {amp}; }} }});"]
    raise ValueError(f"unknown animation kind {kind!r} (tested: rotate, bob, flicker)")


def validate_scene_js(code, backend="webgl"):
    """node --check syntax + knowledge API gate + required scene markers.
    Returns [error strings]; parsing alone never passes."""
    from morphix_validator import validate_js_syntax
    errs = []
    syn = validate_js_syntax(code)
    if syn:
        return [f"syntax: {syn}"]
    errs.extend(check_js_apis(code, backend))
    for marker, why in (
            ("new THREE.WebGLRenderer", "renderer creation"),
            ("renderer.render(", "frame rendering"),
            ("window.__scene_ready", "readiness signal")):
        if marker not in code:
            errs.append(f"missing {why} ({marker!r} not found)")
    return errs


# Gradient-background helper is referenced by compile output; keep the
# template pieces consistent by construction (test asserts presence).
def scene_html(scene_js, three_url="https://unpkg.com/three@0.152.2/build/three.module.js",
               title="Morphix scene"):
    """Standalone scene.html (UI path): importmap + canvas + error overlay."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;padding:0;background:#111622;overflow:hidden}}canvas{{display:block}}#err{{position:fixed;left:8px;top:8px;color:#f66;font:12px monospace;white-space:pre-wrap}}</style>
</head><body>
<div id="err"></div>
<script>window.addEventListener('error', function (e) {{ document.getElementById('err').textContent = String((e && e.message) || e); }});</script>
<script type="importmap">{{"imports":{{"three":{json.dumps(three_url)}}}}}</script>
<script type="module">
import * as THREE from 'three';
const __W__ = window.innerWidth, __H__ = window.innerHeight;
{scene_js.replace('__W__', '__W__').replace('__H__', '__H__')}
</script></body></html>
"""
