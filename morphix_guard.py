# morphix_guard.py
# Anti-hallucination guard for Morphix: validate model JS against the input
# action JSON, and fall back to a deterministic template renderer when the
# model output fails. This guarantees the pipeline never emits wrong output:
# model proposes, guard enforces, template guarantees.
#
# Usage:
#   from morphix_guard import validate_js, render_fallback, enforce
#   errors = validate_js(action, code)
#   final, status, errors = enforce(action, code)  # status: OK | FALLBACK

import json
import re

_BOOL_RE = re.compile(r"\b(True|False|None)\b")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\uf000-\ufffd]")
_DEGEN_MARKERS = ("NdrFc", "{lng", "{lnght", "위원님")

# action -> geometry token that MUST appear (see _GEO_MUST below)
_GEO_MUST = {
    "add_sphere": "SphereGeometry",
    "add_box": "BoxGeometry",
    "add_cylinder": "CylinderGeometry",
    "add_cone": "ConeGeometry",
    "add_torus": "TorusGeometry",
    "add_plane": "PlaneGeometry",
}
_FORBIDDEN_BY_ACTION = {
    "add_sphere": ("PointLight", "ShaderMaterial", "TorusGeometry", "BoxGeometry"),
    "add_box": ("PointLight", "ShaderMaterial", "SphereGeometry"),
    "add_shader": ("PointLight", "DirectionalLight"),
}


def _label_key(label: str) -> str:
    if not label:
        return ""
    # "ball_a1" -> "a1", "cone_4d1a6b" -> "4d1a6b" (model uses uid suffix in var names)
    return label.split("_")[-1] if "_" in label else label


def _safe_id(label: str, fallback="m1") -> str:
    # full label, not just the suffix: ball_01 + sphere_01 must not both
    # collapse to _01 (duplicate const declarations in one scene)
    key = re.sub(r"[^0-9a-zA-Z]", "_", label or fallback)
    if key and key[0].isdigit():
        key = "m_" + key
    return key or fallback


def validate_js(action: dict, code: str) -> list:
    """Return a list of error strings (empty = valid)."""
    errs = []
    if not code or not code.strip():
        return ["empty output"]
    if "```" in code:
        errs.append("markdown backticks")
    for m in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"):
        if m in code:
            errs.append(f"chat marker leak: {m}")
    if _BOOL_RE.search(code):
        errs.append("python literal (True/False/None) in JS")
    if _CJK_RE.search(code):
        errs.append("non-JS characters (CJK/mojibake)")
    for d in _DEGEN_MARKERS:
        if d in code:
            errs.append(f"degenerate run-on marker: {d!r}")
    if code.count("(") != code.count(")"):
        errs.append("unbalanced parens")
    if code.count("[") != code.count("]"):
        errs.append("unbalanced brackets")
    if code.count("{") != code.count("}"):
        errs.append("unbalanced braces")
    if code.count("`") % 2 != 0:
        errs.append("unbalanced shader backticks")

    act = action.get("action", "")
    must = _GEO_MUST.get(act)
    if must and must not in code:
        errs.append(f"missing {must} for {act}")
    for bad in _FORBIDDEN_BY_ACTION.get(act, ()):
        if bad in code:
            errs.append(f"hallucinated {bad} in {act}")

    # grounding: params from the action JSON must appear in the code
    low = code.lower()
    color = str(action.get("color", "")).lower()
    if color and color.startswith("0x") and color not in low:
        errs.append(f"color {action['color']} not grounded in output")
    for k in ("radius", "width", "height", "depth", "intensity", "tube", "angle"):
        if action.get(k) is not None and str(action[k]) not in code:
            # extrude prisms encode width halved (shape spans -w/2..w/2)
            if act == "add_extrude" and k == "width" and str(action[k] / 2) in code:
                continue
            errs.append(f"{k}={action[k]} not grounded in output")
    pos = action.get("position")
    if isinstance(pos, (list, tuple)) and pos:
        flat = ", ".join(str(v) for v in pos)
        nums = [str(v) for v in pos]
        if not all(n in code for n in nums):
            errs.append(f"position {flat} not grounded in output")
    label = action.get("label", "")
    key = _label_key(label)
    if key and key not in code:
        errs.append(f"label suffix {key!r} not grounded in output")

    tail_ok = ("scene.add(" in code) or ("morphix_" in code and ".push(" in code)
    if not tail_ok:
        errs.append("no scene.add()/morphix_*.push() terminator")
    # second-operation leak: only one atomic block expected
    if len(re.findall(r"scene\.add\s*\(", code)) > 3:
        errs.append("multiple scene.add() — not atomic")
    return errs


# ---------------------------------------------------------------- fallback
def render_fallback(action: dict) -> str:
    """Deterministic correct JS for the action. Always valid, always grounded."""
    act = action.get("action", "")
    u = _safe_id(action.get("label", "m1"))
    pos = action.get("position", [0, 0, 0])
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        pos = [0, 0, 0]
    color = action.get("color", "0x888888")

    if act == "add_sphere":
        r = action.get("radius", 0.5)
        return (f"const geo_{u} = new THREE.SphereGeometry({r},32,32);\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.4,metalness:0.3}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_box":
        w = action.get("width", 1)
        h = action.get("height", 1)
        d = action.get("depth", 1)
        return (f"const geo_{u} = new THREE.BoxGeometry({w},{h},{d});\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.6,metalness:0.2}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_cylinder":
        rt = action.get("radiusTop", action.get("radius", 0.5))
        rb = action.get("radiusBottom", action.get("radius", 0.5))
        h = action.get("height", 1)
        return (f"const geo_{u} = new THREE.CylinderGeometry({rt},{rb},{h},32);\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.5,metalness:0.3}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_cone":
        r = action.get("radius", 0.5)
        h = action.get("height", 1)
        return (f"const geo_{u} = new THREE.ConeGeometry({r},{h},32);\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.6,metalness:0.1}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_torus":
        r = action.get("radius", 0.8)
        t = action.get("tube", 0.2)
        return (f"const geo_{u} = new THREE.TorusGeometry({r},{t},16,100);\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.4,metalness:0.5}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_shader" and action.get("preset") == "fresnel":
        inten = action.get("intensity", 2.0)
        return (f"const mat_{u} = new THREE.ShaderMaterial({{\n"
                f"  uniforms: {{\n"
                f"    glowColor: {{ value: new THREE.Color({color}) }},\n"
                f"    intensity: {{ value: {inten} }}\n"
                f"  }},\n"
                f"  vertexShader: `\n"
                f"    varying vec3 vNormal;\n"
                f"    varying vec3 vViewDir;\n"
                f"    void main() {{\n"
                f"      vNormal = normalize(normalMatrix*normal);\n"
                f"      vec4 worldPos = modelViewMatrix*vec4(position,1.0);\n"
                f"      vViewDir = normalize(-worldPos.xyz);\n"
                f"      gl_Position = projectionMatrix*worldPos;\n"
                f"    }}\n"
                f"  `,\n"
                f"  fragmentShader: `\n"
                f"    uniform vec3 glowColor;\n"
                f"    uniform float intensity;\n"
                f"    varying vec3 vNormal;\n"
                f"    varying vec3 vViewDir;\n"
                f"    void main() {{\n"
                f"      float fresnel = pow(1.0-max(dot(vNormal,vViewDir),0.0),3.0);\n"
                f"      gl_FragColor = vec4(glowColor*fresnel*intensity,fresnel);\n"
                f"    }}\n"
                f"  `,\n"
                f"  transparent: true,\n"
                f"  side: THREE.FrontSide,\n"
                f"  depthWrite: false\n"
                f"}});\n"
                f"morphix_shaders.push(mat_{u});")
    if act == "add_light":
        inten = action.get("intensity", 1.0)
        return (f"const li_{u} = new THREE.DirectionalLight({color if color.startswith('0x') else '0xffffff'},{inten});\n"
                f"li_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"li_{u}.castShadow = true;\n"
                f"scene.add(li_{u});")
    if act == "add_plane":
        w = action.get("width", 4)
        h = action.get("height", 4)
        return (f"const geo_{u} = new THREE.PlaneGeometry({w},{h});\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.9,metalness:0,side:THREE.DoubleSide}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.rotation.x = -Math.PI/2;\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_torus_knot":
        r = action.get("radius", 0.6)
        t = action.get("tube", 0.15)
        return (f"const geo_{u} = new THREE.TorusKnotGeometry({r},{t},128,16);\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.3,metalness:0.7}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_extrude":
        d = action.get("depth", 2)
        w = action.get("width", 4)
        h = action.get("height", 1.5)
        return (f"const shape_{u} = new THREE.Shape();\n"
                f"shape_{u}.moveTo({-w / 2},{0});\n"
                f"shape_{u}.lineTo(0,{h});\n"
                f"shape_{u}.lineTo({w / 2},{0});\n"
                f"shape_{u}.closePath();\n"
                f"const geo_{u} = new THREE.ExtrudeGeometry(shape_{u},{{depth:{d},bevelEnabled:false}});\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.7,metalness:0.05}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.castShadow = true;\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_terrain":
        w = action.get("width", 20)
        seg = action.get("segments", 32)
        amp = action.get("amplitude", 1.5)
        freq = action.get("frequency", 0.4)
        return (f"const geo_{u} = new THREE.PlaneGeometry({w},{w},{seg},{seg});\n"
                f"const pp_{u} = geo_{u}.attributes.position;\n"
                f"for(let i=0;i<pp_{u}.count;i++){{\n"
                f"  const xx=pp_{u}.getX(i), zz=pp_{u}.getZ(i);\n"
                f"  pp_{u}.setY(i,Math.sin(xx*{freq})*Math.cos(zz*{freq})*{amp});\n"
                f"}};\n"
                f"geo_{u}.computeVertexNormals();\n"
                f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.95,metalness:0}});\n"
                f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
                f"mesh_{u}.rotation.x = -Math.PI/2;\n"
                f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
                f"mesh_{u}.receiveShadow = true;\n"
                f"scene.add(mesh_{u});")
    if act == "add_particles":
        n = action.get("count", 500)
        s = action.get("size", 0.05)
        return (f"const geo_{u} = new THREE.BufferGeometry();\n"
                f"const vv_{u} = new Float32Array({n}*3);\n"
                f"for(let i=0;i<{n};i++){{\n"
                f"  vv_{u}[i*3]=(Math.random()-0.5)*8;\n"
                f"  vv_{u}[i*3+1]=Math.random()*6;\n"
                f"  vv_{u}[i*3+2]=(Math.random()-0.5)*8;\n"
                f"}};\n"
                f"geo_{u}.setAttribute('position',new THREE.BufferAttribute(vv_{u},3));\n"
                f"const mat_{u} = new THREE.PointsMaterial({{color:{color},size:{s},sizeAttenuation:true,transparent:true,opacity:0.8}});\n"
                f"const pts_{u} = new THREE.Points(geo_{u},mat_{u});\n"
                f"scene.add(pts_{u});")
    # generic atomic mesh fallback — always grounded in label/color/position
    return (f"const geo_{u} = new THREE.BoxGeometry(0.5,0.5,0.5);\n"
            f"const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.6,metalness:0.2}});\n"
            f"const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});\n"
            f"mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});\n"
            f"mesh_{u}.castShadow = true;\n"
            f"mesh_{u}.receiveShadow = true;\n"
            f"scene.add(mesh_{u});")


def enforce(action: dict, code: str):
    """Validate; return (final_code, status, errors). Never returns bad code."""
    errors = validate_js(action, code or "")
    if not errors:
        return code.strip(), "OK", []
    return render_fallback(action), "FALLBACK", errors


if __name__ == "__main__":
    import sys
    act = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"action": "add_sphere"}
    code = sys.stdin.read() if not sys.stdin.isatty() else ""
    print(json.dumps({"errors": validate_js(act, code)}, indent=2))
