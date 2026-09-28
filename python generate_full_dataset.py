"""
Morphix — Full Dataset Generator
Generates ~65,000 training samples covering the entire Three.js universe.
Run: python generate_dataset.py
Output: morphix_train.jsonl, morphix_val.jsonl, morphix_test.jsonl
"""

import json
import random
import uuid
import math

def uid():
    return uuid.uuid4().hex[:6]

def rand_pos(r=3):
    return [round(random.uniform(-r, r), 2),
            round(random.uniform(0, r), 2),
            round(random.uniform(-r, r), 2)]

def rand_color():
    return f"0x{random.randint(0, 0xFFFFFF):06X}"

def rand_rot():
    return round(random.uniform(0, math.pi * 2), 3)

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Morphix Builder — a Three.js 3D printer agent.
You receive ONE action JSON per tick describing a single atomic operation.
Output ONLY raw JavaScript. No markdown, no backticks, no explanation, no comments.
Rules:
- Each output patches the EXISTING scene — never rebuild from scratch.
- Use unique variable names based on the label field to avoid collisions.
- Always use MeshStandardMaterial or MeshPhysicalMaterial for realism.
- Set castShadow = true and receiveShadow = true on every mesh.
- Code must execute inside a Three.js r152 scene with renderer, camera, scene already defined.
- morphix_mixers, morphix_shaders, morphix_physics are global arrays for tracking.
- Never output anything except valid executable JavaScript."""

# ─────────────────────────────────────────────────────────────
# CATEGORY 1 — GEOMETRY
# ─────────────────────────────────────────────────────────────

def gen_box():
    u = uid()
    w = round(random.uniform(0.1, 4), 2)
    h = round(random.uniform(0.1, 4), 2)
    d = round(random.uniform(0.1, 4), 2)
    pos = rand_pos()
    color = rand_color()
    rough = round(random.uniform(0.1, 1.0), 2)
    metal = round(random.uniform(0, 1.0), 2)
    inp = {"action": "add_box", "width": w, "height": h, "depth": d,
           "position": pos, "color": color, "roughness": rough, "metalness": metal, "label": f"box_{u}"}
    out = f"""const geo_{u} = new THREE.BoxGeometry({w},{h},{d});
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:{rough},metalness:{metal}}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_sphere():
    u = uid()
    r = round(random.uniform(0.05, 2.5), 2)
    pos = rand_pos()
    color = rand_color()
    rough = round(random.uniform(0, 1), 2)
    metal = round(random.uniform(0, 1), 2)
    inp = {"action": "add_sphere", "radius": r, "position": pos,
           "color": color, "roughness": rough, "metalness": metal, "label": f"sphere_{u}"}
    out = f"""const geo_{u} = new THREE.SphereGeometry({r},32,32);
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:{rough},metalness:{metal}}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_cylinder():
    u = uid()
    rt = round(random.uniform(0.05, 1.5), 2)
    rb = round(random.uniform(0.05, 1.5), 2)
    h = round(random.uniform(0.1, 4), 2)
    pos = rand_pos()
    color = rand_color()
    inp = {"action": "add_cylinder", "radiusTop": rt, "radiusBottom": rb,
           "height": h, "position": pos, "color": color, "label": f"cyl_{u}"}
    out = f"""const geo_{u} = new THREE.CylinderGeometry({rt},{rb},{h},32);
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.5,metalness:0.3}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_cone():
    u = uid()
    r = round(random.uniform(0.1, 2), 2)
    h = round(random.uniform(0.2, 4), 2)
    pos = rand_pos()
    color = rand_color()
    inp = {"action": "add_cone", "radius": r, "height": h,
           "position": pos, "color": color, "label": f"cone_{u}"}
    out = f"""const geo_{u} = new THREE.ConeGeometry({r},{h},32);
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.6,metalness:0.1}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_torus():
    u = uid()
    r = round(random.uniform(0.3, 2), 2)
    tube = round(random.uniform(0.05, 0.5), 2)
    pos = rand_pos()
    color = rand_color()
    inp = {"action": "add_torus", "radius": r, "tube": tube,
           "position": pos, "color": color, "label": f"torus_{u}"}
    out = f"""const geo_{u} = new THREE.TorusGeometry({r},{tube},16,100);
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.4,metalness:0.5}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_torus_knot():
    u = uid()
    r = round(random.uniform(0.3, 1.5), 2)
    tube = round(random.uniform(0.05, 0.3), 2)
    p = random.randint(2, 5)
    q = random.randint(3, 7)
    color = rand_color()
    pos = rand_pos()
    inp = {"action": "add_torus_knot", "radius": r, "tube": tube,
           "p": p, "q": q, "position": pos, "color": color, "label": f"knot_{u}"}
    out = f"""const geo_{u} = new THREE.TorusKnotGeometry({r},{tube},128,16,{p},{q});
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.3,metalness:0.7}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_plane():
    u = uid()
    w = round(random.uniform(1, 10), 2)
    h = round(random.uniform(1, 10), 2)
    segs = random.randint(1, 32)
    color = rand_color()
    pos = rand_pos(5)
    inp = {"action": "add_plane", "width": w, "height": h,
           "segments": segs, "position": pos, "color": color, "label": f"plane_{u}"}
    out = f"""const geo_{u} = new THREE.PlaneGeometry({w},{h},{segs},{segs});
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.9,metalness:0,side:THREE.DoubleSide}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.rotation.x = -Math.PI/2;
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_lathe():
    u = uid()
    pts = [[round(random.uniform(0.05, 1.2), 2), round(i * 0.25, 2)] for i in range(7)]
    seg = random.randint(12, 64)
    color = rand_color()
    pos = rand_pos()
    inp = {"action": "add_lathe", "points": pts, "segments": seg,
           "position": pos, "color": color, "label": f"lathe_{u}"}
    pts_str = ",".join([f"new THREE.Vector2({p[0]},{p[1]})" for p in pts])
    out = f"""const pts_{u} = [{pts_str}];
const geo_{u} = new THREE.LatheGeometry(pts_{u},{seg});
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.4,metalness:0.5,side:THREE.DoubleSide}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_extrude():
    u = uid()
    depth = round(random.uniform(0.1, 2.5), 2)
    bevel = random.choice([True, False])
    color = rand_color()
    pos = rand_pos()
    sides = random.randint(3, 8)
    pts = []
    for i in range(sides):
        angle = (2 * math.pi / sides) * i
        r = round(random.uniform(0.3, 1.2), 2)
        pts.append([round(r * math.cos(angle), 3), round(r * math.sin(angle), 3)])
    pts_js = "\n".join([f"shape_{u}.{'moveTo' if i == 0 else 'lineTo'}({p[0]},{p[1]});" for i, p in enumerate(pts)])
    inp = {"action": "add_extrude", "depth": depth, "bevel": bevel,
           "sides": sides, "position": pos, "color": color, "label": f"ext_{u}"}
    out = f"""const shape_{u} = new THREE.Shape();
{pts_js}
shape_{u}.closePath();
const geo_{u} = new THREE.ExtrudeGeometry(shape_{u},{{depth:{depth},bevelEnabled:{str(bevel).lower()},bevelSize:0.04,bevelThickness:0.04}});
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.5,metalness:0.2}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
mesh_{u}.castShadow = true;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_tube():
    u = uid()
    r = round(random.uniform(0.02, 0.3), 2)
    segs = random.randint(32, 128)
    color = rand_color()
    pts = [rand_pos(2) for _ in range(4)]
    inp = {"action": "add_tube", "radius": r, "segments": segs,
           "color": color, "label": f"tube_{u}"}
    pts_js = ",\n  ".join([f"new THREE.Vector3({p[0]},{p[1]},{p[2]})" for p in pts])
    out = f"""const curve_{u} = new THREE.CatmullRomCurve3([
  {pts_js}
]);
const geo_{u} = new THREE.TubeGeometry(curve_{u},{segs},{r},8,False);
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.4,metalness:0.6}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.castShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_buffer_geometry():
    u = uid()
    n = random.randint(3, 12)
    verts = []
    for _ in range(n):
        p = rand_pos(1.5)
        verts += p
    color = rand_color()
    inp = {"action": "add_buffer_geometry", "vertex_count": n,
           "color": color, "label": f"buf_{u}"}
    out = f"""const verts_{u} = new Float32Array({verts});
const geo_{u} = new THREE.BufferGeometry();
geo_{u}.setAttribute('position', new THREE.BufferAttribute(verts_{u},3));
geo_{u}.computeVertexNormals();
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},side:THREE.DoubleSide,roughness:0.6,metalness:0.2}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.castShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_instanced():
    u = uid()
    count = random.randint(10, 1000)
    shape = random.choice(["sphere", "box", "cone"])
    color = rand_color()
    spread = round(random.uniform(3, 15), 1)
    inp = {"action": "add_instanced_mesh", "count": count,
           "shape": shape, "spread": spread, "color": color, "label": f"inst_{u}"}
    if shape == "sphere":
        geo_line = "new THREE.SphereGeometry(0.08,8,8)"
    elif shape == "box":
        geo_line = "new THREE.BoxGeometry(0.12,0.12,0.12)"
    else:
        geo_line = "new THREE.ConeGeometry(0.06,0.15,6)"
    out = f"""const geo_{u} = {geo_line};
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.5,metalness:0.3}});
const mesh_{u} = new THREE.InstancedMesh(geo_{u},mat_{u},{count});
const dummy_{u} = new THREE.Object3D();
for(let i=0;i<{count};i++){{
  dummy_{u}.position.set(
    (Math.random()-0.5)*{spread},
    (Math.random()-0.5)*{spread},
    (Math.random()-0.5)*{spread}
  );
  dummy_{u}.rotation.set(Math.random()*Math.PI,Math.random()*Math.PI,0);
  dummy_{u}.scale.setScalar(0.5+Math.random());
  dummy_{u}.updateMatrix();
  mesh_{u}.setMatrixAt(i,dummy_{u}.matrix);
}}
mesh_{u}.instanceMatrix.needsUpdate = true;
mesh_{u}.castShadow = true;
scene.add(mesh_{u});"""
    return inp, out

def gen_terrain():
    u = uid()
    w = random.choice([10, 20, 30])
    segs = random.choice([32, 64, 128])
    color = rand_color()
    amp = round(random.uniform(0.5, 4), 2)
    freq = round(random.uniform(0.2, 1.0), 2)
    inp = {"action": "add_terrain", "width": w, "segments": segs,
           "amplitude": amp, "frequency": freq, "color": color, "label": f"terrain_{u}"}
    out = f"""const geo_{u} = new THREE.PlaneGeometry({w},{w},{segs},{segs});
const pos_{u} = geo_{u}.attributes.position;
for(let i=0;i<pos_{u}.count;i++){{
  const x=pos_{u}.getX(i), z=pos_{u}.getZ(i);
  pos_{u}.setY(i,
    Math.sin(x*{freq})*Math.cos(z*{freq})*{amp}
    +Math.sin(x*{freq*2}+1.2)*{amp*0.4}
    +Math.cos(z*{freq*1.5}+0.7)*{amp*0.3}
  );
}}
geo_{u}.computeVertexNormals();
const mat_{u} = new THREE.MeshStandardMaterial({{color:{color},roughness:0.95,metalness:0,side:THREE.FrontSide}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.rotation.x = -Math.PI/2;
mesh_{u}.receiveShadow = true;
scene.add(mesh_{u});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 2 — PBR MATERIALS
# ─────────────────────────────────────────────────────────────

def gen_metal_material():
    u = uid()
    presets = [
        ("chrome",   "0xC8C8C8", 0.95, 0.05),
        ("copper",   "0xB87333", 0.90, 0.20),
        ("gold",     "0xFFD700", 0.95, 0.08),
        ("steel",    "0x8C8C8C", 0.80, 0.40),
        ("iron",     "0x505050", 0.75, 0.60),
        ("bronze",   "0xCD7F32", 0.85, 0.25),
        ("titanium", "0x878681", 0.88, 0.15),
        ("aluminium","0xA8A9AD", 0.70, 0.55),
    ]
    name, color, metal, rough = random.choice(presets)
    env = round(random.uniform(1.0, 2.5), 1)
    inp = {"action": "set_material", "preset": name, "color": color,
           "metalness": metal, "roughness": rough, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshStandardMaterial({{
  color: {color},
  metalness: {metal},
  roughness: {rough},
  envMapIntensity: {env}
}});"""
    return inp, out

def gen_glass_material():
    u = uid()
    opacity = round(random.uniform(0.05, 0.45), 2)
    ior = round(random.uniform(1.3, 2.0), 2)
    tint = random.choice(["0xffffff", "0xCCEEFF", "0xFFEECC", "0xCCFFCC"])
    inp = {"action": "set_material", "preset": "glass",
           "opacity": opacity, "ior": ior, "tint": tint, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshPhysicalMaterial({{
  color: {tint},
  metalness: 0,
  roughness: 0,
  transmission: 1.0,
  thickness: 0.5,
  ior: {ior},
  transparent: True,
  opacity: {opacity},
  envMapIntensity: 2.0,
  attenuationColor: new THREE.Color({tint}),
  attenuationDistance: 1.0
}});"""
    return inp, out

def gen_skin_material():
    u = uid()
    tones = ["0xFFCBA4", "0xF1C27D", "0xE0AC69", "0xC68642", "0x8D5524"]
    tone = random.choice(tones)
    inp = {"action": "set_material", "preset": "skin",
           "tone": tone, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshPhysicalMaterial({{
  color: {tone},
  metalness: 0,
  roughness: 0.65,
  sheen: 0.3,
  sheenColor: new THREE.Color(0xFF8C69),
  transmission: 0.05,
  thickness: 0.8,
  attenuationColor: new THREE.Color(0xFF6B6B),
  attenuationDistance: 0.5
}});"""
    return inp, out

def gen_fabric_material():
    u = uid()
    color = rand_color()
    sheen = round(random.uniform(0.5, 1.0), 2)
    rough = round(random.uniform(0.7, 1.0), 2)
    inp = {"action": "set_material", "preset": "fabric",
           "color": color, "sheen": sheen, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshPhysicalMaterial({{
  color: {color},
  roughness: {rough},
  metalness: 0,
  sheen: {sheen},
  sheenRoughness: 0.85,
  sheenColor: new THREE.Color({color})
}});"""
    return inp, out

def gen_emissive_material():
    u = uid()
    colors = ["0x00ffff", "0xff00ff", "0x00ff88", "0xff4400",
              "0x4488ff", "0xffff00", "0xff0088", "0x88ff00"]
    color = random.choice(colors)
    intensity = round(random.uniform(0.5, 8), 1)
    inp = {"action": "set_material", "preset": "emissive",
           "color": color, "intensity": intensity, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshStandardMaterial({{
  color: 0x111111,
  emissive: new THREE.Color({color}),
  emissiveIntensity: {intensity},
  roughness: 0.2,
  metalness: 0.8
}});"""
    return inp, out

def gen_subsurface_material():
    u = uid()
    colors = ["0xFFCCCC", "0xFFEECC", "0xCCFFCC", "0xCCCCFF"]
    color = random.choice(colors)
    inp = {"action": "set_material", "preset": "subsurface",
           "color": color, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshPhysicalMaterial({{
  color: {color},
  metalness: 0,
  roughness: 0.4,
  transmission: 0.6,
  thickness: 1.5,
  attenuationColor: new THREE.Color({color}),
  attenuationDistance: 2.0,
  ior: 1.4
}});"""
    return inp, out

def gen_clearcoat_material():
    u = uid()
    color = rand_color()
    cc = round(random.uniform(0.5, 1.0), 2)
    ccr = round(random.uniform(0.0, 0.3), 2)
    inp = {"action": "set_material", "preset": "clearcoat",
           "color": color, "clearcoat": cc, "label": f"mat_{u}"}
    out = f"""const mat_{u} = new THREE.MeshPhysicalMaterial({{
  color: {color},
  metalness: 0.1,
  roughness: 0.4,
  clearcoat: {cc},
  clearcoatRoughness: {ccr},
  envMapIntensity: 1.5
}});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 3 — LIGHTING
# ─────────────────────────────────────────────────────────────

def gen_ambient_light():
    u = uid()
    intensity = round(random.uniform(0.1, 1.5), 2)
    colors = ["0xffffff", "0xFFEECC", "0xCCEEFF", "0xFFCCFF"]
    color = random.choice(colors)
    inp = {"action": "add_light", "type": "ambient",
           "color": color, "intensity": intensity, "label": f"amb_{u}"}
    out = f"""const amb_{u} = new THREE.AmbientLight({color},{intensity});
scene.add(amb_{u});"""
    return inp, out

def gen_directional_light():
    u = uid()
    intensity = round(random.uniform(0.5, 5), 2)
    pos = [round(random.uniform(-10, 10), 1) for _ in range(3)]
    res = random.choice([512, 1024, 2048, 4096])
    inp = {"action": "add_light", "type": "directional",
           "intensity": intensity, "position": pos, "shadow_res": res, "label": f"dir_{u}"}
    out = f"""const dir_{u} = new THREE.DirectionalLight(0xffffff,{intensity});
dir_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
dir_{u}.castShadow = true;
dir_{u}.shadow.mapSize.width = {res};
dir_{u}.shadow.mapSize.height = {res};
dir_{u}.shadow.camera.near = 0.1;
dir_{u}.shadow.camera.far = 100;
dir_{u}.shadow.camera.left = -20;
dir_{u}.shadow.camera.right = 20;
dir_{u}.shadow.camera.top = 20;
dir_{u}.shadow.camera.bottom = -20;
dir_{u}.shadow.bias = -0.001;
scene.add(dir_{u});"""
    return inp, out

def gen_point_light():
    u = uid()
    intensity = round(random.uniform(0.5, 10), 2)
    distance = round(random.uniform(2, 20), 1)
    decay = round(random.uniform(1, 2), 1)
    pos = rand_pos(4)
    colors = ["0xffffff", "0xFFEEAA", "0xAAEEFF", "0xFF8844", "0x44AAFF"]
    color = random.choice(colors)
    inp = {"action": "add_light", "type": "point",
           "color": color, "intensity": intensity, "distance": distance, 
           "decay": decay, "position": pos, "label": f"pt_{u}"}
    out = f"""const pt_{u} = new THREE.PointLight({color},{intensity},{distance},{decay});
pt_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
pt_{u}.castShadow = true;
pt_{u}.shadow.mapSize.width = 1024;
pt_{u}.shadow.mapSize.height = 1024;
scene.add(pt_{u});"""
    return inp, out

def gen_spot_light():
    u = uid()
    intensity = round(random.uniform(1, 10), 2)
    angle = round(random.uniform(0.1, 0.6), 2)
    penumbra = round(random.uniform(0, 0.5), 2)
    pos = rand_pos(5)
    inp = {"action": "add_light", "type": "spot",
           "intensity": intensity, "angle": angle, "penumbra": penumbra,
           "position": pos, "label": f"spot_{u}"}
    out = f"""const spot_{u} = new THREE.SpotLight(0xffffff,{intensity});
spot_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
spot_{u}.angle = {angle};
spot_{u}.penumbra = {penumbra};
spot_{u}.castShadow = true;
spot_{u}.shadow.mapSize.width = 2048;
spot_{u}.shadow.mapSize.height = 2048;
spot_{u}.target.position.set(0,0,0);
scene.add(spot_{u});
scene.add(spot_{u}.target);"""
    return inp, out

def gen_area_light():
    u = uid()
    w = round(random.uniform(1, 6), 1)
    h = round(random.uniform(1, 6), 1)
    intensity = round(random.uniform(1, 15), 1)
    pos = rand_pos(4)
    inp = {"action": "add_light", "type": "area",
           "width": w, "height": h, "intensity": intensity,
           "position": pos, "label": f"area_{u}"}
    out = f"""const area_{u} = new THREE.RectAreaLight(0xffffff,{intensity},{w},{h});
area_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
area_{u}.lookAt(0,0,0);
scene.add(area_{u});"""
    return inp, out

def gen_hemisphere_light():
    u = uid()
    sky_colors = ["0x87CEEB", "0xFFEECC", "0xAADDFF"]
    gnd_colors = ["0x4A7C3F", "0x8B6914", "0x555555"]
    sky = random.choice(sky_colors)
    gnd = random.choice(gnd_colors)
    intensity = round(random.uniform(0.3, 1.5), 2)
    inp = {"action": "add_light", "type": "hemisphere",
           "sky": sky, "ground": gnd, "intensity": intensity, "label": f"hemi_{u}"}
    out = f"""const hemi_{u} = new THREE.HemisphereLight({sky},{gnd},{intensity});
hemi_{u}.position.set(0,20,0);
scene.add(hemi_{u});"""
    return inp, out

def gen_hdri_env():
    u = uid()
    envs = ["studio", "outdoor_day", "night_city", "forest", "sunset", "overcast", "interior"]
    env = random.choice(envs)
    exposure = round(random.uniform(0.5, 2.0), 2)
    inp = {"action": "add_environment", "preset": env,
           "exposure": exposure, "label": f"env_{u}"}
    out = f"""renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = {exposure};
renderer.outputColorSpace = THREE.SRGBColorSpace;
const pmrem_{u} = new THREE.PMREMGenerator(renderer);
pmrem_{u}.compileEquirectangularShader();
new THREE.RGBELoader().load('{env}.hdr', (tex) => {{
  const envMap = pmrem_{u}.fromEquirectangular(tex).texture;
  scene.environment = envMap;
  scene.background = envMap;
  tex.dispose();
  pmrem_{u}.dispose();
}});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 4 — GLSL SHADERS
# ─────────────────────────────────────────────────────────────

def gen_noise_shader():
    u = uid()
    amp = round(random.uniform(0.1, 0.8), 2)
    freq = round(random.uniform(0.5, 4.0), 2)
    colors = ["0.2,0.6,1.0", "0.8,0.3,0.1", "0.1,0.8,0.4", "0.7,0.2,0.8"]
    color = random.choice(colors)
    inp = {"action": "add_shader", "preset": "noise_displacement",
           "amplitude": amp, "frequency": freq, "label": f"noise_{u}"}
    out = f"""const mat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{
    time: {{ value: 0 }},
    amplitude: {{ value: {amp} }},
    frequency: {{ value: {freq} }}
  }},
  vertexShader: `
    uniform float time;
    uniform float amplitude;
    uniform float frequency;
    void main() {{
      vec3 pos = position;
      float n = sin(pos.x*frequency+time)*cos(pos.z*frequency+time*0.7);
      float n2 = sin(pos.y*frequency*1.3+time*0.5)*0.5;
      pos.y += (n+n2)*amplitude;
      gl_Position = projectionMatrix*modelViewMatrix*vec4(pos,1.0);
    }}
  `,
  fragmentShader: `
    uniform float time;
    void main() {{
      float pulse = 0.5+0.5*sin(time*2.0);
      gl_FragColor = vec4({color},1.0)*pulse+vec4({color},1.0)*(1.0-pulse)*0.4;
    }}
  `
}});
morphix_shaders.push(mat_{u});"""
    return inp, out

def gen_fresnel_shader():
    u = uid()
    colors = ["0x00aaff", "0xff4400", "0x00ff88", "0xffaa00", "0xaa00ff"]
    color = random.choice(colors)
    intensity = round(random.uniform(1, 4), 1)
    inp = {"action": "add_shader", "preset": "fresnel",
           "color": color, "intensity": intensity, "label": f"fresnel_{u}"}
    out = f"""const mat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{
    glowColor: {{ value: new THREE.Color({color}) }},
    intensity: {{ value: {intensity} }}
  }},
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewDir;
    void main() {{
      vNormal = normalize(normalMatrix*normal);
      vec4 worldPos = modelViewMatrix*vec4(position,1.0);
      vViewDir = normalize(-worldPos.xyz);
      gl_Position = projectionMatrix*worldPos;
    }}
  `,
  fragmentShader: `
    uniform vec3 glowColor;
    uniform float intensity;
    varying vec3 vNormal;
    varying vec3 vViewDir;
    void main() {{
      float fresnel = pow(1.0-max(dot(vNormal,vViewDir),0.0),3.0);
      gl_FragColor = vec4(glowColor*fresnel*intensity,fresnel);
    }}
  `,
  transparent: True,
  side: THREE.FrontSide,
  depthWrite: False
}});
morphix_shaders.push(mat_{u});"""
    return inp, out

def gen_hologram_shader():
    u = uid()
    inp = {"action": "add_shader", "preset": "hologram", "label": f"holo_{u}"}
    out = f"""const mat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{ time: {{ value: 0 }}, color: {{ value: new THREE.Color(0x00ffff) }} }},
  vertexShader: `
    varying vec2 vUv;
    varying float vY;
    void main() {{
      vUv = uv;
      vY = position.y;
      gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);
    }}
  `,
  fragmentShader: `
    uniform float time;
    uniform vec3 color;
    varying vec2 vUv;
    varying float vY;
    void main() {{
      float scanline = step(0.5,fract(vY*20.0-time*3.0));
      float edge = smoothstep(0.0,0.05,vUv.x)*smoothstep(0.0,0.05,1.0-vUv.x)
                  *smoothstep(0.0,0.05,vUv.y)*smoothstep(0.0,0.05,1.0-vUv.y);
      float flicker = 0.85+0.15*sin(time*17.3);
      gl_FragColor = vec4(color*scanline*flicker*edge,0.7*scanline*edge);
    }}
  `,
  transparent: True,
  side: THREE.DoubleSide,
  depthWrite: False
}});
morphix_shaders.push(mat_{u});"""
    return inp, out

def gen_water_shader():
    u = uid()
    inp = {"action": "add_shader", "preset": "water", "label": f"water_{u}"}
    out = f"""const geo_{u} = new THREE.PlaneGeometry(20,20,64,64);
const mat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{
    time: {{ value: 0 }},
    waterColor: {{ value: new THREE.Color(0x0066AA) }},
    foamColor: {{ value: new THREE.Color(0xAADDFF) }}
  }},
  vertexShader: `
    uniform float time;
    varying vec2 vUv;
    varying float vHeight;
    void main() {{
      vUv = uv;
      vec3 pos = position;
      pos.z += sin(pos.x*2.0+time)*0.15
              +sin(pos.y*3.0+time*1.3)*0.1
              +sin((pos.x+pos.y)*1.5+time*0.8)*0.08;
      vHeight = pos.z;
      gl_Position = projectionMatrix*modelViewMatrix*vec4(pos,1.0);
    }}
  `,
  fragmentShader: `
    uniform vec3 waterColor;
    uniform vec3 foamColor;
    varying vec2 vUv;
    varying float vHeight;
    void main() {{
      float foam = smoothstep(0.1,0.3,vHeight+0.1);
      vec3 color = mix(waterColor,foamColor,foam);
      gl_FragColor = vec4(color,0.85);
    }}
  `,
  transparent: True
}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
mesh_{u}.rotation.x = -Math.PI/2;
mesh_{u}.position.y = 0.0;
scene.add(mesh_{u});
morphix_shaders.push(mat_{u});"""
    return inp, out

def gen_fire_shader():
    u = uid()
    inp = {"action": "add_shader", "preset": "fire", "label": f"fire_{u}"}
    out = f"""const geo_{u} = new THREE.PlaneGeometry(1,2,16,32);
const mat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{ time: {{ value: 0 }} }},
  vertexShader: `
    uniform float time;
    varying vec2 vUv;
    void main() {{
      vUv = uv;
      vec3 pos = position;
      float flicker = sin(pos.y*4.0+time*8.0)*0.08*(1.0-uv.y);
      pos.x += flicker;
      gl_Position = projectionMatrix*modelViewMatrix*vec4(pos,1.0);
    }}
  `,
  fragmentShader: `
    uniform float time;
    varying vec2 vUv;
    void main() {{
      float y = 1.0-vUv.y;
      float heat = smoothstep(0.0,0.4,y)*smoothstep(1.0,0.5,y);
      vec3 fire = mix(vec3(1.0,0.9,0.0),vec3(1.0,0.2,0.0),vUv.y);
      fire = mix(fire,vec3(0.1,0.0,0.8),smoothstep(0.6,1.0,vUv.y));
      float noise = fract(sin(vUv.x*127.1+time*3.0)*43758.5)*0.15;
      gl_FragColor = vec4(fire,heat*(0.8+noise));
    }}
  `,
  transparent: True,
  side: THREE.DoubleSide,
  depthWrite: False
}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
scene.add(mesh_{u});
morphix_shaders.push(mat_{u});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 5 — POST-PROCESSING
# ─────────────────────────────────────────────────────────────

def gen_bloom():
    u = uid()
    strength = round(random.uniform(0.2, 3.0), 2)
    radius = round(random.uniform(0.1, 1.0), 2)
    threshold = round(random.uniform(0.0, 0.8), 2)
    inp = {"action": "add_post", "preset": "bloom",
           "strength": strength, "radius": radius, "threshold": threshold, "label": f"bloom_{u}"}
    out = f"""const composer_{u} = new EffectComposer(renderer);
composer_{u}.addPass(new RenderPass(scene,camera));
const bloom_{u} = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth,window.innerHeight),
  {strength},{radius},{threshold}
);
composer_{u}.addPass(bloom_{u});
window._composer = composer_{u};"""
    return inp, out

def gen_ssao():
    u = uid()
    radius = round(random.uniform(4, 32), 0)
    inp = {"action": "add_post", "preset": "ssao",
           "kernel_radius": radius, "label": f"ssao_{u}"}
    out = f"""const ssao_{u} = new SSAOPass(scene,camera,window.innerWidth,window.innerHeight);
ssao_{u}.kernelRadius = {radius};
ssao_{u}.minDistance = 0.001;
ssao_{u}.maxDistance = 0.15;
composer.addPass(ssao_{u});"""
    return inp, out

def gen_dof():
    u = uid()
    focus = round(random.uniform(1, 20), 1)
    aperture = round(random.uniform(0.000005, 0.0002), 7)
    inp = {"action": "add_post", "preset": "dof",
           "focus": focus, "aperture": aperture, "label": f"dof_{u}"}
    out = f"""const dof_{u} = new BokehPass(scene,camera,{{
  focus: {focus},
  aperture: {aperture},
  maxblur: 0.015
}});
composer.addPass(dof_{u});"""
    return inp, out

def gen_tone_mapping():
    u = uid()
    modes = [
        ("THREE.ACESFilmicToneMapping", round(random.uniform(0.6, 2.0), 2)),
        ("THREE.ReinhardToneMapping", round(random.uniform(1.0, 3.0), 2)),
        ("THREE.CineonToneMapping", round(random.uniform(1.0, 2.5), 2)),
        ("THREE.LinearToneMapping", round(random.uniform(0.8, 1.5), 2)),
    ]
    mode, exp = random.choice(modes)
    inp = {"action": "add_post", "preset": "tone_mapping",
           "mode": mode, "exposure": exp, "label": f"tm_{u}"}
    out = f"""renderer.toneMapping = {mode};
renderer.toneMappingExposure = {exp};
renderer.outputColorSpace = THREE.SRGBColorSpace;"""
    return inp, out

def gen_fog():
    u = uid()
    ftype = random.choice(["linear", "exponential"])
    inp = {"action": "add_post", "preset": f"fog_{ftype}", "label": f"fog_{u}"}
    if ftype == "linear":
        near = round(random.uniform(1, 10), 1)
        far = round(random.uniform(20, 100), 1)
        colors = ["0xcccccc", "0xaaaaaa", "0x87CEEB", "0x333355"]
        color = random.choice(colors)
        out = f"scene.fog = new THREE.Fog({color},{near},{far});"
    else:
        density = round(random.uniform(0.005, 0.08), 4)
        out = f"scene.fog = new THREE.FogExp2(0x888899,{density});"
    return inp, out

def gen_ssr():
    u = uid()
    inp = {"action": "add_post", "preset": "ssr", "label": f"ssr_{u}"}
    out = f"""const ssr_{u} = new SSRPass({{
  renderer, scene, camera,
  width: window.innerWidth,
  height: window.innerHeight,
  groundReflector: null,
  selects: null
}});
ssr_{u}.thickness = 0.018;
ssr_{u}.infiniteThick = False;
ssr_{u}.maxDistance = 0.1;
composer.addPass(ssr_{u});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 6 — ANIMATION
# ─────────────────────────────────────────────────────────────

def gen_keyframe_anim():
    u = uid()
    anim_types = ["position", "rotation", "scale"]
    atype = random.choice(anim_types)
    duration = round(random.uniform(1, 5), 1)
    inp = {"action": "add_animation", "preset": "keyframe",
           "track": atype, "duration": duration, "label": f"kf_{u}"}
    if atype == "position":
        track_type = "VectorKeyframeTrack"
        times = [0, duration/2, duration]
        vals = []
        for _ in range(3):
            vals.extend(rand_pos())
        prop = "mesh_target.position"
    elif atype == "rotation":
        track_type = "QuaternionKeyframeTrack"
        times = [0, duration]
        vals = [0, 0, 0, 1, 0, 1, 0, 0]
        prop = "mesh_target.quaternion"
    else:
        track_type = "VectorKeyframeTrack"
        sc = round(random.uniform(0.5, 2), 2)
        times = [0, duration/2, duration]
        vals = [1, 1, 1, sc, sc, sc, 1, 1, 1]
        prop = "mesh_target.scale"
    out = f"""const kf_{u} = new THREE.{track_type}(
  '{prop}',
  {times},
  {vals}
);
const clip_{u} = new THREE.AnimationClip('anim_{u}',{duration},[kf_{u}]);
const mixer_{u} = new THREE.AnimationMixer(mesh_target);
mixer_{u}.clipAction(clip_{u}).play();
morphix_mixers.push(mixer_{u});"""
    return inp, out

def gen_morph_anim():
    u = uid()
    inp = {"action": "add_animation", "preset": "morph", "label": f"morph_{u}"}
    out = f"""const geo_{u} = new THREE.SphereGeometry(1,32,32);
const pos_{u} = geo_{u}.attributes.position;
const morphVerts_{u} = new Float32Array(pos_{u}.count*3);
for(let i=0;i<pos_{u}.count;i++){{
  morphVerts_{u}[i*3]   = pos_{u}.getX(i)*(1.5+Math.random()*0.3);
  morphVerts_{u}[i*3+1] = pos_{u}.getY(i)*(0.4+Math.random()*0.2);
  morphVerts_{u}[i*3+2] = pos_{u}.getZ(i)*(1.5+Math.random()*0.3);
}}
geo_{u}.morphAttributes.position = [new THREE.Float32BufferAttribute(morphVerts_{u},3)];
const mat_{u} = new THREE.MeshStandardMaterial({{color:0x4488ff,morphTargets:True,roughness:0.4,metalness:0.3}});
const mesh_{u} = new THREE.Mesh(geo_{u},mat_{u});
scene.add(mesh_{u});
morphix_morphs = morphix_morphs||[];
morphix_morphs.push(mesh_{u});"""
    return inp, out

def gen_physics_anim():
    u = uid()
    mass = round(random.uniform(0.5, 5), 1)
    restitution = round(random.uniform(0.3, 0.9), 2)
    inp = {"action": "add_animation", "preset": "physics",
           "mass": mass, "restitution": restitution, "label": f"phys_{u}"}
    out = f"""morphix_physics = morphix_physics||[];
morphix_physics_ticks = morphix_physics_ticks||[];
const body_{u} = {{
  mesh: mesh_target,
  vel: new THREE.Vector3((Math.random()-0.5)*2,Math.random()*3,(Math.random()-0.5)*2),
  mass: {mass},
  restitution: {restitution}
}};
morphix_physics.push(body_{u});
function physicsTick_{u}(delta){{
  body_{u}.vel.y -= 9.8*delta;
  body_{u}.mesh.position.addScaledVector(body_{u}.vel,delta);
  if(body_{u}.mesh.position.y < 0.0){{
    body_{u}.mesh.position.y = 0.0;
    body_{u}.vel.y *= -{restitution};
    body_{u}.vel.x *= 0.92;
    body_{u}.vel.z *= 0.92;
  }}
}}
morphix_physics_ticks.push(physicsTick_{u});"""
    return inp, out

def gen_orbit_anim():
    u = uid()
    radius = round(random.uniform(1, 6), 2)
    speed = round(random.uniform(0.2, 3), 2)
    inp = {"action": "add_animation", "preset": "orbit",
           "radius": radius, "speed": speed, "label": f"orbit_{u}"}
    out = f"""morphix_orbits = morphix_orbits||[];
morphix_orbits.push({{
  mesh: mesh_target,
  radius: {radius},
  speed: {speed},
  phase: Math.random()*Math.PI*2,
  yOffset: mesh_target.position.y
}});"""
    return inp, out

def gen_particle_system():
    u = uid()
    count = random.randint(100, 5000)
    size = round(random.uniform(0.01, 0.1), 3)
    color = rand_color()
    inp = {"action": "add_particles", "count": count,
           "size": size, "color": color, "label": f"parts_{u}"}
    out = f"""const geo_{u} = new THREE.BufferGeometry();
const verts_{u} = new Float32Array({count}*3);
const vels_{u} = [];
for(let i=0;i<{count};i++){{
  verts_{u}[i*3]   = (Math.random()-0.5)*8;
  verts_{u}[i*3+1] = Math.random()*6;
  verts_{u}[i*3+2] = (Math.random()-0.5)*8;
  vels_{u}.push(new THREE.Vector3(
    (Math.random()-0.5)*0.02,
    -0.01-Math.random()*0.03,
    (Math.random()-0.5)*0.02
  ));
}}
geo_{u}.setAttribute('position',new THREE.BufferAttribute(verts_{u},3));
const mat_{u} = new THREE.PointsMaterial({{color:{color},size:{size},sizeAttenuation:True,transparent:True,opacity:0.8}});
const pts_{u} = new THREE.Points(geo_{u},mat_{u});
scene.add(pts_{u});
morphix_particles = morphix_particles||[];
morphix_particles.push({{geo:geo_{u},vels:vels_{u},count:{count}}});"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# CATEGORY 7 — SCENE COMPOSITION
# ─────────────────────────────────────────────────────────────

def gen_lod():
    u = uid()
    shape = random.choice(["sphere", "box"])
    inp = {"action": "add_lod", "shape": shape, "label": f"lod_{u}"}
    segs = [64, 32, 16, 8, 4]
    levels = [0, 5, 10, 20, 40]
    lines = []
    for s, l in zip(segs, levels):
        if shape == "sphere":
            lines.append(f"  lod_{u}.addLevel(new THREE.Mesh(new THREE.SphereGeometry(1,{s},{s}),mat_{u}),{l});")
        else:
            lines.append(f"  lod_{u}.addLevel(new THREE.Mesh(new THREE.BoxGeometry(1,1,1),mat_{u}),{l});")
    out = f"""const mat_{u} = new THREE.MeshStandardMaterial({{color:0x888888,roughness:0.4,metalness:0.5}});
const lod_{u} = new THREE.LOD();
{chr(10).join(lines)}
scene.add(lod_{u});"""
    return inp, out

def gen_env_map():
    u = uid()
    inp = {"action": "add_environment", "preset": "cube_env_map", "label": f"env_{u}"}
    out = f"""const cubeLoader_{u} = new THREE.CubeTextureLoader();
const envMap_{u} = cubeLoader_{u}.load(['px.jpg','nx.jpg','py.jpg','ny.jpg','pz.jpg','nz.jpg']);
scene.background = envMap_{u};
scene.environment = envMap_{u};"""
    return inp, out

def gen_reflection_probe():
    u = uid()
    pos = rand_pos(3)
    inp = {"action": "add_reflection", "position": pos, "label": f"probe_{u}"}
    out = f"""const cubeRT_{u} = new THREE.WebGLCubeRenderTarget(512,{{
  format: THREE.RGBFormat,
  generateMipmaps: True,
  minFilter: THREE.LinearMipmapLinearFilter
}});
const cubeCamera_{u} = new THREE.CubeCamera(0.1,100,cubeRT_{u});
cubeCamera_{u}.position.set({pos[0]},{pos[1]},{pos[2]});
scene.add(cubeCamera_{u});
morphix_cube_cameras = morphix_cube_cameras||[];
morphix_cube_cameras.push(cubeCamera_{u});"""
    return inp, out

def gen_shadow_setup():
    u = uid()
    res = random.choice([1024, 2048, 4096])
    inp = {"action": "setup_shadows", "resolution": res, "label": f"shadow_{u}"}
    out = f"""renderer.shadowMap.enabled = True;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;"""
    return inp, out

def gen_sky_shader():
    u = uid()
    top_colors = ["0x0077FF", "0x001133", "0x224488"]
    bot_colors = ["0xCCEEFF", "0xFF8844", "0x88AACC"]
    top = random.choice(top_colors)
    bot = random.choice(bot_colors)
    inp = {"action": "add_environment", "preset": "sky_shader",
           "top": top, "bottom": bot, "label": f"sky_{u}"}
    out = f"""const skyGeo_{u} = new THREE.SphereGeometry(400,32,15);
const skyMat_{u} = new THREE.ShaderMaterial({{
  uniforms: {{
    topColor: {{ value: new THREE.Color({top}) }},
    bottomColor: {{ value: new THREE.Color({bot}) }},
    offset: {{ value: 33 }},
    exponent: {{ value: 0.6 }}
  }},
  vertexShader: `
    varying vec3 vWorldPos;
    void main() {{
      vec4 wp = modelMatrix*vec4(position,1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);
    }}
  `,
  fragmentShader: `
    uniform vec3 topColor;
    uniform vec3 bottomColor;
    uniform float offset;
    uniform float exponent;
    varying vec3 vWorldPos;
    void main() {{
      float h = normalize(vWorldPos+offset).y;
      gl_FragColor = vec4(mix(bottomColor,topColor,max(pow(max(h,0.0),exponent),0.0)),1.0);
    }}
  `,
  side: THREE.BackSide
}});
scene.add(new THREE.Mesh(skyGeo_{u},skyMat_{u}));"""
    return inp, out

# ─────────────────────────────────────────────────────────────
# REAL-WORLD CHAIN SEQUENCES
# ─────────────────────────────────────────────────────────────

CHAINS = [

# ── BUILDING ──────────────────────────────────────────────────
[
  ({"action":"add_box","width":8,"height":0.3,"depth":6,"position":[0,0,0],"color":"0x777777","label":"foundation"},
"""const geo_foundation = new THREE.BoxGeometry(8,0.3,6);
const mat_foundation = new THREE.MeshStandardMaterial({color:0x777777,roughness:0.9,metalness:0.1});
const mesh_foundation = new THREE.Mesh(geo_foundation,mat_foundation);
mesh_foundation.position.set(0,0,0);
mesh_foundation.receiveShadow = True;
scene.add(mesh_foundation);"""),

  ({"action":"add_box","width":8,"height":4,"depth":6,"position":[0,2.15,0],"color":"0xF5DEB3","label":"walls"},
"""const geo_walls = new THREE.BoxGeometry(8,4,6);
const mat_walls = new THREE.MeshStandardMaterial({color:0xF5DEB3,roughness:0.85,metalness:0});
const mesh_walls = new THREE.Mesh(geo_walls,mat_walls);
mesh_walls.position.set(0,2.15,0);
mesh_walls.castShadow = True;
mesh_walls.receiveShadow = True;
scene.add(mesh_walls);"""),

  ({"action":"add_extrude","depth":6.4,"bevel":False,"position":[0,0,0],"color":"0x8B0000","label":"roof"},
"""const roofShape = new THREE.Shape();
roofShape.moveTo(-4.4,0);
roofShape.lineTo(0,2.2);
roofShape.lineTo(4.4,0);
roofShape.closePath();
const geo_roof = new THREE.ExtrudeGeometry(roofShape,{depth:6.4,bevelEnabled:False});
const mat_roof = new THREE.MeshStandardMaterial({color:0x8B0000,roughness:0.7,metalness:0.05});
const mesh_roof = new THREE.Mesh(geo_roof,mat_roof);
mesh_roof.position.set(-4.4,4.3,-3.2);
mesh_roof.castShadow = True;
scene.add(mesh_roof);"""),

  ({"action":"add_box","width":1.2,"height":2,"depth":0.12,"position":[0,1.15,3.07],"color":"0x8B4513","label":"door"},
"""const geo_door = new THREE.BoxGeometry(1.2,2,0.12);
const mat_door = new THREE.MeshStandardMaterial({color:0x8B4513,roughness:0.5,metalness:0.1});
const mesh_door = new THREE.Mesh(geo_door,mat_door);
mesh_door.position.set(0,1.15,3.07);
mesh_door.castShadow = True;
scene.add(mesh_door);
const geo_knob = new THREE.SphereGeometry(0.05,8,8);
const mat_knob = new THREE.MeshStandardMaterial({color:0xFFD700,roughness:0.1,metalness:0.95});
const mesh_knob = new THREE.Mesh(geo_knob,mat_knob);
mesh_knob.position.set(0.45,1.1,3.14);
scene.add(mesh_knob);"""),

  ({"action":"add_box","width":1.1,"height":1.1,"depth":0.06,"position":[2.5,2.2,3.07],"color":"0xADD8E6","label":"window_right"},
"""const geo_win = new THREE.BoxGeometry(1.1,1.1,0.06);
const mat_win = new THREE.MeshPhysicalMaterial({color:0xCCEEFF,transmission:0.9,roughness:0,metalness:0,ior:1.5,transparent:True,opacity:0.25});
const mesh_win = new THREE.Mesh(geo_win,mat_win);
mesh_win.position.set(2.5,2.2,3.07);
scene.add(mesh_win);
const mat_frame = new THREE.MeshStandardMaterial({color:0xFFFFFF,roughness:0.4,metalness:0.1});
[[-0.55,0],[0.55,0],[0,-0.55],[0,0.55]].forEach(([ox,oy])=>{
  const f = new THREE.Mesh(new THREE.BoxGeometry(0.06,1.1,0.08),mat_frame);
  f.position.set(2.5+ox,2.2+oy,3.1);
  scene.add(f);
});"""),

  ({"action":"add_box","width":1.1,"height":1.1,"depth":0.06,"position":[-2.5,2.2,3.07],"color":"0xADD8E6","label":"window_left"},
"""const mesh_winl = mesh_win.clone();
mesh_winl.position.set(-2.5,2.2,3.07);
scene.add(mesh_winl);"""),

  ({"action":"add_light","type":"directional","intensity":2.5,"position":[10,15,8],"label":"sun"},
"""const sun = new THREE.DirectionalLight(0xFFEECC,2.5);
sun.position.set(10,15,8);
sun.castShadow = True;
sun.shadow.mapSize.width = 2048;
sun.shadow.mapSize.height = 2048;
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 80;
sun.shadow.camera.left = -20;
sun.shadow.camera.right = 20;
sun.shadow.camera.top = 20;
sun.shadow.camera.bottom = -20;
scene.add(sun);
scene.add(new THREE.AmbientLight(0x88AACC,0.4));"""),
],

# ── HUMAN FIGURE ──────────────────────────────────────────────
[
  ({"action":"add_sphere","radius":0.22,"position":[0,1.72,0],"color":"0xFFCBA4","label":"head"},
"""const mat_skin = new THREE.MeshPhysicalMaterial({color:0xFFCBA4,roughness:0.65,metalness:0,sheen:0.3,sheenColor:new THREE.Color(0xFF8C69)});
const mesh_head = new THREE.Mesh(new THREE.SphereGeometry(0.22,32,32),mat_skin);
mesh_head.position.set(0,1.72,0);
mesh_head.castShadow = True;
scene.add(mesh_head);
const geo_eye = new THREE.SphereGeometry(0.03,8,8);
const mat_eye = new THREE.MeshStandardMaterial({color:0x222222,roughness:0.1,metalness:0});
[-0.07,0.07].forEach(x=>{
  const eye = new THREE.Mesh(geo_eye,mat_eye);
  eye.position.set(x,1.75,0.19);
  scene.add(eye);
});"""),

  ({"action":"add_cylinder","radiusTop":0.11,"radiusBottom":0.13,"height":0.18,"position":[0,1.42,0],"color":"0xFFCBA4","label":"neck"},
"""const mesh_neck = new THREE.Mesh(new THREE.CylinderGeometry(0.11,0.13,0.18,16),mat_skin);
mesh_neck.position.set(0,1.42,0);
mesh_neck.castShadow = True;
scene.add(mesh_neck);"""),

  ({"action":"add_box","width":0.52,"height":0.62,"depth":0.24,"position":[0,1.03,0],"color":"0x2244AA","label":"torso"},
"""const mat_shirt = new THREE.MeshPhysicalMaterial({color:0x2244AA,roughness:0.85,metalness:0,sheen:0.4,sheenColor:new THREE.Color(0x1A1A88)});
const mesh_torso = new THREE.Mesh(new THREE.BoxGeometry(0.52,0.62,0.24),mat_shirt);
mesh_torso.position.set(0,1.03,0);
mesh_torso.castShadow = True;
scene.add(mesh_torso);"""),

  ({"action":"add_cylinder","radiusTop":0.055,"radiusBottom":0.07,"height":0.56,"position":[-0.34,0.89,0],"color":"0x2244AA","label":"arm_left"},
"""const mat_arm = new THREE.MeshPhysicalMaterial({color:0xFFCBA4,roughness:0.65,metalness:0,sheen:0.3,sheenColor:new THREE.Color(0xFF8C69)});
const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.07,0.56,12),mat_shirt);
arm.position.set(-0.34,0.89,0);
arm.rotation.z = 0.22;
arm.castShadow = True;
scene.add(arm);
const forearm = new THREE.Mesh(new THREE.CylinderGeometry(0.045,0.055,0.46,12),mat_arm);
forearm.position.set(-0.44,0.55,0);
forearm.rotation.z = 0.35;
scene.add(forearm);
const hand = new THREE.Mesh(new THREE.SphereGeometry(0.055,10,10),mat_arm);
hand.position.set(-0.52,0.32,0);
hand.scale.set(1,0.7,0.8);
scene.add(hand);"""),

  ({"action":"add_cylinder","radiusTop":0.055,"radiusBottom":0.07,"height":0.56,"position":[0.34,0.89,0],"color":"0x2244AA","label":"arm_right"},
"""const arm2 = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.07,0.56,12),mat_shirt);
arm2.position.set(0.34,0.89,0);
arm2.rotation.z = -0.22;
arm2.castShadow = True;
scene.add(arm2);
const forearm2 = new THREE.Mesh(new THREE.CylinderGeometry(0.045,0.055,0.46,12),mat_arm);
forearm2.position.set(0.44,0.55,0);
forearm2.rotation.z = -0.35;
scene.add(forearm2);
const hand2 = new THREE.Mesh(new THREE.SphereGeometry(0.055,10,10),mat_arm);
hand2.position.set(0.52,0.32,0);
hand2.scale.set(1,0.7,0.8);
scene.add(hand2);"""),

  ({"action":"add_box","width":0.48,"height":0.38,"depth":0.22,"position":[0,0.61,0],"color":"0x1A1A2E","label":"hips"},
"""const mat_pants = new THREE.MeshStandardMaterial({color:0x1A1A2E,roughness:0.9,metalness:0});
const mesh_hips = new THREE.Mesh(new THREE.BoxGeometry(0.48,0.38,0.22),mat_pants);
mesh_hips.position.set(0,0.61,0);
mesh_hips.castShadow = True;
scene.add(mesh_hips);"""),

  ({"action":"add_cylinder","radiusTop":0.075,"radiusBottom":0.085,"height":0.44,"position":[-0.14,0.34,0],"color":"0x1A1A2E","label":"leg_left"},
"""const thigh = new THREE.Mesh(new THREE.CylinderGeometry(0.075,0.085,0.44,12),mat_pants);
thigh.position.set(-0.14,0.34,0);
thigh.castShadow = True;
scene.add(thigh);
const shin = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.07,0.44,12),mat_pants);
shin.position.set(-0.14,0.03,0.02);
shin.castShadow = True;
scene.add(shin);
const foot = new THREE.Mesh(new THREE.BoxGeometry(0.1,0.07,0.2),new THREE.MeshStandardMaterial({color:0x111111,roughness:0.8,metalness:0.1}));
foot.position.set(-0.14,-0.18,0.05);
foot.castShadow = True;
scene.add(foot);"""),

  ({"action":"add_cylinder","radiusTop":0.075,"radiusBottom":0.085,"height":0.44,"position":[0.14,0.34,0],"color":"0x1A1A2E","label":"leg_right"},
"""const thigh2 = new THREE.Mesh(new THREE.CylinderGeometry(0.075,0.085,0.44,12),mat_pants);
thigh2.position.set(0.14,0.34,0);
thigh2.castShadow = True;
scene.add(thigh2);
const shin2 = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.07,0.44,12),mat_pants);
shin2.position.set(0.14,0.03,0.02);
shin2.castShadow = True;
scene.add(shin2);
const foot2 = new THREE.Mesh(new THREE.BoxGeometry(0.1,0.07,0.2),new THREE.MeshStandardMaterial({color:0x111111,roughness:0.8,metalness:0.1}));
foot2.position.set(0.14,-0.18,0.05);
foot2.castShadow = True;
scene.add(foot2);"""),
],

# ── TREE ──────────────────────────────────────────────────────
[
  ({"action":"add_cylinder","radiusTop":0.08,"radiusBottom":0.18,"height":2.0,"position":[0,1.0,0],"color":"0x6B3A2A","label":"trunk"},
"""const mat_bark = new THREE.MeshStandardMaterial({color:0x6B3A2A,roughness:0.95,metalness:0});
const mesh_trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.18,2.0,12),mat_bark);
mesh_trunk.position.set(0,1.0,0);
mesh_trunk.castShadow = True;
mesh_trunk.receiveShadow = True;
scene.add(mesh_trunk);"""),

  ({"action":"add_cone","radius":1.4,"height":1.6,"position":[0,2.8,0],"color":"0x228B22","label":"foliage_low"},
"""const mat_leaf = new THREE.MeshStandardMaterial({color:0x228B22,roughness:0.9,metalness:0,side:THREE.FrontSide});
const mesh_fol1 = new THREE.Mesh(new THREE.ConeGeometry(1.4,1.6,16),mat_leaf);
mesh_fol1.position.set(0,2.8,0);
mesh_fol1.castShadow = True;
scene.add(mesh_fol1);"""),

  ({"action":"add_cone","radius":1.05,"height":1.4,"position":[0,3.8,0],"color":"0x2E8B22","label":"foliage_mid"},
"""const mesh_fol2 = new THREE.Mesh(new THREE.ConeGeometry(1.05,1.4,16),
  new THREE.MeshStandardMaterial({color:0x2E8B22,roughness:0.9,metalness:0}));
mesh_fol2.position.set(0,3.8,0);
mesh_fol2.castShadow = True;
scene.add(mesh_fol2);"""),

  ({"action":"add_cone","radius":0.65,"height":1.2,"position":[0,4.65,0],"color":"0x32CD32","label":"foliage_top"},
"""const mesh_fol3 = new THREE.Mesh(new THREE.ConeGeometry(0.65,1.2,16),
  new THREE.MeshStandardMaterial({color:0x32CD32,roughness:0.9,metalness:0}));
mesh_fol3.position.set(0,4.65,0);
mesh_fol3.castShadow = True;
scene.add(mesh_fol3);"""),
],

]

# ─────────────────────────────────────────────────────────────
# ASSEMBLE DATASET
# ─────────────────────────────────────────────────────────────

GENERATORS = [
    (gen_box,              3000),
    (gen_sphere,           2500),
    (gen_cylinder,         2000),
    (gen_cone,             1000),
    (gen_torus,            1000),
    (gen_torus_knot,        500),
    (gen_plane,            1000),
    (gen_lathe,            1000),
    (gen_extrude,          1500),
    (gen_tube,             1000),
    (gen_buffer_geometry,  1000),
    (gen_instanced,         800),
    (gen_terrain,           800),
    (gen_metal_material,   2000),
    (gen_glass_material,   1500),
    (gen_skin_material,     500),
    (gen_fabric_material,   500),
    (gen_emissive_material, 800),
    (gen_subsurface_material,400),
    (gen_clearcoat_material, 500),
    (gen_ambient_light,    1000),
    (gen_directional_light,1500),
    (gen_point_light,      1500),
    (gen_spot_light,       1000),
    (gen_area_light,        800),
    (gen_hemisphere_light,  500),
    (gen_hdri_env,          500),
    (gen_noise_shader,     1500),
    (gen_fresnel_shader,   1000),
    (gen_hologram_shader,   500),
    (gen_water_shader,      500),
    (gen_fire_shader,       500),
    (gen_bloom,            1500),
    (gen_ssao,              500),
    (gen_dof,               500),
    (gen_tone_mapping,     1000),
    (gen_fog,               800),
    (gen_ssr,               300),
    (gen_keyframe_anim,    1500),
    (gen_morph_anim,        500),
    (gen_physics_anim,      500),
    (gen_orbit_anim,        500),
    (gen_particle_system,   800),
    (gen_lod,               500),
    (gen_env_map,           500),
    (gen_reflection_probe,  400),
    (gen_shadow_setup,      500),
    (gen_sky_shader,        500),
]

def main():
    samples = []

    # single-tick samples
    print("Generating single-tick samples...")
    for gen_fn, count in GENERATORS:
        added = 0
        attempts = 0
        while added < count and attempts < count * 3:
            attempts += 1
            try:
                inp, out = gen_fn()
                samples.append({
                    "conversations": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": json.dumps(inp)},
                        {"role": "assistant", "content": out}
                    ]
                })
                added += 1
            except Exception as e:
                pass
        print(f"  {gen_fn.__name__}: {added} samples")

    # chain sequences — each chain repeated many times
    print("Generating chain sequences...")
    chains_per_template = 2500
    for chain in CHAINS:
        for _ in range(chains_per_template):
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            for inp, out in chain:
                history.append({"role": "user",      "content": json.dumps(inp)})
                history.append({"role": "assistant", "content": out})
            samples.append({"conversations": history})
    print(f"  Chain sequences: {len(CHAINS) * chains_per_template} samples")

    random.shuffle(samples)

    # split 90/5/5
    n = len(samples)
    train_end = int(n * 0.90)
    val_end   = int(n * 0.95)

    splits = {
        "train": samples[:train_end],
        "val":   samples[train_end:val_end],
        "test":  samples[val_end:]
    }

    for split_name, split_data in splits.items():
        path = f"morphix_{split_name}.jsonl"
        with open(path, "w") as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data):,} samples → {path}")

    print(f"\nTotal: {n:,} samples")
    print("Done. Run: python train.py")

if __name__ == "__main__":
    main()