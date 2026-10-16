# morphix_image_display.py
# Presentation render of an image reconstruction: the SAME validated
# semantic IR the printer slices, drawn as a clean, realistic Three.js
# scene instead of as deposited beads.
#
#   * smooth geometry — profiles become bevelled ExtrudeGeometry with
#     per-vertex tumblehome (depth tapers with height), revolved parts use
#     spline-smoothed LatheGeometry, boxes are rounded, wheels are real
#     tire + rim + spokes assemblies on a horizontal axle;
#   * physical materials per role — clear-coated paint, tinted glass,
#     rubber, brushed metal rims, emissive lamps, wood;
#   * studio lighting — a procedural "room" environment prefiltered with
#     PMREMGenerator (core r152, no external HDRI), ACES filmic tone
#     mapping, sRGB output, soft PCF shadows on a shadow-catcher floor;
#   * OrbitControls in the UI page (CDN addon, same r152 revision); the
#     verification page omits addons and uses the fitted photo camera so
#     its silhouette can be measured against the reference.
#
# Nothing here changes geometry decisions: dimensions, positions and
# colors come from the IR. Environment context (platform, floor) is
# display-only and only drawn in environment-preserving mode.

import json

THREE_CDN = "https://unpkg.com/three@0.152.2/build/three.module.js"
ADDONS_CDN = "https://unpkg.com/three@0.152.2/examples/jsm/"

DISPLAY_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#0e1116}
canvas{display:block}#hud{position:fixed;left:10px;bottom:8px;font:12px system-ui;
color:#9aa4b2;pointer-events:none}</style>
<script type="importmap">{"imports":{"three":"__THREE__","three/addons/":"__ADDONS__"}}</script>
</head><body><div id="hud"></div>
<script>
// surface runtime failures on the page (sandboxed iframes hide consoles)
// Browser extensions (e.g. MetaMask) inject scripts into every page; their
// failures are not render errors of this scene and are ignored.
const __foreign = (src, stack) => /^(chrome|moz|safari)-extension:/.test(src || '')
  || /(chrome|moz|safari)-extension:\/\//.test(stack || '') || /metamask/i.test(stack || '');
window.addEventListener('error', (e) => {
  if (__foreign(e.filename, e.error && e.error.stack) || /metamask/i.test(e.message || '')) return;
  document.getElementById('hud').textContent = 'render error: ' + e.message; window.__scene_error = e.message;
});
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason, msg = (r && r.message) || String(r);
  if (__foreign('', r && r.stack) || /metamask/i.test(msg)) return;
  document.getElementById('hud').textContent = 'render error: ' + msg; window.__scene_error = msg;
});
</script>
<script type="module">
import * as THREE from 'three';
const D = __DATA__;
const VERIFY = D.verify;
const W = VERIFY ? D.width : window.innerWidth, H = VERIFY ? D.height : window.innerHeight;
const renderer = new THREE.WebGLRenderer({antialias: !VERIFY, preserveDrawingBuffer: true});
renderer.setPixelRatio(VERIFY ? 1 : Math.min(window.devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();

// ---- studio environment (procedural room -> PMREM), core only ----------
function studioRoom() {
  const room = new THREE.Scene();
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({color: 0x7a7f88, side: THREE.BackSide}));
  box.scale.set(30, 16, 30); box.position.y = 7; room.add(box);
  const panel = (x, y, z, sx, sy, sz, k) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({color: new THREE.Color(k, k, k)}));
    m.position.set(x, y, z); m.scale.set(sx, sy, sz); room.add(m);
  };
  panel(0, 14.5, 0, 16, 0.2, 7, 7.0);      // softbox overhead
  panel(-14.5, 6, 0, 0.2, 4, 16, 3.0);     // strip boxes on all walls
  panel(14.5, 6, 0, 0.2, 4, 16, 3.0);
  panel(0, 6, -14.5, 16, 4, 0.2, 2.4);
  panel(0, 6, 14.5, 16, 4, 0.2, 2.4);
  panel(9, 2.5, 9, 4, 3, 0.2, 1.5);        // low kickers for lower panels
  panel(-9, 2.5, -9, 4, 3, 0.2, 1.5);
  return room;
}
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(studioRoom(), 0.035).texture;

if (VERIFY) {
  scene.background = new THREE.Color(parseInt(D.bg, 16));
} else {
  const c = document.createElement('canvas'); c.width = 4; c.height = 256;
  const g = c.getContext('2d'); const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, D.sky[0]); grd.addColorStop(1, D.sky[1]);
  g.fillStyle = grd; g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  scene.background = tex;
}

// ---- lights ----------------------------------------------------------
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(D.center[0] + D.radius * 1.2, D.radius * 3.0, D.center[2] + D.radius * 1.6);
key.target.position.set(...D.center);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
const s = D.radius * 1.6;
Object.assign(key.shadow.camera, {left: -s, right: s, top: s, bottom: -s, near: 0.1, far: D.radius * 10});
key.shadow.bias = -0.0004; key.shadow.radius = 4;
scene.add(key, key.target);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, 0.35));
const rim = new THREE.DirectionalLight(0xbfd4ff, 1.1);     // separation rim
rim.position.set(D.center[0] - D.radius * 2, D.radius * 1.5, D.center[2] - D.radius * 2);
scene.add(rim);

// ---- materials by role ------------------------------------------------
const hex = (h) => new THREE.Color(parseInt(h, 16));
function material(c) {
  const col = hex(c.color);
  if (c.role === 'light') return new THREE.MeshStandardMaterial({color: col, emissive: col, emissiveIntensity: 2.2, roughness: 0.2});
  if (c.material === 'glass') return new THREE.MeshPhysicalMaterial({color: col.clone().lerp(new THREE.Color(0x0a0e13), 0.8), metalness: 0.1, roughness: 0.04, clearcoat: 1, clearcoatRoughness: 0.02, envMapIntensity: 1.6, transparent: true, opacity: 0.92});
  if (c.material === 'tire') return new THREE.MeshStandardMaterial({color: 0x161616, roughness: 0.92, metalness: 0.0});
  if (c.material === 'wood') return new THREE.MeshStandardMaterial({color: col, roughness: 0.62, metalness: 0.0});
  if (D.family === 'vehicle' && c.material === 'body')
    return new THREE.MeshPhysicalMaterial({color: col, metalness: 0.55, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 1.3});
  return new THREE.MeshPhysicalMaterial({color: col, metalness: 0.05, roughness: 0.45, clearcoat: 0.35, clearcoatRoughness: 0.2});
}

// ---- geometry helpers -------------------------------------------------
function chaikin(pts, it) {          // corner rounding for closed polygons
  let p = pts;
  for (let k = 0; k < it; k++) {
    const q = [];
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length];
      q.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      q.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    p = q;
  }
  return p;
}
function extruded(c) {
  const pts0 = c.profile;
  const ys = pts0.map(q => q[1]); const xs = pts0.map(q => q[0]);
  const hh = (Math.max(...ys) - Math.min(...ys)) / 2 || 1e-9;
  const small = Math.min(2 * hh, Math.max(...xs) - Math.min(...xs));
  // corner rounding: full for designed shapes, light on greenhouse panels
  // (keeps straight glass edges), none on traced photo outlines
  const it = c.id === 'silhouette_body' ? 0
    : (c.material === 'glass' || c.role === 'roof') ? 1
    : (c.role === 'wheel' ? 0 : 2);
  const pts = chaikin(pts0, it);
  const shape = new THREE.Shape(pts.map(q => new THREE.Vector2(q[0], q[1])));
  const d0 = c.depth, d1 = (c.depth_top ?? c.depth);
  const flush = c.role === 'roof' || c.material === 'glass';  // greenhouse: tight joins
  const bev = flush ? 0.012 : Math.min(0.07, small * 0.12, Math.min(d0, d1) * 0.12);
  const g = new THREE.ExtrudeGeometry(shape, {depth: 1, bevelEnabled: true,
    bevelThickness: bev, bevelSize: flush ? 0 : bev * 0.6, bevelSegments: flush ? 1 : 5,
    curveSegments: 24, steps: 1});
  g.translate(0, 0, -0.5);
  const p = g.attributes.position;
  // vehicle body: rounded shoulders in the cross-section (presentation of
  // the validated slab; the silhouette check measures the result)
  const shoulder = () => 1;   // linear taper only: caps stay planar
  for (let i = 0; i < p.count; i++) {      // tumblehome: depth follows height
    const y = p.getY(i), t = Math.min(1, Math.max(0, (y + hh) / (2 * hh)));
    const dep = (d0 + (d1 - d0) * t) * shoulder(t);
    const z = p.getZ(i), zs = Math.sign(z) * Math.max(0, Math.abs(z) - 0.5);
    p.setZ(i, Math.sign(z) * Math.min(Math.abs(z), 0.5) * dep + zs);
  }
  g.computeVertexNormals();
  if (c.axis === 'x') g.rotateY(-Math.PI / 2);
  return g;
}
function roundedBox(w, h, d) {
  const r = Math.min(w, h, d) * 0.12;
  const s = new THREE.Shape(); const x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  const b = Math.min(r, d * 0.2);
  const g = new THREE.ExtrudeGeometry(s, {depth: Math.max(1e-3, d - 2 * b), bevelEnabled: true,
    bevelThickness: b, bevelSize: b * 0.5, bevelSegments: 3, curveSegments: 8});
  g.translate(0, 0, -(d - 2 * b) / 2);
  g.computeVertexNormals();
  return g;
}
function wheel(c) {                   // tire + rim + spokes, axle along z
  const xs = c.profile.map(q => q[0]);
  const r = (Math.max(...xs) - Math.min(...xs)) / 2, tw = c.depth;
  const grp = new THREE.Group();
  const prof = [];
  const n = 12, rr = tw * 0.32;
  for (let i = 0; i <= n; i++) {     // rounded tire shoulder cross-section
    const a = -Math.PI / 2 + Math.PI * i / n;
    prof.push(new THREE.Vector2(r - rr + Math.cos(a) * rr, Math.sin(a) * (tw / 2)));
  }
  prof.unshift(new THREE.Vector2(r * 0.66, -tw / 2)); prof.push(new THREE.Vector2(r * 0.66, tw / 2));
  const tire = new THREE.Mesh(new THREE.LatheGeometry(prof, 64), material({...c, material: 'tire'}));
  tire.rotation.x = Math.PI / 2; grp.add(tire);
  const rimMat = new THREE.MeshPhysicalMaterial({color: 0x9aa0a6, metalness: 1.0, roughness: 0.28, clearcoat: 0.5});
  const outer = Math.sign(c.position[2]) || 1;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.66, r * 0.66, tw * 0.7, 48, 1, true), rimMat);
  rim.rotation.x = Math.PI / 2; grp.add(rim);
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.62, tw * 0.08, 48),
    new THREE.MeshStandardMaterial({color: 0x15171a, roughness: 0.6, metalness: 0.4}));
  disc.rotation.x = Math.PI / 2; disc.position.z = outer * tw * 0.2; grp.add(disc);
  for (let k = 0; k < 5; k++) {
    const sp = new THREE.Mesh(new THREE.BoxGeometry(r * 1.18, r * 0.12, tw * 0.1), rimMat);
    sp.rotation.z = k * Math.PI * 2 / 5; sp.position.z = outer * tw * 0.3; grp.add(sp);
  }
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.16, r * 0.16, tw * 0.2, 24), rimMat);
  hub.rotation.x = Math.PI / 2; hub.position.z = outer * tw * 0.32; grp.add(hub);
  return grp;
}
function lathe(c) {
  const v = c.profile.map(q => new THREE.Vector2(q[0], q[1]));
  const pts = v.length >= 4 ? new THREE.SplineCurve(v).getPoints(v.length * 6) : v;
  if (pts[0].x > 0) pts.unshift(new THREE.Vector2(0, pts[0].y));
  const last = pts[pts.length - 1];
  if (last.x > 0) pts.push(new THREE.Vector2(0, last.y));
  return new THREE.LatheGeometry(pts, 96);
}
function inflated(c) {
  // smooth indexed mesh from the half-thickness grid: textured front
  // (photo projected through the measured uv frame), plain back + rim
  const g = c.grid, cell = g.cell, R = g.rows, C = g.cols;
  const hw = C * cell / 2, hh = R * cell / 2, N = R * C;
  const pos = new Float32Array(N * 2 * 3), uv = new Float32Array(N * 2 * 2);
  for (let j = 0; j < R; j++) for (let i = 0; i < C; i++) {
    const k = j * C + i, x = -hw + (i + 0.5) * cell, y = -hh + (j + 0.5) * cell;
    const d = g.half[j][i];
    pos.set([x, y, d], k * 3); pos.set([x, y, -d], (k + N) * 3);
    const gu = g.uv || {u0: 0, du: 0, v0: 0, dv: 0};   // no photo frame: plain colour
    const u = gu.u0 + i * gu.du, v = gu.v0 + j * gu.dv;
    uv.set([u, v], k * 2); uv.set([u, v], (k + N) * 2);
  }
  // triangles over vertices inside the silhouette (3-of-4 corner cells give
  // diagonal edges instead of stair steps); the outline is the set of
  // triangle edges used once, closed by a rim between front and back
  const inV = (k) => g.half[Math.floor(k / C)][k % C] > 0;
  const tris = [];
  for (let j = 0; j < R - 1; j++) for (let i = 0; i < C - 1; i++) {
    const a = j * C + i, b = a + 1, cc = a + C + 1, d = a + C;
    const A = inV(a), B = inV(b), Cc = inV(cc), Dd = inV(d);
    if (A && B && Cc && Dd) { tris.push([a, b, cc], [a, cc, d]); continue; }
    if (A && B && Cc) tris.push([a, b, cc]);
    else if (A && Cc && Dd) tris.push([a, cc, d]);
    else if (A && B && Dd) tris.push([a, b, d]);
    else if (B && Cc && Dd) tris.push([b, cc, d]);
  }
  const ecount = new Map();
  const key = (p, q) => p < q ? p + ',' + q : q + ',' + p;
  for (const t of tris) for (let e = 0; e < 3; e++) {
    const p = t[e], q = t[(e + 1) % 3], kk = key(p, q);
    const v = ecount.get(kk); if (v) v.n++; else ecount.set(kk, {n: 1, p, q});
  }
  const front = [], rest = [];
  for (const [a, b, c2] of tris) { front.push(a, b, c2); rest.push(a + N, c2 + N, b + N); }
  const nbr = new Map();
  for (const {n, p, q} of ecount.values()) {
    if (n !== 1) continue;
    rest.push(p, p + N, q + N, p, q + N, q);
    for (const [u, w] of [[p, q], [q, p]]) { if (!nbr.has(u)) nbr.set(u, []); nbr.get(u).push(w); }
  }
  // smooth the outline (x/y only) so the rim follows the contour, not the grid
  for (let it = 0; it < 4; it++) {
    const upd = [];
    for (const [v, ns] of nbr) {
      let sx = pos[v * 3], sy = pos[v * 3 + 1];
      for (const w of ns) { sx += pos[w * 3]; sy += pos[w * 3 + 1]; }
      upd.push([v, sx / (ns.length + 1), sy / (ns.length + 1)]);
    }
    for (const [v, x, y] of upd) {
      pos[v * 3] = x; pos[v * 3 + 1] = y; pos[(v + N) * 3] = x; pos[(v + N) * 3 + 1] = y;
    }
  }
  // front and back meet exactly on the outline (no stretched seam strip):
  // the silhouette edge becomes a rounded rim instead of a thin wall
  for (const v of nbr.keys()) { pos[v * 3 + 2] = 0; pos[(v + N) * 3 + 2] = 0; }
  const mk = (index) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos.slice(), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv.slice(), 2));
    geo.setIndex(index); geo.computeVertexNormals();
    // how directly each vertex faces the photo axis (|n.z|): the photo is
    // only trusted where it was actually seen
    const n = geo.attributes.normal, f = new Float32Array(n.count);
    for (let i = 0; i < n.count; i++) f[i] = Math.abs(n.getZ(i));
    geo.setAttribute('facing', new THREE.BufferAttribute(f, 1));
    return geo;
  };
  const grp = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({color: hex(c.color), roughness: 0.7, metalness: 0.0, side: THREE.DoubleSide});
  const faceMat = new THREE.MeshStandardMaterial({roughness: 0.62, metalness: 0.0, color: 0xffffff, side: THREE.DoubleSide});
  // the unseen far side is assumed to mirror the photographed side: it
  // carries the same photo (same uv frame), so the object is continuous
  grp.add(new THREE.Mesh(mk(rest), D.texture ? faceMat : body));
  if (D.texture) {
    if (c.tint) faceMat.color = hex(c.tint);   // paint tints the photo
    const tex = new THREE.TextureLoader().load(D.texture, () => { window.__tex_ready = true; }, undefined,
      () => { faceMat.map = null; faceMat.color = hex(c.color); faceMat.needsUpdate = true; window.__tex_ready = true;
              document.getElementById('hud').textContent += ' — photo texture unavailable'; });
    tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; faceMat.map = tex;
    // steep sides (seen edge-on in the photo) would smear one pixel row:
    // blend from the photo to the subject's own colour as the surface turns
    // away from the photo axis
    faceMat.onBeforeCompile = (sh) => {
      sh.uniforms.uBody = {value: hex(c.color)};
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nattribute float facing;\nvarying float vFacing;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvFacing = facing;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vFacing;\nuniform vec3 uBody;')
        .replace('#include <map_fragment>',
                 '#include <map_fragment>\ndiffuseColor.rgb = mix(uBody, diffuseColor.rgb, smoothstep(0.2, 0.62, vFacing));');
    };
  } else { faceMat.color = hex(c.color); window.__tex_ready = true; }
  grp.add(new THREE.Mesh(mk(front), faceMat));
  return grp;
}
function build(c) {
  let obj;
  if (c.primitive === 'inflate') obj = inflated(c);
  else if (c.role === 'wheel' && c.primitive === 'profile') obj = wheel(c);
  else {
    let g;
    if (c.primitive === 'profile') g = extruded(c);
    else if (c.primitive === 'lathe') g = lathe(c);
    else if (c.primitive === 'box') g = roundedBox(...c.dims);
    else if (c.primitive === 'sphere') g = new THREE.SphereGeometry(c.dims[0] / 2, 64, 32);
    else { const t = c.taper || [c.dims[0] / 2, c.dims[0] / 2];
           g = new THREE.CylinderGeometry(t[0], t[1], c.dims[1], 48); }
    obj = new THREE.Mesh(g, material(c));
  }
  obj.position.set(...c.position);
  obj.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return obj;
}
const model = new THREE.Group();
for (const c of D.components) model.add(build(c));
scene.add(model);

// ---- vehicle presentation details (cosmetic, derived from fitted parts,
//      NOT part of the validated IR and never printed) -------------------
function bboxOf(c) {
  const [x, y, z] = c.position, [w, h, d] = c.dims;
  return {x0: x - w / 2, x1: x + w / 2, y0: y - h / 2, y1: y + h / 2, z0: z - d / 2, z1: z + d / 2, w, h, d};
}
if (D.family === 'vehicle' && D.details) {
  const body = D.components.find(c => c.role === 'body' && c.material === 'body');
  const glass = D.components.find(c => c.material === 'glass');
  const lamps = D.components.filter(c => c.role === 'light');
  if (body) {
    const b = bboxOf(body);
    const trim = new THREE.MeshPhysicalMaterial({color: 0x0c0d0f, roughness: 0.45, metalness: 0.2, clearcoat: 0.6});
    const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; model.add(m); return m; };
    const ly = lamps.length ? lamps[0].position[1] : b.y0 + b.h * 0.7;
    // grille between the headlights, bumper lips front and rear
    add(roundedBox(0.08, b.h * 0.26, b.d * 0.46), trim, b.x1 + 0.025, ly - b.h * 0.24, 0);
    add(roundedBox(0.14, 0.12, b.d * 0.92), trim, b.x1 - 0.02, b.y0 + 0.07, 0);
    add(roundedBox(0.14, 0.12, b.d * 0.92), trim, b.x0 + 0.02, b.y0 + 0.07, 0);
    // tail lamps mirror the measured headlight height
    const red = new THREE.MeshStandardMaterial({color: 0x8a0d10, emissive: 0xc0141a, emissiveIntensity: 1.4, roughness: 0.25});
    for (const zs of [1, -1]) add(roundedBox(0.06, 0.13, b.d * 0.26), red, b.x0 - 0.035, ly, zs * b.d * 0.31);
  }
  if (glass) {          // B-pillar: painted strip splitting the side glass
    const g = bboxOf(glass);
    const paint = material({...(body || glass), material: 'body', role: 'body', color: (body || glass).color});
    const pil = new THREE.Mesh(new THREE.BoxGeometry(g.w * 0.07, g.h * 0.98, (glass.depth + (glass.depth_top ?? glass.depth)) / 2 + 0.03), paint);
    pil.position.set(g.x0 + g.w * 0.46, g.y0 + g.h / 2, glass.position[2]);
    model.add(pil);
  }
}

// ---- ground: shadow catcher (+ display-only environment context) ------
if (!VERIFY) {
  scene.fog = new THREE.Fog(new THREE.Color(D.sky[0]), D.radius * 5, D.radius * 14);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(D.radius * 30, 96),
    new THREE.MeshStandardMaterial({color: parseInt(D.floor, 16), roughness: 0.92, metalness: 0.0, envMapIntensity: 0.25}));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  // soft contact shadow (ambient occlusion blob) under the subject
  const cv = document.createElement('canvas'); cv.width = cv.height = 128;
  const cg = cv.getContext('2d'); const rg = cg.createRadialGradient(64, 64, 4, 64, 64, 64);
  rg.addColorStop(0, 'rgba(0,0,0,0.75)'); rg.addColorStop(1, 'rgba(0,0,0,0)');
  cg.fillStyle = rg; cg.fillRect(0, 0, 128, 128);
  const blob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({
    map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false}));
  blob.rotation.x = -Math.PI / 2; blob.position.set(D.center[0], 0.004, D.center[2]);
  blob.scale.set(D.extent[0] * 1.35, D.extent[2] * 1.6, 1); scene.add(blob);
  for (const e of D.env) {       // DISPLAY-ONLY context: never printed
    const m = new THREE.Mesh(new THREE.CylinderGeometry(e.r, e.r, 0.04, 96),
      new THREE.MeshStandardMaterial({color: parseInt(e.color, 16), roughness: 0.7}));
    m.position.set(e.x, 0.02, e.z); m.receiveShadow = true; scene.add(m);
  }
} else {
  const sc = new THREE.Mesh(new THREE.PlaneGeometry(D.radius * 12, D.radius * 12),
    new THREE.ShadowMaterial({opacity: 0.0}));
  sc.rotation.x = -Math.PI / 2; scene.add(sc);
}

// ---- camera -----------------------------------------------------------
const cam = new THREE.PerspectiveCamera(D.camera.fov, W / H, 0.05, 500);
cam.position.set(...D.camera.position);
cam.lookAt(...D.camera.target);
if (!VERIFY) {
  let ctl = {update() {}};
  if (!D.noControls) {
    const {OrbitControls} = await import('three/addons/controls/OrbitControls.js');
    ctl = new OrbitControls(cam, renderer.domElement);
    ctl.target.set(...D.camera.target); ctl.enableDamping = true;
    ctl.maxPolarAngle = Math.PI * 0.495; ctl.minDistance = D.radius * 1.2; ctl.maxDistance = D.radius * 8;
  }
  document.getElementById('hud').textContent = D.caption + ' — drag to orbit, scroll to zoom';
  window.addEventListener('resize', () => {
    cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  (function loop() { ctl.update(); renderer.render(scene, cam); requestAnimationFrame(loop); })();
  window.__done = true;
} else {
  // wait for the photo texture (if any) so the measured render is final
  const t0 = performance.now();
  (function wait() {
    if (window.__tex_ready || !D.components.some(c => c.primitive === 'inflate')
        || performance.now() - t0 > 8000) {
      renderer.render(scene, cam); window.__done = true;
    } else requestAnimationFrame(wait);
  })();
}
</script></body></html>
"""


def _ui_camera(comps, cam):
    """UI opening camera: the photo's side of the subject at a flattering
    three-quarter angle (a head-on or tail-on photo view hides length), a
    little above the subject. The verification page keeps the exact fitted
    photo camera."""
    import morphix_image_shapes as SH
    yaw = cam["yaw"]
    a = abs(((yaw + 180) % 360) - 180)
    if any(c.get("primitive") == "inflate" for c in comps):
        # inflated photo subject: turn ~28 deg off the photo axis so the
        # rounded volume reads, while the textured front stays in view
        return SH.camera_for(comps, yaw - 28.0, 12.0, margin=1.05)
    if a < 30 or a > 150:
        sgn = 1 if yaw >= 0 else -1
        yaw = (40 if a < 30 else 140) * sgn
    c = SH.camera_for(comps, yaw, max(14.0, cam["elevation"]), margin=1.0)
    return c


def display_data(ir, verify=False, width=640, height=480, bg="0xD8DDE2",
                 texture=None):
    import morphix_image_shapes as SH
    comps = ir["components"]
    lo, hi = SH.scene_bounds(comps)
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    radius = 0.5 * sum((b - a) ** 2 for a, b in zip(lo, hi)) ** 0.5
    # objects from text runs / edits may lack the image-only fields
    ir = dict(ir)
    ir.setdefault("environment", {"mode": "object-only", "elements": []})
    ir.setdefault("method", {"builder": "printed", "kind": "object"})
    if not ir.get("camera"):
        ir["camera"] = SH.camera_for(comps, 40.0, 18.0)
    env = []
    if ir["environment"]["mode"] == "preserve":
        for e in ir["environment"]["elements"]:
            if e.get("build") and e["kind"] == "platform":
                env.append({"r": round(max(hi[0] - lo[0], hi[2] - lo[2]) * 0.62,
                                       3), "color": e["color"],
                            "x": center[0], "z": center[2]})
    sky = ("#2a2f37", "#1b1e23")    # horizon == fog colour: no seam
    floor = "0x2a2f36"
    if ir["environment"]["mode"] == "preserve":
        els = {e["kind"]: e for e in ir["environment"]["elements"]}
        if els.get("backdrop"):
            c = els["backdrop"]["color"]
            sky = ("#" + c[2:], "#" + c[2:])
        if els.get("floor"):
            floor = els["floor"]["color"]
    sub = ir.get("subject") or {}
    caption = (f"{sub.get('label') or ir['category']} — "
               f"{ir['method']['builder']}/{ir['method']['kind']}"
               + (" — grille/bumpers/tail lamps/pillar are presentation "
                  "details, not printed" if sub.get("family") == "vehicle"
                  else ""))
    return {"verify": verify, "width": width, "height": height, "bg": bg,
            "sky": list(sky), "floor": floor, "env": env,
            "family": sub.get("family"),
            "camera": ir["camera"] if verify else _ui_camera(comps,
                                                             ir["camera"]),
            "center": center, "radius": radius, "caption": caption,
            "extent": [b - a for a, b in zip(lo, hi)],
            "texture": texture,
            "details": True,
            "components": [_present(c) for c in comps]}


def _present(c):
    out = {k: v for k, v in c.items()
           if k in ("id", "role", "primitive", "profile", "depth",
                    "depth_top", "axis", "dims", "position", "material",
                    "color", "taper", "grid", "texture", "tint")}
    if c.get("role") == "light":
        # lamps sit flush with the bevelled skin instead of inside it
        x = out["position"][0]
        out["position"] = [x + (0.07 if x >= 0 else -0.07)] +             list(out["position"][1:])
    return out


def display_html(ir, verify=False, three_url=THREE_CDN,
                 addons_url=ADDONS_CDN, **kw):
    data = display_data(ir, verify=verify, **kw)
    title = "Morphix model — " + str((ir.get("subject") or {}).get("label")
                                     or ir["category"])
    return (DISPLAY_TEMPLATE.replace("__TITLE__", title)
            .replace("__THREE__", three_url)
            .replace("__ADDONS__", addons_url)
            .replace("__DATA__", json.dumps(data)))
