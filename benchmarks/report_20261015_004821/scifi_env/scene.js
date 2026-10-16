import * as THREE from "three";
function __rng(seed) {
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

const __anims = [];
function __tag(o, id) { o.userData.morphix_id = id; return o; }
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(1);
renderer.setSize(__W__, __H__);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070C);
scene.fog = new THREE.FogExp2(0x0A1220, 0.03);
__bakeEnvironment(scene, renderer);
const camera = new THREE.PerspectiveCamera(60, __W__ / __H__, 0.1, 300);
camera.position.set(0, 2.2, 12);
camera.lookAt(0, 2.2, 0);
const __light_sky = new THREE.HemisphereLight(0x3A5A8C, 0x05070C, 0.5);
scene.add(__light_sky);
const __light_key = new THREE.SpotLight(0xBFD9FF, 4.0, 40, 0.6, 0.5);
__light_key.position.set(0, 9, 4);
scene.add(__light_key);
const __light_strip_light_0 = new THREE.PointLight(0x33E0FF, 1.2, 14, 2);
__light_strip_light_0.position.set(-5.5, 3.4, -8);
scene.add(__light_strip_light_0);
const __light_strip_light_1 = new THREE.PointLight(0x33E0FF, 1.2, 14, 2);
__light_strip_light_1.position.set(-5.5, 3.4, 0);
scene.add(__light_strip_light_1);
const __light_strip_light_2 = new THREE.PointLight(0x33E0FF, 1.2, 14, 2);
__light_strip_light_2.position.set(-5.5, 3.4, 8);
scene.add(__light_strip_light_2);
const __n_floor = new THREE.Mesh(new THREE.PlaneGeometry(16, 40, 1, 1), new THREE.MeshStandardMaterial({ color: 0x0C1119, roughness: 0.25, metalness: 0.8 }));
__n_floor.position.set(0, 0, 0);
__n_floor.rotation.set(-1.5707963, 0, 0);
__n_floor.scale.set(1, 1, 1);
__n_floor.receiveShadow = true;
scene.add(__tag(__n_floor, 'floor'));
const __n_arch_0 = new THREE.Mesh(new THREE.TorusGeometry(5.2, 0.35, 12, 40), new THREE.MeshStandardMaterial({ color: 0x1A2433, roughness: 0.5, metalness: 0.7 }));
__n_arch_0.position.set(0, 0.4, -8);
__n_arch_0.rotation.set(0, 0, 0);
__n_arch_0.scale.set(1, 1, 1);
scene.add(__tag(__n_arch_0, 'arch_0'));
const __n_strip_0 = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 6.0), new THREE.MeshBasicMaterial({ color: 0x33E0FF }));
__n_strip_0.position.set(-5.4, 3.4, -8);
__n_strip_0.rotation.set(0, 0, 0);
__n_strip_0.scale.set(1, 1, 1);
scene.add(__tag(__n_strip_0, 'strip_0'));
const __n_arch_1 = new THREE.Mesh(new THREE.TorusGeometry(5.2, 0.35, 12, 40), new THREE.MeshStandardMaterial({ color: 0x1A2433, roughness: 0.5, metalness: 0.7 }));
__n_arch_1.position.set(0, 0.4, 0);
__n_arch_1.rotation.set(0, 0, 0);
__n_arch_1.scale.set(1, 1, 1);
scene.add(__tag(__n_arch_1, 'arch_1'));
const __n_strip_1 = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 6.0), new THREE.MeshBasicMaterial({ color: 0x33E0FF }));
__n_strip_1.position.set(-5.4, 3.4, 0);
__n_strip_1.rotation.set(0, 0, 0);
__n_strip_1.scale.set(1, 1, 1);
scene.add(__tag(__n_strip_1, 'strip_1'));
const __n_arch_2 = new THREE.Mesh(new THREE.TorusGeometry(5.2, 0.35, 12, 40), new THREE.MeshStandardMaterial({ color: 0x1A2433, roughness: 0.5, metalness: 0.7 }));
__n_arch_2.position.set(0, 0.4, 8);
__n_arch_2.rotation.set(0, 0, 0);
__n_arch_2.scale.set(1, 1, 1);
scene.add(__tag(__n_arch_2, 'arch_2'));
const __n_strip_2 = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 6.0), new THREE.MeshBasicMaterial({ color: 0x33E0FF }));
__n_strip_2.position.set(-5.4, 3.4, 8);
__n_strip_2.rotation.set(0, 0, 0);
__n_strip_2.scale.set(1, 1, 1);
scene.add(__tag(__n_strip_2, 'strip_2'));
const __n_core = new THREE.Mesh(new THREE.SphereGeometry(1.1, 32, 32), new THREE.MeshStandardMaterial({ color: 0x66D9FF, roughness: 0.2, metalness: 0.1, emissiveIntensity: 1.2, emissive: 0x1A5A77 }));
__n_core.position.set(0, 2.4, -12);
__n_core.rotation.set(0, 0, 0);
__n_core.scale.set(1, 1, 1);
scene.add(__tag(__n_core, 'core'));
function __bakeEnvironment(scene, renderer) {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 32;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 32);
  grad.addColorStop(0, '#1e3a5c');
  grad.addColorStop(0.55, '#3a6a9e');
  grad.addColorStop(1, '#05070c');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromEquirectangular(tex).texture;
  tex.dispose(); pm.dispose();
}
__anims.push({ update: (function () { const __y0 = __n_core.position.y; return function (dt, t) { __n_core.position.y = __y0 + Math.sin(t * 0.9) * 0.25; }; })() });
__anims.push({ update: function (dt, t) { __light_strip_light_1.intensity = 1.2 + Math.sin(t * 6.0) * 0.5; } });
let __frames = 0;
window.__scene_error = null;
window.addEventListener('error', function (e) { window.__scene_error = String((e && e.message) || e); });
const __clock = new THREE.Clock();
function __tick() {
  const dt = Math.min(__clock.getDelta(), 0.05);
  const t = __clock.elapsedTime;
  for (const a of __anims) { a.update(dt, t); }
  renderer.render(scene, camera);
  if (++__frames >= 3) {
    let __inst = 0;
    scene.traverse(function (o) {
      if (o.isInstancedMesh) { __inst += o.count; }
      else if (o.isPoints && o.geometry && o.geometry.attributes.position) {
        __inst += o.geometry.attributes.position.count; }
    });
    window.__scene_stats = { frames: __frames, nodes: scene.children.length, drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles, instances: __inst };
    window.__scene_ready = true;
  } else { requestAnimationFrame(__tick); }
}
window.__introspectScene = function () {
  const out = {};
  const __box = new THREE.Box3();
  scene.updateMatrixWorld(true);
  scene.traverse(function (o) {
    const id = o.userData && o.userData.morphix_id;
    if (!id) return;
    if (o.isMesh || o.isInstancedMesh || o.isPoints) {
      __box.setFromObject(o, true);
      if (__box.isEmpty()) {
        out[id] = { present: true, empty: true, visible: !!o.visible };
      } else {
        out[id] = { present: true, empty: false, visible: !!o.visible,
          bbox: [__box.min.toArray(), __box.max.toArray()] };
      }
    } else {
      out[id] = out[id] || { present: true, empty: true, visible: !!o.visible, group: true };
    }
  });
  return out;
};
__tick();
