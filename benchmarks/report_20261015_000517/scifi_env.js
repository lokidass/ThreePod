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

__installGradient(scene, '#3a4a63', '#232c40', '#111622');
const camera = new THREE.PerspectiveCamera(45, __W__ / __H__, 0.1, 500);
camera.position.set(5, 3.2, 7);
camera.lookAt(0, 0.8, 0);
const __light_sun = new THREE.DirectionalLight(0xFFF2E0, 1.6);
__light_sun.position.set(6, 10, 7);
__light_sun.castShadow = true;
__light_sun.shadow.mapSize.set(1024, 1024);
__light_sun.shadow.camera.left = -15;
__light_sun.shadow.camera.right = 15;
__light_sun.shadow.camera.top = 15;
__light_sun.shadow.camera.bottom = -15;
scene.add(__light_sun);
const __light_sky = new THREE.HemisphereLight(0xBDD7F2, 0x334455, 0.7);
scene.add(__light_sky);
const __light_rim = new THREE.DirectionalLight(0x88AAFF, 0.8);
__light_rim.position.set(-6, 4, -6);
scene.add(__light_rim);
const __n_floor = new THREE.Mesh(new THREE.CircleGeometry(14, 64), new THREE.MeshStandardMaterial({ color: 0x39414F, roughness: 0.9, metalness: 0.0 }));
__n_floor.position.set(0, 0, 0);
__n_floor.rotation.set(-1.5707963, 0, 0);
__n_floor.scale.set(1, 1, 1);
__n_floor.receiveShadow = true;
scene.add(__n_floor);
const __n_pedestal = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.3, 1.0, 48), new THREE.MeshStandardMaterial({ color: 0x2A3242, roughness: 0.35, metalness: 0.1 }));
__n_pedestal.position.set(0, 0.5, 0);
__n_pedestal.rotation.set(0, 0, 0);
__n_pedestal.scale.set(1, 1, 1);
__n_pedestal.castShadow = true;
__n_pedestal.receiveShadow = true;
scene.add(__n_pedestal);
const __n_hero = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.0, 1.0), new THREE.MeshStandardMaterial({ color: 0xE8A33D, roughness: 0.4, metalness: 0.2 }));
__n_hero.position.set(0, 1.7, 0);
__n_hero.rotation.set(0, 0.6, 0);
__n_hero.scale.set(1, 1, 1);
__n_hero.castShadow = true;
scene.add(__n_hero);
__anims.push({ update: function (dt, t) { __n_hero.rotation.y += dt * 0.4; } });
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
__tick();
