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

function __installGradient(scene, top, mid, bot, id) {
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
  scene.add(__tag(sky, id || 'sky'));
}

__installGradient(scene, '#2e4a68', '#8fa9c4', '#141c28', 'sky');
scene.fog = new THREE.FogExp2(0x8FA9C4, 0.02);
const camera = new THREE.PerspectiveCamera(55, __W__ / __H__, 0.1, 400);
camera.position.set(14, 7, 18);
camera.lookAt(0, 2, 0);
const __light_sun = new THREE.DirectionalLight(0xFFF2E0, 1.6);
__light_sun.position.set(6, 10, 7);
__light_sun.castShadow = true;
__light_sun.shadow.mapSize.set(1024, 1024);
__light_sun.shadow.camera.left = -30;
__light_sun.shadow.camera.right = 30;
__light_sun.shadow.camera.top = 30;
__light_sun.shadow.camera.bottom = -30;
scene.add(__light_sun);
const __light_sky = new THREE.HemisphereLight(0xBDD7F2, 0x334455, 0.7);
scene.add(__light_sky);
const __light_mountains_sun = new THREE.DirectionalLight(0xFFF2E0, 1.6);
__light_mountains_sun.position.set(6, 10, 7);
scene.add(__light_mountains_sun);
const __light_mountains_sky = new THREE.HemisphereLight(0xBDD7F2, 0x334455, 0.7);
scene.add(__light_mountains_sky);
const __n_ground = new THREE.Mesh(new THREE.CircleGeometry(32, 48), new THREE.MeshStandardMaterial({ color: 0x3D4A33, roughness: 1.0, metalness: 0.0 }));
__n_ground.position.set(0, 0, 0);
__n_ground.rotation.set(-1.5707963, 0, 0);
__n_ground.scale.set(1, 1, 1);
__n_ground.receiveShadow = true;
scene.add(__tag(__n_ground, 'ground'));
const __n_trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.09, 0.14, 1.6, 7), new THREE.MeshStandardMaterial({ color: 0x6B4A2F, roughness: 0.95, metalness: 0.0 }), 2000);
{ const __r = __rng(2371);
  const __m = new THREE.Matrix4();
  const __p = new THREE.Vector3();
  const __q = new THREE.Quaternion();
  const __s = new THREE.Vector3();
  const __e = new THREE.Euler();
  let __placed = 0, __guard = 0;
  while (__placed < 2000 && __guard++ < 2000 * 40) {
    const __a = __r() * Math.PI * 2;
    const __rad = Math.sqrt(__r()) * 26;
    __p.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);
    if (Math.hypot(__p.x, __p.z) < 4.0) continue;
    __p.y = 0.8;
    __e.set(0, __r() * Math.PI * 2, 0);
    __q.setFromEuler(__e);
    const __sj = 1 + (__r() * 2 - 1) * 0.35;
    __s.set(__sj, __sj, __sj);
    __m.compose(__p, __q, __s);
    __n_trunks.setMatrixAt(__placed, __m);
    __placed++;
  }
  __n_trunks.instanceMatrix.needsUpdate = true;
  __n_trunks.count = __placed;
}
__n_trunks.castShadow = true;
scene.add(__tag(__n_trunks, 'trunks'));
const __n_canopies = new THREE.InstancedMesh(new THREE.ConeGeometry(1.0, 2.6, 8), new THREE.MeshStandardMaterial({ color: 0x2F5B2F, roughness: 0.9, metalness: 0.0 }), 2000);
{ const __r = __rng(3895);
  const __m = new THREE.Matrix4();
  const __p = new THREE.Vector3();
  const __q = new THREE.Quaternion();
  const __s = new THREE.Vector3();
  const __e = new THREE.Euler();
  let __placed = 0, __guard = 0;
  while (__placed < 2000 && __guard++ < 2000 * 40) {
    const __a = __r() * Math.PI * 2;
    const __rad = Math.sqrt(__r()) * 26;
    __p.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);
    if (Math.hypot(__p.x, __p.z) < 4.0) continue;
    __p.y = 2.6;
    __e.set(0, __r() * Math.PI * 2, 0);
    __q.setFromEuler(__e);
    const __sj = 1 + (__r() * 2 - 1) * 0.35;
    __s.set(__sj, __sj, __sj);
    __m.compose(__p, __q, __s);
    __n_canopies.setMatrixAt(__placed, __m);
    __placed++;
  }
  __n_canopies.instanceMatrix.needsUpdate = true;
  __n_canopies.count = __placed;
}
__n_canopies.castShadow = true;
scene.add(__tag(__n_canopies, 'canopies'));
function __terrainH(x, z) { return __fbm(x * 0.035 + 7.3, z * 0.035 + 2.9, 11, 4) * 9; }
const __n_mountains_terrain = (function () {
  const __g = new THREE.PlaneGeometry(90, 90, 96, 96);
  __g.rotateX(-Math.PI / 2);
  const __pos = __g.attributes.position;
  for (let i = 0; i < __pos.count; i++) {
    __pos.setY(i, __terrainH(__pos.getX(i), __pos.getZ(i)) - 9 * 0.35);
  }
  __g.computeVertexNormals();
  return new THREE.Mesh(__g, new THREE.MeshStandardMaterial({ color: 0x5A6B4A, roughness: 0.95, metalness: 0.0 }));
})();
__n_mountains_terrain.position.set(0, 0, 0);
__n_mountains_terrain.rotation.set(0, 0, 0);
__n_mountains_terrain.scale.set(1, 1, 1);
__n_mountains_terrain.receiveShadow = true;
scene.add(__tag(__n_mountains_terrain, 'mountains_terrain'));
__n_mountains_terrain.position.set(0, 0, 0);
__n_mountains_terrain.rotation.set(0, 0, 0);
__n_mountains_terrain.scale.set(1, 1, 1);
__n_mountains_terrain.receiveShadow = true;
scene.add(__tag(__n_mountains_terrain, 'mountains_terrain'));
__installGradient(scene, '#27435f', '#9db8d6', '#1a2433', 'mountains_sky');
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
