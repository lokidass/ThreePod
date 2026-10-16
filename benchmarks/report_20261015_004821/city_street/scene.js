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

__installGradient(scene, '#0e1626', '#2c3e58', '#05080e', 'sky');
scene.fog = new THREE.FogExp2(0x2C3E58, 0.016);
const camera = new THREE.PerspectiveCamera(55, __W__ / __H__, 0.1, 600);
camera.position.set(34, 22, 40);
camera.lookAt(0, 6, 0);
const __light_moon = new THREE.DirectionalLight(0x9FB8DD, 0.7);
__light_moon.position.set(-20, 30, 10);
scene.add(__light_moon);
const __light_sky = new THREE.HemisphereLight(0x2C3E58, 0x05080E, 0.5);
scene.add(__light_sky);
const __n_ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120, 1, 1), new THREE.MeshStandardMaterial({ color: 0x0A0E16, roughness: 0.9, metalness: 0.1 }));
__n_ground.position.set(0, 0, 0);
__n_ground.rotation.set(-1.5707963, 0, 0);
__n_ground.scale.set(1, 1, 1);
__n_ground.receiveShadow = true;
scene.add(__tag(__n_ground, 'ground'));
const __n_towers = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x1B2740, roughness: 0.6, metalness: 0.3, emissiveIntensity: 0.55, emissive: 0xFFC873 }), 64);
{ const __r = __rng(2370);
  const __m = new THREE.Matrix4();
  const __p = new THREE.Vector3();
  const __q = new THREE.Quaternion();
  const __s = new THREE.Vector3();
  const __e = new THREE.Euler();
  let __placed = 0, __guard = 0;
  while (__placed < 64 && __guard++ < 64 * 40) {
    __p.set((__r() * 2 - 1) * 25.6, 0, (__r() * 2 - 1) * 25.6);
    __p.y = 7.0;
    __e.set(0, __r() * Math.PI * 2, 0);
    __q.setFromEuler(__e);
    const __sxz = 1.6 + __r() * 1.6;
    const __sy = 14 * (0.25 + 0.75 * __r());
    __s.set(__sxz, __sy, __sxz);
    __p.y = __sy / 2;
    __m.compose(__p, __q, __s);
    __n_towers.setMatrixAt(__placed, __m);
    __placed++;
  }
  __n_towers.instanceMatrix.needsUpdate = true;
  __n_towers.count = __placed;
}
scene.add(__tag(__n_towers, 'towers'));
const __n_lamp_post_pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 3.4, 20), new THREE.MeshStandardMaterial({ color: 0x22262E }));
__n_lamp_post_pole.position.set(0, 1.7, 0);
__n_lamp_post_pole.rotation.set(0, 0, 0);
__n_lamp_post_pole.scale.set(1, 1, 1);
__n_lamp_post_pole.castShadow = true;
__n_lamp_post_pole.receiveShadow = true;
scene.add(__tag(__n_lamp_post_pole, 'lamp_post_pole'));
const __n_lamp_post_arm = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.08, 0.08), new THREE.MeshStandardMaterial({ color: 0x22262E }));
__n_lamp_post_arm.position.set(0.4, 3.3, 0);
__n_lamp_post_arm.rotation.set(0, 0, 0);
__n_lamp_post_arm.scale.set(1, 1, 1);
__n_lamp_post_arm.castShadow = true;
__n_lamp_post_arm.receiveShadow = true;
scene.add(__tag(__n_lamp_post_arm, 'lamp_post_arm'));
const __n_lamp_post_head = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.16, 0.2), new THREE.MeshBasicMaterial({ color: 0xFFD9A0 }));
__n_lamp_post_head.position.set(0.8, 3.2199999999999998, 0);
__n_lamp_post_head.rotation.set(0, 0, 0);
__n_lamp_post_head.scale.set(1, 1, 1);
__n_lamp_post_head.castShadow = true;
__n_lamp_post_head.receiveShadow = true;
scene.add(__tag(__n_lamp_post_head, 'lamp_post_head'));
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
