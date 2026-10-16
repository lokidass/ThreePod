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

__installGradient(scene, '#2c4a6e', '#cf9a5a', '#1a2030', 'sky');
scene.fog = new THREE.FogExp2(0xCF9A5A, 0.008);
__bakeEnvironment(scene, renderer);
const camera = new THREE.PerspectiveCamera(55, __W__ / __H__, 0.1, 600);
camera.position.set(16, 6, 22);
camera.lookAt(0, 0, 0);
const __light_sun = new THREE.DirectionalLight(0xFFD9A0, 1.8);
__light_sun.position.set(-14, 7, -18);
scene.add(__light_sun);
const __light_sky = new THREE.HemisphereLight(0xCF9A5A, 0x1A2030, 0.5);
scene.add(__light_sky);
const __light_particles_fill = new THREE.AmbientLight(0x334455, 0.8);
scene.add(__light_particles_fill);
const __n_lake = new THREE.Mesh(new THREE.PlaneGeometry(60, 60, 1, 1), new THREE.MeshStandardMaterial({ color: 0x1E3A55, roughness: 0.08, metalness: 0.9, envMapIntensity: 1.2 }));
__n_lake.position.set(0, 0, 0);
__n_lake.rotation.set(-1.5707963, 0, 0);
__n_lake.scale.set(1, 1, 1);
__n_lake.receiveShadow = true;
scene.add(__tag(__n_lake, 'lake'));
const __n_mist = (function () {
  const __r = __rng(1139);
  const __pos = new Float32Array(800 * 3);
  for (let i = 0; i < 800; i++) {
    __pos[i * 3] = (__r() * 2 - 1) * 10.0;
    __pos[i * 3 + 1] = __r() * 6;
    __pos[i * 3 + 2] = (__r() * 2 - 1) * 10.0;
  }
  const __g = new THREE.BufferGeometry();
  __g.setAttribute('position', new THREE.BufferAttribute(__pos, 3));
  return new THREE.Points(__g, new THREE.PointsMaterial({ color: 0xE8D9C0, size: 0.5, sizeAttenuation: true, transparent: true, opacity: 0.9, depthWrite: false }));
})();
__n_mist.position.set(0, 0, 0);
__n_mist.rotation.set(0, 0, 0);
__n_mist.scale.set(1, 1, 1);
scene.add(__tag(__n_mist, 'mist'));
const __n_particles_embers = (function () {
  const __r = __rng(14448);
  const __pos = new Float32Array(4000 * 3);
  for (let i = 0; i < 4000; i++) {
    __pos[i * 3] = (__r() * 2 - 1) * 10.0;
    __pos[i * 3 + 1] = __r() * 6;
    __pos[i * 3 + 2] = (__r() * 2 - 1) * 10.0;
  }
  const __g = new THREE.BufferGeometry();
  __g.setAttribute('position', new THREE.BufferAttribute(__pos, 3));
  return new THREE.Points(__g, new THREE.PointsMaterial({ color: 0xFFB066, size: 0.14, sizeAttenuation: true, transparent: true, opacity: 0.9, depthWrite: false }));
})();
__n_particles_embers.position.set(0, 0, 0);
__n_particles_embers.rotation.set(0, 0, 0);
__n_particles_embers.scale.set(1, 1, 1);
scene.add(__tag(__n_particles_embers, 'particles_embers'));
const __n_particles_stars = (function () {
  const __r = __rng(13030);
  const __pos = new Float32Array(2000 * 3);
  for (let i = 0; i < 2000; i++) {
    __pos[i * 3] = (__r() * 2 - 1) * 10.0;
    __pos[i * 3 + 1] = __r() * 6;
    __pos[i * 3 + 2] = (__r() * 2 - 1) * 10.0;
  }
  const __g = new THREE.BufferGeometry();
  __g.setAttribute('position', new THREE.BufferAttribute(__pos, 3));
  return new THREE.Points(__g, new THREE.PointsMaterial({ color: 0x9FC0FF, size: 0.1, sizeAttenuation: true, transparent: true, opacity: 0.9, depthWrite: false }));
})();
__n_particles_stars.position.set(0, 0, 0);
__n_particles_stars.rotation.set(0, 0, 0);
__n_particles_stars.scale.set(1, 1, 1);
scene.add(__tag(__n_particles_stars, 'particles_stars'));
const __n_coral_bed_cones = new THREE.InstancedMesh(new THREE.ConeGeometry(0.5, 1.8, 7), new THREE.MeshStandardMaterial({ color: 0xE86A8A, roughness: 0.7, metalness: 0.0 }), 60);
{ const __r = __rng(12506);
  const __m = new THREE.Matrix4();
  const __p = new THREE.Vector3();
  const __q = new THREE.Quaternion();
  const __s = new THREE.Vector3();
  const __e = new THREE.Euler();
  let __placed = 0, __guard = 0;
  while (__placed < 60 && __guard++ < 60 * 40) {
    const __a = __r() * Math.PI * 2;
    const __rad = Math.sqrt(__r()) * 14;
    __p.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);
    __p.y = 0.9;
    __e.set(0, __r() * Math.PI * 2, 0);
    __q.setFromEuler(__e);
    const __sj = 1 + (__r() * 2 - 1) * 0.45;
    __s.set(__sj, __sj, __sj);
    __m.compose(__p, __q, __s);
    __n_coral_bed_cones.setMatrixAt(__placed, __m);
    __placed++;
  }
  __n_coral_bed_cones.instanceMatrix.needsUpdate = true;
  __n_coral_bed_cones.count = __placed;
}
scene.add(__tag(__n_coral_bed_cones, 'coral_bed_cones'));
const __n_coral_bed_bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.35, 10, 8), new THREE.MeshStandardMaterial({ color: 0xF2A541, roughness: 0.6, metalness: 0.0 }), 30);
{ const __r = __rng(12499);
  const __m = new THREE.Matrix4();
  const __p = new THREE.Vector3();
  const __q = new THREE.Quaternion();
  const __s = new THREE.Vector3();
  const __e = new THREE.Euler();
  let __placed = 0, __guard = 0;
  while (__placed < 30 && __guard++ < 30 * 40) {
    const __a = __r() * Math.PI * 2;
    const __rad = Math.sqrt(__r()) * 14;
    __p.set(Math.cos(__a) * __rad, 0, Math.sin(__a) * __rad);
    __p.y = 0.5;
    __e.set(0, __r() * Math.PI * 2, 0);
    __q.setFromEuler(__e);
    const __sj = 1 + (__r() * 2 - 1) * 0.45;
    __s.set(__sj, __sj, __sj);
    __m.compose(__p, __q, __s);
    __n_coral_bed_bulbs.setMatrixAt(__placed, __m);
    __placed++;
  }
  __n_coral_bed_bulbs.instanceMatrix.needsUpdate = true;
  __n_coral_bed_bulbs.count = __placed;
}
scene.add(__tag(__n_coral_bed_bulbs, 'coral_bed_bulbs'));
const __n_free_snake = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xffffff }));
__n_free_snake.position.set(0, 0, 0);
__n_free_snake.rotation.set(0, 0, 0);
__n_free_snake.scale.set(1, 1, 1);
scene.add(__tag(__n_free_snake, 'free_snake'));
function __bakeEnvironment(scene, renderer) {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 32;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 32);
  grad.addColorStop(0, '#2c4a6e');
  grad.addColorStop(0.55, '#cf9a5a');
  grad.addColorStop(1, '#1a2030');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromEquirectangular(tex).texture;
  tex.dispose(); pm.dispose();
}
__anims.push({ update: function (dt, t) { __n_particles_embers.rotation.y += dt * 0.02; } });
__anims.push({ update: function (dt, t) { __n_particles_stars.rotation.y += dt * -0.008; } });
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
