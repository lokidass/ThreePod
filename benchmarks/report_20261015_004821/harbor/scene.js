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
const __n_boat_hull = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.7, 1.2), new THREE.MeshStandardMaterial({ color: 0x6E4A2F, roughness: 0.7 }));
__n_boat_hull.position.set(0, 0.35, 0);
__n_boat_hull.rotation.set(0, 0, 0);
__n_boat_hull.scale.set(1, 1, 1);
__n_boat_hull.castShadow = true;
__n_boat_hull.receiveShadow = true;
scene.add(__tag(__n_boat_hull, 'boat_hull'));
const __n_boat_gunwale_l = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.12, 0.1), new THREE.MeshStandardMaterial({ color: 0x4A3320 }));
__n_boat_gunwale_l.position.set(0, 0.76, 0.6);
__n_boat_gunwale_l.rotation.set(0, 0, 0);
__n_boat_gunwale_l.scale.set(1, 1, 1);
__n_boat_gunwale_l.castShadow = true;
__n_boat_gunwale_l.receiveShadow = true;
scene.add(__tag(__n_boat_gunwale_l, 'boat_gunwale_l'));
const __n_boat_gunwale_r = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.12, 0.1), new THREE.MeshStandardMaterial({ color: 0x4A3320 }));
__n_boat_gunwale_r.position.set(0, 0.76, -0.6);
__n_boat_gunwale_r.rotation.set(0, 0, 0);
__n_boat_gunwale_r.scale.set(1, 1, 1);
__n_boat_gunwale_r.castShadow = true;
__n_boat_gunwale_r.receiveShadow = true;
scene.add(__tag(__n_boat_gunwale_r, 'boat_gunwale_r'));
const __n_boat_mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.07, 2.6, 20), new THREE.MeshStandardMaterial({ color: 0x5A4632 }));
__n_boat_mast.position.set(0.3, 2.0, 0);
__n_boat_mast.rotation.set(0, 0, 0);
__n_boat_mast.scale.set(1, 1, 1);
__n_boat_mast.castShadow = true;
__n_boat_mast.receiveShadow = true;
scene.add(__tag(__n_boat_mast, 'boat_mast'));
const __n_boat_sail = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.9, 1, 1), new THREE.MeshStandardMaterial({ color: 0xEDE6DA, roughness: 0.85, side: THREE.DoubleSide }));
__n_boat_sail.position.set(-0.45, 2.2, 0);
__n_boat_sail.rotation.set(0, 1.5707963, 0);
__n_boat_sail.scale.set(1, 1, 1);
__n_boat_sail.castShadow = true;
scene.add(__tag(__n_boat_sail, 'boat_sail'));
const __n_pier_deck = new THREE.Mesh(new THREE.BoxGeometry(8.0, 0.14, 1.6), new THREE.MeshStandardMaterial({ color: 0x7A6248, roughness: 0.85 }));
__n_pier_deck.position.set(0, 0.5, 0);
__n_pier_deck.rotation.set(0, 0, 0);
__n_pier_deck.scale.set(1, 1, 1);
__n_pier_deck.castShadow = true;
__n_pier_deck.receiveShadow = true;
scene.add(__tag(__n_pier_deck, 'pier_deck'));
const __n_pier_pile_0 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_0.position.set(-3.5, -0.1, -0.65);
__n_pier_pile_0.rotation.set(0, 0, 0);
__n_pier_pile_0.scale.set(1, 1, 1);
__n_pier_pile_0.castShadow = true;
__n_pier_pile_0.receiveShadow = true;
scene.add(__tag(__n_pier_pile_0, 'pier_pile_0'));
const __n_pier_pile_1 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_1.position.set(-3.5, -0.1, 0.65);
__n_pier_pile_1.rotation.set(0, 0, 0);
__n_pier_pile_1.scale.set(1, 1, 1);
__n_pier_pile_1.castShadow = true;
__n_pier_pile_1.receiveShadow = true;
scene.add(__tag(__n_pier_pile_1, 'pier_pile_1'));
const __n_pier_pile_2 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_2.position.set(-1.5, -0.1, -0.65);
__n_pier_pile_2.rotation.set(0, 0, 0);
__n_pier_pile_2.scale.set(1, 1, 1);
__n_pier_pile_2.castShadow = true;
__n_pier_pile_2.receiveShadow = true;
scene.add(__tag(__n_pier_pile_2, 'pier_pile_2'));
const __n_pier_pile_3 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_3.position.set(-1.5, -0.1, 0.65);
__n_pier_pile_3.rotation.set(0, 0, 0);
__n_pier_pile_3.scale.set(1, 1, 1);
__n_pier_pile_3.castShadow = true;
__n_pier_pile_3.receiveShadow = true;
scene.add(__tag(__n_pier_pile_3, 'pier_pile_3'));
const __n_pier_pile_4 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_4.position.set(0.5, -0.1, -0.65);
__n_pier_pile_4.rotation.set(0, 0, 0);
__n_pier_pile_4.scale.set(1, 1, 1);
__n_pier_pile_4.castShadow = true;
__n_pier_pile_4.receiveShadow = true;
scene.add(__tag(__n_pier_pile_4, 'pier_pile_4'));
const __n_pier_pile_5 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_5.position.set(0.5, -0.1, 0.65);
__n_pier_pile_5.rotation.set(0, 0, 0);
__n_pier_pile_5.scale.set(1, 1, 1);
__n_pier_pile_5.castShadow = true;
__n_pier_pile_5.receiveShadow = true;
scene.add(__tag(__n_pier_pile_5, 'pier_pile_5'));
const __n_pier_pile_6 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_6.position.set(2.5, -0.1, -0.65);
__n_pier_pile_6.rotation.set(0, 0, 0);
__n_pier_pile_6.scale.set(1, 1, 1);
__n_pier_pile_6.castShadow = true;
__n_pier_pile_6.receiveShadow = true;
scene.add(__tag(__n_pier_pile_6, 'pier_pile_6'));
const __n_pier_pile_7 = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.4, 20), new THREE.MeshStandardMaterial({ color: 0x4E3B28 }));
__n_pier_pile_7.position.set(2.5, -0.1, 0.65);
__n_pier_pile_7.rotation.set(0, 0, 0);
__n_pier_pile_7.scale.set(1, 1, 1);
__n_pier_pile_7.castShadow = true;
__n_pier_pile_7.receiveShadow = true;
scene.add(__tag(__n_pier_pile_7, 'pier_pile_7'));
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
