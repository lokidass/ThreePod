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

__installGradient(scene, '#2c4258', '#a8becf', '#1c2634', 'sky');
scene.fog = new THREE.FogExp2(0x9DB8D6, 0.012);
const camera = new THREE.PerspectiveCamera(55, __W__ / __H__, 0.1, 500);
camera.position.set(16, 9, 20);
camera.lookAt(0, 2.5, 0);
const __light_sun = new THREE.DirectionalLight(0xFFF2E0, 1.6);
__light_sun.position.set(6, 10, 7);
__light_sun.castShadow = true;
__light_sun.shadow.mapSize.set(1024, 1024);
__light_sun.shadow.camera.left = -45;
__light_sun.shadow.camera.right = 45;
__light_sun.shadow.camera.top = 45;
__light_sun.shadow.camera.bottom = -45;
scene.add(__light_sun);
const __light_sky = new THREE.HemisphereLight(0xBDD7F2, 0x334455, 0.7);
scene.add(__light_sky);
function __terrainH(x, z) { return __fbm(x * 0.035 + 7.3, z * 0.035 + 2.9, 11, 4) * 9; }
const __n_terrain = (function () {
  const __g = new THREE.PlaneGeometry(90, 90, 96, 96);
  __g.rotateX(-Math.PI / 2);
  const __pos = __g.attributes.position;
  for (let i = 0; i < __pos.count; i++) {
    __pos.setY(i, __terrainH(__pos.getX(i), __pos.getZ(i)) - 9 * 0.35);
  }
  __g.computeVertexNormals();
  return new THREE.Mesh(__g, new THREE.MeshStandardMaterial({ color: 0x5A6B4A, roughness: 0.95, metalness: 0.0 }));
})();
__n_terrain.position.set(0, 0, 0);
__n_terrain.rotation.set(0, 0, 0);
__n_terrain.scale.set(1, 1, 1);
__n_terrain.receiveShadow = true;
scene.add(__tag(__n_terrain, 'terrain'));
__n_terrain.position.set(0, 0, 0);
__n_terrain.rotation.set(0, 0, 0);
__n_terrain.scale.set(1, 1, 1);
__n_terrain.receiveShadow = true;
scene.add(__tag(__n_terrain, 'terrain'));
__installGradient(scene, '#27435f', '#9db8d6', '#1a2433', 'sky');
const __n_castle_keep_wall_n = new THREE.Mesh(new THREE.BoxGeometry(5.0, 3.0, 0.5), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_wall_n.position.set(0, 1.5, 2.5);
__n_castle_keep_wall_n.rotation.set(0, 0, 0);
__n_castle_keep_wall_n.scale.set(1, 1, 1);
__n_castle_keep_wall_n.castShadow = true;
__n_castle_keep_wall_n.receiveShadow = true;
scene.add(__tag(__n_castle_keep_wall_n, 'castle_keep_wall_n'));
const __n_castle_keep_merlon_0_0 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_0.position.set(-2.15, 3.17, 2.5);
__n_castle_keep_merlon_0_0.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_0.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_0.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_0, 'castle_keep_merlon_0_0'));
const __n_castle_keep_merlon_0_1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_1.position.set(-1.45, 3.17, 2.5);
__n_castle_keep_merlon_0_1.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_1.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_1.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_1, 'castle_keep_merlon_0_1'));
const __n_castle_keep_merlon_0_2 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_2.position.set(-0.75, 3.17, 2.5);
__n_castle_keep_merlon_0_2.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_2.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_2.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_2, 'castle_keep_merlon_0_2'));
const __n_castle_keep_merlon_0_3 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_3.position.set(-0.050000000000000266, 3.17, 2.5);
__n_castle_keep_merlon_0_3.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_3.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_3.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_3, 'castle_keep_merlon_0_3'));
const __n_castle_keep_merlon_0_4 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_4.position.set(0.6499999999999999, 3.17, 2.5);
__n_castle_keep_merlon_0_4.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_4.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_4.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_4, 'castle_keep_merlon_0_4'));
const __n_castle_keep_merlon_0_5 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_5.position.set(1.35, 3.17, 2.5);
__n_castle_keep_merlon_0_5.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_5.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_5.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_5, 'castle_keep_merlon_0_5'));
const __n_castle_keep_merlon_0_6 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_0_6.position.set(2.0499999999999994, 3.17, 2.5);
__n_castle_keep_merlon_0_6.rotation.set(0, 0, 0);
__n_castle_keep_merlon_0_6.scale.set(1, 1, 1);
__n_castle_keep_merlon_0_6.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_0_6, 'castle_keep_merlon_0_6'));
const __n_castle_keep_wall_s = new THREE.Mesh(new THREE.BoxGeometry(5.0, 3.0, 0.5), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_wall_s.position.set(0, 1.5, -2.5);
__n_castle_keep_wall_s.rotation.set(0, 0, 0);
__n_castle_keep_wall_s.scale.set(1, 1, 1);
__n_castle_keep_wall_s.castShadow = true;
__n_castle_keep_wall_s.receiveShadow = true;
scene.add(__tag(__n_castle_keep_wall_s, 'castle_keep_wall_s'));
const __n_castle_keep_merlon_1_0 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_0.position.set(-2.15, 3.17, -2.5);
__n_castle_keep_merlon_1_0.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_0.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_0.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_0, 'castle_keep_merlon_1_0'));
const __n_castle_keep_merlon_1_1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_1.position.set(-1.45, 3.17, -2.5);
__n_castle_keep_merlon_1_1.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_1.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_1.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_1, 'castle_keep_merlon_1_1'));
const __n_castle_keep_merlon_1_2 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_2.position.set(-0.75, 3.17, -2.5);
__n_castle_keep_merlon_1_2.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_2.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_2.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_2, 'castle_keep_merlon_1_2'));
const __n_castle_keep_merlon_1_3 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_3.position.set(-0.050000000000000266, 3.17, -2.5);
__n_castle_keep_merlon_1_3.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_3.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_3.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_3, 'castle_keep_merlon_1_3'));
const __n_castle_keep_merlon_1_4 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_4.position.set(0.6499999999999999, 3.17, -2.5);
__n_castle_keep_merlon_1_4.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_4.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_4.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_4, 'castle_keep_merlon_1_4'));
const __n_castle_keep_merlon_1_5 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_5.position.set(1.35, 3.17, -2.5);
__n_castle_keep_merlon_1_5.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_5.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_5.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_5, 'castle_keep_merlon_1_5'));
const __n_castle_keep_merlon_1_6 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_1_6.position.set(2.0499999999999994, 3.17, -2.5);
__n_castle_keep_merlon_1_6.rotation.set(0, 0, 0);
__n_castle_keep_merlon_1_6.scale.set(1, 1, 1);
__n_castle_keep_merlon_1_6.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_1_6, 'castle_keep_merlon_1_6'));
const __n_castle_keep_wall_e = new THREE.Mesh(new THREE.BoxGeometry(0.5, 3.0, 5.0), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_wall_e.position.set(2.5, 1.5, 0);
__n_castle_keep_wall_e.rotation.set(0, 0, 0);
__n_castle_keep_wall_e.scale.set(1, 1, 1);
__n_castle_keep_wall_e.castShadow = true;
__n_castle_keep_wall_e.receiveShadow = true;
scene.add(__tag(__n_castle_keep_wall_e, 'castle_keep_wall_e'));
const __n_castle_keep_merlon_2_0 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_0.position.set(2.5, 3.17, -2.15);
__n_castle_keep_merlon_2_0.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_0.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_0.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_0, 'castle_keep_merlon_2_0'));
const __n_castle_keep_merlon_2_1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_1.position.set(2.5, 3.17, -1.45);
__n_castle_keep_merlon_2_1.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_1.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_1.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_1, 'castle_keep_merlon_2_1'));
const __n_castle_keep_merlon_2_2 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_2.position.set(2.5, 3.17, -0.75);
__n_castle_keep_merlon_2_2.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_2.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_2.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_2, 'castle_keep_merlon_2_2'));
const __n_castle_keep_merlon_2_3 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_3.position.set(2.5, 3.17, -0.050000000000000266);
__n_castle_keep_merlon_2_3.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_3.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_3.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_3, 'castle_keep_merlon_2_3'));
const __n_castle_keep_merlon_2_4 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_4.position.set(2.5, 3.17, 0.6499999999999999);
__n_castle_keep_merlon_2_4.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_4.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_4.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_4, 'castle_keep_merlon_2_4'));
const __n_castle_keep_merlon_2_5 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_5.position.set(2.5, 3.17, 1.35);
__n_castle_keep_merlon_2_5.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_5.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_5.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_5, 'castle_keep_merlon_2_5'));
const __n_castle_keep_merlon_2_6 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_2_6.position.set(2.5, 3.17, 2.0499999999999994);
__n_castle_keep_merlon_2_6.rotation.set(0, 0, 0);
__n_castle_keep_merlon_2_6.scale.set(1, 1, 1);
__n_castle_keep_merlon_2_6.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_2_6, 'castle_keep_merlon_2_6'));
const __n_castle_keep_wall_w = new THREE.Mesh(new THREE.BoxGeometry(0.5, 3.0, 5.0), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_wall_w.position.set(-2.5, 1.5, 0);
__n_castle_keep_wall_w.rotation.set(0, 0, 0);
__n_castle_keep_wall_w.scale.set(1, 1, 1);
__n_castle_keep_wall_w.castShadow = true;
__n_castle_keep_wall_w.receiveShadow = true;
scene.add(__tag(__n_castle_keep_wall_w, 'castle_keep_wall_w'));
const __n_castle_keep_merlon_3_0 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_0.position.set(-2.5, 3.17, -2.15);
__n_castle_keep_merlon_3_0.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_0.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_0.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_0, 'castle_keep_merlon_3_0'));
const __n_castle_keep_merlon_3_1 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_1.position.set(-2.5, 3.17, -1.45);
__n_castle_keep_merlon_3_1.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_1.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_1.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_1, 'castle_keep_merlon_3_1'));
const __n_castle_keep_merlon_3_2 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_2.position.set(-2.5, 3.17, -0.75);
__n_castle_keep_merlon_3_2.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_2.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_2.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_2, 'castle_keep_merlon_3_2'));
const __n_castle_keep_merlon_3_3 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_3.position.set(-2.5, 3.17, -0.050000000000000266);
__n_castle_keep_merlon_3_3.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_3.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_3.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_3, 'castle_keep_merlon_3_3'));
const __n_castle_keep_merlon_3_4 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_4.position.set(-2.5, 3.17, 0.6499999999999999);
__n_castle_keep_merlon_3_4.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_4.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_4.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_4, 'castle_keep_merlon_3_4'));
const __n_castle_keep_merlon_3_5 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_5.position.set(-2.5, 3.17, 1.35);
__n_castle_keep_merlon_3_5.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_5.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_5.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_5, 'castle_keep_merlon_3_5'));
const __n_castle_keep_merlon_3_6 = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0x8D8D94, roughness: 0.9 }));
__n_castle_keep_merlon_3_6.position.set(-2.5, 3.17, 2.0499999999999994);
__n_castle_keep_merlon_3_6.rotation.set(0, 0, 0);
__n_castle_keep_merlon_3_6.scale.set(1, 1, 1);
__n_castle_keep_merlon_3_6.receiveShadow = true;
scene.add(__tag(__n_castle_keep_merlon_3_6, 'castle_keep_merlon_3_6'));
const __n_castle_keep_turret_ne = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 4.2, 20), new THREE.MeshStandardMaterial({ color: 0x7D7D84, roughness: 0.9 }));
__n_castle_keep_turret_ne.position.set(2.5, 2.1, 2.5);
__n_castle_keep_turret_ne.rotation.set(0, 0, 0);
__n_castle_keep_turret_ne.scale.set(1, 1, 1);
__n_castle_keep_turret_ne.castShadow = true;
__n_castle_keep_turret_ne.receiveShadow = true;
scene.add(__tag(__n_castle_keep_turret_ne, 'castle_keep_turret_ne'));
const __n_castle_keep_cap_ne = new THREE.Mesh(new THREE.ConeGeometry(0.95, 1.1, 12), new THREE.MeshStandardMaterial({ color: 0x4E463C, roughness: 0.85 }));
__n_castle_keep_cap_ne.position.set(2.5, 4.75, 2.5);
__n_castle_keep_cap_ne.rotation.set(0, 0, 0);
__n_castle_keep_cap_ne.scale.set(1, 1, 1);
__n_castle_keep_cap_ne.castShadow = true;
scene.add(__tag(__n_castle_keep_cap_ne, 'castle_keep_cap_ne'));
const __n_castle_keep_turret_se = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 4.2, 20), new THREE.MeshStandardMaterial({ color: 0x7D7D84, roughness: 0.9 }));
__n_castle_keep_turret_se.position.set(2.5, 2.1, -2.5);
__n_castle_keep_turret_se.rotation.set(0, 0, 0);
__n_castle_keep_turret_se.scale.set(1, 1, 1);
__n_castle_keep_turret_se.castShadow = true;
__n_castle_keep_turret_se.receiveShadow = true;
scene.add(__tag(__n_castle_keep_turret_se, 'castle_keep_turret_se'));
const __n_castle_keep_cap_se = new THREE.Mesh(new THREE.ConeGeometry(0.95, 1.1, 12), new THREE.MeshStandardMaterial({ color: 0x4E463C, roughness: 0.85 }));
__n_castle_keep_cap_se.position.set(2.5, 4.75, -2.5);
__n_castle_keep_cap_se.rotation.set(0, 0, 0);
__n_castle_keep_cap_se.scale.set(1, 1, 1);
__n_castle_keep_cap_se.castShadow = true;
scene.add(__tag(__n_castle_keep_cap_se, 'castle_keep_cap_se'));
const __n_castle_keep_turret_nw = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 4.2, 20), new THREE.MeshStandardMaterial({ color: 0x7D7D84, roughness: 0.9 }));
__n_castle_keep_turret_nw.position.set(-2.5, 2.1, 2.5);
__n_castle_keep_turret_nw.rotation.set(0, 0, 0);
__n_castle_keep_turret_nw.scale.set(1, 1, 1);
__n_castle_keep_turret_nw.castShadow = true;
__n_castle_keep_turret_nw.receiveShadow = true;
scene.add(__tag(__n_castle_keep_turret_nw, 'castle_keep_turret_nw'));
const __n_castle_keep_cap_nw = new THREE.Mesh(new THREE.ConeGeometry(0.95, 1.1, 12), new THREE.MeshStandardMaterial({ color: 0x4E463C, roughness: 0.85 }));
__n_castle_keep_cap_nw.position.set(-2.5, 4.75, 2.5);
__n_castle_keep_cap_nw.rotation.set(0, 0, 0);
__n_castle_keep_cap_nw.scale.set(1, 1, 1);
__n_castle_keep_cap_nw.castShadow = true;
scene.add(__tag(__n_castle_keep_cap_nw, 'castle_keep_cap_nw'));
const __n_castle_keep_turret_sw = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 4.2, 20), new THREE.MeshStandardMaterial({ color: 0x7D7D84, roughness: 0.9 }));
__n_castle_keep_turret_sw.position.set(-2.5, 2.1, -2.5);
__n_castle_keep_turret_sw.rotation.set(0, 0, 0);
__n_castle_keep_turret_sw.scale.set(1, 1, 1);
__n_castle_keep_turret_sw.castShadow = true;
__n_castle_keep_turret_sw.receiveShadow = true;
scene.add(__tag(__n_castle_keep_turret_sw, 'castle_keep_turret_sw'));
const __n_castle_keep_cap_sw = new THREE.Mesh(new THREE.ConeGeometry(0.95, 1.1, 12), new THREE.MeshStandardMaterial({ color: 0x4E463C, roughness: 0.85 }));
__n_castle_keep_cap_sw.position.set(-2.5, 4.75, -2.5);
__n_castle_keep_cap_sw.rotation.set(0, 0, 0);
__n_castle_keep_cap_sw.scale.set(1, 1, 1);
__n_castle_keep_cap_sw.castShadow = true;
scene.add(__tag(__n_castle_keep_cap_sw, 'castle_keep_cap_sw'));
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
