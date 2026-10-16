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
scene.background = new THREE.Color(0x2A2620);
const camera = new THREE.PerspectiveCamera(50, __W__ / __H__, 0.1, 200);
camera.position.set(4.2, 2.8, 5.2);
camera.lookAt(0, 0.8, 0);
const __light_fill = new THREE.HemisphereLight(0xBDD7F2, 0x334455, 0.7);
scene.add(__light_fill);
const __light_key = new THREE.DirectionalLight(0xFFF2E0, 1.4);
__light_key.position.set(6, 10, 7);
scene.add(__light_key);
const __n_table_set_top = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.09, 1.3), new THREE.MeshStandardMaterial({ color: 0x9C6B3F, roughness: 0.6 }));
__n_table_set_top.position.set(0, 0.735, 0);
__n_table_set_top.rotation.set(0, 0, 0);
__n_table_set_top.scale.set(1, 1, 1);
__n_table_set_top.castShadow = true;
__n_table_set_top.receiveShadow = true;
scene.add(__tag(__n_table_set_top, 'table_set_top'));
const __n_table_set_leg_front_left = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.6900000000000001, 0.12), new THREE.MeshStandardMaterial({ color: 0x9C6B3F, roughness: 0.7 }));
__n_table_set_leg_front_left.position.set(0.9400000000000001, 0.34500000000000003, 0.49);
__n_table_set_leg_front_left.rotation.set(0, 0, 0);
__n_table_set_leg_front_left.scale.set(1, 1, 1);
__n_table_set_leg_front_left.castShadow = true;
__n_table_set_leg_front_left.receiveShadow = true;
scene.add(__tag(__n_table_set_leg_front_left, 'table_set_leg_front_left'));
const __n_table_set_leg_front_right = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.6900000000000001, 0.12), new THREE.MeshStandardMaterial({ color: 0x9C6B3F, roughness: 0.7 }));
__n_table_set_leg_front_right.position.set(0.9400000000000001, 0.34500000000000003, -0.49);
__n_table_set_leg_front_right.rotation.set(0, 0, 0);
__n_table_set_leg_front_right.scale.set(1, 1, 1);
__n_table_set_leg_front_right.castShadow = true;
__n_table_set_leg_front_right.receiveShadow = true;
scene.add(__tag(__n_table_set_leg_front_right, 'table_set_leg_front_right'));
const __n_table_set_leg_rear_left = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.6900000000000001, 0.12), new THREE.MeshStandardMaterial({ color: 0x9C6B3F, roughness: 0.7 }));
__n_table_set_leg_rear_left.position.set(-0.9400000000000001, 0.34500000000000003, 0.49);
__n_table_set_leg_rear_left.rotation.set(0, 0, 0);
__n_table_set_leg_rear_left.scale.set(1, 1, 1);
__n_table_set_leg_rear_left.castShadow = true;
__n_table_set_leg_rear_left.receiveShadow = true;
scene.add(__tag(__n_table_set_leg_rear_left, 'table_set_leg_rear_left'));
const __n_table_set_leg_rear_right = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.6900000000000001, 0.12), new THREE.MeshStandardMaterial({ color: 0x9C6B3F, roughness: 0.7 }));
__n_table_set_leg_rear_right.position.set(-0.9400000000000001, 0.34500000000000003, -0.49);
__n_table_set_leg_rear_right.rotation.set(0, 0, 0);
__n_table_set_leg_rear_right.scale.set(1, 1, 1);
__n_table_set_leg_rear_right.castShadow = true;
__n_table_set_leg_rear_right.receiveShadow = true;
scene.add(__tag(__n_table_set_leg_rear_right, 'table_set_leg_rear_right'));
const __n_room_shell_floor = new THREE.Mesh(new THREE.BoxGeometry(9, 0.18, 8), new THREE.MeshStandardMaterial({ color: 0x8A6F4D, roughness: 0.8 }));
__n_room_shell_floor.position.set(0, -0.09, 0);
__n_room_shell_floor.rotation.set(0, 0, 0);
__n_room_shell_floor.scale.set(1, 1, 1);
__n_room_shell_floor.receiveShadow = true;
scene.add(__tag(__n_room_shell_floor, 'room_shell_floor'));
const __n_room_shell_wall_back = new THREE.Mesh(new THREE.BoxGeometry(9, 3.2, 0.18), new THREE.MeshStandardMaterial({ color: 0xC9BFAE, roughness: 0.9 }));
__n_room_shell_wall_back.position.set(0, 1.6, -4.0);
__n_room_shell_wall_back.rotation.set(0, 0, 0);
__n_room_shell_wall_back.scale.set(1, 1, 1);
__n_room_shell_wall_back.receiveShadow = true;
scene.add(__tag(__n_room_shell_wall_back, 'room_shell_wall_back'));
const __n_room_shell_wall_side = new THREE.Mesh(new THREE.BoxGeometry(0.18, 3.2, 8), new THREE.MeshStandardMaterial({ color: 0xC9BFAE, roughness: 0.9 }));
__n_room_shell_wall_side.position.set(-4.5, 1.6, 0);
__n_room_shell_wall_side.rotation.set(0, 0, 0);
__n_room_shell_wall_side.scale.set(1, 1, 1);
__n_room_shell_wall_side.receiveShadow = true;
scene.add(__tag(__n_room_shell_wall_side, 'room_shell_wall_side'));
const __n_room_shell_window_frame = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.3, 0.1), new THREE.MeshStandardMaterial({ color: 0x5A4632, roughness: 0.6 }));
__n_room_shell_window_frame.position.set(1.2, 1.8, -3.94);
__n_room_shell_window_frame.rotation.set(0, 0, 0);
__n_room_shell_window_frame.scale.set(1, 1, 1);
__n_room_shell_window_frame.castShadow = true;
__n_room_shell_window_frame.receiveShadow = true;
scene.add(__tag(__n_room_shell_window_frame, 'room_shell_window_frame'));
const __n_room_shell_window_glass = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.1, 0.06), new THREE.MeshBasicMaterial({ color: 0x9DB8D6 }));
__n_room_shell_window_glass.position.set(1.2, 1.8, -3.93);
__n_room_shell_window_glass.rotation.set(0, 0, 0);
__n_room_shell_window_glass.scale.set(1, 1, 1);
__n_room_shell_window_glass.castShadow = true;
__n_room_shell_window_glass.receiveShadow = true;
scene.add(__tag(__n_room_shell_window_glass, 'room_shell_window_glass'));
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
