# Morphix Scene Engine (Phase 3) — architecture

## Two execution targets

| concern | printer mode (existing) | scene mode (this phase) |
|---|---|---|
| goal | incremental solid construction | rich interactive environments |
| IR | `morphix_ir.py` Object IR (category solids) | `morphix_scene_ir.py` Scene IR (graph+behavior) |
| generation | thinker footprints + deterministic slicer | tested templates + thinker system selection |
| validation | guard + `node --check` + mesh/fill/contact gates | knowledge API gate + `node --check` + browser smoke |
| render | `render_deposit` boxes/spheres only | full r152 scene graph, fog, instancing, animation |
| tests | `tests/test_*.py` + `visual_geometry/` (272 green) | `tests/test_scene_*.py` + `test_knowledge.py` |

Sharing rule: scene effects (fog, particles, shaders, sky, crowds) are
never deposition geometry. Only `printable_nodes()` — explicit
`printable=True` on a printer-compatible constructor — may cross the
boundary, and crossing is one-way (scene -> printer parts list, never
scene effects -> toolpath).

## Dependencies and compatibility authority

- Three.js **r152 core**, vendored at
  `tests/visual_geometry/vendor/three.module.js` (409 exports, frozen in
  `morphix_three_knowledge.py:VENDOR_EXPORTS`). This file — not model
  memory — decides which APIs exist.
- UI scene pages load `three@0.152.2` from unpkg (same revision).
- Browser validation: headless Chromium (Playwright) + SwiftShader,
  640x480, pixelRatio 1, antialias off — same rig as the printer visual
  suite, extended by `tests/scene_browser.py` (waits for
  `window.__scene_ready`, captures `__scene_error`/`__scene_stats`).
- Models: DeepSeek-R1 1.5B (thinker/planner), Qwen2.5-Coder-7B (builder),
  Moondream (vision). The planner uses the thinker for system selection
  only; geometry comes from tested templates.

## Missing capabilities (explicit, not silent)

- **No WebGPU renderer, no TSL, no RenderPipeline** in r152 core.
  `api_supported()` rejects them with the WebGL analogue named.
- **No examples/jsm addons offline** (OrbitControls, EffectComposer,
  GLTFLoader, UltraHDRLoader). UI pages may import them via CDN
  importmap; the offline harness omits them (fixed cameras instead).
- **No external assets in generated scenes.** HDRI environments are
  replaced by procedural equirect canvases + PMREM; GLTF subjects by
  original procedural compositions. Anything else requires a
  user-supplied asset URL (not implemented in this phase).

## Reference benchmark analysis

`webgpu_custom_fog_background` (three.js dev) dissected 2026-10-13:

| reference feature | Morphix scene treatment |
|---|---|
| TSL `rangeFogFactor(2.7, 4)` post graph | built-in `THREE.Fog(0x4080cc, 2.7, 4)` — same range numbers |
| ACES tone mapping in post | `ACESFilmicToneMapping` + sRGB output on the renderer |
| Royal Esplanade HDRI environment | procedural equirect canvas + PMREMGenerator |
| DamagedHelmet GLTF hero | original procedural hero (shell + visor + rim torus) |
| OrbitControls (addons) | fixed composed camera offline; CDN controls in UI pages |
| WebGPURenderer / TSL / RenderPipeline | explicitly unsupported on r152 (gated + documented) |

A second benchmark (`fog_landscape`: terrain + instanced pines + custom
height fog via a tested `onBeforeCompile` injection) covers procedural
landscape treatment. Both benchmarks assert render success, zero console
errors, measured fog presence, subject visibility, determinism
(SSIM >= 0.999 on re-render), and performance budgets — never bare
pixel comparison.

## Module map (new this phase)

- `morphix_scene_ir.py` — Scene IR schema, validation, budgets, seeds.
- `morphix_three_knowledge.py` — frozen r152 exports + curated entries,
  `api_supported`, `check_js_apis`.
- `morphix_scene_templates.py` — 11 tested templates + `FOG_BENCHMARK_NOTES`.
- `morphix_scene_compile.py` — IR -> JS (WebGL only), `validate_scene_js`,
  `scene_html` UI wrapper.
- `morphix_scene_plan.py` — thinker plan (schema + bounded retry) ->
  deterministic template composition.
- `server.py: /api/scenes, /api/scene-templates` — scene runs into
  `runs/<id>/{scene.html,scene.js,scene_ir.json,report.json}`.
- `frontend` scene mode — template catalogue, seed/prompt, backend badge,
  node/instance/validation panels.
- `tests/scene_browser.py`, `test_scene_{ir,knowledge,compile,templates,
  plan,isolation,benchmark,api}.py`.
