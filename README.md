# ThreePod — Morphix 3D Builder

Natural-language prompt → **DeepSeek Thinker** → atomic action JSONs →
**Qwen Builder** → Guard → **validated** Three.js scene (`scene.js` + `scene.html`).

Every step is streamed live to the browser and validated before it is shown:
think → propose → build → guard → `node --check` → commit. A step that cannot
validate fails loudly instead of being silently replaced.

## Deployment (Docker Compose, recommended)

The app and Ollama ship as one compose project. On first boot the entrypoint
automatically pulls the **real** models (`deepseek-r1:1.5b`, `qwen2.5-coder:7b`)
into the Ollama volume.

```bash
docker compose up --build
# open http://localhost:5000
```

Configuration is via env vars (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` (compose) | Ollama endpoint |
| `MORPHIX_THINKER_MODEL` | `deepseek-r1:1.5b` | planner model |
| `MORPHIX_BUILDER_MODEL` | `qwen2.5-coder:7b` | builder model |
| `MORPHIX_STRICT_MODELS` | `true` | **always real models, fail loudly** |
| `MORPHIX_BUILDER_RETRIES` | `2` | re-asks per block on validation failure |
| `MORPHIX_OLLAMA_TIMEOUT` | `180` | per-request timeout (s) |
| `PORT` / `WORKERS` / `THREADS` | `5000` / `1` / `8` | WSGI tuning |

Model pulls in the compose service are one-shot; rerun
`docker compose up -d` or the entrypoint will skip models already present.

## Local development

### Web UI (React)

The UI lives in `web/` (React + JavaScript/JSX + Vite). Build it once and the
Flask server serves it at `/` (the previous static UI stays at `/classic/`):

```bash
cd web && npm install && npm run build
```

For UI development with hot reload, run the API (`python server.py --port 5000`)
and `npm run dev` in `web/` (http://localhost:5173, proxies `/api` and `/runs`).
The Docker image builds the UI in a separate Node stage.

### Server


```bash
python server.py --port 5000      # then open http://localhost:5000
```

Uses the same env vars; defaults to your local Ollama at `localhost:11434`.
The `template` builder and rule-based planner are disabled in strict mode (the
default). To see rule/template fallbacks again for experiments, set
`MORPHIX_STRICT_MODELS=false`.

## What "always real models" means

- The thinker MUST be a DeepSeek model; the builder MUST be Qwen (or the local
  LoRA). The UI only offers real models.
- If Ollama is down or a model returns unusable output, the run **fails loudly**
  and the error is streamed to the user — no silent rule plan, no template
  geometry substituted.
- Validation still catches model mistakes: per action a schema check, live-geometry
  state checks (bounds / collision / stacking), the anti-hallucination guard, and a
  real `node --check`. A rejected action gets validator feedback and the model is
  re-asked up to `MORPHIX_BUILDER_RETRIES` times before the run fails.

## Live step visibility

`POST /api/jobs` starts a run in a background thread and returns a job id. Every
phase is pushed as a Server-Sent Event:

```
GET /api/jobs/<id>/stream          # SSE: start, plan, step, think, validate,
                                   # commit, reject, warn, error, done
GET /api/jobs/<id>/events          # pollable JSON of the same events
GET /api/jobs/<id>                 # job status + final result
```

The frontend renders these live and updates the 3D viewer after every commit, so
you watch the scene grow step by validated step.

## API

`GET /api/health`, `POST /api/think`, `POST /api/jobs` (+ stream/events/cancel),
`POST /api/generate` (sync, for scripts), `POST /api/image` (multipart:
image + grid + mode), `GET /api/runs`, `GET /api/runs/<id>`,
`GET /runs/<id>/scene.html|scene.js|actions.json|report.json`.

## CLI pipeline (real models)

```bash
python morphix_pipeline.py "a red sphere on a blue box with warm sun light" --incremental --out ./out_step
python morphix_pipeline.py "a gold ring with glow" --out ./out
```

Output per run (`--out` dir): `actions.json`, `scene.js`, `scene.html`, `report.json`.
`report.json` carries per-block guard + `node --check` results; the CLI exits
non-zero if anything fails validation.

## Image → 3D + printer-order growth

- Upload an image in the browser (or `POST /api/image`, or
  `python morphix_image.py photo.png --grid 12 --mode relief`): it becomes a
  voxel relief/mosaic printed **cell by cell in serpentine order from one
  corner** — like a 3D printer. Luminance drives column height.
- Tick "describe with vision model" to have **moondream** caption the picture
  into a scene prompt.
- All incremental growth follows a **print head**: new pieces must attach
  within reach of existing geometry (`morphix_state.py`,
  `morphix_validator.py:validate_against_state`).

## Components

| File | Role |
|---|---|
| `morphix_config.py` | env config, Ollama client, strict-mode policy |
| `morphix_thinker.py` | DeepSeek-R1 planner (Ollama) → action JSONs |
| `morphix_guard.py` | Anti-hallucination guard per block |
| `morphix_validator.py` | Action schema + `node --check` + scene assembly report |
| `morphix_state.py` | Live geometry state (stacking, collisions, print head) |
| `morphix_pipeline.py` | Full run with per-step validation + `emit(event)` hook |
| `morphix_jobs.py` | Background jobs + SSE event stream |
| `morphix_path.py` | Point-by-point printer core: toolhead state, segment validation, deterministic path generators, engine rendering |
| `morphix_vision.py` | Vision inspector: close-up screenshots + GOOD/DRIFTING/BROKEN verdicts (advisory only) |
| `demo_print.py` | Deterministic offline print demo (no models) → scene + `train.jsonl` |
| `server.py` | Flask backend (WSGI-ready) |
| `test_qwen7b.py` | LoRA builder single-action runner |
| `train_qwen7b_qlora.py` | QLoRA retrain (GPU machine, see requirements-dev) |

## Setup notes

- Thinker needs `ollama pull deepseek-r1:1.5b`; builder needs `ollama pull qwen2.5-coder:7b`.
- `node` must be on PATH — it performs the real JavaScript syntax check of every block.
- LoRA builder needs the HF cache for `Qwen/Qwen2.5-Coder-7B` +
  `./qwen7b_qlora_morphix/final_adapter` plus GPU tooling
  (`pip install -r requirements-dev.txt`).

## Cancelling a run

- The UI has a **Stop** button next to Generate; it calls
  `POST /api/jobs/<id>/cancel`.
- The worker checks for cancellation at the top of every step (incremental)
  or block (whole-scene). The in-flight model call finishes, then the run
  aborts with `cancelled by user` — no further commits happen.
- Already committed layers are preserved and the partial `scene.js` /
  `scene.html` in `runs/<id>/` stays viewable; the run is loadable from
  History like any other.
- Cancelling a finished or unknown job returns `409` / `404`.

## Automated tests

Offline, deterministic, no models or credentials needed:

```bash
python -m pytest tests/ -q
```

`tests/test_core.py` covers action-schema validation, `node --check`
syntax gating, the anti-hallucination guard, geometry-state rules, cancel,
and the HTTP surface. `tests/test_path.py` covers the printer protocol
(§11.1–8, 11–13); `tests/test_print.py` covers the print controller
(budgets §11.9–10, component gate, toolhead handoff §11.14);
`tests/test_vision.py` covers verdict parsing, streak warnings, and the
advisory-only wiring (BROKEN can never fail a run).

## Point-by-point printing (printer protocol)

`morphix_path.py` implements true segment-level construction alongside the
primitive pipeline (the two modes are separate: primitive mode still uses
`add_box`/`add_sphere`/…; printer mode forbids them — see below).

Printer mode permits exactly six construction actions (lowercase only):

- `move_to` {position, layer, label} — travel, no deposition
- `deposit_segment` {from, to, bead_width, bead_height, material, layer, label}
- `deposit_point` {position, size, material, layer, label}
- `complete_path` / `complete_layer` {layer, label}
- `complete_object` {label}

Rules:

- One authoritative toolhead position; deposits must start exactly there
  (`POS_TOL`); a failed step never advances it. The builder proposes
  endpoints — it never moves the toolhead by returning JSON.
- Deposition is y-locked to the active layer plane (`LAYER_HEIGHT = 0.2`);
  `layer` must equal the active layer — no skipping ahead. Beads are drawn
  centered on the layer plane (toolpath-centerline convention, as in slicer
  previews); no hidden offsets are ever applied.
- Validated materials only: `body` / `accent` / `support` (engine palette).
- `complete_layer` needs ≥ 1 segment; `complete_object` needs every layer
  AND every planned component completed (tracked by slicer-stamped tags:
  missing / partial / completed — never label guessing).
- Hard budgets, all reported: 2 planner attempts, builder attempts, repair
  retries, attempt ceiling, `segment_cap` (default 400), `layer_cap`
  (default 32), `deadline_s` (default 1800). Rejects consume budget but
  never commit; hitting a limit stops loudly with the reason.
- Checkpoints: `PathState.checkpoint()` / `PathState.restore()` resume
  from the last committed action.
- Deterministic generators (`trace_rect`, `zigzag_infill`, `trace_ngon`)
  produce ground-truth state → action sequences; the engine renders validated
  deposits itself (`render_deposit`) — the model proposes, the engine draws.

Deterministic demo (no models, ~1s):

```bash
python demo_print.py --out ./out_print   # scene + train.jsonl
```

Live path print (thinks once, then prints every segment with validation):

```bash
python morphix_pipeline.py "red box 2 wide 1 deep 0.4 tall on the ground" --path --out ./out_path
python morphix_pipeline.py "..." --path --guided --segment-cap 12  # per-step builder proposals
```

or press **Print path** in the UI (`POST /api/jobs` with `mode: "path"`,
`guided: true` for per-step inference). Commits stream over SSE exactly
like primitive mode, so the viewport grows segment by segment; **Stop**
cancels between segments and `path.json` checkpoints every commit
(`--resume` replays one).

## Vision inspections (watching the print)

Every N deposits (`--vision-every`, default 5, `0` disables; 👁 input in
the UI) the pipeline screenshots a close-up `inspect.html` of the working
scene with headless Chromium and asks the vision model (moondream,
auto-detected) whether the print is going well:

- Verdicts stream live as 👁 chips and land in the report (`vision` list
  with `shot_*.png` files kept per run): GOOD / DRIFTING / BROKEN.
- Two consecutive BROKENs raise a visible warning suggesting Stop.
- Vision is advisory only: it can never fail or redirect a run — the
  geometric validator stays the sole hard gate, and any vision outage
  disables inspections with a warning instead of touching the print.
- Cost note: the vision model swaps VRAM with the builder on small GPUs,
  so frequent inspections slow the run; raise the interval if needed.

`out_print/train.jsonl` rows (`state` + `plan` + `layer` → `action`) are
the training-data format for the segment-level builder: current state,
plan, and active layer in, exactly one next action out.

## Toolpath optimization + success gate

The slicer emits fine bead steps; `optimize_toolpath` (in `morphix_path.py`)
folds them before printing — collinear runs merge (capped at 5 m),
zero-length segments and redundant travel drop, contours visit
nearest-first per layer, oversize segments split. Geometry is re-validated;
any defect falls back to the raw script loudly. Counts land in
`report.budgets.path_optimization` (typical car: ~55-60% fewer actions,
identical deposited length).

`report.success` is the only headline verdict: execution valid AND geometry
verified AND goal complete AND no fidelity warnings. The CLI exits non-zero
unless `success` is true. Timeouts and reject-budget aborts write an explicit
partial report (`status: incomplete`, committed/total/remaining) plus a
checkpoint — resume with `--resume` without duplicating segments.

## Visual geometry regression (`tests/visual_geometry/`)

Deterministic mesh + rendering checks that reject visibly wrong models even
when every action validates. No new dependencies (numpy/PIL/playwright/
chromium already in the stack; three.js r152 is vendored, no CDN).

```bash
pytest tests/visual_geometry -m "not visual and not live"  # fast geometry
pytest tests/visual_geometry -m visual                      # renders + SSIM
pytest tests/ -m "not live"                                 # full suite
pytest tests/visual_geometry -m live                        # real adapters
python tests/visual_geometry/regen_baselines.py             # report-only diff
python tests/visual_geometry/regen_baselines.py --approve   # bless new shots
```

How it works: fixtures build through the production compiler, committed
deposits become triangle meshes (`morphix_mesh.py`, also wired into
`run_path_print` finalize as fidelity warnings), and each fixture renders
from 5 fixed viewpoints (front/side/three-quarter/rear/top) with fixed
lights and materials. Full-frame SSIM (≥0.985) plus per-component crop
SSIM (≥0.95, localizes the failing part) gate acceptance; mismatches save
the actual render and an amplified diff under `.actual/` (git-ignored —
baselines are only ever written by `--approve`).#   T h r e e P o d  
 