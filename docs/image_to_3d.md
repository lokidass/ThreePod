# Image → 3D (open-ended reconstruction)

Entry points: `morphix_reconstruct.run_reconstruction` (server, CLI) →
`morphix_image3d.run`. The previous closed-category implementation is kept as
`run_reconstruction_legacy` (its tests still run against it).

```
photo ─► morphix_image_seg      subject vs backdrop / floor / platform / light
                                pool / shadow / clutter / occluder (GrabCut)
      ─► morphix_image_vision   bounded natural questions; every reply is
                                classified (json | sentence | bare_word |
                                degenerate | malformed | empty); the subject
                                comes only from a sentence subject or a json
                                field; failures: unavailable | malformed |
                                low_confidence | ambiguous
      ─► builder                family builder fitted to the silhouette
                                (vehicle, table, chair, stool, house,
                                quadruped) — or, for ANY other subject or an
                                unidentified one, its own measured outline
                                (revolve if symmetric, else extrude)
      ─► semantic IR v2         subject, silhouette, viewpoint, components
                                (stage-ordered: silhouette → main volume →
                                structure → secondary → surface → camera),
                                appearance/palette, environment (never
                                printed), uncertainty — validate_image_ir
      ─► printer                add_profile / add_lathe / box / cylinder,
                                sliced and committed by run_path_print with
                                the measured palette
      ─► verification           headless render of the print, an id pass,
                                the smooth BufferGeometry preview and the
                                realistic model, all at the fitted photo
                                camera; measured checks vs the reference
      ─► bounded repair (≤ 2)   refit excluding the failing preset, then the
                                measured silhouette
      ─► verdict                reconstructed | approximation | failed
```

## Verdicts

| status | meaning |
|---|---|
| `reconstructed` | subject identified, family builder used, every critical check passes |
| `approximation` | built from the subject's measured outline (no builder for it, or vision could not name it); shape verified, depth/identity assumed |
| `failed` | critical checks still fail after repairs — reported with the failing checks, never shown as success |

`report.success` is true only when printing completed and the verdict is not
`failed`. A vision model's description is never used as proof: every verdict
needs the render to agree with the photo by measurement.

## Measured checks

Critical: silhouette IoU (bbox-normalized, 1-px tolerance) ≥ 0.62, aspect
within ±25 %, subject framed, planned-visible parts rendered and placed
where the plan puts them, family placement rules (wheels low and spread,
cabin above body, legs under the top, roof above walls, …), realistic-model
silhouette ≥ 0.62, no environment geometry in object-only mode.
Major (reported, not fatal): lamp position vs the photo's bright spots,
ground contacts vs the photo, body paint ΔE ≤ 32, clean page.

## Views

`runs/<id>/model.html` is the realistic presentation of the same IR
(bevelled ExtrudeGeometry with tumblehome, tire + rim wheels, physical
clear-coat paint / glass / rubber, PMREM studio environment, ACES tone
mapping, soft shadows, OrbitControls). Vehicle grille / bumpers / tail lamps
/ B-pillar are cosmetic presentation details, labelled on the page and never
printed. `scene.html` is the printed bead result. Evidence for every run is
under `runs/<id>/trace/` (reference, mask, vision replies, IR, component
plan, renders, `compare.png`, verification JSON).

## Tests

`tests/image3d/` (separate from scene and printer suites): vision adapter
kinds, segmentation (platform, light pool, clutter, low contrast), printer
primitives, IR contract, and end-to-end cases — real Urus photo with
recorded moondream replies, side car on a platform, SUV with clutter,
occluded car, table, chair, house, cat, vase (revolve), teapot (no builder),
malformed vision, missing model, wrong forced category, preserve mode.
Evidence lands in `tests/image3d/.artifacts/<case>/`.

## Editing with a prompt

Any run that has an object (`object.json`: image reconstructions, text-built
objects, earlier edits) can be edited from the edit bar under the viewer or
`POST /api/runs/<id>/edit {"prompt": "..."}`.

`morphix_edit.py` turns the prompt into validated operations — `color`,
`scale` (all / x length / y height / z width), `material`, `remove` — aimed at
the whole object, a part role (`wheel`, `roof`, `leg`, …) or a material
(`windows` → glass). Common phrasings are parsed by rules ("make the wheels
bigger and paint it red", "30% taller", "twice as big", "remove the tail");
anything else is translated by the local builder model and validated the same
way — a part the object does not have is rejected with the list of real parts.

Every edit is applied to the validated IR, re-printed and verified as a **new
run** (`mode: "edit"`, `base_run`, `edit_history`); the original is never
modified. "Paint it red" repaints the object's paint material, not its tires or
glass; on photo-textured subjects paint tints the photo so detail is kept.
