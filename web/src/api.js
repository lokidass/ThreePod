// Client for the Morphix Flask API. Reports are large, evolving JSON
// documents produced by the pipeline; the UI reads only the fields it needs.

export class ApiError extends Error {
  constructor(msg, status, stage) {
    super(msg);
    this.status = status;
    this.stage = stage;
  }
}

async function call(path, init) {
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(data.error || `HTTP ${r.status}`, r.status, data.stage);
  return data;
}

const postJson = (path, body) =>
  call(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const api = {
  health: () => call("/api/health"),
  runs: () => call("/api/runs"),
  run: (id) => call(`/api/runs/${id}`),
  sceneTemplates: () => call("/api/scene-templates"),
  startJob: (body) => postJson("/api/jobs", body),
  cancelJob: (id) => postJson(`/api/jobs/${id}/cancel`, {}),
  scene: (body) => postJson("/api/scenes", body),
  image: (fd) => call("/api/image", { method: "POST", body: fd }),
  parts: (id) => call(`/api/runs/${id}/parts`),
  text3d: (prompt) => postJson("/api/text3d", { prompt }),
  editRun: (id, prompt) => postJson(`/api/runs/${id}/edit`, { prompt }),
  exists: async (url) => {
    try {
      const r = await fetch(url, { method: "HEAD" });
      return r.ok;
    } catch {
      return false;
    }
  },
};

/** Headline verdict for a finished run: { tone, title, detail }. Image runs
 *  report the MEASURED agreement with the photo, never the printer's syntax
 *  status. Tones: ok | warn | bad | info | live | idle. */
export function verdictFor(report) {
  if (report.mode === "edit") {
    const sub = report.image?.subject || {};
    const name = sub.label || sub.category || report.ir_full?.category || "object";
    return report.success
      ? { tone: "ok", title: `Edited ${name}`, detail: `“${report.prompt}” · re-printed and verified` }
      : { tone: "bad", title: "Edit printed with problems", detail: `“${report.prompt}”` };
  }
  if (report.mode === "text-3d") {
    const sub = report.image?.subject || {};
    return report.success
      ? { tone: "ok", title: `Built ${sub.label || sub.category}`, detail: `${sub.family} shape · ${report.image?.builder?.preset} preset · print verified` }
      : { tone: "bad", title: "Build failed verification", detail: sub.label };
  }
  if (report.mode === "image-reconstruct") {
    const sub = report.image?.subject || {};
    const name = sub.identified ? sub.label || sub.category || "object" : "unidentified object";
    const m = report.visual_verification?.metrics;
    const detail = m?.iou !== undefined ? `Silhouette IoU ${m.iou} · aspect ×${m.aspect_ratio}` : undefined;
    switch (report.reconstruction_status) {
      case "reconstructed":
        return { tone: "ok", title: `Reconstructed ${name}`, detail };
      case "approximation":
        return { tone: "warn", title: `Silhouette approximation of ${name}`, detail };
      default:
        return { tone: "bad", title: "Reconstruction failed verification", detail: name };
    }
  }
  if (report.mode === "scene") {
    const s = report.summary || {};
    return report.ok
      ? { tone: "ok", title: "Environment validated", detail: `${s.nodes ?? 0} nodes · ${s.instances ?? 0} instances · ${report.backend ?? "webgl"}` }
      : { tone: "bad", title: "Scene failed validation" };
  }
  const comps = report.components || [];
  const done = comps.filter((c) => c.status === "completed").length;
  const detail = comps.length
    ? `${done}/${comps.length} components · ${report.deposits ?? report.blocks?.length ?? 0} deposits`
    : `${report.blocks?.length ?? 0} blocks`;
  if (report.ok && report.success !== false && report.complete !== false)
    return { tone: "ok", title: "Build validated", detail };
  if (report.ok) return { tone: "warn", title: report.complete === false ? "Build incomplete" : "Built with quality warnings", detail };
  return { tone: "bad", title: "Build failed validation", detail };
}
