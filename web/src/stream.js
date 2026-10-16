// Server-Sent Events from /api/jobs/<id>/stream -> normalized timeline
// entries { id, tone, kind, title, detail, json }. Every pipeline event the
// backend emits gets a readable title, a tone, and (where useful) the raw
// JSON for inspection.

const EVENTS = [
  "start", "plan", "step", "think", "validate", "commit", "reject", "warn",
  "goal", "compile", "spec", "ir", "repair", "vision", "block_error",
  "block_fail", "layer", "component", "optimize", "done", "cancelled", "error",
];

let seq = 0;
const item = (tone, kind, title, detail, json) => ({ id: ++seq, tone, kind, title, detail, json });

function describe(name, e) {
  switch (name) {
    case "start":
      return item("info", name, "Run started", `builder ${e.builder_model || e.builder} · thinker ${e.thinker_model}${e.strict ? " · strict real models" : ""}`);
    case "plan":
      return item("info", name, `Plan: ${e.max_steps} steps`, e.decided_by ? `decided by ${e.decided_by}${e.layers ? ` · ${e.layers} layers` : ""}` : undefined);
    case "step":
      return e.phase === "done" ? item("ok", name, "Thinker finished early", e.reason) : null;
    case "think":
      if (e.phase === "action" && e.status === "ENGINE") return null; // engine steps are summarized by commits
      return e.action
        ? item("info", name, `Proposal: ${e.action.action}`, e.status, e.action)
        : item("info", name, `Plan proposed (${e.n_actions ?? "?"} actions)`, e.status, e.actions);
    case "validate": {
      const errs = [...(e.guard_errors || []), ...(e.syntax_errors || [])];
      return errs.length
        ? item("bad", name, `Validator rejected ${e.label || e.action}`, `attempt ${e.attempt}: ${errs.join("; ")}`)
        : item("ok", name, `Validated ${e.label || e.action}`, "guard OK · node --check clean");
    }
    case "reject":
      return item("bad", name, `Step ${e.step} rejected`, (e.errors || []).join("; "), e.action);
    case "warn":
      return item("warn", name, "Warning", e.message);
    case "goal":
      return item((e.missing || []).length ? "warn" : "ok", name, "Goal coverage",
        `covered ${(e.covered || []).join(", ") || "—"}${(e.missing || []).length ? ` · missing ${e.missing.join(", ")}` : ""}`);
    case "compile":
      return item(e.status === "ok" ? "ok" : "bad", name, `Compiled ${e.component}`,
        e.status === "ok" ? `layers ${(e.layers || []).join(", ")}${e.warning ? ` · ${e.warning}` : ""}` : e.error);
    case "spec":
      return item((e.errors || []).length ? "bad" : "ok", name, `Fidelity gate (${e.spec || "generic"})`,
        [...(e.errors || []), ...(e.warnings || [])].join("; ") || "passed");
    case "ir":
      return item("info", name, `Object IR: ${e.ir?.category ?? e.category ?? ""}`,
        e.components ? `${e.components.length} components` : undefined, e.ir || e);
    case "repair":
      return item("info", name, `Repaired ${e.component}`, e.reason);
    case "vision":
      return item(e.verdict === "GOOD" ? "ok" : e.verdict === "BROKEN" ? "bad" : "info", name, `Vision check: ${e.verdict}`, e.note);
    case "block_error":
      return item("warn", name, `Attempt ${e.attempt} failed`, e.error);
    case "block_fail":
      return item("bad", name, `Validation failed: ${e.label || e.action}`, (e.errors || []).join("; "));
    case "layer":
      return item("ok", name, `Layer ${e.layer} complete`, `${e.committed}/${e.total} actions committed`);
    case "component":
      return item("ok", name, `Component ${e.component} complete`, `${e.deposits} deposits · layers ${(e.layers || []).join(", ")}`);
    case "optimize":
      return item("info", name, "Toolpath optimized", `${e.before} → ${e.after} actions`);
    default:
      return null;
  }
}

/** Subscribe to a job. handlers: { onItem(item), onCommit(runId, n),
 *  onDone(runId, report, sceneJs), onEnd(reason, tone) }. Returns close(). */
export function openStream(jobId, h) {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  let committed = 0;
  let finished = false;
  for (const name of EVENTS) {
    es.addEventListener(name, (ev) => {
      let e;
      try {
        e = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (name === "commit") {
        committed++;
        if (committed % 25 === 1 || e.action?.action === "complete_layer")
          h.onCommit(String(e.run_id || jobId), committed);
        return;
      }
      if (name === "done") {
        finished = true;
        h.onDone(String(e.run_id || jobId), e.report || {}, String(e.scene_js || ""));
        return;
      }
      if (name === "cancelled") {
        finished = true;
        h.onItem(item("warn", name, "Cancelled", `${committed} committed actions preserved`));
        h.onEnd("Cancelled — committed work preserved", "warn");
        return;
      }
      if (name === "error") {
        finished = true;
        h.onItem(item("bad", name, "Run failed", String(e.message || "")));
        h.onEnd(String(e.message || "run failed"), "bad");
        return;
      }
      const it = describe(name, e);
      if (it) h.onItem(it);
    });
  }
  es.addEventListener("end", () => {
    if (!finished) h.onEnd("Stream ended", "idle");
    es.close();
  });
  return () => es.close();
}
