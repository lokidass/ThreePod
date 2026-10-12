// Morphix frontend: prompt -> job -> SSE live step stream -> 3D viewer.
// Every step (think/build/guard/validate/commit/reject) is rendered live and
// every block is validated before it is shown as committed.
const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const r = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

// ---------------------------------------------------------------- health
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const strict = h.strict ? " <b class='strict'>strict real models</b>" : "";
    const vis = h.vision_ready
      ? ` · <b class="ok">vision ${h.vision_model || "?"}</b>`
      : ` · <b class="no">no vision model</b>`;
    if (h.ready) {
      $("health").innerHTML =
        `<b class="ok">thinker ${h.thinker_model}</b> · <b class="ok">builder ${h.builder_model}</b>${vis}${strict}`;
    } else {
      $("health").innerHTML =
        `<b class="no">thinker ${h.thinker_ready ? "ok" : "missing"}</b> · ` +
        `<b class="no">builder ${h.builder_ready ? "ok" : "missing"}</b>${strict}`;
    }
    $("generate").disabled = !h.ready;
    if (!h.ready) $("status").textContent =
      "models not ready — install deepseek-r1:1.5b and qwen2.5-coder:7b in Ollama";
  } catch {
    $("health").textContent = "server unreachable";
    $("generate").disabled = true;
  }
}

async function refreshHistory(select) {
  try {
    const { runs } = await api("/api/runs");
    const cur = select.value;
    select.innerHTML = `<option value="">— runs —</option>` +
      runs.map((id) => `<option value="${id}">${id}</option>`).join("");
    if (runs.includes(cur)) select.value = cur;
    else if (runs.length && !select.value) { select.value = runs[0]; loadRun(runs[0]); }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------- rendering bits
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function badge(status) {
  const s = status || "?";
  const ok = /\/OK$/.test(s) || s === "OK" || s.startsWith("MODEL ");
  return `<span class="badge ${ok ? "ok" : "fb"}">${esc(s)}</span>`;
}

function jsonPre(obj) {
  return `<pre>${esc(JSON.stringify(obj, null, 1))}</pre>`;
}

function chip(cls, html) {
  return `<div class="chip ${cls}">${html}</div>`;
}

function irChip(ir) {
  if (!ir) return "object IR: n/a";
  const rows = (ir.hierarchy || []).map((c) =>
    `${"&nbsp;&nbsp;".repeat(c.depth || 0)}<code>${esc(c.id)}</code> ` +
    `<span class="dim">${esc(c.role || c.material || "")}${c.observed ? " · seen" : ""}</span>`
  ).join("<br>");
  return `<b>object IR</b> — category <b>${esc(ir.category)}</b> · source ${esc(ir.source)} · ` +
    `confidence ${esc(ir.confidence)} · bbox ${esc((ir.bbox || []).join("×"))}` +
    (rows ? `<div>${rows}</div>` : "");
}

// ------------------------------------------------------------- live stream
let es = null;
let jobId = null;
let committed = 0;
let rejected = 0;
let liveCount = 0;
let finished = false;

function setRunning(running) {
  $("generate").disabled = running;
  $("printpath").disabled = running;
  const gs = $("genscene");
  if (gs) gs.disabled = running;
  $("stop").disabled = !running;
}

async function refreshSceneTemplates() {
  try {
    const d = await api("/api/scene-templates");
    $("scenetemplate").innerHTML = (d.templates || []).map((t) =>
      `<option value="${esc(t.name)}">${esc(t.name)} — ${t.nodes} nodes` +
      (t.instances ? `, ${t.instances} inst` : "") + `</option>`).join("");
  } catch {
    $("scenetemplate").innerHTML = `<option value="">catalogue unreachable</option>`;
  }
}

function showSceneRun(d) {
  const rep = d.report || {};
  const ir = d.ir || {};
  const v = $("verdict");
  v.className = "verdict " + (rep.ok ? "ok" : "bad");
  v.textContent = rep.ok
    ? `✔ scene validated — ${(ir.nodes || 0)} nodes · ${(ir.instances || 0)} instances · backend ${(rep.backend || "?")}`
    : `✖ scene failed — ${esc((rep.validation && rep.validation.js_errors || []).join("; "))}`;
  $("scene").src = `/runs/${d.run_id}/scene.html?rev=${Date.now()}`;
  $("open").href = `/runs/${d.run_id}/scene.html`;
  $("code").textContent = d.scene_js || "";
  $("tab-report").innerHTML =
    `<div>mode: <b>scene</b> (rich environment — not printer deposition)</div>
     <div>backend: <b>${esc(rep.backend || "?")}</b> · three r${esc(rep.three_revision || "?")}</div>
     <div>plan: <b>${esc((rep.plan && (rep.plan.templates || []).join(" + ")) || "")}</b> ` +
    `<span class="dim">via ${esc((rep.plan && rep.plan.source) || "?")}</span></div>
     <div>🧩 nodes <b>${ir.nodes || 0}</b> · lights <b>${ir.lights || 0}</b> · ` +
    `instances <b>${ir.instances || 0}</b> · animations <b>${ir.animations || 0}</b></div>
     <div>printable solids: <b>${esc((ir.printable || []).join(", ") || "—")}</b> ` +
    `<span class="dim">(effects excluded by construction)</span></div>
     <div>validation: <span class="ok"><b>IR + JS clean</b></span></div>`;
}

$("genscene").onclick = async () => {
  const btn = $("genscene");
  btn.disabled = true;
  $("tab-steps").innerHTML = "";
  $("verdict").className = "verdict live";
  $("verdict").textContent = "composing scene from tested templates…";
  $("status").textContent = "generating WebGL environment…";
  try {
    const body = { seed: parseInt($("sceneseed").value || "11", 10) };
    if ($("sceneprompt").value.trim()) body.prompt = $("sceneprompt").value.trim();
    else if ($("scenetemplate").value) body.template = $("scenetemplate").value;
    else throw new Error("pick a template or describe a scene");
    const d = await api("/api/scenes", body);
    showSceneRun(d);
    $("scenecap").textContent =
      `scene ${d.run_id} OK — ${(d.ir.nodes || 0)} nodes, ${(d.ir.instances || 0)} instances`;
    $("status").textContent = `environment ready: ${d.run_id}`;
    refreshHistory($("history"));
    $("history").value = d.run_id;
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = `✖ scene failed — ${e.message}`;
  } finally {
    btn.disabled = false;
  }
};

function openStream() {
  if (es) { es.close(); es = null; }
  $("tab-steps").innerHTML = "";
  committed = 0; rejected = 0; liveCount = 0; finished = false;
  es = new EventSource(`/api/jobs/${jobId}/stream`);
  const actions = {
    start: (e) => append(String.raw`start`,
      chip("info",
        `<b>run started</b> · strict=${e.strict ? "yes" : "no"} · ` +
        `builder=${esc(e.builder_model)} · thinker=${esc(e.thinker_model)}`)),
    plan: (e) => append(String.raw`plan`,
      chip("info", `plan budget: <b>${e.max_steps}</b> steps (decided by ${e.decided_by})`)),
    step: (e) => {
      if (e.phase === "think") append(String.raw`step ${e.step} think`,
        chip("wait", `step ${e.step} — thinker is proposing the next action…`));
      if (e.phase === "done") append(String.raw`done`,
        chip("ok", `thinker finished early: ${esc(e.reason || "done")}`));
    },
    think: (e) => append(String.raw`think ${e.step}`,
      chip("act",
        `<b>step ${e.step}</b> proposal: <code>${esc(e.action.action)}</code> ${badge(e.status)}` +
        jsonPre(e.action))),
    validate: (e) => {
      const errs = [...(e.guard_errors || []), ...(e.syntax_errors || [])];
      if (errs.length) {
        append(String.raw`validate ${e.step}`,
          chip("bad", `step ${e.step} — validator REJECTED <code>${esc(e.label)}</code> ` +
            `(attempt ${e.attempt}: ${esc(errs.join("; "))})`));
      } else {
        append(String.raw`validate ${e.step}`,
          chip("ok", `step ${e.step} — validator OK (guard ${esc(e.guard_status)}, node --check clean)`));
      }
    },
    commit: (e) => {
      committed++;
      append(String.raw`commit ${e.step}`,
        `<div class="step done">
           <div class="head"><span><b>step ${e.step}</b> committed · ${esc(e.action.action)} <code>${esc(e.action.label || "")}</code></span>${badge(e.status)}</div>${jsonPre(e.action)}
         </div>`);
      // live 3D: the server rewrites scene.html after every commit
      $("scene").src = `/runs/${e.run_id}/scene.html?rev=${liveCount++}`;
      $("verdict").className = "verdict live";
      $("verdict").textContent = `🔨 step ${e.step} committed · ${committed} block(s) validated`;
    },
    reject: (e) => {
      rejected++;
      append(String.raw`reject ${e.step}`,
        chip("bad", `step ${e.step} — rejected (${esc((e.errors || []).join("; "))})` + jsonPre(e.action)));
      $("verdict").className = "verdict live";
      $("verdict").textContent = `step ${e.step} rejected — retrying with feedback`;
    },
    warn: (e) => append(String.raw`warn`, chip("warn", `⚠ ${esc(e.message)}`)),
    goal: (e) => {
      const cov = (e.covered || []).map(esc).join(", ") || "—";
      const miss = (e.missing || []).map(esc).join(", ") || "none — goal covered";
      append(String.raw`goal ${e.step || ""}`,
        chip("info", `🎯 goal coverage — covered: <b>${cov}</b> · missing: <b>${miss}</b>`));
    },
    compile: (e) => {
      const ok = e.status === "ok";
      const layers = (e.layers || []).join(", ");
      append(String.raw`compile ${esc(e.component || "")}`,
        chip(ok ? "ok" : "bad", ok
          ? `🏗 compiled <code>${esc(e.component || "")}</code> — layers [${esc(layers)}]${e.warning ? ` · ⚠ seated: ${esc(e.warning)}` : ""}`
          : `🏗 <code>${esc(e.component || "")}</code> failed: ${esc(e.error || "")}`));
    },
    spec: (e) => append(String.raw`spec`,
      chip((e.errors || []).length ? "bad" : "ok",
        ((e.errors || []).length
          ? `📐 plan fails ${(esc(e.spec || "generic"))} fidelity: ${esc(e.errors.join("; "))}`
          : `📐 plan passes ${(esc(e.spec || "generic"))} fidelity`) +
        ((e.warnings || []).length ? ` · ⚠ ${esc(e.warnings.join("; "))}` : ""))),
    ir: (e) => append("ir", chip("info", irChip(e.ir))),
    vision_input: (e) => append("vision_input", chip("info",
      `🖼 image analysed as an <b>${esc(e.category)}</b> (source: ${esc(e.source)})` +
      (e.vision_available
        ? ` · vision ${esc(e.vision_model || "")}`
        : ` · <b>no vision model</b> — demonstration template`) +
      (e.confidence != null ? ` · confidence ${esc(String(e.confidence))}` : "") +
      (e.components && e.components.length
        ? ` · parts: ${e.components.map((c) => esc(c.name || c)).join(", ")}` : "") +
      (e.materials && e.materials.length
        ? ` · materials: ${e.materials.map(esc).join(", ")}` : ""))),
    vision_retry: (e) => append("vision_retry", chip("warn",
      `🔁 re-analysing image (attempt ${e.attempt || "?"}): ` +
      `${e.detected ? `saw <b>${esc(e.detected)}</b> — ` : ""}${esc(e.problem || "")}`)),
    repair: (e) => append(String.raw`repair ${esc(e.component || "")}`,
      chip("info", `🔧 repaired <code>${esc(e.component || "")}</code>: ${esc(e.reason || "")}`)),
    vision: (e) => {
      const cls = e.verdict === "GOOD" ? "ok" : e.verdict === "BROKEN" ? "bad" : "info";
      append(String.raw`vision ${e.step || ""}`,
        chip(cls, `👁 vision [${esc(e.verdict)}] — ${esc(e.note)} <span class="dim">(${e.deposits} deposits · ${esc(e.model || "")})</span>`));
    },
    block_error: (e) => append(String.raw`error`, chip("warn",
      `attempt ${e.attempt} failed for <code>${esc(e.action)}</code>: ${esc(e.error)}`)),
    block_fail: (e) => append(String.raw`fail`, chip("bad",
      `❌ validation failed for <code>${esc(e.label || e.action)}</code>: ${esc((e.errors || []).join("; "))}`)),
    done: (e) => { finished = true; finalize(e); },
    cancelled: (e) => {
      finished = true;
      append(String.raw`cancelled`, chip("warn",
        `⏹ cancelled by user — ${committed} committed layer(s) preserved. ` +
        `Partial scene stays viewable.`));
      $("verdict").className = "verdict live";
      $("verdict").textContent = `⏹ cancelled — ${committed} committed block(s) kept`;
      $("status").textContent = `cancelled: ${jobId} (${committed} committed)`;
      setRunning(false);
      refreshHistory($("history"));
      if (jobId) { $("history").value = jobId; loadRun(jobId); }
    },
    error: (e) => {
      finished = true;
      append(String.raw`fatal`, chip("bad", `❌ run failed: ${esc(e.message)}`));
      $("verdict").className = "verdict bad";
      $("verdict").textContent = `✖ run failed — ${e.message}`;
      $("status").textContent = `error: ${e.message}`;
      setRunning(false);
    },
  };
  for (const [name, fn] of Object.entries(actions)) {
    es.addEventListener(name, (ev) => { try { fn(JSON.parse(ev.data)); } catch (err) { /* ignore */ } });
  }
  // terminal frame from the server: if neither done nor error fired (e.g.
  // the run was cancelled before its next step), close out the UI here.
  es.addEventListener("end", (ev) => {
    try {
      const p = JSON.parse(ev.data);
      if (!finished && p.error) {
        finished = true;
        $("verdict").className = "verdict live";
        $("verdict").textContent = `⏹ stopped — ${p.error}`;
        $("status").textContent = `stopped: ${p.error}`;
        setRunning(false);
      } else if (!finished) {
        finished = true;
        setRunning(false);
      }
    } catch (err) { /* ignore */ }
    es.close();
  });
  es.onerror = () => {
    // network-level error from the browser; EventSource will reconnect on its own
    if (es.readyState === EventSource.CLOSED) {
      // truly closed -> nothing more will come
    }
  };
}

function append(key, html) {
  const el = $("tab-steps");
  const box = document.createElement("div");
  box.dataset.key = key;
  box.innerHTML = html;
  el.appendChild(box);
  el.scrollTop = el.scrollHeight;
}

function finalize(ev) {
  finished = true;
  setRunning(false);
  const report = ev.report || {};
  showRun(ev.run_id, report.blocks || [], ev.scene_js || "", report, committed, rejected);
  $("verdict").className = "verdict " + (report.ok ? "ok" : "bad");
  const comps = report.components || [];
  const compDone = comps.filter((c) => c.status === "completed").length;
  const compTxt = comps.length ? ` · ${compDone}/${comps.length} components` : "";
  const partial = report.complete === false ? ", INCOMPLETE" : "";
  $("verdict").textContent = report.ok
    ? `✔ validator: OK — ${committed} committed block(s)${compTxt}${partial}, node --check clean`
    : `✖ validator: FAIL — see Report`;
  $("status").textContent = report.ok ? `done: ${ev.run_id} OK${compTxt}${partial}` : `done: ${ev.run_id} FAIL (see Report)`;
  refreshHistory($("history"));
  $("history").value = ev.run_id;
}

// ------------------------------------------------------------- final view
function showRun(runId, actions, sceneJs, report, committedCount, rejectedCount) {
  const n = (report.blocks || []).length;
  const bad = (report.blocks || []).filter((b) => !b.ok).length;
  const verb = committedCount !== undefined
    ? `${committedCount} committed · ${rejectedCount || 0} rejected`
    : `${n} blocks`;
  const v = $("verdict");
  v.className = "verdict " + (report.ok ? "ok" : "bad");
  v.textContent = report.ok
    ? `✔ validator: OK — ${verb}, node --check clean (${report.think_status || report.mode || ""})`
    : `✖ validator: FAIL — ${bad}/${n} bad blocks (see Report)`;
  if (sceneJs) { $("scene").src = `/runs/${runId}/scene.html?rev=${Date.now()}`; }
  $("open").href = `/runs/${runId}/scene.html`;
  $("code").textContent = sceneJs;
  if (!document.querySelector(`[data-key="final"]`)) {
    append("final", chip(report.ok ? "ok" : "info",
      `<b>final report</b> — ${report.ok ? "everything validated" : "see Report tab for failures"}`));
  }
  const statuses = report.block_statuses || [];
  $("tab-steps").dataset.finalActions = JSON.stringify(actions);
  const rows = (report.blocks || []).map((b) => {
    const errs = [...(b.guard_errors || []), ...(b.syntax_errors || [])];
    return `<div>${errs.length ? `<span class="bad">✖ ${esc(b.action)} ${esc(b.label)}: ${esc(errs.join("; "))}</span>` : `<span class="ok">✔ ${esc(b.action)} ${esc(b.label)}</span>`}</div>`;
  }).join("");
  const top = [...(report.action_errors || []), ...(report.scene_errors || []), ...(report.syntax_scene_js || [])];
  $("tab-report").innerHTML =
    `<div>think: <b>${esc(report.think_status || "n/a")}</b>${report.mode ? ` · mode: ${esc(report.mode)}` : ""}</div>
     <div>overall: ${report.ok ? `<span class="ok"><b>OK</b></span>` : `<span class="bad"><b>FAIL</b></span>`}${report.complete === false ? ` · <span class="bad"><b>INCOMPLETE</b></span>` : ""}${report.quality_accepted === false ? ` · <span class="bad"><b>quality rejected</b></span>` : report.quality_accepted ? ` · <span class="ok"><b>quality accepted</b></span>` : ""}</div>
     ${(report.goal_parts || []).length ? `<div>🎯 goal: covered <b>${esc((report.goal_covered || []).join(", "))}</b>${(report.goal_missing || []).length ? ` · missing <span class="bad"><b>${esc(report.goal_missing.join(", "))}</b></span>` : ` · <span class="ok"><b>all parts built</b></span>`}</div>` : ""}
     ${(report.components || []).length ? `<div>🧩 components: ${(report.components || []).map((c) => { const good = c.status === "completed" && c.verified !== false; return `<span class="${good ? "ok" : "bad"}" title="${esc((c.checks || []).filter((k) => !k.ok).map((k) => k.name).join(", "))}">${esc(c.component)}=${esc(c.status)}${c.verified === false ? "✖" : ""} ${c.committed_deposits}/${c.planned_deposits}</span>`; }).join(" · ")}</div>` : ""}
     <div class="dim">statuses: ${statuses.map((s) => badge(s)).join(" ")}</div>
     ${report.budgets ? `<div class="dim">budgets: plan×${report.budgets.planner_attempts} · builder×${report.budgets.builder_attempts} · repairs×${report.budgets.repair_retries} · committed ${report.budgets.committed_actions}/${report.budgets.segment_cap} · layers ${report.budgets.layers}/${report.budgets.layer_cap} · deadline ${report.budgets.deadline_s}s</div>` : ""}
     ${top.map((e) => `<div class="bad">✖ ${esc(e)}</div>`).join("")}${rows}
     ${irPanel(report)}`;
}

function irPanel(report) {
  const ir = report && report.ir;
  const img = report && report.image;
  if (!ir && !img) return "";
  let html = `<hr>`;
  if (ir) {
    const rows = (ir.hierarchy || []).map((c) =>
      `<div style="padding-left:${(c.depth || 0) * 12}px">` +
      `${c.observed ? "👁" : "▫"} <code>${esc(c.id)}</code> ` +
      `<span class="dim">${esc(c.role || "")} · ${esc(c.material || "")} · ${esc((c.dims || []).join("×"))}</span></div>`
    ).join("");
    html += `<div>🧱 object IR — category <b>${esc(ir.category)}</b> · source ${esc(ir.source)} · ` +
      `confidence ${esc(ir.confidence)} · bbox ${esc((ir.bbox || []).join("×"))}</div>`;
    if (rows) html += `<div>${rows}</div>`;
    const asm = ir.assumptions || [];
    if (asm.length) html += `<div class="dim">assumptions: ${asm.map(esc).join("; ")}</div>`;
  }
  if (img) {
    html += `<div>🖼 image: <b>${esc(img.category)}</b> via ${esc(img.source)}` +
      `${img.vision_available ? ` (vision ${esc(img.vision_model || "")})` : " — no vision model"}</div>`;
    (img.warnings || []).forEach((w) => { html += `<div class="bad">⚠ ${esc(w)}</div>`; });
  }
  const fid = report.image_fidelity;
  if (fid) {
    const checks = (fid.checks || []).map((c) =>
      `<span class="${c.ok ? "ok" : "bad"}">${esc(c.name)}=${c.ok ? "OK" : "FAIL"}</span>`).join(" · ");
    html += `<div>📷 photo consistency: ${checks || "—"} ` +
      (fid.verified ? `<span class="ok"><b>verified</b></span>` : `<span class="bad"><b>not verified</b></span>`) + `</div>`;
    (fid.unsupported || []).forEach((u) => {
      html += `<div class="dim">? not determined by a single photo: ${esc(u)}</div>`;
    });
  }
  return html;
}

async function loadRun(runId) {
  const d = await api(`/api/runs/${runId}`);
  showRun(d.run_id, d.actions, d.scene_js, d.report);
  $("tab-steps").innerHTML = `<p class="dim">loaded run ${runId} — see Report/Code; re-run to stream steps live.</p>`;
  $("status").textContent = `loaded ${runId}`;
}

// ------------------------------------------------------------ generation
async function stopRun() {
  if (!jobId || finished) return;
  $("stop").disabled = true;
  $("status").textContent = `requesting cancel for ${jobId}…`;
  try {
    await api(`/api/jobs/${jobId}/cancel`, {});
  } catch (e) {
    $("status").textContent = `cancel failed: ${e.message}`;
    $("stop").disabled = false;
  }
}

async function generate() {
  const btn = $("generate");
  btn.disabled = true;
  $("stop").disabled = false;
  $("verdict").className = "verdict live";
  $("verdict").textContent = "starting…";
  $("status").textContent = "creating job…";
  try {
    const d = await api("/api/jobs", {
      prompt: $("prompt").value,
      builder: $("builder").value,
      builder_model: $("builder").value === "lora" ? "" : "qwen2.5-coder:7b",
      thinker_model: $("thinker").value,
      // incremental ON = printer-deposition mode in the backend (path);
      // OFF = whole-scene primitive mode. The checkbox selects the mode,
      // not just a frontend hint.
      mode: $("incremental").checked ? "path" : "primitive",
      guided: $("guided").checked,
      vision_every: parseInt($("visionevery").value || "5", 10),
      incremental: $("incremental").checked,
      max_steps: null,
      cap: 24,
    });
    jobId = d.job_id;
    // point the viewport at THIS run immediately: its scene starts as an
    // empty grid (written at job creation), so a rejected first step never
    // displays the previous run's geometry.
    $("scene").src = `/runs/${jobId}/scene.html`;
    $("open").href = `/runs/${jobId}/scene.html`;
    $("code").textContent = "";
    $("status").textContent = `streaming run ${jobId} — watching every step…`;
    openStream();
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = `✖ could not start run — ${e.message}`;
    setRunning(false);
  }
}

document.querySelectorAll("nav button").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $(`tab-${b.dataset.tab}`).classList.add("active");
  };
});
$("generate").onclick = generate;
$("stop").onclick = stopRun;
$("printpath").onclick = async () => {
  setRunning(true);
  $("verdict").className = "verdict live";
  $("verdict").textContent = "starting path print…";
  $("status").textContent = "creating path job…";
  try {
    const d = await api("/api/jobs", {
      prompt: $("prompt").value,
      builder: $("builder").value,
      builder_model: $("builder").value === "lora" ? "" : "qwen2.5-coder:7b",
      thinker_model: $("thinker").value,
      mode: "path",
      guided: $("guided").checked,
      vision_every: parseInt($("visionevery").value || "5", 10),
    });
    jobId = d.job_id;
    $("scene").src = `/runs/${jobId}/scene.html`;
    $("open").href = `/runs/${jobId}/scene.html`;
    $("code").textContent = "";
    $("status").textContent = `printing path ${jobId} — one validated segment at a time…`;
    openStream();
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = `✖ could not start print — ${e.message}`;
    setRunning(false);
  }
};
$("history").onchange = (e) => { if (e.target.value) loadRun(e.target.value); };
$("copy").onclick = async () => {
  await navigator.clipboard.writeText($("code").textContent);
  $("status").textContent = "scene.js copied";
};
$("imgfile").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("imgprev").src = URL.createObjectURL(f);
  $("imgprev").style.display = "block";
};
$("printimg").onclick = async () => {
  const f = $("imgfile").files[0];
  if (!f) { $("status").textContent = "choose an image first"; return; }
  const btn = $("printimg");
  btn.disabled = true;
  $("status").textContent = "printing image…";
  try {
    const fd = new FormData();
    fd.append("image", f);
    fd.append("grid", $("imggrid").value);
    fd.append("mode", $("imgmode").value);
    if ($("imgdesc").checked) fd.append("describe", "1");
    const r = await fetch("/api/image", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    if (d.description) $("imgcap").textContent = `vision: ${d.description}`;
    showRun(d.run_id, d.actions, d.scene_js, d.report);
    $("status").textContent = d.report.ok ? `printed: ${d.run_id} OK` : "printed with errors (see Report)";
    refreshHistory($("history"));
    $("history").value = d.run_id;
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
};
$("reconstruct").onclick = async () => {
  const f = $("imgfile").files[0];
  if (!f) { $("status").textContent = "choose an image first"; return; }
  const btn = $("reconstruct");
  btn.disabled = true;
  $("tab-steps").innerHTML = "";
  $("status").textContent = "analysing the photo as an object…";
  $("verdict").className = "verdict live";
  $("verdict").textContent = "semantic reconstruction: identify → IR → print";
  try {
    const fd = new FormData();
    fd.append("image", f);
    fd.append("mode", "reconstruct");
    if ($("imgcategory").value.trim()) fd.append("category", $("imgcategory").value.trim());
    if ($("visionmodel").value.trim()) fd.append("vision_model", $("visionmodel").value.trim());
    const r = await fetch("/api/image", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) {
      if (r.status === 422 && d.needs_category) {
        const seen = d.detected_category ? ` (saw “${d.detected_category}”)` : "";
        const cands = (d.candidates || []).join(", ");
        $("status").textContent = "needs a category — type it below and retry";
        $("imgcap").textContent =
          `could not classify confidently${seen} — choose a category` +
          (cands ? `: ${cands}` : "");
        $("verdict").className = "verdict bad";
        $("verdict").textContent = `✖ needs a category — ${d.error}`;
        return;
      }
      throw new Error(d.error || `HTTP ${r.status}`);
    }
    const rep = d.report || {};
    if (rep.image && rep.image.source === "image-demo") {
      $("imgcap").textContent = "no vision model — used a demonstration template (see Report)";
    } else if (rep.image) {
      $("imgcap").textContent = `analysed as ${rep.image.category} (${rep.image.source})`;
    }
    showRun(d.run_id, d.actions, d.scene_js, rep);
    $("status").textContent = rep.ok
      ? `reconstructed ${rep.image ? rep.image.category : "object"}: ${d.run_id} OK`
      : "reconstruction had errors (see Report)";
    refreshHistory($("history"));
    $("history").value = d.run_id;
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = `✖ reconstruction failed — ${e.message}`;
  } finally {
    btn.disabled = false;
  }
};

function applyGenMode() {
  const m = $("genmode").value;
  $("text-controls").hidden = m !== "text";
  $("scene-controls").hidden = m !== "scene";
  $("image-controls").hidden = m === "text" || m === "scene";
  $("img-options-semantic").hidden = m !== "image_semantic";
  $("img-options-relief").hidden = m !== "image_relief";
  $("status").textContent = m === "text"
    ? "text → 3D: the thinker plans, the engine prints layer by layer"
    : m === "scene"
      ? "scene → 3D environment: tested templates compose a WebGL world (no printer involved)"
      : m === "image_semantic"
        ? "image → 3D: the photo is analysed as an object, then printed as named parts"
        : "image relief: pixels become a printer-ordered heightmap (not full 3D)";
}
$("genmode").onchange = applyGenMode;
applyGenMode();

refreshHealth();
refreshHistory($("history"));
refreshSceneTemplates();