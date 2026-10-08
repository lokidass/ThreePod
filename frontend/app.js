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
    if (h.ready) {
      $("health").innerHTML =
        `<b class="ok">thinker ${h.thinker_model}</b> · <b class="ok">builder ${h.builder_model}</b>${strict}`;
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

// ------------------------------------------------------------- live stream
let es = null;
let jobId = null;
let committed = 0;
let rejected = 0;
let liveCount = 0;

function openStream() {
  if (es) { es.close(); es = null; }
  $("tab-steps").innerHTML = "";
  committed = 0; rejected = 0; liveCount = 0;
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
    block_error: (e) => append(String.raw`error`, chip("warn",
      `attempt ${e.attempt} failed for <code>${esc(e.action)}</code>: ${esc(e.error)}`)),
    block_fail: (e) => append(String.raw`fail`, chip("bad",
      `❌ validation failed for <code>${esc(e.label || e.action)}</code>: ${esc((e.errors || []).join("; "))}`)),
    done: (e) => finalize(e),
    error: (e) => {
      append(String.raw`fatal`, chip("bad", `❌ run failed: ${esc(e.message)}`));
      $("verdict").className = "verdict bad";
      $("verdict").textContent = `✖ run failed — ${e.message}`;
      $("status").textContent = `error: ${e.message}`;
    },
  };
  for (const [name, fn] of Object.entries(actions)) {
    es.addEventListener(name, (ev) => { try { fn(JSON.parse(ev.data)); } catch (err) { /* ignore */ } });
  }
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
  const report = ev.report || {};
  showRun(ev.run_id, report.blocks || [], ev.scene_js || "", report, committed, rejected);
  $("verdict").className = "verdict " + (report.ok ? "ok" : "bad");
  $("verdict").textContent = report.ok
    ? `✔ validator: OK — ${committed} committed block(s), node --check clean`
    : `✖ validator: FAIL — see Report`;
  $("status").textContent = report.ok ? `done: ${ev.run_id} OK` : `done: ${ev.run_id} FAIL (see Report)`;
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
     <div>overall: ${report.ok ? `<span class="ok"><b>OK</b></span>` : `<span class="bad"><b>FAIL</b></span>`}</div>
     <div class="dim">statuses: ${statuses.map((s) => badge(s)).join(" ")}</div>
     ${top.map((e) => `<div class="bad">✖ ${esc(e)}</div>`).join("")}${rows}`;
}

async function loadRun(runId) {
  const d = await api(`/api/runs/${runId}`);
  showRun(d.run_id, d.actions, d.scene_js, d.report);
  $("tab-steps").innerHTML = `<p class="dim">loaded run ${runId} — see Report/Code; re-run to stream steps live.</p>`;
  $("status").textContent = `loaded ${runId}`;
}

// ------------------------------------------------------------ generation
async function generate() {
  const btn = $("generate");
  btn.disabled = true;
  $("verdict").className = "verdict live";
  $("verdict").textContent = "starting…";
  $("status").textContent = "creating job…";
  try {
    const d = await api("/api/jobs", {
      prompt: $("prompt").value,
      builder: $("builder").value,
      builder_model: $("builder").value === "lora" ? "" : "qwen2.5-coder:7b",
      thinker_model: $("thinker").value,
      incremental: $("incremental").checked,
      max_steps: null,
      cap: 24,
    });
    jobId = d.job_id;
    $("status").textContent = `streaming run ${jobId} — watching every step…`;
    openStream();
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("verdict").className = "verdict bad";
    $("verdict").textContent = `✖ could not start run — ${e.message}`;
    btn.disabled = false;
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
refreshHealth();
refreshHistory($("history"));