import { useEffect, useRef, useState } from "react";
import { Activity, ClipboardCheck, Code2, Copy, FileText, ListChecks } from "lucide-react";
import { Badge, Empty, Segmented, Stat, ToneIcon } from "./ui";

// tab: "activity" | "report" | "code"
export default function Inspector({ timeline, run, tab, setTab, onOpenRun }) {
  return (
    <aside className="panel inspector">
      <Segmented
        value={tab}
        onChange={setTab}
        options={[
          { value: "activity", label: "Activity", icon: <Activity size={15} /> },
          { value: "report", label: "Report", icon: <FileText size={15} /> },
          { value: "code", label: "Code", icon: <Code2 size={15} /> },
        ]}
      />
      <div className="inspector-body">
        {tab === "activity" && <Timeline items={timeline} />}
        {tab === "report" && (run ? <ReportView run={run} onOpenRun={onOpenRun} /> : <Empty icon={<ClipboardCheck size={26} />} title="No report yet" />)}
        {tab === "code" && <CodeView run={run} />}
      </div>
    </aside>
  );
}

function Timeline({ items }) {
  const end = useRef(null);
  useEffect(() => {
    // braces matter: newer Chrome returns a Promise from scrollIntoView, and
    // an effect must return nothing or a cleanup function
    end.current?.scrollIntoView({ block: "end" });
  }, [items.length]);
  if (!items.length)
    return (
      <Empty icon={<Activity size={26} />} title="No activity">
        Start a run — every planning, validation and commit step streams here live.
      </Empty>
    );
  return (
    <ol className="timeline">
      {items.map((it) => (
        <li key={it.id} className={`tl tl-${it.tone}`}>
          <ToneIcon tone={it.tone} size={15} />
          <div className="tl-body">
            <div className="tl-title">{it.title}</div>
            {it.detail && <div className="tl-detail">{it.detail}</div>}
            {it.json !== undefined && (
              <details>
                <summary>JSON</summary>
                <pre>{JSON.stringify(it.json, null, 1)}</pre>
              </details>
            )}
          </div>
        </li>
      ))}
      <div ref={end} />
    </ol>
  );
}

function CheckRow({ c }) {
  const tone = c.ok ? "ok" : c.severity === "critical" ? "bad" : "warn";
  return (
    <div className={`check check-${tone}`}>
      <ToneIcon tone={tone} size={14} />
      <div>
        <div className="check-name">
          {String(c.name ?? "check").replace(/_/g, " ")}
          {c.severity && <span className="check-sev">{c.severity}</span>}
        </div>
        {c.detail && <div className="check-detail">{c.detail}</div>}
      </div>
    </div>
  );
}

function ImageReport({ run, r }) {
  const img = r.image || {};
  const sub = img.subject || {};
  const v = r.visual_verification || {};
  const checks = v.checks || [];
  const failed = checks.filter((c) => !c.ok);
  const unc = r.ir_full?.uncertainty || {};
  const status = r.reconstruction_status || "failed";
  return (
    <div className="report">
      <div className="report-hero">
        <Badge tone={status === "reconstructed" ? "ok" : status === "approximation" ? "warn" : "bad"}>{status}</Badge>
        <h3>{sub.identified ? sub.label || sub.category : "Unidentified object"}</h3>
        {sub.decided_by && <p className="muted small">{sub.decided_by}</p>}
      </div>
      <div className="stats">
        <Stat label="Silhouette IoU" value={v.metrics?.iou ?? "—"} tone={v.metrics?.iou !== undefined ? (v.metrics.iou >= 0.62 ? "ok" : "bad") : undefined} />
        <Stat label="Aspect" value={v.metrics?.aspect_ratio !== undefined ? `×${v.metrics.aspect_ratio}` : "—"} />
        <Stat label="Builder" value={`${img.builder?.method ?? "—"} / ${img.builder?.kind ?? ""}`} />
        <Stat label="Vision" value={img.vision?.status ?? "—"} tone={img.vision?.status === "ok" || img.vision?.status === "manual" ? "ok" : "warn"} />
      </div>
      <figure className="compare">
        <img src={`/runs/${run.run_id}/trace/compare.png`} alt="reference, mask, renders" />
        <figcaption>Reference · subject mask · renders measured against the photo</figcaption>
      </figure>
      <h4><ListChecks size={15} /> Measured checks <span className="muted">{checks.length - failed.length}/{checks.length} pass</span></h4>
      <div className="checks">
        {[...failed, ...checks.filter((c) => c.ok)].map((c) => <CheckRow key={c.name} c={c} />)}
      </div>
      {unc.assumptions?.length || unc.occluded?.length ? (
        <>
          <h4>Assumptions &amp; uncertainty</h4>
          <ul className="bullets">
            {(unc.assumptions || []).map((a) => <li key={a}>{a}</li>)}
            {(unc.occluded || []).map((a) => <li key={a}>Not visible: {a}</li>)}
          </ul>
        </>
      ) : null}
      {(r.warnings || []).length > 0 && (
        <>
          <h4>Warnings</h4>
          <ul className="bullets warn">{r.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </>
      )}
    </div>
  );
}

function BuildReport({ r }) {
  const comps = r.components || [];
  const bad = (r.blocks || []).filter((b) => !b.ok);
  return (
    <div className="report">
      <div className="stats">
        <Stat label="Validation" value={r.ok ? "OK" : "FAIL"} tone={r.ok ? "ok" : "bad"} />
        <Stat label="Complete" value={r.complete === false ? "No" : "Yes"} tone={r.complete === false ? "bad" : "ok"} />
        <Stat label="Quality" value={r.quality_accepted ? "Accepted" : r.quality_accepted === false ? "Rejected" : "—"} tone={r.quality_accepted ? "ok" : r.quality_accepted === false ? "warn" : undefined} />
        <Stat label="Deposits" value={r.deposits ?? r.blocks?.length ?? "—"} />
      </div>
      {(r.goal_parts || []).length > 0 && (
        <p className="small">
          Goal coverage: <b>{(r.goal_covered || []).join(", ") || "—"}</b>
          {(r.goal_missing || []).length > 0 && <span className="t-bad"> · missing {r.goal_missing.join(", ")}</span>}
        </p>
      )}
      {comps.length > 0 && (
        <>
          <h4>Components</h4>
          <div className="comps">
            {comps.map((c) => {
              const pct = c.planned_deposits ? Math.min(100, (100 * c.committed_deposits) / c.planned_deposits) : 0;
              const good = c.status === "completed" && c.verified !== false;
              return (
                <div key={c.component} className="comp">
                  <div className="comp-head">
                    <ToneIcon tone={good ? "ok" : "bad"} size={13} />
                    <code>{c.component}</code>
                    <span className="muted">{c.committed_deposits}/{c.planned_deposits}</span>
                  </div>
                  <div className="bar"><span style={{ width: `${pct}%` }} className={good ? "" : "bar-bad"} /></div>
                </div>
              );
            })}
          </div>
        </>
      )}
      {bad.length > 0 && (
        <>
          <h4>Failed blocks</h4>
          <ul className="bullets warn">
            {bad.map((b) => <li key={b.label}>{b.action} {b.label}: {[...(b.guard_errors || []), ...(b.syntax_errors || [])].join("; ")}</li>)}
          </ul>
        </>
      )}
      {(r.warnings || []).length > 0 && (
        <>
          <h4>Warnings</h4>
          <ul className="bullets warn">{r.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </>
      )}
    </div>
  );
}

function SceneReport({ r }) {
  const s = r.summary || {};
  return (
    <div className="report">
      <div className="stats">
        <Stat label="Nodes" value={s.nodes ?? 0} />
        <Stat label="Instances" value={s.instances ?? 0} />
        <Stat label="Lights" value={s.lights ?? 0} />
        <Stat label="Animations" value={s.animations ?? 0} />
      </div>
      <p className="small">Plan: <b>{(r.plan?.templates || []).join(" + ") || "—"}</b> <span className="muted">via {r.plan?.source}</span></p>
      <p className="small">Backend: <b>{r.backend || "webgl"}</b> · three r152 · IR + JS validated</p>
    </div>
  );
}

function EditReport({ r, onOpenRun }) {
  return (
    <div className="report">
      <div className="report-hero">
        <Badge tone={r.success ? "ok" : "bad"}>edited</Badge>
        <h3>“{r.prompt}”</h3>
        <p className="muted small">
          Version of{" "}
          <button className="link" onClick={() => onOpenRun?.(r.base_run)}>{r.base_run}</button>
          {" "}· understood by {r.edit_parser}
        </p>
      </div>
      <h4>Changes applied</h4>
      <ul className="bullets">{(r.edit_log || []).map((l) => <li key={l}>{l}</li>)}</ul>
      {(r.edit_history || []).length > 1 && (
        <>
          <h4>Edit history</h4>
          <ol className="bullets">{r.edit_history.map((h, i) => <li key={i}>“{h.prompt}”</li>)}</ol>
        </>
      )}
      <BuildReport r={r} />
    </div>
  );
}

function ReportView({ run, onOpenRun }) {
  const r = run.report || {};
  if (r.mode === "edit") return <EditReport r={r} onOpenRun={onOpenRun} />;
  if (r.mode === "image-reconstruct") return <ImageReport run={run} r={r} />;
  if (r.mode === "scene") return <SceneReport r={r} />;
  return <BuildReport r={r} />;
}

function CodeView({ run }) {
  const [copied, setCopied] = useState(false);
  if (!run?.scene_js) return <Empty icon={<Code2 size={26} />} title="No code yet" />;
  return (
    <div className="code">
      <div className="code-head">
        <span className="muted small">scene.js · {run.scene_js.length.toLocaleString()} chars</span>
        <button
          className="btn btn-ghost btn-sm"
          onClick={async () => {
            await navigator.clipboard.writeText(run.scene_js);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          <Copy size={14} /> {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>{run.scene_js.length > 200000 ? run.scene_js.slice(0, 200000) + "\n// … truncated in view" : run.scene_js}</pre>
    </div>
  );
}
