import { useCallback, useEffect, useRef, useState } from "react";
import { api, verdictFor } from "./api";
import { openStream } from "./stream";
import TopBar from "./components/TopBar";
import CreatePanel from "./components/CreatePanel";
import Viewer from "./components/Viewer";
import Inspector from "./components/Inspector";
import EditBar from "./components/EditBar";
import ErrorBoundary from "./components/ErrorBoundary";

const IDLE = { tone: "idle", title: "Ready", detail: "Describe an object, drop a photo, or open a run" };
let localSeq = 100000;

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(false);
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState(null);
  const [view, setView] = useState("scene"); // "model" | "scene"
  const [hasModel, setHasModel] = useState(false);
  const [rev, setRev] = useState(0);
  const [liveRun, setLiveRun] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [busy, setBusy] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [verdict, setVerdict] = useState(IDLE);
  const [tab, setTab] = useState("activity");
  const [mode, setMode] = useState("text");
  const closeStream = useRef(null);

  const note = (tone, title, detail) =>
    setTimeline((t) => [...t, { id: ++localSeq, tone, kind: "ui", title, detail }]);

  const refreshHealth = useCallback(() => {
    api.health()
      .then((h) => {
        setHealth(h);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));
  }, []);
  const refreshRuns = useCallback(() => api.runs().then((d) => setRuns(d.runs)).catch(() => undefined), []);

  const showRun = useCallback(async (r, switchTab = true) => {
    setRun(r);
    setLiveRun(null);
    // image reconstructions and edits have a realistic model.html (runs made
    // before the realistic presentation existed do not)
    const model = await api.exists(`/runs/${r.run_id}/model.html`);
    setHasModel(model);
    setView(model ? "model" : "scene");
    setRev((x) => x + 1);
    setVerdict(verdictFor(r.report || {}));
    if (switchTab) setTab("report");
  }, []);

  const openRun = useCallback(async (id) => {
    try {
      const r = await api.run(id);
      setTimeline([]);
      await showRun(r);
    } catch (e) {
      setVerdict({ tone: "bad", title: "Could not open run", detail: e.message });
    }
  }, [showRun]);

  useEffect(() => {
    refreshHealth();
    refreshRuns();
    const t = setInterval(refreshHealth, 30000);
    return () => clearInterval(t);
  }, [refreshHealth, refreshRuns]);

  // ------------------------------------------------------------ actions
  const startJob = async (body, label) => {
    closeStream.current?.();
    setTimeline([]);
    setTab("activity");
    setBusy(label);
    setRun(null);
    setHasModel(false);
    setVerdict({ tone: "live", title: `${label}…`, detail: "Planning, validating and committing step by step" });
    try {
      const { job_id } = await api.startJob(body);
      setJobId(job_id);
      setLiveRun(job_id);
      setView("scene");
      setRev((x) => x + 1);
      closeStream.current = openStream(job_id, {
        onItem: (it) => setTimeline((t) => [...t.slice(-400), it]),
        onCommit: (runId, n) => {
          setLiveRun(runId);
          setRev((x) => x + 1);
          setVerdict({ tone: "live", title: `${label}…`, detail: `${n} actions committed and validated` });
        },
        onDone: async (runId) => {
          setBusy(null);
          setJobId(null);
          refreshRuns();
          try {
            await showRun(await api.run(runId), false);
          } catch {
            setVerdict({ tone: "warn", title: "Finished", detail: runId });
          }
          note("ok", "Run finished", runId);
        },
        onEnd: (reason, tone) => {
          setBusy(null);
          setJobId(null);
          setVerdict({ tone, title: reason });
          refreshRuns();
        },
      });
    } catch (e) {
      setBusy(null);
      setVerdict({ tone: "bad", title: "Could not start run", detail: e.message });
    }
  };

  const stop = async () => {
    if (!jobId) return;
    note("warn", "Cancel requested", "the in-flight step finishes, then the run stops");
    try {
      await api.cancelJob(jobId);
    } catch (e) {
      note("bad", "Cancel failed", e.message);
    }
  };

  const scene = async (body) => {
    setBusy("Composing environment");
    setTimeline([]);
    setVerdict({ tone: "live", title: "Composing environment…", detail: "Tested templates on WebGL" });
    try {
      const d = await api.scene(body);
      note("ok", "Environment generated", d.run_id);
      await showRun({ run_id: d.run_id, actions: [], scene_js: d.scene_js, report: { ...d.report, summary: d.ir } });
      refreshRuns();
    } catch (e) {
      setVerdict({ tone: "bad", title: `Scene failed${e.stage ? ` (${e.stage})` : ""}`, detail: e.message });
      note("bad", "Scene failed", e.message);
    } finally {
      setBusy(null);
    }
  };

  const image = async (fd, label) => {
    setBusy(label);
    setTimeline([]);
    setTab("activity");
    setRun(null);
    setHasModel(false);
    const steps = fd.get("mode") === "reconstruct"
      ? ["Segmenting subject and environment", "Identifying the subject", "Fitting shape to the silhouette", "Printing layer by layer", "Rendering and measuring against the photo"]
      : ["Sampling the image grid", "Printing the relief"];
    steps.forEach((s) => note("live", s));
    setVerdict({ tone: "live", title: `${label}…`, detail: steps.join(" → ") });
    try {
      const d = await api.image(fd);
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "ok" } : x)));
      if (d.description) note("info", "Vision description", d.description);
      note("ok", "Done", d.run_id);
      await showRun({ run_id: d.run_id, actions: d.actions, scene_js: d.scene_js, report: d.report });
      refreshRuns();
    } catch (e) {
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "idle" } : x)));
      note("bad", `Failed${e.stage ? ` at ${e.stage}` : ""}`, e.message);
      setVerdict({ tone: "bad", title: `Failed${e.stage ? ` at ${e.stage}` : ""}`, detail: e.message });
    } finally {
      setBusy(null);
    }
  };

  const edit = async (prompt) => {
    if (!run) return;
    const base = run.run_id;
    setBusy("Applying edit");
    setTab("activity");
    setTimeline([]);
    note("live", `Editing: “${prompt}”`, "parse → apply to the validated shape → re-print → verify");
    setVerdict({ tone: "live", title: "Applying edit…", detail: `“${prompt}”` });
    try {
      const d = await api.editRun(base, prompt);
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "ok" } : x)));
      (d.report?.edit_log || []).forEach((l) => note("ok", l));
      note("ok", "New version", `${d.run_id} (from ${base})`);
      await showRun({ run_id: d.run_id, actions: [], scene_js: d.scene_js, report: d.report }, false);
      refreshRuns();
    } catch (e) {
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "idle" } : x)));
      note("bad", "Edit not applied", e.message);
      setVerdict({ tone: "bad", title: "Edit not applied", detail: e.message });
    } finally {
      setBusy(null);
    }
  };

  const text3d = async (prompt) => {
    setBusy("Building");
    setTimeline([]);
    setTab("activity");
    setRun(null);
    setHasModel(false);
    note("live", `Building “${prompt}”`, "subject → shape builder → colours & sizes → print → verify");
    setVerdict({ tone: "live", title: "Building…", detail: `“${prompt}”` });
    try {
      const d = await api.text3d(prompt);
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "ok" } : x)));
      (d.report?.text_log || []).forEach((l) => note("ok", l));
      note("ok", "Done", d.run_id);
      await showRun({ run_id: d.run_id, actions: [], scene_js: d.scene_js, report: d.report }, false);
      refreshRuns();
    } catch (e) {
      setTimeline((t) => t.map((x) => (x.tone === "live" ? { ...x, tone: "idle" } : x)));
      note(e.status === 422 ? "warn" : "bad", e.status === 422 ? "Not built" : "Failed", e.message);
      setVerdict({ tone: e.status === 422 ? "warn" : "bad", title: e.status === 422 ? "Not built — use Image mode for this subject" : "Build failed", detail: e.message });
    } finally {
      setBusy(null);
    }
  };

  const shownRun = liveRun || run?.run_id || null;
  const src = shownRun ? `/runs/${shownRun}/${liveRun ? "scene" : view}.html?rev=${rev}` : null;

  return (
    <div className="app">
      <TopBar
        health={health}
        healthError={healthError}
        runs={runs}
        currentRun={run?.run_id}
        onOpenRun={openRun}
        onRefresh={() => {
          refreshHealth();
          refreshRuns();
        }}
      />
      <main className="workspace">
        <ErrorBoundary name="Create panel">
          <CreatePanel
            health={health}
            busy={busy}
            canStop={!!jobId}
            mode={mode}
            setMode={setMode}
            h={{ onJob: startJob, onText3d: text3d, onScene: scene, onImage: image, onStop: stop }}
          />
        </ErrorBoundary>
        <ErrorBoundary name="Viewer">
          <Viewer
            src={src}
            verdict={verdict}
            view={view}
            setView={(v) => {
              setView(v);
              setRev((x) => x + 1);
            }}
            hasModel={hasModel && !liveRun}
            busy={busy}
            footer={!liveRun && <EditBar run={run} busy={busy} onEdit={edit} onOpenRun={openRun} />}
          />
        </ErrorBoundary>
        <ErrorBoundary name="Inspector">
          <Inspector timeline={timeline} run={run} tab={tab} setTab={setTab} onOpenRun={openRun} />
        </ErrorBoundary>
      </main>
    </div>
  );
}
