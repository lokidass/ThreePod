import { useEffect, useRef, useState } from "react";
import { ChevronDown, Clock, History, RefreshCw } from "lucide-react";

function runTime(id) {
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(id);
  if (!m) return id;
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function Pill({ ok, label, value }) {
  return (
    <span className={`pill ${ok ? "pill-ok" : "pill-bad"}`} title={`${label}: ${value}`}>
      <span className="dot" />
      <span className="pill-label">{label}</span>
      <span className="pill-value">{value}</span>
    </span>
  );
}

export default function TopBar({ health, healthError, runs, currentRun, onOpenRun, onRefresh }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="logo">M</div>
        <div>
          <div className="brand-name">Morphix Studio</div>
          <div className="brand-sub">validated 3D generation</div>
        </div>
      </div>

      <div className="topbar-status">
        {healthError && <Pill ok={false} label="server" value="unreachable" />}
        {health && (
          <>
            <Pill ok={health.thinker_ready} label="thinker" value={health.thinker_model} />
            <Pill ok={health.builder_ready} label="builder" value={health.builder_model} />
            <Pill ok={health.vision_ready} label="vision" value={health.vision_model || "none"} />
            {health.strict && <span className="strict">strict · real models</span>}
          </>
        )}
        <button className="icon-btn" title="Refresh status" onClick={onRefresh}>
          <RefreshCw size={15} />
        </button>
      </div>

      <div className="runs" ref={ref}>
        <button className="btn btn-ghost" onClick={() => setOpen((v) => !v)}>
          <History size={16} />
          <span>{currentRun ? runTime(currentRun) : "Runs"}</span>
          <ChevronDown size={14} />
        </button>
        {open && (
          <div className="runs-menu">
            <div className="runs-head">Recent runs</div>
            {runs.length === 0 && <div className="runs-empty">No runs yet</div>}
            {runs.slice(0, 60).map((id) => (
              <button
                key={id}
                className={`runs-item ${id === currentRun ? "active" : ""}`}
                onClick={() => {
                  setOpen(false);
                  onOpenRun(id);
                }}
              >
                <Clock size={13} />
                <span>{runTime(id)}</span>
                <code>{id}</code>
              </button>
            ))}
          </div>
        )}
      </div>
    </header>
  );
}
