import { useEffect, useState } from "react";
import { GitBranch, Wand2 } from "lucide-react";
import { api } from "../api";

// Edit the shown object with a prompt. Each edit is re-printed and verified
// as a new version; the original run is never modified.
export default function EditBar({ run, busy, onEdit, onOpenRun }) {
  const [parts, setParts] = useState(null);
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    setParts(null);
    if (!run?.run_id) return;
    let alive = true;
    api.parts(run.run_id)
      .then((p) => alive && setParts(p))
      .catch(() => alive && setParts({ editable: false, roles: [] }));
    return () => {
      alive = false;
    };
  }, [run?.run_id]);

  if (!run || !parts?.editable) return null;

  // natural part names: "wheels", "headlights", "roof"
  const NAMES = { light: "headlight", cabin: "windows", tabletop: "table top", wall: "walls" };
  const name = (r) => {
    const n = NAMES[r] || r;
    return (parts.counts?.[r] || 1) > 1 && !n.endsWith("s") ? `${n}s` : n;
  };
  const roles = (parts.roles || []).filter((r) => r !== "body");
  const ideas = [
    "paint it red",
    "make it 20% taller",
    ...(roles[0] ? [`make the ${name(roles[0])} bigger`] : []),
    ...(roles[1] ? [`paint the ${name(roles[1])} white`] : []),
    ...(roles.length > 2 ? [`remove the ${name(roles[roles.length - 1])}`] : []),
  ].slice(0, 5);

  const submit = () => {
    const p = prompt.trim();
    if (p && !busy) onEdit(p);
  };
  const base = run.report?.base_run;

  return (
    <div className="editbar">
      <div className="editbar-row">
        <Wand2 size={16} className="t-live" />
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder={`Edit this ${parts.category || "object"} — e.g. “make the wheels bigger and paint it blue”`}
          disabled={!!busy}
        />
        <button className="btn btn-primary" disabled={!!busy || !prompt.trim()} onClick={submit}>
          Apply edit
        </button>
      </div>
      <div className="editbar-meta">
        {ideas.map((x) => (
          <button key={x} className="chip-btn" disabled={!!busy} onClick={() => setPrompt(x)}>{x}</button>
        ))}
        {base && (
          <button className="version" onClick={() => onOpenRun(base)} title="Open the version this was edited from">
            <GitBranch size={13} /> edited from {base}
          </button>
        )}
      </div>
    </div>
  );
}
