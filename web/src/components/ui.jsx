import { AlertTriangle, CheckCircle2, CircleDashed, Info, Loader2, XCircle } from "lucide-react";

// Tones used across the UI: ok | warn | bad | info | live | idle.
export function ToneIcon({ tone, size = 16 }) {
  switch (tone) {
    case "ok":
      return <CheckCircle2 size={size} className="t-ok" />;
    case "warn":
      return <AlertTriangle size={size} className="t-warn" />;
    case "bad":
      return <XCircle size={size} className="t-bad" />;
    case "live":
      return <Loader2 size={size} className="t-live spin" />;
    case "info":
      return <Info size={size} className="t-info" />;
    default:
      return <CircleDashed size={size} className="t-idle" />;
  }
}

export function Badge({ tone, children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function Segmented({ value, options, onChange, size = "md" }) {
  return (
    <div className={`segmented segmented-${size}`} role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          title={o.title}
          aria-selected={o.value === value}
          className={o.value === value ? "active" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.icon}
          <span>{o.label}</span>
        </button>
      ))}
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function Toggle({ checked, onChange, label, hint }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track"><span className="toggle-thumb" /></span>
      <span className="toggle-text">
        <span>{label}</span>
        {hint && <small>{hint}</small>}
      </span>
    </label>
  );
}

export function Stat({ label, value, tone }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${tone ? `t-${tone}` : ""}`}>{value}</span>
    </div>
  );
}

export function Empty({ icon, title, children }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}
