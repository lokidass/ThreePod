import { useRef } from "react";
import { Box, ExternalLink, Layers, Maximize2, Sparkles } from "lucide-react";
import { Empty, Segmented, ToneIcon } from "./ui";

// view: "model" (realistic presentation) | "scene" (printed / generated scene)
export default function Viewer({ src, verdict, view, setView, hasModel, busy, footer }) {
  const frame = useRef(null);
  return (
    <section className="viewer">
      <div className="viewer-head">
        <div className={`verdict verdict-${verdict.tone}`}>
          <ToneIcon tone={verdict.tone} size={18} />
          <div>
            <div className="verdict-title">{verdict.title}</div>
            {verdict.detail && <div className="verdict-detail">{verdict.detail}</div>}
          </div>
        </div>
        <div className="viewer-actions">
          {hasModel && (
            <Segmented
              size="sm"
              value={view}
              onChange={setView}
              options={[
                { value: "model", label: "Realistic", icon: <Sparkles size={14} /> },
                { value: "scene", label: "Printed", icon: <Layers size={14} /> },
              ]}
            />
          )}
          {src && (
            <>
              <a className="icon-btn" href={src} target="_blank" rel="noopener" title="Open in a new tab">
                <ExternalLink size={16} />
              </a>
              <button className="icon-btn" title="Fullscreen" onClick={() => frame.current?.requestFullscreen?.()}>
                <Maximize2 size={16} />
              </button>
            </>
          )}
        </div>
      </div>
      <div className="viewer-stage">
        {src ? (
          <iframe ref={frame} key={src} src={src} title="3D view" sandbox="allow-scripts" allow="fullscreen" />
        ) : (
          <Empty icon={<Box size={30} />} title="Nothing to show yet">
            Describe an object, drop a photo, or open a previous run.
          </Empty>
        )}
        {busy && (
          <div className="stage-busy">
            <ToneIcon tone="live" size={16} /> {busy}…
          </div>
        )}
      </div>
      {footer}
    </section>
  );
}
