import { useEffect, useRef, useState } from "react";
import { Box, Grid3x3, Image as ImageIcon, Mountain, Play, Sparkles, Square, Type, Upload, X } from "lucide-react";
import { api } from "../api";
import { Field, Segmented, Toggle } from "./ui";

// Modes: "text" | "image" | "scene" | "relief"
const EXAMPLES = ["a red car with four wheels", "a wooden chair", "a blue box with a sphere on top", "a small house with a gable roof"];

// h: { onJob(body, label), onScene(body), onImage(formData, label), onStop() }
export default function CreatePanel({ health, busy, canStop, mode, setMode, h }) {
  return (
    <aside className="panel create">
      <div className="panel-head">
        <h2>Create</h2>
        <p>Every step is validated before it is shown.</p>
      </div>
      <Segmented
        value={mode}
        onChange={setMode}
        options={[
          { value: "text", label: "Text", icon: <Type size={15} /> },
          { value: "image", label: "Image", icon: <ImageIcon size={15} /> },
          { value: "scene", label: "Scene", icon: <Mountain size={15} /> },
          { value: "relief", label: "Relief", icon: <Grid3x3 size={15} /> },
        ]}
      />
      <div className="panel-body">
        {mode === "text" && <TextForm health={health} busy={busy} h={h} />}
        {mode === "image" && <ImageForm busy={busy} h={h} relief={false} />}
        {mode === "relief" && <ImageForm busy={busy} h={h} relief />}
        {mode === "scene" && <SceneForm busy={busy} h={h} />}
      </div>
      {canStop && (
        <div className="panel-foot">
          <button className="btn btn-danger btn-block" onClick={h.onStop}>
            <Square size={14} /> Stop run
          </button>
          <p className="muted small">Committed work is preserved and stays viewable.</p>
        </div>
      )}
    </aside>
  );
}

function TextForm({ health, busy, h }) {
  const [prompt, setPrompt] = useState("a blue box with a sphere on top");
  const [builder, setBuilder] = useState("ollama");
  const [thinker, setThinker] = useState("deepseek-r1:1.5b");
  const [mode, setMode] = useState("realistic"); // "realistic" | "path" | "primitive"
  const [guided, setGuided] = useState(false);
  const [visionEvery, setVisionEvery] = useState(5);
  const ready = !!health?.ready;
  return (
    <div className="form">
      <Field label="Describe the object">
        <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="a wooden table with four legs" />
      </Field>
      <div className="chips">
        {EXAMPLES.map((x) => (
          <button key={x} className="chip-btn" onClick={() => setPrompt(x)}>{x}</button>
        ))}
      </div>
      <div className="grid-2">
        <Field label="Builder">
          <select value={builder} onChange={(e) => setBuilder(e.target.value)}>
            <option value="ollama">qwen2.5-coder:7b</option>
            <option value="lora">LoRA (fine-tuned)</option>
          </select>
        </Field>
        <Field label="Thinker">
          <select value={thinker} onChange={(e) => setThinker(e.target.value)}>
            <option value="deepseek-r1:1.5b">deepseek-r1:1.5b</option>
            <option value="morphix-thinker">morphix-thinker</option>
          </select>
        </Field>
      </div>
      <Field label="Construction">
        <Segmented
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: "realistic", label: "Realistic object", title: "Shape builders + realistic model, editable" },
            { value: "path", label: "LLM print", title: "The thinker plans primitives, printed layer by layer" },
            { value: "primitive", label: "LLM scene" },
          ]}
        />
      </Field>
      {mode === "realistic" && (
        <p className="muted small">Builds vehicles, tables, chairs, stools, houses and four-legged animals from your words
          (colours, materials, sizes). For any other subject use <b>Image</b> mode with a photo.</p>
      )}
      {mode === "path" && (
        <>
          <Toggle checked={guided} onChange={setGuided} label="Guided printing" hint="The builder proposes every move (slow)" />
          <Field label="Vision inspection" hint="Screenshot + vision verdict every N deposits (0 = off)">
            <input type="number" min={0} max={50} value={visionEvery} onChange={(e) => setVisionEvery(+e.target.value)} />
          </Field>
        </>
      )}
      <button
        className="btn btn-primary btn-block btn-lg"
        disabled={!!busy || (mode !== "realistic" && !ready) || !prompt.trim()}
        title={mode !== "realistic" && !ready ? "Models not ready — install them in Ollama" : undefined}
        onClick={() => mode === "realistic" ? h.onText3d(prompt) :
          h.onJob({
            prompt,
            builder,
            builder_model: builder === "lora" ? "" : "qwen2.5-coder:7b",
            thinker_model: thinker,
            mode,
            guided,
            vision_every: visionEvery,
            incremental: mode === "path",
            max_steps: null,
            cap: 24,
          }, mode === "path" ? "Printing" : "Building")
        }
      >
        <Sparkles size={16} /> Generate
      </button>
      {!ready && health && <p className="muted small">Models not ready — install deepseek-r1:1.5b and qwen2.5-coder:7b in Ollama.</p>}
    </div>
  );
}

function Dropzone({ file, setFile }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return undefined;
    }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return (
    <div
      className={`dropzone ${drag ? "drag" : ""} ${file ? "has-file" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f && f.type.startsWith("image/")) setFile(f);
      }}
    >
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => setFile(e.target.files?.[0] || null)} />
      {url ? (
        <>
          <img src={url} alt="reference" />
          <button
            className="icon-btn dz-clear"
            title="Remove image"
            onClick={(e) => {
              e.stopPropagation();
              setFile(null);
            }}
          >
            <X size={14} />
          </button>
          <div className="dz-name">{file?.name}</div>
        </>
      ) : (
        <div className="dz-empty">
          <Upload size={22} />
          <div><b>Drop a photo</b> or click to browse</div>
          <small>PNG, JPG or WEBP · any subject</small>
        </div>
      )}
    </div>
  );
}

function ImageForm({ busy, h, relief }) {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState("");
  const [vision, setVision] = useState("");
  const [env, setEnv] = useState("object-only"); // "object-only" | "preserve"
  const [grid, setGrid] = useState("12");
  const [rmode, setRmode] = useState("relief");
  const [describe, setDescribe] = useState(false);
  return (
    <div className="form">
      <Dropzone file={file} setFile={setFile} />
      {!relief ? (
        <>
          <div className="grid-2">
            <Field label="Subject" hint="Optional — any word">
              <input value={subject} onChange={(e) => setSubject(e.target.value)} placeholder="auto-detect" />
            </Field>
            <Field label="Vision model" hint="Optional">
              <input value={vision} onChange={(e) => setVision(e.target.value)} placeholder="auto" />
            </Field>
          </div>
          <Field label="Environment">
            <Segmented
              size="sm"
              value={env}
              onChange={setEnv}
              options={[
                { value: "object-only", label: "Object only" },
                { value: "preserve", label: "Keep platform / floor", title: "Display-only context, never printed" },
              ]}
            />
          </Field>
          <button
            className="btn btn-primary btn-block btn-lg"
            disabled={!!busy || !file}
            onClick={() => {
              const fd = new FormData();
              fd.append("image", file);
              fd.append("mode", "reconstruct");
              fd.append("environment", env);
              if (subject.trim()) fd.append("category", subject.trim());
              if (vision.trim()) fd.append("vision_model", vision.trim());
              h.onImage(fd, "Reconstructing");
            }}
          >
            <Box size={16} /> Reconstruct in 3D
          </button>
          <ol className="pipeline">
            <li>Segment the subject from its background</li>
            <li>Identify it (any subject — no fixed list)</li>
            <li>Fit shape, print layer by layer</li>
            <li>Render &amp; measure against the photo</li>
          </ol>
        </>
      ) : (
        <>
          <div className="grid-2">
            <Field label="Grid">
              <select value={grid} onChange={(e) => setGrid(e.target.value)}>
                <option value="8">8 × 8</option>
                <option value="12">12 × 12</option>
                <option value="16">16 × 16</option>
                <option value="24">24 × 24</option>
              </select>
            </Field>
            <Field label="Style">
              <select value={rmode} onChange={(e) => setRmode(e.target.value)}>
                <option value="relief">Relief</option>
                <option value="mosaic">Mosaic</option>
              </select>
            </Field>
          </div>
          <Toggle checked={describe} onChange={setDescribe} label="Describe with vision model" />
          <button
            className="btn btn-primary btn-block btn-lg"
            disabled={!!busy || !file}
            onClick={() => {
              const fd = new FormData();
              fd.append("image", file);
              fd.append("grid", grid);
              fd.append("mode", rmode);
              if (describe) fd.append("describe", "1");
              h.onImage(fd, "Printing relief");
            }}
          >
            <Play size={16} /> Print relief
          </button>
          <p className="muted small">Pixels become a printer-ordered heightmap — a relief, not a full 3D object.</p>
        </>
      )}
    </div>
  );
}

function SceneForm({ busy, h }) {
  const [templates, setTemplates] = useState([]);
  const [tpl, setTpl] = useState("");
  const [seed, setSeed] = useState(11);
  const [prompt, setPrompt] = useState("");
  useEffect(() => {
    api.sceneTemplates()
      .then((d) => {
        setTemplates(d.templates.filter((t) => !t.error));
        setTpl((v) => v || d.templates[0]?.name || "");
      })
      .catch(() => setTemplates([]));
  }, []);
  return (
    <div className="form">
      <Field label="Template">
        <select value={tpl} onChange={(e) => setTpl(e.target.value)}>
          {templates.length === 0 && <option value="">Loading catalogue…</option>}
          {templates.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name} — {t.nodes} nodes{t.instances ? `, ${t.instances} instances` : ""}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Or describe a scene" hint="Uses the thinker to pick tested systems">
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="misty forest at dusk" />
      </Field>
      <Field label="Seed">
        <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
      </Field>
      <button
        className="btn btn-primary btn-block btn-lg"
        disabled={!!busy || (!tpl && !prompt.trim())}
        onClick={() => h.onScene(prompt.trim() ? { prompt: prompt.trim(), seed } : { template: tpl, seed })}
      >
        <Mountain size={16} /> Generate environment
      </button>
      <p className="muted small">Rich WebGL worlds (fog, instancing, water, animation) on three r152 — never sent to the printer.</p>
    </div>
  );
}
