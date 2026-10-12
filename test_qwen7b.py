# test_qwen7b.py
# Run the fine-tuned Morphix adapter: action JSON in -> Three.js JavaScript out.
# Loads the cached 7B base in 4-bit + your trained LoRA adapter (no merge needed).
#
# Usage:
#   python test_qwen7b.py                        # runs a few built-in sample actions
#   python test_qwen7b.py '{"action":"add_sphere","radius":0.5,"color":"0xFF0000","label":"s1"}'

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from morphix_guard import enforce

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"  # default for newly trained adapters (see train_qwen7b_qlora.py)
ADAPTER = "./qwen7b_qlora_morphix/final_adapter"


def resolve_base():
    """Use the base recorded in the trained adapter when present.

    The old final_adapter was trained on Qwen2.5-Coder-7B (non-Instruct);
    forcing Instruct underneath it would mismatch. New retrains use Instruct.
    """
    import os as _os
    cfg = _os.path.join(ADAPTER, "adapter_config.json")
    try:
        if _os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                base = json.load(f).get("base_model_name_or_path")
            if base:
                return base
    except Exception:
        pass
    return BASE

SYSTEM = (
    "You are Morphix Builder — a Three.js 3D printer agent.\n"
    "You receive ONE action JSON per tick describing a single atomic operation.\n"
    "Output ONLY raw JavaScript. No markdown, no backticks, no explanation, no comments.\n"
    "\n"
    "Rules:\n"
    "- Each output patches the EXISTING scene — never rebuild from scratch.\n"
    "- Use unique variable names based on the label field to avoid collisions.\n"
    "- Always use MeshStandardMaterial or MeshPhysicalMaterial for realism.\n"
    "- Set castShadow = true and receiveShadow = true on every mesh.\n"
    "- Code must execute inside a Three.js r152 scene with renderer, camera, scene already defined.\n"
    "- morphix_mixers, morphix_shaders, morphix_physics are global arrays for tracking.\n"
    "- Never output anything except valid executable JavaScript."
)

SAMPLES = [
    {"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0], "color": "0xFF0000", "label": "ball_a1"},
    {"action": "add_shader", "preset": "fresnel", "color": "0x00ffcc", "intensity": 2.0, "label": "glow_b2"},
]


def clean_js(text: str) -> str:
    """Truncate run-on generation to the last valid JS terminator.

    The adapter (125 steps on the non-Instruct base) learned the code pattern
    but not a strong <|im_end|> stop, so greedy decoding runs past the end
    into degenerate loops (NdrFc / {lng...). Every training example ends with
    scene.add(...); or morphix_*.push(...); — cut there.
    """
    import re
    # cut embedded chat turn markers if they leak through
    for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker)[0]
    # find last valid statement end
    ends = [m.end() for m in re.finditer(
        r"(?:scene\.add\([^;]*\);|morphix_\w+\.push\([^;]*\);)", text)]
    if ends:
        text = text[:ends[-1]]
    # drop a trailing partial line (e.g. "const geo_b2 =") left by the cut
    lines = text.rstrip().split("\n")
    if lines and not lines[-1].rstrip().endswith((";", "});", "}", "`);")):
        # keep partial only if it is a complete single-statement line
        if len(lines[-1].strip()) < 30 or "=" in lines[-1][-20:]:
            lines = lines[:-1]
    return "\n".join(lines).strip()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = resolve_base()
    print(f"Base: {base}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    print("Loading base 7B (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if os.path.isdir(ADAPTER):
        print(f"Attaching adapter: {ADAPTER}")
        model = PeftModel.from_pretrained(model, ADAPTER)
    else:
        print(f"⚠️  No adapter at {ADAPTER} — running the BASE model (untrained).")
    model.eval()

    actions = [json.loads(sys.argv[1])] if len(sys.argv) > 1 else SAMPLES

    for act in actions:
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{json.dumps(act)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tok(prompt, return_tensors="pt").to(0)
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        # shaders/post/animation are long blocks (~350 tokens); geometry is short
        budget = 384 if act.get("action") in (
            "add_shader", "add_post", "add_animation", "add_particles") else 256
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,               # deterministic for a code task
                repetition_penalty=1.15,
                no_repeat_ngram_size=8,        # kills the NdrFc / {lng... loops
                pad_token_id=tok.eos_token_id,
                # stop on <|im_end|> (chat turn end) OR the real EOS <|endoftext|>
                eos_token_id=[im_end_id, tok.eos_token_id],
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = clean_js(gen)
        final, status, errors = enforce(act, gen)
        print("\n" + "=" * 70)
        print("ACTION:", json.dumps(act))
        print(f"STATUS: {status}" + (f" ({'; '.join(errors)})" if errors else ""))
        print("-" * 70)
        print(final.strip())


if __name__ == "__main__":
    main()
