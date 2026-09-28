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

BASE = "Qwen/Qwen2.5-Coder-7B"
ADAPTER = "./qwen7b_qlora_morphix/final_adapter"

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


def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    print("Loading base 7B (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
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
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,               # deterministic for a code task
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
                # stop on <|im_end|> (chat turn end) OR the real EOS <|endoftext|>
                eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id],
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print("ACTION:", json.dumps(act))
        print("-" * 70)
        print(gen.strip())


if __name__ == "__main__":
    main()
