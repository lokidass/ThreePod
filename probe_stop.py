# probe_stop.py
# Diagnostic: does the trained adapter ever want to emit <|im_end|> after the code?
#
# We generate GREEDILY but deliberately DO NOT stop on <|im_end|>, so we can watch,
# at every step, how much probability mass the model puts on the stop token vs. the
# token greedy actually picked. eos is left as <|endoftext|> and min_new_tokens forces
# the run past the natural end of the code block -- neither masks <|im_end|> (151645),
# so its probability is observed untouched.
#
# Verdict:
#   - p(im_end) spikes high (top-1/top-3) right at the end of the code  -> "close but losing"
#     (rare, since greedy would then pick it) or a decoding/config issue -> cheap inference fix.
#   - p(im_end) stays tiny everywhere                                    -> stop token never learned
#     -> retrain (Instruct base and/or EOS reinforcement + more epochs).

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn.functional as F
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

SAMPLE = {"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0],
          "color": "0xFF0000", "label": "ball_a1"}

MAX_NEW = 100     # enough to sail past the natural end of a short block
FORCE_MIN = 90    # force generation past <|im_end|> so we can watch its probability


def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")   # 151645
    EOT = tok.convert_tokens_to_ids("<|endoftext|>")   # 151643
    print(f"IM_END id={IM_END}  EOT id={EOT}")

    print("Loading base 7B (4-bit) + adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{json.dumps(SAMPLE)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(0)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            min_new_tokens=FORCE_MIN,        # masks EOS(=eot) only; im_end left untouched
            do_sample=False,
            repetition_penalty=1.0,          # see the model's TRUE distribution
            eos_token_id=EOT,                # do NOT stop on im_end, so we can observe it
            pad_token_id=EOT,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seq = out.sequences[0]
    gen_ids = seq[inputs["input_ids"].shape[1]:]
    scores = out.scores  # tuple(len=steps) of [1, vocab] logits

    print("\nstep | chosen token          | p(chosen) | p(im_end)  rank | p(eot)")
    print("-" * 78)
    best_imend, best_step = 0.0, -1
    for t, logit in enumerate(scores):
        probs = F.softmax(logit[0].float(), dim=-1)
        chosen = int(gen_ids[t])
        p_chosen = probs[chosen].item()
        p_im = probs[IM_END].item()
        p_eot = probs[EOT].item()
        # rank of im_end (1 = argmax)
        rank_im = int((probs > p_im).sum().item()) + 1
        if p_im > best_imend:
            best_imend, best_step = p_im, t
        piece = tok.decode([chosen]).replace("\n", "\\n")
        mark = "  <== im_end top-5" if rank_im <= 5 else ""
        print(f"{t:4d} | {piece[:20]:20s} | {p_chosen:8.4f} | {p_im:9.6f} {rank_im:5d} | {p_eot:8.6f}{mark}")

    print("-" * 78)
    print(f"max p(im_end) over {len(scores)} steps = {best_imend:.6f} at step {best_step}")
    print("\nFull greedy text (special tokens shown):")
    print(tok.decode(gen_ids, skip_special_tokens=False)[:600])


if __name__ == "__main__":
    main()
