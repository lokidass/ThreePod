# morphix_lora_worker.py
# Persistent LoRA inference worker: loads the user's fine-tuned Qwen 7B +
# adapter ONCE, then serves build requests over stdio as newline-JSON.
#
# Isolation rationale: bitsandbytes/transformers native code can segfault the
# process (observed exit -1073741819 on this 6GB card). In a child process a
# crash kills only the worker; the parent detects the death, reports the true
# cause, and the Flask server keeps serving. Never import this module into the
# server process (it pulls torch/CUDA into the importer).
#
# Protocol (all lines UTF-8 JSON, flushed):
#   parent -> child: {"id": 1, "prompt": "<full chat prompt>",
#                     "max_new_tokens": 256}\n
#   child  -> parent: {"id": 1, "code": "<raw decoded text>"}\n
#   child  -> parent: {"id": 1, "error": "<WhatFailed>: ..."}\n
# Child prints MORPHIX_LORA_READY after load. EOF on stdin exits quietly.

import json
import os
import sys


def _load():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_qwen7b as T
    base = T.resolve_base()
    # A 6 GB card cannot hold 7B-nf4 + staging buffers next to the desktop's
    # own VRAM use: bitsandbytes then segfaults instead of raising OOM. Cap
    # GPU memory (MORPHIX_LORA_GPU_GB, default: free VRAM minus a margin) and
    # let accelerate place the overflow layers in system RAM.
    free_gb = torch.cuda.mem_get_info(0)[0] / 2 ** 30
    gpu_gb = float(os.environ.get("MORPHIX_LORA_GPU_GB",
                                  max(1.5, round(free_gb - 1.2, 1))))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True)
    print(f"[lora-worker] loading {base} + adapter (GPU cap {gpu_gb} GiB of "
          f"{free_gb:.1f} free) ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        max_memory={0: f"{gpu_gb}GiB", "cpu": "14GiB"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[lora-worker] placement: "
          f"{sorted(set(str(v) for v in model.hf_device_map.values()))}",
          file=sys.stderr, flush=True)
    if os.path.isdir(T.ADAPTER):
        model = PeftModel.from_pretrained(model, T.ADAPTER)
    model.eval()
    return tok, model, torch, tok.convert_tokens_to_ids("<|im_end|>")


def main():
    tok, model, torch, im_end = _load()
    print("MORPHIX_LORA_READY", flush=True)
    rid = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"id": rid, "error": f"bad request: {e}"}),
                  flush=True)
            continue
        rid = req.get("id", 0)
        try:
            inputs = tok(req["prompt"], return_tensors="pt").to(0)
            budget = int(req.get("max_new_tokens", 256))
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=budget, do_sample=False,
                    repetition_penalty=1.15, no_repeat_ngram_size=8,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=[im_end, tok.eos_token_id])
            code = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            print(json.dumps({"id": rid, "code": code}), flush=True)
        except Exception as e:
            print(json.dumps({"id": rid,
                              "error": f"{type(e).__name__}: {e}"}),
                  flush=True)


if __name__ == "__main__":
    main()
