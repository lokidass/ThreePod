# preflight_vram.py
# Ground-truth test: does Qwen2.5-Coder-7B actually load and train in 4-bit on this GPU?
# Loads GPU-ONLY (no CPU offload, so the true footprint is visible), applies LoRA,
# and runs ONE real forward+backward+optimizer step at the target sequence length.
# Prints peak VRAM so we know if training is feasible before committing to a full run.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen2.5-Coder-7B"
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512

def vram():
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9

free0, total0 = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total {total0/1e9:.2f} GB | free right now {free0/1e9:.2f} GB")
print(f"Target sequence length: {SEQ}\n")

gc.collect(); torch.cuda.empty_cache()

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading 7B in 4-bit (GPU-only)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map={"": 0},          # force everything onto GPU 0 — no offload masking
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
except torch.cuda.OutOfMemoryError as e:
    print(f"\n>>> OOM DURING LOAD. 7B 4-bit weights do not fit in free VRAM.")
    print(f">>> {str(e)[:200]}")
    sys.exit(2)

model.config.use_cache = False
a, _ = vram()
print(f"[after load] allocated {a:.2f} GB")

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[after LoRA] allocated {vram()[0]:.2f} GB | trainable params {trainable/1e6:.1f}M")

# one real example at target length
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
d = json.loads(open("qwen_train_clean.jsonl", encoding="utf-8").readline())
user = d["conversations"][0]["value"]; asst = d["conversations"][1]["value"]
text = (f"<|im_start|>system\n{d['system']}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{asst}<|im_end|>\n")
ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ, padding="max_length")
input_ids = ids["input_ids"].to(0)
attn = ids["attention_mask"].to(0)
labels = input_ids.clone()
labels[attn == 0] = -100

import bitsandbytes as bnb_optim
opt = bnb_optim.optim.PagedAdamW8bit(
    [p for p in model.parameters() if p.requires_grad], lr=2e-4
)

print(f"\nRunning one training step at seq={SEQ}, batch=1 ...")
try:
    model.train()
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    out.loss.backward()
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
except torch.cuda.OutOfMemoryError as e:
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n>>> OOM DURING TRAINING STEP at seq={SEQ}. peak before OOM ~{peak:.2f} GB")
    print(f">>> Try a shorter sequence: python preflight_vram.py 384   (or 256)")
    sys.exit(3)

alloc, peak = vram()
free_now, _ = torch.cuda.mem_get_info()
print(f"\n=== RESULT (seq={SEQ}) ===")
print(f"loss: {out.loss.item():.4f}")
print(f"peak allocated during step: {peak:.2f} GB")
print(f"currently free VRAM:        {free_now/1e9:.2f} GB")
print(f">>> FEASIBLE at seq={SEQ}. Headroom: {free_now/1e9:.2f} GB.")
