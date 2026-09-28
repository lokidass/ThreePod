# train_qwen7b_qlora.py
# QLoRA fine-tune of Qwen2.5-Coder-7B for the Morphix Three.js agent, on a 6 GB RTX 4050.
#
# HONEST NOTE: 7B needs ~9.9 GB peak; this card has 6.44 GB. It runs only because Windows
# spills the overflow into system RAM over PCIe, so steps are SLOW (expect a long run).
# This script is tuned to minimize that spillover and to checkpoint often so you can
# stop/resume safely. Close other GPU apps (browser, WhatsApp) before launching.
#
# Usage:
#   python train_qwen7b_qlora.py                     # full run, 1 epoch
#   python train_qwen7b_qlora.py --max_steps 4 --grad_accum 1   # quick timing smoke test
#   python train_qwen7b_qlora.py --resume            # resume from last checkpoint
#   python train_qwen7b_qlora.py --max_seq_len 512   # cheaper/faster steps, more truncation

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import gc
import time
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------- config
CONFIG = {
    "model_name": "Qwen/Qwen2.5-Coder-7B",   # already cached locally; keep as-is
    "train_file": "qwen_train_clean.jsonl",
    "val_file": "qwen_val_clean.jsonl",
    "output_dir": "./qwen7b_qlora_morphix",
    "max_seq_len": 1024,     # covers ~90% of examples fully; dynamic padding keeps short ones cheap
    "epochs": 1,
    "batch_size": 1,
    "grad_accum": 16,        # effective batch = 16
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "save_steps": 25,        # frequent -> safe stop/resume on a long run
    "logging_steps": 2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps", type=int, default=-1, help="cap optimizer steps (for smoke test)")
    p.add_argument("--grad_accum", type=int, default=CONFIG["grad_accum"])
    p.add_argument("--max_seq_len", type=int, default=CONFIG["max_seq_len"])
    p.add_argument("--epochs", type=float, default=CONFIG["epochs"])
    p.add_argument("--max_samples", type=int, default=-1, help="limit training samples")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    return p.parse_args()


# ---------------------------------------------------------------- data
def load_examples(path, tok, max_len, max_samples=-1):
    """Build input_ids/labels with completion-only loss masking (prompt tokens = -100)."""
    kept, dropped_no_completion = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            system = d.get("system", "")
            user = asst = None
            for m in d["conversations"]:
                if m["from"] == "user":
                    user = m["value"]
                elif m["from"] == "assistant":
                    asst = m["value"]
            if user is None or asst is None:
                continue

            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            completion = f"{asst}<|im_end|>"

            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]

            input_ids = (p_ids + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]

            # if truncation ate the whole completion, the example teaches nothing -> drop
            if all(t == -100 for t in labels):
                dropped_no_completion += 1
                continue

            kept.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            })
    print(f"  {path}: {len(kept)} usable examples ({dropped_no_completion} dropped: prompt exceeded max_len)")
    return Dataset.from_list(kept)


# ---------------------------------------------------------------- timing callback
class StepTimer(TrainerCallback):
    def __init__(self):
        self.t0 = None
        self.count = 0
    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.t0 is not None:
            dt = now - self.t0
            self.count += 1
            if self.count <= 5 or state.global_step % 10 == 0:
                free, total = torch.cuda.mem_get_info()
                print(f"    step {state.global_step}: {dt:.1f}s/step | "
                      f"VRAM free {free/1e9:.2f} GB | peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        self.t0 = now


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA not available"

    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)} | free {free/1e9:.2f} / {total/1e9:.2f} GB")
    if free / 1e9 < 5.0:
        print("  ⚠️  <5 GB free. Close browser/WhatsApp/etc. to reduce RAM-spillover and speed up steps.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gc.collect(); torch.cuda.empty_cache()

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Building datasets (completion-only masking)...")
    train_ds = load_examples(CONFIG["train_file"], tok, args.max_seq_len, args.max_samples)
    eval_ds = load_examples(CONFIG["val_file"], tok, args.max_seq_len, max_samples=64)

    print("Loading 7B in 4-bit NF4 (double-quant)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tok, padding=True, label_pad_token_id=-100, return_tensors="pt"
    )

    targs = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=CONFIG["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",          # optimizer states paged to CPU RAM (off the GPU)
        max_grad_norm=0.3,
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,          # safest on Windows
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[StepTimer()],
    )

    n_steps = (len(train_ds) * args.epochs) / args.grad_accum if args.max_steps < 0 else args.max_steps
    print(f"\nStarting training. ~{int(n_steps)} optimizer steps "
          f"({len(train_ds)} samples, grad_accum {args.grad_accum}, seq<= {args.max_seq_len}).")
    print("Watch the first few 's/step' lines to estimate total time.\n")

    trainer.train(resume_from_checkpoint=args.resume)

    print("Saving LoRA adapter + tokenizer...")
    trainer.model.save_pretrained(os.path.join(CONFIG["output_dir"], "final_adapter"))
    tok.save_pretrained(os.path.join(CONFIG["output_dir"], "final_adapter"))
    print(f"Done. Adapter at {CONFIG['output_dir']}/final_adapter")


if __name__ == "__main__":
    main()
