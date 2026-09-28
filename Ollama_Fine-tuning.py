"""
Morphix — Qwen2.5-Coder 1.5B Fine-tune
Optimized for RTX 4050 6GB VRAM laptop GPU.

Setup:
  pip install unsloth transformers datasets trl accelerate bitsandbytes -q

Run:
  python train_qwen_4050.py

Estimated:
  - VRAM usage  : ~4.8GB peak (safe on 6GB)
  - Time/epoch  : ~2-3 hrs on 4050 for 10k samples
  - Total time  : ~6-9 hrs for 3 epochs
"""

import os, json, torch, gc
from pathlib import Path

# ── CONFIG — tuned for 6GB VRAM ───────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # fits in 6GB at 4-bit
OUTPUT_DIR    = "./morphix-qwen-adapter"
MERGED_DIR    = "./morphix-qwen-merged"
DATASET_DIR   = "../dataset"

MAX_SEQ_LEN   = 512    # keep low — saves VRAM dramatically
BATCH_SIZE    = 1      # must be 1 on 6GB
GRAD_ACCUM    = 16     # effective batch = 16
EPOCHS        = 3
LR            = 2e-4
LORA_RANK     = 8      # lower rank = less VRAM
LORA_ALPHA    = 16
WARMUP_STEPS  = 50
LOG_STEPS     = 10
EVAL_STEPS    = 100
SAVE_STEPS    = 100
MAX_SAMPLES   = 10000  # start with 10k, increase if stable
# ─────────────────────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Make sure CUDA drivers are installed.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {name}")
    print(f"VRAM: {vram:.1f} GB")
    if vram < 5.5:
        print("WARNING: Less than 5.5GB VRAM detected. Reduce MAX_SEQ_LEN to 256.")
    return vram

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_model():
    from unsloth import FastLanguageModel

    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,   # float16 not bfloat16 — better on 4050
        load_in_4bit=True,
        # These reduce VRAM further:
        rope_scaling=None,
        fix_tokenizer=True,
    )

    print("Attaching LoRA adapters (rank=8 for 6GB)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        # Only train the most impactful layers to save VRAM
        target_modules=[
            "q_proj", "v_proj",          # attention
            "gate_proj", "up_proj",       # FFN
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30% VRAM
        random_state=42,
        use_rslora=True,   # rank-stabilized LoRA — better quality at low rank
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.1f}%)")
    return model, tokenizer

def format_conversation(sample):
    """
    Convert conversations to Qwen2.5 chat format.
    Qwen uses: <|im_start|>role\ncontent<|im_end|>\n
    """
    convs = sample.get("conversations", [])
    text = ""
    for turn in convs:
        role    = turn["role"]
        content = turn["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}

def load_dataset_split(path, max_samples=None):
    from datasets import load_dataset as hf_load

    ds = hf_load("json", data_files=str(path), split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Filter out samples that are too long (saves VRAM on long chain sequences)
    # We'll estimate by character count first (faster than tokenizing)
    ds = ds.filter(lambda s: sum(len(t["content"]) for t in s["conversations"]) < MAX_SEQ_LEN * 3)

    ds = ds.map(format_conversation, remove_columns=ds.column_names)
    return ds

def load_data():
    train_path = Path(DATASET_DIR) / "morphix_train.jsonl"
    val_path   = Path(DATASET_DIR) / "morphix_val.jsonl"

    # If full dataset not generated yet, check for any .jsonl
    if not train_path.exists():
        jsonl_files = list(Path(DATASET_DIR).glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No dataset found in {DATASET_DIR}.\n"
                "Run: python dataset/generate_dataset.py first."
            )
        # Use first jsonl found, split manually
        print(f"Using {jsonl_files[0]} and splitting 90/10...")
        from datasets import load_dataset as hf_load
        ds = hf_load("json", data_files=str(jsonl_files[0]), split="train")
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds   = split["test"]
    else:
        train_ds = load_dataset_split(train_path, max_samples=MAX_SAMPLES)
        val_ds   = load_dataset_split(val_path,   max_samples=MAX_SAMPLES // 10)

    train_ds = train_ds.map(lambda s: format_conversation(s) if "text" not in s else s)
    val_ds   = val_ds.map(lambda s: format_conversation(s) if "text" not in s else s)

    print(f"Train: {len(train_ds):,} samples")
    print(f"Val  : {len(val_ds):,} samples")
    return train_ds, val_ds

def get_training_args():
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,

        # Batch settings — critical for 6GB
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Memory optimizations
        gradient_checkpointing=True,
        optim="adamw_8bit",        # 8-bit optimizer — saves ~1GB VRAM
        fp16=True,                 # float16 on 4050
        bf16=False,

        # LR schedule
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,

        # Logging & saving
        logging_steps=LOG_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Speed
        dataloader_num_workers=0,   # 0 on Windows laptop
        group_by_length=True,       # groups similar lengths → less padding → faster
        ddp_find_unused_parameters=False,

        # Disable things that waste VRAM
        report_to="none",
        push_to_hub=False,

        # Resume if interrupted
        resume_from_checkpoint=True if os.path.exists(OUTPUT_DIR) else False,
    )

def train(model, tokenizer, train_ds, val_ds):
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=get_training_args(),
        packing=True,       # packs multiple short samples into one sequence → faster
        dataset_num_proc=1, # low on laptop
    )

    # Print VRAM before training
    free_vram()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved  = torch.cuda.memory_reserved(0) / 1e9
    print(f"\nVRAM before training: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved")
    print("Starting training...\n")

    trainer_stats = trainer.train()

    print(f"\nTraining complete.")
    print(f"Runtime      : {trainer_stats.metrics['train_runtime']/3600:.1f} hours")
    print(f"Samples/sec  : {trainer_stats.metrics['train_samples_per_second']:.1f}")
    print(f"Final loss   : {trainer_stats.metrics['train_loss']:.4f}")
    return trainer

def save_adapter(model, tokenizer):
    print(f"\nSaving LoRA adapter → {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Adapter saved.")

def merge_and_export(model, tokenizer):
    print(f"\nMerging adapter into base model → {MERGED_DIR}")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved.")

    print("\nExporting GGUF Q4_K_M for Ollama...")
    model.save_pretrained_gguf(
        MERGED_DIR,
        tokenizer,
        quantization_method="q4_k_m",
    )
    gguf_path = Path(MERGED_DIR) / "model-Q4_K_M.gguf"
    print(f"GGUF saved → {gguf_path}")
    print("\nRun: ollama create morphix-builder -f Modelfile")

def main():
    print("=" * 60)
    print("  Morphix Builder — Qwen2.5-Coder Fine-tune")
    print("  Target: RTX 4050 6GB VRAM")
    print("=" * 60)

    vram = check_gpu()
    model, tokenizer = load_model()

    free_vram()

    train_ds, val_ds = load_data()
    trainer = train(model, tokenizer, train_ds, val_ds)

    save_adapter(model, tokenizer)
    merge_and_export(model, tokenizer)

    print("\n" + "=" * 60)
    print("  Done! Next: run scripts/setup_ollama.sh")
    print("=" * 60)

if __name__ == "__main__":
    main()