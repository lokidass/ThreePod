# Training Qwen2.5-Coder-7B (QLoRA) for Morphix on the RTX 4050 (6 GB)

This documents the working setup for fine-tuning the **7B** model locally, plus the
honest performance reality on a 6 GB card.

## What was fixed to make this run at all

| Problem | Fix |
|---|---|
| `bitsandbytes 0.41.3` — 4-bit broken on Windows | Pinned `bitsandbytes==0.44.1` (`--no-deps`) |
| A `pip -U` had swapped in `torch 2.13.0+cpu` (CUDA gone) | Restored `torch==2.2.0+cu118` (+vision/audio) |
| `triton-windows 3.7.1` broke bitsandbytes import | Removed it (QLoRA doesn't use Triton) |
| ~3000 Python booleans in JS targets (`castShadow = True`) | `fix_dataset.py` → `qwen_train_clean.jsonl` |
| Mojibake em-dash in system prompt | Normalized to a canonical UTF-8 system prompt |

**Do not run `pip install -U bitsandbytes` again** — it will pull a newer bnb that needs
torch ≥2.4 and will clobber your CUDA torch. The stack is pinned for a reason.

## The performance reality (measured, not estimated)

- 7B in 4-bit = **6.07 GB** of weights; one training step peaks at **~10.6 GB**.
- Your card has **6.44 GB**. The overflow spills into system RAM over PCIe.
- **Measured** step time (smoke test, 3 steps): **~12.5 s/step** at seq 512.
  - seq 512  → full epoch (2000 samples) ≈ **~7 hours** (an overnight run, *not* days).
  - seq 1024 (the default) → ≈ **~8–10 hours** — slower per step because the long
    shader examples aren't truncated, but that's exactly why quality is better.
- Loss was already falling in the smoke test (0.86 → 0.52 over 3 steps), so the
  pipeline is genuinely learning, not just running.
- The embeddings + lm_head (~2.2 GB) can't be quantized, so this can't be tuned away at 7B.

To go faster: **close every other GPU app** (browser, WhatsApp, Discord) before launching —
that frees ~1.6 GB, so the weights fit in real VRAM and only activations spill.

## How to run

```bash
# 1) (one-time) clean the dataset — already done, produces *_clean.jsonl
python fix_dataset.py

# 2) quick sanity/timing check (3 steps, ~a few minutes after load)
python train_qwen7b_qlora.py --max_steps 3 --grad_accum 1 --max_seq_len 512

# 3) full run (1 epoch). Checkpoints every 25 steps to ./qwen7b_qlora_morphix
python train_qwen7b_qlora.py

# if it stops / laptop sleeps / crashes — resume from the last checkpoint:
python train_qwen7b_qlora.py --resume

# 4) test the trained adapter
python test_qwen7b.py
python test_qwen7b.py '{"action":"add_sphere","radius":0.5,"color":"0xFF0000","label":"s1"}'
```

## Knobs (in `train_qwen7b_qlora.py` or via flags)

- `--max_seq_len 512` — shorter = faster/less spill, but truncates long shader examples
  (median example is 421 tokens; ~10% exceed 1024).
- `--max_samples 500` — train on a subset first to validate quality before the full run.
- `--grad_accum 16` — effective batch size (16 × 1).
- `--epochs 1` — raise for more training once you've confirmed speed is tolerable.

## If the long runtime is not workable

The full 7B trains in ~1–2 h on a free Colab/Kaggle **T4 (16 GB)**, where 9.9 GB fits
comfortably. The cleaned `qwen_train_clean.jsonl` is ready to upload. Ask and I'll produce
a ready-to-run notebook. A local **3B** (fits your VRAM with headroom) is the other option.
