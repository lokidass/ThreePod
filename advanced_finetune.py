# train_qwen_25_nobnb.py
# Complete Qwen 2.5 Coder training WITHOUT bitsandbytes

import json
import os
import gc
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

print("🔍 Checking GPU...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA not available!")
    exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION - Qwen 2.5 Coder 1.5B (fits in 6GB)
# ─────────────────────────────────────────────────────────

CONFIG = {
    "model_name": "Qwen/Qwen2.5-Coder-1.5B",  # 1.5B fits in 6GB VRAM
    "dataset_path": "qwen_train.jsonl",
    "output_dir": "./qwen_25_nobnb_model",
    "epochs": 1,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "lora_r": 8,
    "lora_alpha": 16,
    "max_samples": 1000,
}

# ─────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────

def load_qwen_dataset(path, max_samples=None):
    print("📚 Loading dataset...")
    
    if not os.path.exists(path):
        print(f"❌ Dataset not found: {path}")
        return None
    
    data = []
    
    # Try multiple encodings
    encodings = ['cp1252', 'latin-1', 'utf-8']
    
    for encoding in encodings:
        try:
            print(f"  Trying encoding: {encoding}")
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        item = json.loads(line)
                        text = ""
                        if item.get('system'):
                            text += f"<|im_start|>system\n{item['system']}<|im_end|>\n"
                        for msg in item['conversations']:
                            if msg['from'] == 'user':
                                text += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
                            elif msg['from'] == 'assistant':
                                text += f"<|im_start|>assistant\n{msg['value']}<|im_end|>\n"
                        data.append({"text": text})
                    except:
                        continue
            
            if data:
                print(f"✅ Successfully loaded {len(data)} samples using {encoding}")
                return Dataset.from_list(data)
                
        except:
            continue
    
    if len(data) == 0:
        print("❌ No valid data found")
        return None
    
    return Dataset.from_list(data)

dataset = load_qwen_dataset(CONFIG['dataset_path'], max_samples=CONFIG['max_samples'])

if dataset is None:
    print("❌ Failed to load dataset. Exiting.")
    exit(1)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ─────────────────────────────────────────────────────────────
# TOKENIZE
# ─────────────────────────────────────────────────────────

print("🔤 Loading tokenizer...")

# Try loading with trust_remote_code
try:
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        trust_remote_code=True,
        padding_side="right",
    )
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"⚠️ Failed: {e}")
    print("Trying instruct version...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            trust_remote_code=True,
            padding_side="right",
        )
        CONFIG['model_name'] = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        print("✅ Tokenizer loaded from instruct version")
    except Exception as e2:
        print(f"❌ All tokenizer attempts failed: {e2}")
        exit(1)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=CONFIG['max_seq_length'],
    )

print("🔤 Tokenizing dataset...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# ─────────────────────────────────────────────────────────────
# LOAD MODEL - Full precision (no bitsandbytes)
# ─────────────────────────────────────────────────────────

print(f"🔄 Loading {CONFIG['model_name']}...")

gc.collect()
torch.cuda.empty_cache()

# Load with float16 (full precision, no quantization)
try:
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────
# SETUP LORA
# ─────────────────────────────────────────────────────────

print("🔄 Setting up LoRA...")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

print("🏋️ Starting training...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    warmup_steps=20,
    learning_rate=CONFIG['learning_rate'],
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    bf16=False,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n💾 GPU Memory before training:")
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

try:
    trainer.train()
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ─────────────────────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────────────────

print("💾 Saving model...")
os.makedirs(CONFIG['output_dir'], exist_ok=True)
model.save_pretrained(f"{CONFIG['output_dir']}/lora_adapter")
tokenizer.save_pretrained(f"{CONFIG['output_dir']}/tokenizer")

print(f"✅ Model saved to {CONFIG['output_dir']}")

# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────

print("🧪 Testing the model...")

def test_generation(prompt):
    test_prompt = f"""<|im_start|>system
You are Morphix Builder — a Three.js 3D printer agent.
Output ONLY raw JavaScript. No markdown, no backticks, no explanation.

<|im_start|>user
{json.dumps(prompt)}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    return response

test_prompt = {"action": "add_sphere", "radius": 0.5, "position": [0, 1, 0], "color": "0xFF0000", "label": "test"}
print(f"Input: {test_prompt}")
try:
    result = test_generation(test_prompt)
    print("Generated code:")
    print(result[:500])
except Exception as e:
    print(f"Test failed: {e}")

print("\n✅ Training Complete!")

# ─────────────────────────────────────────────────────────────
# CREATE OLLAMA MODELFIL
# ─────────────────────────────────────────────────────────

system_prompt = """You are Morphix Builder — a Three.js 3D printer agent.
You receive ONE action JSON per tick describing a single atomic operation.
Output ONLY raw JavaScript. No markdown, no backticks, no explanation, no comments.

Rules:
- Each output patches the EXISTING scene — never rebuild from scratch.
- Use unique variable names based on the label field to avoid collisions.
- Always use MeshStandardMaterial or MeshPhysicalMaterial for realism.
- Set castShadow = true and receiveShadow = true on every mesh.
- Code must execute inside a Three.js r152 scene with renderer, camera, scene already defined.
- morphix_mixers, morphix_shaders, morphix_physics are global arrays for tracking.
- Never output anything except valid executable JavaScript."""

modelfile = f"""FROM {CONFIG['output_dir']}/lora_adapter

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE \"\"\"
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"
"""

with open("Modelfile", "w") as f:
    f.write(modelfile)

print("\n📋 To create the Ollama model, run:")
print("   ollama create morphix-qwen-25 -f Modelfile")
print("\n📋 To test the model, run:")
print("   ollama run morphix-qwen-25 '{\"action\":\"add_sphere\",\"radius\":0.5,\"color\":\"0xFF0000\"}'")