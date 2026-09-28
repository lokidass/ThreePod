import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B", trust_remote_code=True)

print("=== special tokens ===")
print("eos_token     :", repr(tok.eos_token), "id=", tok.eos_token_id)
print("pad_token     :", repr(tok.pad_token), "id=", tok.pad_token_id)
for t in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
    print(f"convert_tokens_to_ids({t!r}) =", tok.convert_tokens_to_ids(t))

print("\n=== does <|im_end|> stay a single token when glued to code? ===")
comp = "scene.add(mesh_a1);<|im_end|>"
ids = tok(comp, add_special_tokens=False)["input_ids"]
print("tail ids  :", ids[-6:])
print("tail toks :", tok.convert_ids_to_tokens(ids[-6:]))
IMEND = tok.convert_tokens_to_ids("<|im_end|>")
print("final id == <|im_end|> id ?", ids[-1] == IMEND)

print("\n=== truncation audit over qwen_train_clean.jsonl @ max_len=1024 ===")
MAXLEN = 1024
n = trunc = lost_final_imend = 0
lengths = []
with open("qwen_train_clean.jsonl", "r", encoding="utf-8") as f:
    for line in f:
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
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(completion, add_special_tokens=False)["input_ids"]
        full = len(p) + len(c)
        lengths.append(full)
        input_ids = (p + c)[:MAXLEN]
        n += 1
        if full > MAXLEN:
            trunc += 1
            if input_ids[-1] != IMEND:
                lost_final_imend += 1

lengths.sort()
def pct(q):
    return lengths[min(len(lengths) - 1, int(q * len(lengths)))]
print(f"examples                : {n}")
print(f"len p50/p90/p95/p99/max : {pct(.5)}/{pct(.9)}/{pct(.95)}/{pct(.99)}/{lengths[-1]}")
print(f"truncated (> {MAXLEN})     : {trunc}  ({100*trunc/n:.1f}%)")
print(f"  -> lost final <|im_end|>: {lost_final_imend}  (these teach 'do not stop')")
