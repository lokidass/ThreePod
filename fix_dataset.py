# fix_dataset.py
# Repairs two systematic bugs in the Qwen training data:
#   1. Python booleans leaking into the generated JavaScript:
#        castShadow = True  ->  castShadow = true
#        transparent: False ->  transparent: false
#      (~3000 occurrences; every mesh sets castShadow = True). Invalid JS at runtime.
#   2. Mojibake in the `system` field (em-dash rendered as U+FFFD). We replace the
#      system prompt wholesale with a clean canonical version so every example is
#      byte-identical and valid UTF-8.
#
# Output: qwen_train_clean.jsonl / qwen_val_clean.jsonl (originals untouched).

import json
import re
import os

# Canonical, clean system prompt (matches advanced_finetune.py, proper em-dashes).
CANON_SYSTEM = (
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

# Standalone Python literals -> JS. \b ensures we never touch identifiers like isTrue.
_BOOL_RE = re.compile(r"\b(True|False|None)\b")
_MAP = {"True": "true", "False": "false", "None": "null"}


def fix_js(code: str):
    n = len(_BOOL_RE.findall(code))
    return _BOOL_RE.sub(lambda m: _MAP[m.group(1)], code), n


def clean_file(src: str, dst: str):
    if not os.path.exists(src):
        print(f"  SKIP (missing): {src}")
        return
    total_lines = kept = fixes = sys_fixed = 0
    with open(src, "r", encoding="utf-8", errors="replace") as f_in, \
         open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"  ! unparseable line skipped: {e}")
                continue

            # 1) canonical system prompt
            if item.get("system") != CANON_SYSTEM:
                sys_fixed += 1
            item["system"] = CANON_SYSTEM

            # 2) fix booleans in assistant turns only (user turns are action JSON)
            for msg in item.get("conversations", []):
                if msg.get("from") == "assistant":
                    msg["value"], n = fix_js(msg["value"])
                    fixes += n

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")
    print(f"    lines: {kept}/{total_lines} kept | boolean fixes: {fixes} | system normalized: {sys_fixed}")


if __name__ == "__main__":
    print("Cleaning Qwen dataset (booleans + system prompt)...")
    clean_file("qwen_train.jsonl", "qwen_train_clean.jsonl")
    clean_file("qwen_val.jsonl", "qwen_val_clean.jsonl")

    # Verify no standalone Python literals remain in the cleaned assistant code.
    print("\nVerification (should all be 0):")
    for path in ("qwen_train_clean.jsonl", "qwen_val_clean.jsonl"):
        if not os.path.exists(path):
            continue
        bad = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for msg in item.get("conversations", []):
                    if msg.get("from") == "assistant":
                        bad += len(_BOOL_RE.findall(msg["value"]))
        print(f"  {path}: residual Python literals in assistant code = {bad}")
    print("\nDone.")
