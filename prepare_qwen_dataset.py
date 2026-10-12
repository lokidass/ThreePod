# prepare_qwen_dataset.py
# Prepare dataset specifically for Qwen format

import json
import os

def prepare_qwen_dataset(input_file="morphix_train.jsonl", output_file="qwen_train.jsonl", max_samples=None):
    """
    Convert dataset to Qwen chat format
    Qwen uses: [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
    """
    
    if not os.path.exists(input_file):
        print(f"[ERROR] {input_file} not found. Please run generate_dataset.py first.")
        return
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                conversations = item['conversations']
                
                # Extract system prompt separately
                system_prompt = ""
                messages = []
                
                for msg in conversations:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    elif msg['role'] == 'user':
                        messages.append({"from": "user", "value": msg['content']})
                    elif msg['role'] == 'assistant':
                        messages.append({"from": "assistant", "value": msg['content']})
                
                # Qwen format expects this structure
                converted_data.append({
                    "conversations": messages,
                    "system": system_prompt
                })
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Save in Qwen format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"[OK] Converted {len(converted_data)} samples to {output_file}")
    return converted_data

if __name__ == "__main__":
    # Convert full dataset
    prepare_qwen_dataset("morphix_train.jsonl", "qwen_train.jsonl", max_samples=None)
    prepare_qwen_dataset("morphix_val.jsonl", "qwen_val.jsonl", max_samples=None)
    prepare_qwen_dataset("morphix_test.jsonl", "qwen_test.jsonl", max_samples=None)