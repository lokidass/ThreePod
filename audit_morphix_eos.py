# audit_morphix_eos.py
# STEP 1 — Audit every .jsonl dataset in the project for the EOS/stop-token situation.
#
# Reports, per file:
#   - schema detected  ({role,content}  vs  {from,value} + system)
#   - number of samples
#   - how many assistant turns literally end with "<|im_end|>"
#   - how many do NOT (candidates for the "missing stop token" problem)
#   - whether any assistant value already contains an embedded <|im_end|> mid-text
#   - JSON parse failures / mojibake (U+FFFD) counts
#
# Read-only. Prints a table. Does NOT modify anything.

import json
import glob
import os

IM_END = "<|im_end|>"
REPL = "�"  # mojibake replacement char, e.g. broken em-dash


def assistant_turns(obj):
    """Yield assistant text from either schema."""
    convs = obj.get("conversations")
    if not isinstance(convs, list):
        return
    for m in convs:
        if not isinstance(m, dict):
            continue
        role = m.get("from") or m.get("role")
        val = m.get("value") if "value" in m else m.get("content")
        if role == "assistant" and isinstance(val, str):
            yield val


def detect_schema(obj):
    convs = obj.get("conversations", [])
    if convs and isinstance(convs[0], dict):
        if "from" in convs[0]:
            return "from/value" + ("+system" if "system" in obj else "")
        if "role" in convs[0]:
            return "role/content"
    return "unknown"


def audit(path):
    n = 0
    n_asst = 0
    ends_imend = 0
    missing_imend = 0
    embedded_mid = 0
    parse_fail = 0
    mojibake = 0
    schema = "?"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                parse_fail += 1
                continue
            if schema == "?":
                schema = detect_schema(obj)
            if REPL in line:
                mojibake += 1
            for val in assistant_turns(obj):
                n_asst += 1
                stripped = val.rstrip()
                if stripped.endswith(IM_END):
                    ends_imend += 1
                else:
                    missing_imend += 1
                # embedded but not at the very end
                if IM_END in stripped[:-len(IM_END)] if len(stripped) > len(IM_END) else False:
                    embedded_mid += 1
    return {
        "file": os.path.basename(path),
        "schema": schema,
        "samples": n,
        "asst_turns": n_asst,
        "ends_imend": ends_imend,
        "missing_imend": missing_imend,
        "embedded_mid": embedded_mid,
        "parse_fail": parse_fail,
        "mojibake_lines": mojibake,
    }


def main():
    files = sorted(glob.glob("*.jsonl"))
    if not files:
        print("No .jsonl files found in", os.getcwd())
        return
    rows = [audit(p) for p in files]
    hdr = f"{'file':30s} {'schema':18s} {'samples':>8s} {'asst':>7s} {'ends<im_end>':>13s} {'MISSING':>9s} {'mojibake':>9s} {'parsefail':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:30s} {r['schema']:18s} {r['samples']:8d} {r['asst_turns']:7d} "
              f"{r['ends_imend']:13d} {r['missing_imend']:9d} {r['mojibake_lines']:9d} {r['parse_fail']:9d}")
    print("-" * len(hdr))
    tot_samples = sum(r["samples"] for r in rows)
    tot_missing = sum(r["missing_imend"] for r in rows)
    print(f"\nTOTAL samples across all files : {tot_samples}")
    print(f"TOTAL assistant turns missing <|im_end|> : {tot_missing}")
    print("\nNOTE: train_qwen7b_qlora.py appends <|im_end|> to the completion at train time,")
    print("so 'missing' in the *.clean/qwen files is EXPECTED for that trainer, not a bug per se.")


if __name__ == "__main__":
    main()
