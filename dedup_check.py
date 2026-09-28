# dedup_check.py  -- pure-python, no model load; instant.
# Reports: exact duplicates (full pair / user / assistant), contradictions,
# action-type distribution, and STRUCTURAL template diversity (how many distinct
# code shapes hide behind the randomized numbers/hex/labels).

import json, re
from collections import Counter, defaultdict

PATH = "qwen_train_clean.jsonl"

rows = []
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        u = a = None
        for m in d["conversations"]:
            if m["from"] == "user":
                u = m["value"]
            elif m["from"] == "assistant":
                a = m["value"]
        if u is not None and a is not None:
            rows.append((u, a))

n = len(rows)
users = [u for u, a in rows]
comps = [a for u, a in rows]
pairs = [u + "\x00" + a for u, a in rows]


def report(name, items):
    c = Counter(items)
    dups = sum(v - 1 for v in c.values() if v > 1)
    print(f"  {name:28s}: {n} total | {len(c)} unique | {dups} dup rows ({100*dups/n:.1f}%)")
    return c


print(f"loaded {n} examples from {PATH}\n")
print("=== exact duplicates ===")
report("full record (user+asst)", pairs)
cu = report("user prompt only", users)
report("assistant only", comps)

# same input -> different output (label collisions aside)
u2a = defaultdict(set)
for u, a in rows:
    u2a[u].add(a)
contra = {u: s for u, s in u2a.items() if len(s) > 1}
print(f"\n  user prompts w/ >1 distinct completion (contradictions): {len(contra)}")

print("\n=== action distribution ===")
acts = Counter()
for u in users:
    try:
        acts[json.loads(u).get("action", "?")] += 1
    except Exception:
        acts["<unparseable>"] += 1
for k, v in acts.most_common():
    print(f"  {k:22s} {v:5d}  ({100*v/n:.1f}%)")


def norm(a):
    a = re.sub(r'0x[0-9a-fA-F]+', 'HEX', a)
    a = re.sub(r'-?\d+\.\d+', 'N', a)
    a = re.sub(r'-?\d+', 'N', a)
    a = re.sub(r'_[0-9a-f]{6}\b', '_ID', a)   # labels like _51e67e
    a = re.sub(r'_[a-z]+\d+\b', '_ID', a)     # labels like _a1 / _ball1
    return a


cn = Counter(norm(a) for a in comps)
print(f"\n=== structural template diversity (numbers/hex/labels normalized) ===")
print(f"  distinct completion shapes: {len(cn)}  (out of {n} examples)")
print("  top shapes:")
for shape, cnt in cn.most_common(12):
    head = shape.split("\n")[0][:64]
    print(f"    {cnt:5d}  {head}")
