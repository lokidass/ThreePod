import json
import urllib.error
import urllib.request


def probe(opts, label):
    payload = json.dumps({
        "model": "deepseek-r1:1.5b", "stream": False,
        "prompt": "Say OK and nothing else.",
        "options": opts,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        print(f"{label}: OK {d.get('response', '')[:80]!r}")
    except urllib.error.HTTPError as e:
        print(f"{label}: HTTP", e.code, e.read().decode()[:200])


probe({"temperature": 0.0, "num_ctx": 512}, "default-gpu")
probe({"temperature": 0.0, "num_ctx": 512, "num_gpu": 0}, "cpu-pinned")
