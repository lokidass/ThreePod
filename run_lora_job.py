import json
import time
import urllib.request

BASE = "http://127.0.0.1:5000"


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


job = post("/api/jobs", {"prompt": "a red sphere", "builder": "lora",
                          "incremental": True})
jid = job["job_id"]
print("job:", jid, flush=True)
seen = 0
while True:
    time.sleep(15)
    snap = get(f"/api/jobs/{jid}/events")
    evs = snap.get("events", snap) if isinstance(snap, dict) else snap
    for ev in evs[seen:]:
        t = ev.get("type", "?")
        msg = (ev.get("message") or ev.get("action")
               or ev.get("status") or "")
        print(f"[{t}] {str(msg)[:130]}", flush=True)
    seen = len(evs)
    st = get(f"/api/jobs/{jid}")
    if st.get("done"):
        res = st.get("result") or {}
        rep = res.get("report", {})
        print("DONE ok:", rep.get("ok"),
              "| statuses:", rep.get("block_statuses"))
        print("think:", rep.get("think_status"))
        print("error:", st.get("error"))
        break
